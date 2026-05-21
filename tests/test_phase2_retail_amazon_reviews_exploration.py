import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/explore_retail_amazon_reviews.py"
SOURCE_PLAN_PATH = ROOT / "data/sources/retail_amazon_reviews_source_plan.json"
REVIEW_SCHEMA_SAMPLE_PATH = (
    ROOT / "data/real_world_samples/retail_amazon_reviews_schema_sample.jsonl"
)
METADATA_SCHEMA_SAMPLE_PATH = (
    ROOT / "data/real_world_samples/retail_amazon_metadata_schema_sample.jsonl"
)
DOC_PATH = ROOT / "docs/36_phase2_retail_amazon_reviews_exploration.md"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("explore_retail_amazon_reviews", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def test_retail_source_plan_exists_and_parses() -> None:
    plan = json.loads(SOURCE_PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["do_not_download_full_dataset"] is True
    assert plan["controlled_sampling_required"] is True
    assert plan["target_exploration_sample_size"] == 1000
    assert {"rating", "title", "text", "asin", "parent_asin"}.issubset(
        set(plan["expected_review_fields"])
    )
    assert {"main_category", "average_rating", "details", "parent_asin"}.issubset(
        set(plan["expected_metadata_fields"])
    )


def test_schema_samples_exist_and_are_marked_examples() -> None:
    if not REVIEW_SCHEMA_SAMPLE_PATH.exists() or not METADATA_SCHEMA_SAMPLE_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--write-schema-samples"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    rows = _read_jsonl(REVIEW_SCHEMA_SAMPLE_PATH) + _read_jsonl(METADATA_SCHEMA_SAMPLE_PATH)
    assert rows
    for row in rows:
        assert row["source_type"] == "schema_example"
        assert row["not_real_customer_data"] is True
        assert row["not_for_benchmark_claims"] is True


def test_profile_fields() -> None:
    module = _load_module()
    rows = [
        {"asin": "A1", "rating": 5, "text": "Great product", "verified_purchase": True},
        {"asin": "A2", "rating": "4", "title": "Good", "verified_purchase": False},
    ]

    profile = module.profile_fields(rows)

    assert profile["row_count"] == 2
    assert {"asin", "rating", "text", "title", "verified_purchase"}.issubset(
        set(profile["fields_seen"])
    )
    assert profile["missing_counts"]["text"] == 1
    assert profile["type_counts_by_field"]["rating"]["int"] == 1
    assert profile["type_counts_by_field"]["rating"]["str"] == 1
    assert profile["example_values"]["asin"] == "A1"
    assert profile["unique_counts"]["asin"] == 2


def test_profile_text_fields() -> None:
    module = _load_module()
    rows = [
        {
            "rating": 1,
            "title": "Broken charger",
            "text": "The charger arrived broken and damaged. I want a refund.",
            "verified_purchase": True,
            "helpful_vote": 3,
        },
        {
            "rating": 5,
            "title": "Works well",
            "text": "Works well and I recommend the product quality.",
            "verified_purchase": False,
            "helpful_vote": 0,
        },
    ]

    profile = module.profile_text_fields(rows)

    assert profile["text_field_present_count"] == 2
    assert profile["empty_text_count"] == 0
    assert profile["rating_distribution"] == {"1": 1, "5": 1}
    assert profile["helpful_vote_summary"]["max"] == 3.0
    assert any(item["term"] == "charger" for item in profile["top_unigrams"])
    issue_terms = {item["term"]: item["count"] for item in profile["frequent_issue_terms"]}
    assert issue_terms["broken"] == 1
    assert issue_terms["refund"] == 1


def test_profile_quality() -> None:
    module = _load_module()
    rows = [
        {
            "asin": "A1",
            "user_id": "u1",
            "timestamp": 1,
            "parent_asin": "P1",
            "rating": 6,
            "text": "bad",
        },
        {
            "asin": "A1",
            "user_id": "u1",
            "timestamp": 1,
            "parent_asin": "",
            "rating": None,
            "text": "<b>bad</b> email test@example.com",
        },
    ]

    quality = module.profile_quality(rows)

    assert quality["duplicate_review_key_count"] == 1
    assert quality["missing_parent_asin_count"] == 1
    assert quality["missing_rating_count"] == 1
    assert quality["invalid_rating_count"] == 2
    assert quality["very_short_review_count"] == 2
    assert quality["pii_like_pattern_count"] == 1
    assert quality["html_like_text_count"] == 1


def test_sanitize_review_row_hashes_user_id() -> None:
    module = _load_module()
    row = {
        "rating": 5,
        "title": "Works",
        "text": "Works well",
        "images": [],
        "asin": "A1",
        "parent_asin": "P1",
        "user_id": "raw_user_123",
        "timestamp": 1,
        "verified_purchase": True,
        "helpful_vote": 2,
    }

    sanitized = module.sanitize_review_row(row)

    assert "user_id" not in sanitized
    assert sanitized["user_id_hash"]
    assert sanitized["user_id_hash"] != "raw_user_123"
    assert sanitized["source_type"] == "real_sample"
    assert sanitized["sanitized"] is True
    assert sanitized["asin"] == "A1"


def test_sanitize_metadata_row_preserves_expected_fields() -> None:
    module = _load_module()
    row = {
        "main_category": "All_Beauty",
        "title": "Example Product",
        "average_rating": 4.5,
        "rating_number": 10,
        "features": ["feature one"],
        "description": ["description"],
        "price": "10.00",
        "images": [],
        "videos": [],
        "store": "Example Store",
        "categories": ["All_Beauty"],
        "details": {"Color": "Blue"},
        "parent_asin": "P1",
        "bought_together": [],
    }

    sanitized = module.sanitize_metadata_row(row)

    assert sanitized["main_category"] == "All_Beauty"
    assert sanitized["title"] == "Example Product"
    assert sanitized["features"] == ["feature one"]
    assert sanitized["source_type"] == "real_sample"
    assert sanitized["sanitized"] is True


def test_build_hf_parquet_filenames_all_beauty() -> None:
    module = _load_module()

    filenames = module.build_hf_parquet_filenames("All_Beauty")

    assert filenames["reviews_file"] == "raw_review_All_Beauty/full-00000-of-00001.parquet"
    assert filenames["metadata_file"] == "raw_meta_All_Beauty/full-00000-of-00001.parquet"


def test_read_parquet_limited_with_mock_pyarrow() -> None:
    module = _load_module()
    captured: dict[str, Any] = {}

    class FakeBatch:
        def to_pylist(self) -> list[dict[str, Any]]:
            return [{"row": 1}, {"row": 2}, {"row": 3}]

    class FakeParquetFile:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def iter_batches(self, batch_size: int) -> list[FakeBatch]:
            captured["batch_size"] = batch_size
            return [FakeBatch()]

    class FakePyArrowParquet:
        ParquetFile = FakeParquetFile

    original_import_module = module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        if name == "pyarrow.parquet":
            return FakePyArrowParquet
        return original_import_module(name)

    with mock.patch.object(module.importlib, "import_module", side_effect=fake_import_module):
        rows = module.read_parquet_limited(Path("reviews.parquet"), 2)

    assert rows == [{"row": 1}, {"row": 2}]
    assert captured["path"] == Path("reviews.parquet")
    assert captured["batch_size"] == 2


def _load_from_huggingface_args(
    module: Any,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> Any:
    args = [
        "--load-from-huggingface",
        "--sample-limit",
        "1",
        "--metadata-limit",
        "1",
        "--output-reviews-sample",
        str(tmp_path / "reviews.jsonl"),
        "--output-metadata-sample",
        str(tmp_path / "metadata.jsonl"),
        "--output-report",
        str(tmp_path / "report.json"),
        "--field-profile-output",
        str(tmp_path / "field.json"),
        "--text-profile-output",
        str(tmp_path / "text.json"),
        "--quality-report-output",
        str(tmp_path / "quality.json"),
        "--plots-dir",
        str(tmp_path / "plots"),
        "--word-views-dir",
        str(tmp_path / "word_views"),
    ]
    if extra_args:
        args.extend(extra_args)
    return module.build_parser().parse_args(args)


def _run_mocked_parquet_load(
    module: Any,
    tmp_path: Path,
    extra_args: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    args = _load_from_huggingface_args(module, tmp_path, extra_args)
    downloaded_files: list[str] = []

    def fake_download(_repo_id: str, filename: str) -> Path:
        downloaded_files.append(filename)
        return tmp_path / filename.replace("/", "_")

    def fake_read(path: Path, limit: int) -> list[dict[str, Any]]:
        if "raw_review" in path.name or "reviews" in path.name:
            rows = [
                {
                    "rating": 5,
                    "title": "Works",
                    "text": "Works well for the sampled product.",
                    "asin": "A1",
                    "parent_asin": "P1",
                    "user_id": "raw_user",
                    "timestamp": 1,
                    "verified_purchase": True,
                    "helpful_vote": 0,
                }
            ]
        else:
            rows = [
                {
                    "main_category": "All_Beauty",
                    "title": "Sample Product",
                    "parent_asin": "P1",
                    "features": ["feature"],
                }
            ]
        return rows[:limit]

    with (
        mock.patch.object(module, "download_hf_dataset_file", side_effect=fake_download),
        mock.patch.object(module, "read_parquet_limited", side_effect=fake_read),
        mock.patch.object(
            module,
            "_import_hf_load_dataset",
            side_effect=AssertionError("datasets loader should not be used"),
        ),
    ):
        summary = module.run_huggingface_load(args)

    return summary, downloaded_files


def test_hf_loader_missing_datasets_graceful() -> None:
    module = _load_module()

    with mock.patch.object(module.importlib, "import_module", side_effect=ImportError):
        try:
            module._import_hf_load_dataset()
        except RuntimeError as exc:
            assert "Install datasets to use --use-datasets-loader" in str(exc)
            assert "--explore-local" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")


def test_load_from_huggingface_uses_direct_parquet_by_default(tmp_path: Path) -> None:
    module = _load_module()

    summary, downloaded_files = _run_mocked_parquet_load(module, tmp_path)
    rows = _read_jsonl(tmp_path / "reviews.jsonl")

    assert summary["loader"] == "direct_parquet"
    assert summary["hf_repo_id"] == "McAuley-Lab/Amazon-Reviews-2023"
    assert summary["reviews_sample_count"] == 1
    assert summary["metadata_sample_count"] == 1
    assert downloaded_files == [
        "raw_review_All_Beauty/full-00000-of-00001.parquet",
        "raw_meta_All_Beauty/full-00000-of-00001.parquet",
    ]
    assert "user_id" not in rows[0]
    assert rows[0]["user_id_hash"]


def test_reviews_file_override(tmp_path: Path) -> None:
    module = _load_module()

    summary, downloaded_files = _run_mocked_parquet_load(
        module,
        tmp_path,
        ["--reviews-file", "custom/reviews.parquet"],
    )

    assert summary["reviews_file"] == "custom/reviews.parquet"
    assert downloaded_files[0] == "custom/reviews.parquet"


def test_metadata_file_override(tmp_path: Path) -> None:
    module = _load_module()

    summary, downloaded_files = _run_mocked_parquet_load(
        module,
        tmp_path,
        ["--metadata-file", "custom/metadata.parquet"],
    )

    assert summary["metadata_file"] == "custom/metadata.parquet"
    assert downloaded_files[1] == "custom/metadata.parquet"


def test_dry_run_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "dry_run"
    assert summary["sampling_plan"]["controlled_sampling_required"] is True


def test_write_schema_samples_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--write-schema-samples"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert REVIEW_SCHEMA_SAMPLE_PATH.exists()
    assert METADATA_SCHEMA_SAMPLE_PATH.exists()


def test_explore_local_with_tmp_files(tmp_path: Path) -> None:
    reviews_path = tmp_path / "reviews.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    report_path = tmp_path / "report.json"
    field_profile_path = tmp_path / "field_profile.json"
    text_profile_path = tmp_path / "text_profile.json"
    quality_path = tmp_path / "quality.json"
    plots_dir = tmp_path / "plots"
    word_views_dir = tmp_path / "word_views"

    reviews = [
        {
            "rating": 5,
            "title": "Works well",
            "text": "Works well and I recommend it for quality.",
            "asin": "A1",
            "parent_asin": "P1",
            "user_id": "u1",
            "timestamp": 1,
            "verified_purchase": True,
            "helpful_vote": 1,
        },
        {
            "rating": 1,
            "title": "Broken",
            "text": "The item arrived broken and damaged.",
            "asin": "A2",
            "parent_asin": "P2",
            "user_id": "u2",
            "timestamp": 2,
            "verified_purchase": False,
            "helpful_vote": 0,
        },
    ]
    metadata = [{"main_category": "All_Beauty", "title": "Example", "parent_asin": "P1"}]
    reviews_path.write_text(
        "".join(json.dumps(row) + "\n" for row in reviews),
        encoding="utf-8",
    )
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--explore-local",
            "--reviews-input",
            str(reviews_path),
            "--metadata-input",
            str(metadata_path),
            "--output-report",
            str(report_path),
            "--field-profile-output",
            str(field_profile_path),
            "--text-profile-output",
            str(text_profile_path),
            "--quality-report-output",
            str(quality_path),
            "--plots-dir",
            str(plots_dir),
            "--word-views-dir",
            str(word_views_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.exists()
    assert field_profile_path.exists()
    assert text_profile_path.exists()
    assert quality_path.exists()
    assert (word_views_dir / "top_unigrams.txt").exists()
    assert (word_views_dir / "metadata_product_titles_preview.txt").exists()
    assert (word_views_dir / "metadata_features_preview.txt").exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reviews_row_count"] == 2
    assert report["metadata_row_count"] == 1


def test_explore_local_outputs_word_views(tmp_path: Path) -> None:
    reviews_path = tmp_path / "reviews.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    output_paths = {
        "report": tmp_path / "report.json",
        "field": tmp_path / "field.json",
        "text": tmp_path / "text.json",
        "quality": tmp_path / "quality.json",
        "plots": tmp_path / "plots",
        "words": tmp_path / "word_views",
    }
    reviews_path.write_text(
        json.dumps(
            {
                "rating": 1,
                "title": "Broken item",
                "text": "The item arrived broken and damaged.",
                "asin": "A1",
                "parent_asin": "P1",
                "user_id_hash": "hash1",
                "timestamp": 1,
                "verified_purchase": True,
                "helpful_vote": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "main_category": "All_Beauty",
                "title": "Example Product",
                "features": ["Feature A", "Feature B"],
                "parent_asin": "P1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--explore-local",
            "--reviews-input",
            str(reviews_path),
            "--metadata-input",
            str(metadata_path),
            "--output-report",
            str(output_paths["report"]),
            "--field-profile-output",
            str(output_paths["field"]),
            "--text-profile-output",
            str(output_paths["text"]),
            "--quality-report-output",
            str(output_paths["quality"]),
            "--plots-dir",
            str(output_paths["plots"]),
            "--word-views-dir",
            str(output_paths["words"]),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for filename in [
        "top_unigrams.txt",
        "top_bigrams.txt",
        "issue_terms.txt",
        "low_rating_issue_preview.txt",
        "high_rating_positive_preview.txt",
        "longest_reviews_preview.txt",
        "shortest_reviews_preview.txt",
        "metadata_product_titles_preview.txt",
        "metadata_features_preview.txt",
    ]:
        assert (output_paths["words"] / filename).exists()


def test_quality_report_counts_sanitized_user_id() -> None:
    module = _load_module()
    rows = [
        {
            "asin": "A1",
            "user_id_hash": "hash1",
            "timestamp": 1,
            "parent_asin": "P1",
            "rating": 5,
            "title": "Works",
            "text": "This product works well.",
        }
    ]

    quality = module.profile_quality(rows)

    assert quality["raw_user_id_present_count"] == 0
    assert quality["sanitized_user_id_hash_present_count"] == 1
    assert quality["missing_text_count"] == 0
    assert quality["missing_title_count"] == 0


def test_docs_include_eda_and_controlled_sampling() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "controlled sampling" in docs
    assert "plots" in docs
    assert "word views" in docs
    assert "no RAG" in docs
    assert "2A-6B" in docs


def test_docs_include_2a6b_loader() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-6B" in docs
    assert "--load-from-huggingface" in docs
    assert "user_id_hash" in docs
    assert "controlled real-data loading" in docs.lower()


def test_docs_include_direct_parquet_loading() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Direct Parquet Loading" in docs
    assert "huggingface_hub" in docs
    assert "pyarrow" in docs
    assert "Dataset scripts are no longer supported" in docs
