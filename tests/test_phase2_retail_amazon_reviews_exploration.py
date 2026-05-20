import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reviews_row_count"] == 2
    assert report["metadata_row_count"] == 1


def test_docs_include_eda_and_controlled_sampling() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "controlled sampling" in docs
    assert "plots" in docs
    assert "word views" in docs
    assert "no RAG" in docs
    assert "2A-6B" in docs
