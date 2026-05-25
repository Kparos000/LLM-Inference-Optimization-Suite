import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from inference_bench.context_corpora import (
    CHUNK_BUILDERS,
    VERTICALS,
    build_context_corpora,
    build_corpus_registry,
    read_jsonl,
)
from inference_bench.context_schema import ContextRecord

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def fixture_rows(vertical: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if vertical == "airline":
        kb = [
            {
                "allowed_to_commit": True,
                "body": "Canada Air policy allows eligible 24-hour cancellation refunds.",
                "doc_id": "CA-POL-001",
                "document_type": "policy",
                "metadata": {"base_doc_id": "CA-POL-001", "fictional_airline": "Canada Air"},
                "source_type": "synthetic_public_inspired",
                "tags": ["airline", "refund"],
                "title": "Cancellation Policy",
                "vertical": "airline",
            }
        ]
    elif vertical == "healthcare_admin":
        kb = [
            {
                "allowed_to_commit": True,
                "body": (
                    "MapleCare scheduling staff collect visit reason and avoid clinical triage."
                ),
                "doc_id": "MCH-POL-001",
                "document_type": "procedure",
                "metadata": {
                    "base_doc_id": "MCH-POL-001",
                    "fictional_provider": "MapleCare Health",
                },
                "source_type": "synthetic_public_inspired",
                "tags": ["healthcare-admin", "scheduling", "triage"],
                "title": "Scheduling Procedure",
                "vertical": "healthcare_admin",
            }
        ]
    elif vertical == "retail":
        kb = [
            {
                "allowed_to_commit": True,
                "body": "Review evidence for a towel. Rating 5.0. Issue signals: size and texture.",
                "doc_id": "retail_review_001",
                "document_type": "review_evidence",
                "metadata": {
                    "category": "Home_and_Kitchen",
                    "parent_asin": "B000TEST",
                    "product_title": "Test Towel",
                    "rating": 5.0,
                },
                "source_type": "derived",
                "tags": ["retail", "review"],
                "title": "Test Towel Review",
                "vertical": "retail",
            }
        ]
    elif vertical == "finance":
        kb = [
            {
                "allowed_to_commit": True,
                "body": "Curated XBRL fact evidence for Test Corp: Revenue = 100 USD.",
                "doc_id": "finance_kb_xbrl_TEST_revenue",
                "document_type": "xbrl_fact_evidence",
                "metadata": {
                    "company_name": "Test Corp",
                    "concept": "Revenue",
                    "ticker": "TEST",
                },
                "source_type": "derived",
                "tags": ["finance", "xbrl"],
                "title": "TEST XBRL Revenue",
                "vertical": "finance",
            }
        ]
    elif vertical == "research_ai":
        kb = [
            {
                "allowed_to_commit": True,
                "body": (
                    "ABSTRACT This paper studies efficient long-context inference for LLM serving."
                ),
                "doc_id": "research_ai_kb_001_abstract",
                "document_type": "paper_section",
                "metadata": {
                    "paper_id": "paper_test_001",
                    "section_title": "Abstract",
                    "section_type": "abstract",
                    "title": "Efficient Long-Context Inference",
                },
                "source_type": "derived",
                "tags": ["research_ai", "abstract"],
                "title": "Efficient Long-Context Inference - Abstract",
                "vertical": "research_ai",
            }
        ]
    else:
        raise AssertionError(f"Unexpected vertical: {vertical}")

    doc_id = str(kb[0]["doc_id"])
    gold = [
        {
            "prompt_id": f"{vertical}_fixture_001",
            "reference_answer": "Use the cited evidence.",
            "required_doc_ids": [doc_id],
            "required_evidence_ids": [doc_id],
            "required_chunk_ids": [doc_id],
            "must_include": [doc_id],
            "must_not_include": ["unsupported claim"],
            "task_type": "answer_grounded",
            "vertical": vertical,
        }
    ]
    return kb, gold


def make_fixture_dataset(root: Path) -> Path:
    for vertical in VERTICALS:
        kb, gold = fixture_rows(vertical)
        prompts = [
            {
                "prompt_id": f"{vertical}_fixture_001",
                "question": "Answer using the cited evidence.",
                "expected_output_format": "text",
                "vertical": vertical,
            }
        ]
        write_jsonl(root / vertical / f"{vertical}_kb_2000.jsonl", kb)
        write_jsonl(root / vertical / f"{vertical}_gold_2000.jsonl", gold)
        write_jsonl(root / vertical / f"{vertical}_prompts_2000.jsonl", prompts)
    return root


def test_corpus_registry_loads(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    registry = build_corpus_registry(dataset_root, tmp_path / "out")

    assert registry["entries"]
    assert registry["dataset_root"] == str(dataset_root)


def test_each_vertical_has_registered_corpus(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    registry = build_corpus_registry(dataset_root, tmp_path / "out")
    benchmark_entries = [
        entry for entry in registry["entries"] if entry["corpus_role"] == "benchmark_kb"
    ]

    assert {entry["vertical"] for entry in benchmark_entries} == set(VERTICALS)


def test_each_chunk_builder_creates_valid_context_records() -> None:
    for vertical in VERTICALS:
        kb, gold = fixture_rows(vertical)
        referenced_ids = set(gold[0]["required_doc_ids"])
        records = CHUNK_BUILDERS[vertical](kb, referenced_ids)

        assert records
        assert all(isinstance(record, ContextRecord) for record in records)
        assert all(record.vertical == vertical for record in records)


def test_context_ids_are_unique(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    result = build_context_corpora(dataset_root=dataset_root, output_root=tmp_path / "out")

    for vertical_report in result["report"]["by_vertical"].values():
        assert vertical_report["context_ids_unique"] is True


def test_required_fields_exist(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    build_context_corpora(dataset_root=dataset_root, output_root=tmp_path / "out")
    rows = read_jsonl(tmp_path / "out" / "corpora" / "airline_context_corpus.jsonl")

    assert rows
    assert {
        "context_id",
        "vertical",
        "source_id",
        "parent_id",
        "chunk_id",
        "chunk_strategy",
        "source_type",
        "title",
        "text",
        "metadata",
        "token_estimate",
        "provenance",
        "is_gold_linked",
    }.issubset(rows[0])


def test_generated_context_records_validate_against_schema(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    build_context_corpora(dataset_root=dataset_root, output_root=tmp_path / "out")

    for vertical in VERTICALS:
        rows = read_jsonl(tmp_path / "out" / "corpora" / f"{vertical}_context_corpus.jsonl")
        assert rows
        for row in rows:
            ContextRecord(**row)


def test_finance_metadata_warnings_are_explicit_if_fields_are_missing(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    result = build_context_corpora(dataset_root=dataset_root, output_root=tmp_path / "out")
    finance_report = result["report"]["by_vertical"]["finance"]

    assert finance_report["missing_metadata_warnings"]
    assert any("finance" in warning for warning in finance_report["missing_metadata_warnings"])
    assert finance_report["finance_metadata_summary"]["ticker"]["present"] > 0
    assert finance_report["finance_metadata_summary"]["concept"]["present"] > 0


def test_research_ai_section_metadata_is_preserved_when_available(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    result = build_context_corpora(dataset_root=dataset_root, output_root=tmp_path / "out")
    research_report = result["report"]["by_vertical"]["research_ai"]
    rows = read_jsonl(tmp_path / "out" / "corpora" / "research_ai_context_corpus.jsonl")

    assert rows[0]["metadata"]["paper_id"] == "paper_test_001"
    assert rows[0]["metadata"]["section_type"] == "abstract"
    assert (
        "abstract"
        in research_report["research_ai_section_metadata_summary"][
            "preserved_expected_section_types"
        ]
    )


def test_output_jsonl_files_are_valid_in_temp_directory(tmp_path: Path) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    build_context_corpora(dataset_root=dataset_root, output_root=tmp_path / "out")

    for vertical in VERTICALS:
        rows = read_jsonl(tmp_path / "out" / "corpora" / f"{vertical}_context_corpus.jsonl")
        assert len(rows) >= 1


def test_script_can_run_on_small_fixture_without_touching_real_promoted_data(
    tmp_path: Path,
) -> None:
    dataset_root = make_fixture_dataset(tmp_path / "dataset")
    output_root = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "phase3" / "build_context_corpora.py"),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_root / "corpus_registry.json").exists()
    assert (output_root / "corpora" / "finance_context_corpus.jsonl").exists()
