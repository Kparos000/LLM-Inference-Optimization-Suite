import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/explore_phase2a_promoted_dataset.py"
DOC_PATH = ROOT / "docs/53_phase2a_10000_dataset_eda.md"
DATASET_ROOT = ROOT / "data/scaleup_2000_full"
MANIFEST_PATH = DATASET_ROOT / "phase2a_2000_full_manifest.json"
OUTPUT_DIR = ROOT / "data/generated/phase2a/eda_test_cache"
CORPUS_PATH = OUTPUT_DIR / "research_ai_test_corpus.jsonl"

_EDA_SUMMARY: dict[str, Any] | None = None


def _write_test_corpus() -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "corpus_id": "paper_a_method",
            "paper_id": "paper_a",
            "section_id": "paper_a_method",
            "section_type": "method",
            "text": "A mocked retrieval corpus section for EDA coverage.",
            "word_count": 8,
        },
        {
            "corpus_id": "paper_b_results",
            "paper_id": "paper_b",
            "section_id": "paper_b_results",
            "section_type": "results",
            "text": "A second mocked retrieval corpus section for EDA coverage.",
            "word_count": 9,
        },
    ]
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_eda() -> dict[str, Any]:
    global _EDA_SUMMARY
    if _EDA_SUMMARY is not None:
        return _EDA_SUMMARY

    _write_test_corpus()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset-root",
            str(DATASET_ROOT),
            "--write-report",
            "--output-dir",
            str(OUTPUT_DIR),
            "--no-make-plots",
            "--research-ai-retrieval-corpus",
            str(CORPUS_PATH),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert isinstance(summary, dict)
    _EDA_SUMMARY = summary
    return summary


def _read_report(name: str) -> dict[str, Any]:
    _run_eda()
    path = OUTPUT_DIR / name
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_eda_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_eda_cli_on_promoted_10000_dataset() -> None:
    summary = _run_eda()

    assert summary["phase"] == "2A-16C"
    assert summary["total_prompt_count"] == 10000
    assert summary["total_gold_count"] == 10000
    assert (OUTPUT_DIR / "phase2a_10000_dataset_inventory.json").exists()
    assert (OUTPUT_DIR / "phase2a_10000_dataset_summary.csv").exists()


def test_eda_inventory_counts_match_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = _read_report("phase2a_10000_dataset_inventory.json")

    assert inventory["total_prompt_count"] == manifest["total_prompt_count"]
    assert inventory["total_gold_count"] == manifest["total_gold_count"]
    assert inventory["total_kb_count"] == manifest["total_kb_count"]
    assert inventory["missing_files"] == []


def test_eda_alignment_report_has_required_fields() -> None:
    report = _read_report("phase2a_alignment_report.json")

    assert "critical_issue_count" in report
    assert "alignment_clean" in report
    assert "by_vertical" in report
    for vertical in ["airline", "healthcare_admin", "retail", "finance", "research_ai"]:
        assert "missing_gold_for_prompts" in report["by_vertical"][vertical]
        assert "orphan_gold_records" in report["by_vertical"][vertical]
        assert "answerable_records_without_evidence" in report["by_vertical"][vertical]


def test_eda_evidence_reuse_report_has_required_fields() -> None:
    report = _read_report("phase2a_evidence_reuse_report.json")

    assert "by_vertical" in report
    for vertical, row in report["by_vertical"].items():
        assert row["gold_count"] == 2000, vertical
        assert "evidence_coverage_rate" in row
        assert "top_reused_evidence_ids" in row
        assert "max_evidence_reuse_share" in row
        assert "unused_kb_count" in row


def test_eda_safety_report_has_required_fields() -> None:
    report = _read_report("phase2a_safety_report.json")

    assert "critical_issue_count" in report
    assert "safety_clean" in report
    assert "by_vertical" in report
    for vertical, row in report["by_vertical"].items():
        assert "hygiene_hits" in row, vertical
        assert "domain_flags" in row, vertical
        assert "issue_count" in row, vertical


def test_eda_word_views_created() -> None:
    _run_eda()
    word_view = OUTPUT_DIR / "word_views/finance_prompt_terms.txt"

    assert word_view.exists()
    assert "Top unigrams" in word_view.read_text(encoding="utf-8")


def test_eda_research_ai_reports_full_corpus_if_available() -> None:
    inventory = _read_report("phase2a_10000_dataset_inventory.json")
    research_ai = inventory["vertical_specific"]["research_ai"]

    assert research_ai["full_retrieval_corpus_exists"] is True
    assert research_ai["full_retrieval_corpus_count"] == 2
    assert research_ai["promoted_benchmark_kb_count"] == 1600


def test_eda_docs_include_commands() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-16C" in text
    assert "explore_phase2a_promoted_dataset.py" in text
    assert "--dataset-root data/scaleup_2000_full --write-report" in text
    assert "RAG" in text
    assert "embeddings" in text
    assert "inference" in text
