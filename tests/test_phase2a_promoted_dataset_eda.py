import json
import shutil
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

VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
COMMON_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "they",
    "can",
    "should",
    "what",
    "how",
    "using",
    "only",
    "cited",
    "evidence",
    "records",
    "record",
    "scenario",
    "selected",
    "answer",
    "question",
    "request",
    "help",
}

INTERACTIVE_FILES = [
    "inventory_prompts_gold_kb_by_vertical.html",
    "status_distribution_by_vertical.html",
    "output_format_by_vertical.html",
    "task_type_mix_by_vertical.html",
    "prompt_length_boxplot.html",
    "gold_length_boxplot.html",
    "kb_length_boxplot.html",
    "workload_shape_by_vertical.html",
    "evidence_reuse_by_vertical.html",
    "vertical_task_heatmap.html",
    "vertical_status_heatmap.html",
]

STATIC_PLOTS = [
    "inventory_prompts_gold_kb_by_vertical.png",
    "kb_rows_by_vertical.png",
    "prompts_by_vertical.png",
    "gold_by_vertical.png",
    "status_distribution_by_vertical.png",
    "output_format_by_vertical.png",
    "task_type_mix_by_vertical.png",
    "prompt_length_by_vertical.png",
    "gold_length_by_vertical.png",
    "kb_length_by_vertical.png",
    "workload_shape_by_vertical.png",
    "evidence_reuse_by_vertical.png",
]

_EDA_SUMMARY: dict[str, Any] | None = None


def _run_eda() -> dict[str, Any]:
    global _EDA_SUMMARY
    if _EDA_SUMMARY is not None:
        return _EDA_SUMMARY

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    missing_corpus = OUTPUT_DIR / "missing_research_ai_full_sections_corpus.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset-root",
            str(DATASET_ROOT),
            "--write-report",
            "--output-dir",
            str(OUTPUT_DIR),
            "--research-ai-retrieval-corpus",
            str(missing_corpus),
            "--research-ai-retrieval-manifest",
            str(OUTPUT_DIR / "missing_manifest.json"),
            "--research-ai-retrieval-quality-report",
            str(OUTPUT_DIR / "missing_quality.json"),
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
    assert path.exists(), path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_eda_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_eda_cli_on_promoted_10000_dataset() -> None:
    summary = _run_eda()

    assert summary["phase"] == "2A-16R"
    assert summary["total_prompt_count"] == 10000
    assert summary["total_gold_count"] == 10000
    assert summary["total_kb_count"] == 4740
    assert summary["vertical_count"] == 5


def test_eda_creates_dashboard_html() -> None:
    _run_eda()
    dashboard = OUTPUT_DIR / "dashboard/phase2a_10000_overview.html"
    markdown = OUTPUT_DIR / "dashboard/phase2a_10000_overview.md"

    assert dashboard.exists()
    assert "Phase 2A promoted 10,000-record dataset EDA" in dashboard.read_text(encoding="utf-8")
    assert markdown.exists()


def test_eda_creates_interactive_plot_files() -> None:
    _run_eda()

    for filename in INTERACTIVE_FILES:
        path = OUTPUT_DIR / "interactive" / filename
        assert path.exists(), filename
        assert path.stat().st_size > 0, filename


def test_eda_creates_static_plot_files_or_reports_plot_skip() -> None:
    _run_eda()
    skipped = OUTPUT_DIR / "plots/plots_skipped.md"

    if skipped.exists():
        assert "Matplotlib is unavailable" in skipped.read_text(encoding="utf-8")
        return

    for filename in STATIC_PLOTS:
        path = OUTPUT_DIR / "plots" / filename
        assert path.exists(), filename
        assert path.stat().st_size > 0, filename


def test_eda_creates_word_views() -> None:
    _run_eda()

    for vertical in VERTICALS:
        assert (OUTPUT_DIR / "word_views" / f"{vertical}_clean_terms.txt").exists()
        assert (OUTPUT_DIR / "word_views" / f"{vertical}_tfidf_terms.txt").exists()
        assert (OUTPUT_DIR / "word_clouds" / f"{vertical}_wordcloud.png").exists()


def test_cleaned_word_views_do_not_have_common_stopwords_at_top() -> None:
    _run_eda()

    for vertical in VERTICALS:
        path = OUTPUT_DIR / "word_views" / f"{vertical}_clean_terms.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        start = lines.index("### top_clean_unigrams") + 1
        terms: list[str] = []
        for line in lines[start:]:
            if not line.strip():
                break
            terms.append(line.split("\t", maxsplit=1)[0])
        assert not set(terms[:20]) & COMMON_STOPWORDS, vertical


def test_eda_creates_per_vertical_html_pages() -> None:
    _run_eda()

    for vertical in VERTICALS:
        path = OUTPUT_DIR / "verticals" / vertical / f"{vertical}_eda.html"
        assert path.exists(), vertical
        assert "Representative Prompts" in path.read_text(encoding="utf-8")


def test_finance_vertical_folder_exists() -> None:
    _run_eda()

    assert (OUTPUT_DIR / "verticals/finance/finance_eda.html").exists()


def test_research_ai_vertical_folder_exists() -> None:
    _run_eda()

    page = OUTPUT_DIR / "verticals/research_ai/research_ai_eda.html"
    assert page.exists()
    text = page.read_text(encoding="utf-8")
    assert "promoted_benchmark_kb_count" in text
    assert "retrieval corpus" in text.lower()


def test_inventory_counts_match_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = _read_report("phase2a_10000_dataset_inventory.json")

    assert inventory["total_prompt_count"] == manifest["total_prompt_count"]
    assert inventory["total_gold_count"] == manifest["total_gold_count"]
    assert inventory["total_kb_count"] == manifest["total_kb_count"]
    assert inventory["missing_files"] == []
    assert inventory["all_manifest_counts_match"] is True


def test_alignment_report_has_required_fields() -> None:
    report = _read_report("phase2a_alignment_report.json")

    assert "critical_issue_count" in report
    assert "warning_count" in report
    assert "issue_list" in report
    for vertical in VERTICALS:
        row = report["by_vertical"][vertical]
        assert "missing_gold_for_prompts" in row
        assert "orphan_gold_without_prompt" in row
        assert "duplicate_prompt_ids" in row
        assert "duplicate_gold_prompt_ids" in row
        assert "answerable_prompts_without_evidence" in row
        assert "negative_prompts_without_must_not_include" in row
        assert "prompt_gold_output_format_mismatch" in row


def test_evidence_reuse_report_has_required_fields() -> None:
    report = _read_report("phase2a_evidence_reuse_report.json")

    for vertical, row in report["by_vertical"].items():
        assert vertical in VERTICALS
        assert row["gold_count"] == 2000
        assert "evidence_coverage_rate" in row
        assert "average_evidence_ids_per_prompt" in row
        assert "top_20_reused_evidence_ids" in row
        assert "max_evidence_reuse_share" in row
        assert "unused_kb_count" in row
        assert "unused_kb_share" in row
        assert row["evidence_reuse_concentration_label"] in {"low", "medium", "high"}


def test_safety_report_has_required_fields() -> None:
    report = _read_report("phase2a_safety_report.json")

    assert "critical_issue_count" in report
    assert "warning_count" in report
    assert "flag_counts_by_vertical" in report
    assert "issue_list" in report
    for vertical in VERTICALS:
        row = report["by_vertical"][vertical]
        assert "critical_flag_counts" in row
        assert "warning_flag_counts" in row
        assert "safety_flag_count" in row


def test_workload_shape_report_has_required_fields() -> None:
    report = _read_report("phase2a_workload_shape_report.json")

    assert "context_heavy_vertical_ranking" in report
    for vertical in VERTICALS:
        row = report["by_vertical"][vertical]
        assert "estimated_prompt_tokens" in row
        assert "estimated_kb_tokens" in row
        assert "estimated_expected_output_tokens" in row
        assert "total_estimated_input_tokens" in row
        assert "output_format_mix" in row
        assert "single_evidence_prompt_share" in row
        assert "multi_evidence_prompt_share" in row
        assert row["likely_inference_cost_pressure"] in {"low", "medium", "high"}


def test_results_readmes_explain_smoke_test_artifacts() -> None:
    results_readme = (ROOT / "results/README.md").read_text(encoding="utf-8")
    figures_readme = (ROOT / "results/figures/README.md").read_text(encoding="utf-8")

    for text in [results_readme, figures_readme]:
        assert "smoke-test" in text
        assert "optimization" in text
        assert "none" in text
        assert "data/generated/phase2a/eda" in text


def test_eda_docs_include_dashboard_and_word_clouds() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-16R" in text
    assert "phase2a_10000_overview.html" in text
    assert "word_clouds" in text
    assert "word cloud" in text.lower()
    assert "Streamlit" in text
    assert "RAG" in text
    assert "embeddings" in text
    assert "inference" in text
