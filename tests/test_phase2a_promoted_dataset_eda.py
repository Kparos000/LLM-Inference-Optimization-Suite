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
DEFAULT_OUTPUT_DIR = ROOT / "data/generated/dataset_10000"
OUTPUT_DIR = ROOT / "data/generated/dataset_10000_test_cache"
FINANCE_OUTPUT_DIR = ROOT / "data/generated/finance_test_cache"
LEGACY_OUTPUT_DIR = ROOT / "data/generated/phase2a/eda"
OLD_PUBLIC_PARENT_DIR = ROOT / "data/generated/eda"
OLD_PUBLIC_OUTPUT_DIR = ROOT / "data/generated/eda/dataset_10000"
OUTPUT_PREFIX = "dataset_10000_eda"

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
    "them",
    "their",
    "can",
    "could",
    "should",
    "would",
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "using",
    "use",
    "used",
    "only",
    "based",
    "cited",
    "evidence",
    "records",
    "record",
    "scenario",
    "scenarios",
    "selected",
    "answer",
    "question",
    "request",
    "asks",
    "asked",
    "help",
    "provide",
    "explain",
    "summarize",
    "identify",
    "determine",
    "describe",
    "compare",
    "create",
    "extract",
    "generate",
    "return",
    "output",
    "response",
    "support",
    "agent",
    "customer",
    "user",
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
    if FINANCE_OUTPUT_DIR.exists():
        shutil.rmtree(FINANCE_OUTPUT_DIR)

    (OUTPUT_DIR / "dashboard").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "word_views").mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "phase2a_10000_dataset_inventory.json",
        "phase2a_10000_dataset_summary.csv",
        "phase2a_prompt_profile.json",
        "phase2a_kb_profile.json",
        "phase2a_gold_profile.json",
        "phase2a_alignment_report.json",
        "phase2a_evidence_reuse_report.json",
        "phase2a_safety_report.json",
        "phase2a_workload_shape_report.json",
        "phase2a_eda_summary.md",
        "phase2a_vertical_specific_report.json",
    ]:
        (OUTPUT_DIR / stale_name).write_text("stale\n", encoding="utf-8")
    (OUTPUT_DIR / "dashboard/phase2a_10000_overview.html").write_text("stale\n", encoding="utf-8")
    (OUTPUT_DIR / "dashboard/phase2a_10000_overview.md").write_text("stale\n", encoding="utf-8")
    for vertical in VERTICALS:
        (OUTPUT_DIR / "word_views" / f"{vertical}_prompt_terms.txt").write_text(
            "stale\n", encoding="utf-8"
        )

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
            "--finance-output-dir",
            str(FINANCE_OUTPUT_DIR),
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


def test_eda_default_output_dir_is_public_dataset_10000() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'DEFAULT_OUTPUT_DIR = Path("data/generated/dataset_10000")' in text
    assert 'DEFAULT_OUTPUT_DIR = Path("data/generated/eda/dataset_10000")' not in text
    assert 'DEFAULT_OUTPUT_DIR = Path("data/generated/phase2a/eda")' not in text


def test_eda_does_not_write_public_outputs_under_phase2a_by_default() -> None:
    legacy_public_report = LEGACY_OUTPUT_DIR / f"{OUTPUT_PREFIX}_inventory.json"
    before_mtime = (
        legacy_public_report.stat().st_mtime_ns if legacy_public_report.exists() else None
    )

    _run_eda()

    assert "phase2a" not in str(DEFAULT_OUTPUT_DIR).replace("\\", "/")
    if before_mtime is None:
        assert not legacy_public_report.exists()
    else:
        assert legacy_public_report.stat().st_mtime_ns == before_mtime


def test_eda_cli_still_runs_on_promoted_10000_dataset() -> None:
    summary = _run_eda()

    assert summary["phase"] == "2A-16R"
    assert summary["total_prompt_count"] == 10000
    assert summary["total_gold_count"] == 10000
    assert summary["total_kb_count"] == 4740
    assert summary["vertical_count"] == 5
    assert summary["output_dir"].endswith("data\\generated\\dataset_10000_test_cache") or summary[
        "output_dir"
    ].endswith("data/generated/dataset_10000_test_cache")
    assert summary["finance_output_dir"].endswith("data\\generated\\finance_test_cache") or summary[
        "finance_output_dir"
    ].endswith("data/generated/finance_test_cache")


def test_eda_uses_public_facing_10000_record_names() -> None:
    _run_eda()
    dashboard = (OUTPUT_DIR / f"dashboard/{OUTPUT_PREFIX}_overview.html").read_text(
        encoding="utf-8"
    )
    readme = (OUTPUT_DIR / "README.md").read_text(encoding="utf-8")

    assert "10,000-record dataset EDA" in dashboard
    assert "10,000-Record Dataset EDA" in readme
    assert "Phase 2A promoted dataset EDA" not in dashboard
    assert "phase2a_10000" not in readme


def test_dashboard_uses_dataset_10000_eda_name() -> None:
    _run_eda()
    dashboard = OUTPUT_DIR / f"dashboard/{OUTPUT_PREFIX}_overview.html"
    markdown = OUTPUT_DIR / f"dashboard/{OUTPUT_PREFIX}_overview.md"

    assert dashboard.exists()
    assert "10,000-record dataset EDA" in dashboard.read_text(encoding="utf-8")
    assert "dataset_10000_eda" in markdown.read_text(encoding="utf-8")
    assert markdown.exists()


def test_dashboard_renamed_to_dataset_10000_eda_overview() -> None:
    _run_eda()

    assert (OUTPUT_DIR / f"dashboard/{OUTPUT_PREFIX}_overview.html").exists()
    assert (OUTPUT_DIR / f"dashboard/{OUTPUT_PREFIX}_overview.md").exists()
    assert not (OUTPUT_DIR / "dashboard/phase2a_10000_overview.html").exists()
    assert not (OUTPUT_DIR / "dashboard/phase2a_10000_overview.md").exists()


def test_eda_creates_interactive_plot_files() -> None:
    _run_eda()

    for filename in INTERACTIVE_FILES:
        path = OUTPUT_DIR / "interactive" / filename
        assert path.exists(), filename
        assert path.stat().st_size > 0, filename


def test_wordclouds_or_term_visuals_created() -> None:
    _run_eda()

    for vertical in VERTICALS:
        assert (OUTPUT_DIR / "word_clouds" / f"{vertical}_wordcloud.png").exists()
        assert (OUTPUT_DIR / "term_visuals" / f"{vertical}_top_terms_bar.html").exists()
        assert (OUTPUT_DIR / "term_visuals" / f"{vertical}_term_treemap.html").exists()


def test_retail_term_charts_have_specific_titles() -> None:
    _run_eda()

    general = (OUTPUT_DIR / "term_visuals/retail_top_terms_bar.html").read_text(encoding="utf-8")
    assert "Retail - Clean Top Review Terms Across Promoted 10,000-Record Dataset" in general
    category_chart = OUTPUT_DIR / "term_visuals/all_beauty_top_terms_bar.html"
    assert category_chart.exists()
    category_text = category_chart.read_text(encoding="utf-8")
    assert "Retail - All Beauty Top Review Terms" in category_text


def test_json_reports_use_dataset_10000_eda_prefix() -> None:
    _run_eda()

    expected = [
        "inventory.json",
        "summary.csv",
        "prompt_profile.json",
        "kb_profile.json",
        "gold_profile.json",
        "alignment_report.json",
        "evidence_reuse_report.json",
        "safety_report.json",
        "workload_shape_report.json",
        "summary.md",
    ]
    for suffix in expected:
        assert (OUTPUT_DIR / f"{OUTPUT_PREFIX}_{suffix}").exists(), suffix


def test_old_phase2a_output_filenames_not_created() -> None:
    _run_eda()

    stale_paths = [
        "phase2a_10000_dataset_inventory.json",
        "phase2a_10000_dataset_summary.csv",
        "phase2a_prompt_profile.json",
        "phase2a_kb_profile.json",
        "phase2a_gold_profile.json",
        "phase2a_alignment_report.json",
        "phase2a_evidence_reuse_report.json",
        "phase2a_safety_report.json",
        "phase2a_workload_shape_report.json",
        "phase2a_eda_summary.md",
        "phase2a_vertical_specific_report.json",
    ]
    for stale_path in stale_paths:
        assert not (OUTPUT_DIR / stale_path).exists(), stale_path


def test_cleanup_legacy_eda_flag_removes_duplicate_old_eda_dirs() -> None:
    _run_eda()
    (LEGACY_OUTPUT_DIR / "dashboard").mkdir(parents=True, exist_ok=True)
    (LEGACY_OUTPUT_DIR / "word_views").mkdir(parents=True, exist_ok=True)
    old_report = LEGACY_OUTPUT_DIR / "phase2a_safety_report.json"
    old_public_report = LEGACY_OUTPUT_DIR / f"{OUTPUT_PREFIX}_inventory.json"
    old_dashboard = LEGACY_OUTPUT_DIR / "dashboard/phase2a_10000_overview.html"
    old_public_dashboard = LEGACY_OUTPUT_DIR / f"dashboard/{OUTPUT_PREFIX}_overview.html"
    old_terms = LEGACY_OUTPUT_DIR / "word_views/retail_prompt_terms.txt"
    unrelated = LEGACY_OUTPUT_DIR / "unrelated_keep.txt"
    OLD_PUBLIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old_public_wrapper_report = OLD_PUBLIC_OUTPUT_DIR / f"{OUTPUT_PREFIX}_inventory.json"
    for path in [
        old_report,
        old_public_report,
        old_dashboard,
        old_public_dashboard,
        old_terms,
        unrelated,
        old_public_wrapper_report,
    ]:
        path.write_text("stale\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset-root",
            str(DATASET_ROOT),
            "--write-report",
            "--output-dir",
            str(OUTPUT_DIR),
            "--finance-output-dir",
            str(FINANCE_OUTPUT_DIR),
            "--cleanup-legacy-eda",
            "--research-ai-retrieval-corpus",
            str(OUTPUT_DIR / "missing_research_ai_full_sections_corpus.jsonl"),
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
    assert not old_report.exists()
    assert not old_public_report.exists()
    assert not old_dashboard.exists()
    assert not old_public_dashboard.exists()
    assert not old_terms.exists()
    assert not LEGACY_OUTPUT_DIR.exists()
    assert not OLD_PUBLIC_OUTPUT_DIR.exists()
    assert not OLD_PUBLIC_PARENT_DIR.exists()
    assert not unrelated.exists()


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
        assert (OUTPUT_DIR / "word_views" / f"{vertical}_domain_terms.txt").exists()
        assert (OUTPUT_DIR / "word_views" / f"{vertical}_tfidf_terms.txt").exists()
        assert (OUTPUT_DIR / "word_clouds" / f"{vertical}_wordcloud.png").exists()


def test_word_views_use_clean_terms_not_prompt_terms() -> None:
    _run_eda()

    for vertical in VERTICALS:
        assert (OUTPUT_DIR / "word_views" / f"{vertical}_clean_terms.txt").exists()
        assert not (OUTPUT_DIR / "word_views" / f"{vertical}_prompt_terms.txt").exists()


def _top_clean_terms(vertical: str) -> list[str]:
    path = OUTPUT_DIR / "word_views" / f"{vertical}_clean_terms.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("### top_clean_unigrams") + 1
    terms: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            break
        terms.append(line.split("\t", maxsplit=1)[0])
    return terms


def test_cleaned_terms_remove_common_stopwords() -> None:
    _run_eda()

    for vertical in VERTICALS:
        assert not set(_top_clean_terms(vertical)[:20]) & COMMON_STOPWORDS, vertical


def test_top_clean_terms_do_not_start_with_filler_words() -> None:
    _run_eda()

    for vertical in VERTICALS:
        first_terms = _top_clean_terms(vertical)[:5]
        assert first_terms, vertical
        assert all(term not in COMMON_STOPWORDS for term in first_terms), vertical


def test_eda_creates_per_vertical_html_pages() -> None:
    _run_eda()

    for vertical in VERTICALS:
        path = OUTPUT_DIR / "verticals" / vertical / f"{vertical}_eda.html"
        assert path.exists(), vertical
        assert "Representative Prompts" in path.read_text(encoding="utf-8")


def test_finance_vertical_outputs_exist() -> None:
    _run_eda()

    assert (OUTPUT_DIR / "verticals/finance/finance_eda.html").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_eda.html").exists()
    assert (FINANCE_OUTPUT_DIR / "index.html").exists()
    assert (FINANCE_OUTPUT_DIR / "README.md").exists()
    assert (OUTPUT_DIR / "word_clouds/finance_wordcloud.png").exists()
    assert (OUTPUT_DIR / "word_views/finance_clean_terms.txt").exists()
    assert (OUTPUT_DIR / "word_views/finance_domain_terms.txt").exists()
    assert (OUTPUT_DIR / "word_views/finance_tfidf_terms.txt").exists()


def test_finance_term_visuals_exist() -> None:
    _run_eda()

    assert (OUTPUT_DIR / "term_visuals/finance_top_terms_bar.html").exists()
    assert (OUTPUT_DIR / "term_visuals/finance_term_treemap.html").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_top_terms_bar.html").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_term_treemap.html").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_wordcloud.png").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_clean_terms.txt").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_domain_terms.txt").exists()
    assert (FINANCE_OUTPUT_DIR / "finance_tfidf_terms.txt").exists()
    finance_page = (OUTPUT_DIR / "verticals/finance/finance_eda.html").read_text(encoding="utf-8")
    finance_entrypoint = (FINANCE_OUTPUT_DIR / "finance_eda.html").read_text(encoding="utf-8")
    assert "ticker_coverage" in finance_page
    assert "filing_form_coverage" in finance_page
    assert "xbrl_concept_coverage" in finance_page
    assert "../../term_visuals/" not in finance_entrypoint
    assert "finance_top_terms_bar.html" in finance_entrypoint


def test_research_ai_vertical_folder_exists() -> None:
    _run_eda()

    page = OUTPUT_DIR / "verticals/research_ai/research_ai_eda.html"
    assert page.exists()
    text = page.read_text(encoding="utf-8")
    assert "promoted_benchmark_kb_count" in text
    assert "retrieval corpus" in text.lower()


def test_inventory_counts_match_manifest() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = _read_report(f"{OUTPUT_PREFIX}_inventory.json")

    assert inventory["total_prompt_count"] == manifest["total_prompt_count"]
    assert inventory["total_gold_count"] == manifest["total_gold_count"]
    assert inventory["total_kb_count"] == manifest["total_kb_count"]
    assert inventory["missing_files"] == []
    assert inventory["all_manifest_counts_match"] is True


def test_alignment_report_has_required_fields() -> None:
    report = _read_report(f"{OUTPUT_PREFIX}_alignment_report.json")

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
    report = _read_report(f"{OUTPUT_PREFIX}_evidence_reuse_report.json")

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
    report = _read_report(f"{OUTPUT_PREFIX}_safety_report.json")

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
    report = _read_report(f"{OUTPUT_PREFIX}_workload_shape_report.json")

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


def test_results_readme_separates_experiment_results_from_dataset_eda() -> None:
    results_readme = (ROOT / "results/README.md").read_text(encoding="utf-8")
    figures_readme = (ROOT / "results/figures/README.md").read_text(encoding="utf-8")
    raw_readme = (ROOT / "results/raw/README.md").read_text(encoding="utf-8")
    processed_readme = (ROOT / "results/processed/README.md").read_text(encoding="utf-8")

    for text in [results_readme, figures_readme, raw_readme, processed_readme]:
        assert "smoke-test" in text
        assert "data/generated/dataset_10000" in text
        assert "inference/benchmark outputs" in text or "experiment results" in text
        assert "dataset EDA" in text


def test_results_raw_processed_figures_readmes_exist() -> None:
    assert (ROOT / "results/raw/README.md").exists()
    assert (ROOT / "results/processed/README.md").exists()
    assert (ROOT / "results/figures/README.md").exists()


def test_eda_docs_include_dashboard_and_word_clouds() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "10,000-Record Dataset EDA" in text
    assert f"{OUTPUT_PREFIX}_overview.html" in text
    assert "word_clouds" in text
    assert "word cloud" in text.lower()
    assert "Streamlit" in text
    assert "RAG" in text
    assert "embeddings" in text
    assert "inference" in text


def test_docs_explain_phase2a_only_as_internal_stage() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Internally, this corresponds to Phase 2A data preparation" in text
    assert "10,000-record dataset EDA" in text
    assert "Phase 2A promoted dataset EDA" not in text
    assert "phase2a_10000" not in text
