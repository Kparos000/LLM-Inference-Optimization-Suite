import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/49_phase2a_2000_scaleup_generator.md"
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]

EXPECTED_STATUS = {
    "airline": {"answer": 1800, "escalate": 160, "spam_or_fraud": 40},
    "healthcare_admin": {
        "answer": 1760,
        "escalate": 160,
        "safety_boundary": 40,
        "spam_or_fraud": 20,
        "out_of_scope": 20,
    },
    "retail": {
        "answer": 1780,
        "insufficient_evidence": 70,
        "escalate": 70,
        "spam_or_low_quality": 60,
        "out_of_scope": 20,
    },
    "finance": {"answer": 1840, "insufficient_evidence": 80, "escalate": 80},
    "research_ai": {
        "answer": 1800,
        "insufficient_evidence": 80,
        "escalate": 80,
        "out_of_scope": 40,
    },
}

EXPECTED_OUTPUT = {
    "airline": {"text": 1520, "json": 280, "markdown_table": 200},
    "healthcare_admin": {"text": 1560, "json": 280, "markdown_table": 160},
    "retail": {"text": 1480, "json": 320, "markdown_table": 200},
    "finance": {"text": 1240, "json": 400, "markdown_table": 360},
    "research_ai": {"text": 1440, "json": 280, "markdown_table": 280},
}

KB_RANGES = {
    "airline": (300, 500),
    "healthcare_admin": (300, 500),
    "retail": (1000, 2000),
    "finance": (1500, 2500),
    "research_ai": (1600, 2000),
}


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_phase2a_scaleup", SCRIPT_PATH)
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


def _generate_vertical(vertical: str) -> dict[str, Any]:
    prompts = ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_prompts_2000.jsonl"
    gold = ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_gold_2000.jsonl"
    kb = ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_kb_2000.jsonl"
    report = ROOT / f"data/generated/phase2a/scaleup_reports/{vertical}_scaleup_2000_report.json"
    if prompts.exists() and gold.exists() and kb.exists() and report.exists():
        report_payload = json.loads(report.read_text(encoding="utf-8"))
        return {
            "vertical": vertical,
            "target_per_vertical": 2000,
            "prompt_count": report_payload["prompt_count"],
            "gold_count": report_payload["gold_count"],
            "kb_count": report_payload["kb_count"],
            "critical_issue_count": report_payload["critical_issue_count"],
            "warning_count": report_payload["warning_count"],
            "prompts_path": str(prompts.relative_to(ROOT)),
            "gold_path": str(gold.relative_to(ROOT)),
            "kb_path": str(kb.relative_to(ROOT)),
            "report_path": str(report.relative_to(ROOT)),
        }

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            vertical,
            "--target-per-vertical",
            "2000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert isinstance(summary, dict)
    return summary


def test_2000_generation_cli_for_all_verticals() -> None:
    for vertical in VERTICALS:
        summary = _generate_vertical(vertical)
        assert summary["vertical"] == vertical
        assert summary["target_per_vertical"] == 2000
        assert summary["prompt_count"] == 2000
        assert summary["gold_count"] == 2000
        assert summary["critical_issue_count"] == 0
        assert summary["warning_count"] == 0


def test_2000_counts_and_status_distributions() -> None:
    for vertical in VERTICALS:
        summary = _generate_vertical(vertical)
        prompts = _read_jsonl(ROOT / summary["prompts_path"])
        gold = _read_jsonl(ROOT / summary["gold_path"])
        assert len(prompts) == 2000
        assert len(gold) == 2000
        assert prompts[0]["prompt_id"] == f"{vertical}_scaleup_2000_0001"
        assert prompts[-1]["prompt_id"] == f"{vertical}_scaleup_2000_2000"
        assert Counter(prompt["expected_status"] for prompt in prompts) == EXPECTED_STATUS[vertical]
        assert {prompt["prompt_id"] for prompt in prompts} == {row["prompt_id"] for row in gold}


def test_2000_output_format_distributions() -> None:
    for vertical in VERTICALS:
        summary = _generate_vertical(vertical)
        prompts = _read_jsonl(ROOT / summary["prompts_path"])
        assert (
            Counter(prompt["expected_output_format"] for prompt in prompts)
            == EXPECTED_OUTPUT[vertical]
        )


def test_2000_kb_target_ranges() -> None:
    module = _load_module()
    for vertical in VERTICALS:
        summary = _generate_vertical(vertical)
        prompts = _read_jsonl(ROOT / summary["prompts_path"])
        gold = _read_jsonl(ROOT / summary["gold_path"])
        kb = _read_jsonl(ROOT / summary["kb_path"])
        kb_min, kb_max = KB_RANGES[vertical]
        assert kb_min <= len(kb) <= kb_max
        assert summary["kb_count"] == len(kb)
        assert module.validate_evidence_coverage(gold, kb) == []
        assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_2000_linguistic_variation() -> None:
    for vertical in VERTICALS:
        summary = _generate_vertical(vertical)
        report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
        assert report["linguistic_variation_rate"] >= 0.60
        assert report["most_common_question_template_share"] <= 0.40
        assert report["warning_count"] == 0


def test_2000_domain_safety_checks() -> None:
    for vertical in VERTICALS:
        summary = _generate_vertical(vertical)
        prompts = _read_jsonl(ROOT / summary["prompts_path"])
        gold = _read_jsonl(ROOT / summary["gold_path"])
        kb = _read_jsonl(ROOT / summary["kb_path"])
        combined = json.dumps([prompts, gold, kb], ensure_ascii=True).lower()
        assert "c:\\\\users" not in combined
        assert "/home/" not in combined
        if vertical == "healthcare_admin":
            assert "treatment instructions" in combined
            assert "diagnose the patient" not in combined
        if vertical == "finance":
            assert "investment advice" not in combined
            assert (
                "price target"
                not in json.dumps(
                    [prompt["question"] for prompt in prompts], ensure_ascii=True
                ).lower()
            )
        if vertical == "retail":
            assert "user_id" not in combined
            assert "all_beauty product <asin>" not in combined
            assert "synthetic support policy is amazon policy" not in combined
            assert "official amazon policy" not in combined
        if vertical == "research_ai":
            assert "according to my knowledge" not in combined
            assert "fabricated paper claim" not in combined
            assert "full paper text dump" not in combined
        if vertical == "airline":
            assert "verification bypass" in combined
            assert "guaranteed cash compensation" not in combined


def test_clean_checkout_fallbacks_do_not_require_ignored_raw_artifacts() -> None:
    module = _load_module()

    original_finance_sections = module.build_finance_section_kb_rows_from_manifest
    original_finance_xbrl = module.build_finance_xbrl_inventory_kb_rows
    module.build_finance_section_kb_rows_from_manifest = lambda: []
    module.build_finance_xbrl_inventory_kb_rows = lambda limit: []
    try:
        finance_seed_kb = _read_jsonl(ROOT / "data/kb/finance/kb_sample.jsonl")
        finance_kb = module.expand_finance_kb_rows(
            finance_seed_kb,
            target_kb_count=1500,
            target_per_vertical=2000,
        )
    finally:
        module.build_finance_section_kb_rows_from_manifest = original_finance_sections
        module.build_finance_xbrl_inventory_kb_rows = original_finance_xbrl

    original_research_sections = module.build_research_ai_section_kb_rows_from_manifest
    module.build_research_ai_section_kb_rows_from_manifest = lambda limit: []
    try:
        research_seed_kb = _read_jsonl(ROOT / "data/kb/research_ai/kb_sample.jsonl")
        research_kb = module.expand_research_ai_kb_rows(
            research_seed_kb,
            target_per_vertical=2000,
        )
    finally:
        module.build_research_ai_section_kb_rows_from_manifest = original_research_sections

    assert 1500 <= len(finance_kb) <= 2500
    assert 1600 <= len(research_kb) <= 2000
    assert len(module.research_ai_contexts_by_paper(research_kb)) >= 40
