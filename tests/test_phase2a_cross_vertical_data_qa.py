import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/audit_phase2a_seed_data.py"
DOC_PATH = ROOT / "docs/38_phase2a_cross_vertical_data_qa.md"
REPORT_PATH = ROOT / "data/generated/phase2a/phase2a_cross_vertical_qa_report.json"
SUMMARY_PATH = ROOT / "data/generated/phase2a/phase2a_cross_vertical_qa_summary.csv"
ISSUE_LOG_PATH = ROOT / "data/generated/phase2a/phase2a_issue_log.jsonl"


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_phase2a_seed_data", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_audit_target_files_exist() -> None:
    module = _load_audit_module()
    for target in module.VERTICAL_TARGETS.values():
        assert target["prompts"].exists()
        assert target["kb"].exists()
        assert target["gold"].exists()


def test_load_jsonl_helper(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

    rows = _load_audit_module().read_jsonl(path)

    assert rows == [{"a": 1}, {"b": 2}]


def test_prompt_gold_alignment_helper() -> None:
    module = _load_audit_module()
    prompts = [{"prompt_id": "p1"}]
    gold = [{"prompt_id": "p1"}]

    matching_issues = module.validate_prompt_gold_alignment(
        prompts,
        gold,
        vertical="test",
    )
    assert not [issue for issue in matching_issues if issue["severity"] == "critical"]

    orphan_issues = module.validate_prompt_gold_alignment(
        prompts,
        [{"prompt_id": "p2"}],
        vertical="test",
    )
    assert any(issue["check_name"] == "orphan_gold_record" for issue in orphan_issues)
    assert any(issue["severity"] == "critical" for issue in orphan_issues)


def test_hygiene_scan_detects_private_path() -> None:
    private_path = "\\".join(["C:", "Users", "name", "secret.txt"])
    findings = _load_audit_module().scan_hygiene_text(f"local file {private_path}")

    assert ("critical", "private Windows path") in findings


def test_retail_synthetic_policy_check() -> None:
    module = _load_audit_module()
    kb = [
        {
            "doc_id": "retail_policy_bad",
            "document_type": "support_policy",
            "body": "Return handling guidance.",
        }
    ]

    issues = module.check_retail_specific(kb, [])

    assert issues
    assert issues[0]["severity"] == "critical"
    assert issues[0]["check_name"] == "retail_synthetic_policy_label"


def test_scale_up_readiness_logic() -> None:
    module = _load_audit_module()
    ready = module.calculate_scale_up_readiness(
        {
            "critical_issues": 0,
            "prompt_count": 40,
            "gold_count": 40,
            "kb_count": 25,
            "negative_status_count": 1,
            "answerable_without_evidence": 0,
            "has_eda_or_source_report": True,
        }
    )
    blocked = module.calculate_scale_up_readiness(
        {
            "critical_issues": 1,
            "prompt_count": 40,
            "gold_count": 40,
            "kb_count": 25,
            "negative_status_count": 1,
            "answerable_without_evidence": 0,
            "has_eda_or_source_report": True,
        }
    )

    assert ready["ready_for_250_scale"] is True
    assert blocked["ready_for_250_scale"] is False


def test_run_audit_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--run-audit"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-7"
    assert summary["verticals_audited"] == 5
    assert REPORT_PATH.exists()
    assert SUMMARY_PATH.exists()
    assert ISSUE_LOG_PATH.exists()


def test_docs_include_audit_command() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")
    lowered = docs.lower()

    assert "Phase 2A-7" in docs
    assert "cross-vertical" in lowered
    assert "--run-audit" in docs
    assert "ready_for_250_scale" in docs
