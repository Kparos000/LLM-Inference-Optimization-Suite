import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/audit_finance_evidence_reuse.py"
PLAN_SCRIPT_PATH = ROOT / "scripts/phase2/plan_phase2a_1000_scaleup.py"
DOC_PATH = ROOT / "docs/44_phase2a_1000_scaleup_plan.md"


def _load_finance_module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_finance_evidence_reuse", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plan_module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_phase2a_1000_scaleup", PLAN_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finance_evidence_reuse_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_finance_evidence_reuse_cli(tmp_path: Path) -> None:
    output_report = tmp_path / "finance_reuse_report.json"
    output_csv = tmp_path / "finance_reuse_by_doc.csv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--run-audit",
            "--output-report",
            str(output_report),
            "--output-csv",
            str(output_csv),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-12D"
    assert summary["total_prompts"] == 250
    assert summary["total_gold"] == 250
    assert output_report.exists()
    assert output_csv.exists()


def test_reuse_risk_low_medium_high_logic() -> None:
    module = _load_finance_module()

    assert module.reuse_risk_from_share(0.21) == "high"
    assert module.reuse_risk_from_share(0.20) == "medium"
    assert module.reuse_risk_from_share(0.10) == "medium"
    assert module.reuse_risk_from_share(0.099) == "low"


def test_reuse_audit_detects_overused_doc(tmp_path: Path) -> None:
    module = _load_finance_module()
    prompts = [
        {
            "prompt_id": f"finance_test_{index:04d}",
            "ticker": "AAPL" if index < 5 else "MSFT",
            "filing_form": "10-K" if index < 5 else "10-Q",
            "task_type": "answer_grounded",
        }
        for index in range(10)
    ]
    gold = [
        {
            "prompt_id": f"finance_test_{index:04d}",
            "required_doc_ids": ["finance_doc_overused" if index < 3 else f"finance_doc_{index}"],
        }
        for index in range(10)
    ]

    report, doc_rows = module.build_audit_report(
        prompts=prompts,
        gold=gold,
        kb_rows=[{"doc_id": "finance_doc_overused"}],
        output_csv=tmp_path / "doc_rows.csv",
    )

    assert report["evidence_reuse_risk"] == "high"
    assert report["ready_for_1000_finance_generation"] is False
    assert report["max_doc_reuse_count"] == 3
    assert doc_rows[0]["doc_id"] == "finance_doc_overused"


def test_planner_clears_finance_warning_when_reuse_audit_ready() -> None:
    module = _load_plan_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")
    finance_report = {
        "phase": "2A-12D",
        "evidence_reuse_risk": "low",
        "ready_for_1000_finance_generation": True,
    }

    report = module.build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
        promoted_manifest_path=Path("data/scaleup/phase2a_250_manifest.json"),
        finance_evidence_reuse_report=finance_report,
        finance_evidence_reuse_report_path=Path(
            "data/generated/phase2a/scaleup_reports/finance_evidence_reuse_audit_report.json"
        ),
    )

    finance = report["per_vertical_readiness"]["finance"]
    assert finance["evidence_reuse_audit_ready"] is True
    assert finance["evidence_reuse_risk"] == "low"
    assert not any(
        warning == "finance:evidence_reuse_audit_required_before_generation"
        for warning in report["warnings"]
    )


def test_docs_or_readme_mentions_finance_reuse_audit() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Finance Evidence Reuse Audit" in docs
    assert "python scripts/phase2/audit_finance_evidence_reuse.py --run-audit" in docs
    assert "20%" in docs
