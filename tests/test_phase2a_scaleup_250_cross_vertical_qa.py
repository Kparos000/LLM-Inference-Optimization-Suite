import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/audit_phase2a_scaleup_250.py"
GENERATOR_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/42_phase2a_250_cross_vertical_qa.md"
REPORT_PATH = (
    ROOT / "data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_report.json"
)
SUMMARY_PATH = (
    ROOT / "data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_summary.csv"
)
ISSUE_LOG_PATH = ROOT / "data/generated/phase2a/scaleup_reports/phase2a_250_issue_log.jsonl"


def _load_audit_module() -> Any:
    spec = importlib.util.spec_from_file_location("audit_phase2a_scaleup_250", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_files(vertical: str) -> list[Path]:
    return [
        ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_prompts_250.jsonl",
        ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_gold_250.jsonl",
        ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_kb_250.jsonl",
    ]


def _ensure_generated_candidates() -> None:
    for vertical in ["airline", "healthcare_admin", "retail", "research_ai", "finance"]:
        if all(path.exists() for path in _candidate_files(vertical)):
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--generate-vertical",
                "--vertical",
                vertical,
                "--target-per-vertical",
                "250",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_audit_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_cross_vertical_qa_cli() -> None:
    _ensure_generated_candidates()

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--run-audit"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-10"
    assert summary["total_prompt_count"] == 1250
    assert summary["total_gold_count"] == 1250
    assert summary["verticals_audited"] == 5
    assert REPORT_PATH.exists()
    assert SUMMARY_PATH.exists()
    assert ISSUE_LOG_PATH.exists()


def test_audit_detects_missing_file(tmp_path: Path) -> None:
    module = _load_audit_module()
    targets = {
        "finance": {
            "prompts": tmp_path / "missing_prompts.jsonl",
            "gold": tmp_path / "missing_gold.jsonl",
            "kb": tmp_path / "missing_kb.jsonl",
        }
    }

    missing = module.find_missing_candidate_files(targets)
    message = module.missing_files_error(missing)

    assert len(missing) == 3
    assert "Missing generated 250-scale candidate file" in message
    assert "--generate-vertical --vertical finance --target-per-vertical 250" in message


def test_audit_detects_alignment_issue() -> None:
    module = _load_audit_module()

    issues = module.validate_prompt_gold_alignment(
        [{"prompt_id": "p1"}],
        [{"prompt_id": "p2"}],
        vertical="test",
        prompt_file="prompts.jsonl",
        gold_file="gold.jsonl",
    )

    assert any(issue["check_name"] == "missing_gold_record" for issue in issues)
    assert any(issue["check_name"] == "orphan_gold_record" for issue in issues)
    assert any(issue["severity"] == "critical" for issue in issues)


def test_audit_detects_hygiene_issue() -> None:
    module = _load_audit_module()

    findings = module.scan_hygiene_text("Local file C:\\Users\\name\\data.txt has a token.")

    assert ("critical", "private Windows path") in findings
    assert ("critical", "token reference") in findings


def test_audit_requires_linguistic_variation() -> None:
    module = _load_audit_module()
    prompts = [
        {"question": f"Same finance question scenario {index}", "vertical": "finance"}
        for index in range(10)
    ]

    metrics, issues = module.validate_linguistic_variation(
        prompts,
        vertical="finance",
        file="finance_prompts.jsonl",
    )

    assert metrics["linguistic_variation_rate"] == 0.0
    assert any(issue["check_name"] == "linguistic_variation_gate" for issue in issues)
    assert any(issue["severity"] == "critical" for issue in issues)


def test_audit_domain_specific_checks() -> None:
    module = _load_audit_module()

    healthcare_issues = module.run_domain_specific_checks(
        "healthcare_admin",
        [{"prompt_id": "h1", "question": "Patient asks for urgent clinical symptom help."}],
        [
            {
                "prompt_id": "h1",
                "expected_status": "answer",
                "reference_answer": "Diagnose the patient and give treatment advice.",
            }
        ],
        [],
    )
    finance_issues = module.run_domain_specific_checks(
        "finance",
        [{"prompt_id": "f1", "question": "Should I invest based on this filing?"}],
        [
            {
                "prompt_id": "f1",
                "expected_status": "answer",
                "reference_answer": "This is investment advice: you should buy the stock.",
                "required_doc_ids": ["finance_doc"],
            }
        ],
        [],
    )
    retail_issues = module.run_domain_specific_checks(
        "retail",
        [{"prompt_id": "r1", "question": "raw user_id 123 asks for help."}],
        [],
        [
            {
                "doc_id": "retail_policy",
                "document_type": "support_policy",
                "body": "Support policy without synthetic benchmark label.",
            }
        ],
    )

    issues = healthcare_issues + finance_issues + retail_issues
    check_names = {issue["check_name"] for issue in issues}
    assert "healthcare_clinical_advice" in check_names
    assert "healthcare_safety_status" in check_names
    assert "finance_investment_advice" in check_names
    assert "retail_raw_user_id" in check_names
    assert "retail_synthetic_policy_label" in check_names


def test_docs_include_qa_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")
    lowered = docs.lower()

    assert "Phase 2A-10" in docs
    assert "--run-audit" in docs
    assert "linguistic variation" in lowered
    assert "promotion_ready" in docs
