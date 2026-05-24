import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/45_phase2a_1000_scaleup_generator.md"


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
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            vertical,
            "--target-per-vertical",
            "1000",
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


def test_airline_1000_generation_cli() -> None:
    summary = _generate_vertical("airline")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "airline"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"
    assert report["promotion_required_before_next_checkpoint"] == "checkpoint_2000"


def test_healthcare_1000_generation_cli() -> None:
    summary = _generate_vertical("healthcare_admin")
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert summary["vertical"] == "healthcare_admin"
    assert summary["prompt_count"] == 1000
    assert summary["gold_count"] == 1000
    assert summary["critical_issue_count"] == 0
    assert (ROOT / summary["prompts_path"]).exists()
    assert (ROOT / summary["gold_path"]).exists()
    assert (ROOT / summary["kb_path"]).exists()
    assert report["target_per_vertical"] == 1000
    assert report["checkpoint"] == "checkpoint_1000"
    assert report["generation_scope"] == "local_candidate_generation"
    assert report["promotion_required_before_next_checkpoint"] == "checkpoint_2000"


def test_airline_1000_counts_and_distribution() -> None:
    summary = _generate_vertical("airline")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert prompts[0]["prompt_id"] == "airline_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "airline_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 900,
        "escalate": 80,
        "spam_or_fraud": 20,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 760,
        "json": 140,
        "markdown_table": 100,
    }
    assert {"ticket_change", "cancellation_refund", "baggage_delay"}.issubset(
        {str(prompt["support_type"]) for prompt in prompts}
    )
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_healthcare_1000_counts_and_distribution() -> None:
    summary = _generate_vertical("healthcare_admin")
    prompts = _read_jsonl(ROOT / summary["prompts_path"])
    gold = _read_jsonl(ROOT / summary["gold_path"])
    kb = _read_jsonl(ROOT / summary["kb_path"])

    assert len(prompts) == 1000
    assert len(gold) == 1000
    assert prompts[0]["prompt_id"] == "healthcare_admin_scaleup_1000_0001"
    assert prompts[-1]["prompt_id"] == "healthcare_admin_scaleup_1000_1000"
    assert Counter(prompt["expected_status"] for prompt in prompts) == {
        "answer": 880,
        "escalate": 80,
        "safety_boundary": 20,
        "spam_or_fraud": 10,
        "out_of_scope": 10,
    }
    assert Counter(prompt["expected_output_format"] for prompt in prompts) == {
        "text": 780,
        "json": 140,
        "markdown_table": 80,
    }
    assert {
        "appointment_booking",
        "billing_question",
        "insurance_verification",
        "medical_records_request",
        "prior_authorization_status",
        "portal_access",
        "telehealth_setup",
    }.issubset({str(prompt["support_type"]) for prompt in prompts})
    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_airline_healthcare_1000_kb_targets() -> None:
    module = _load_module()
    for vertical in ["airline", "healthcare_admin"]:
        summary = _generate_vertical(vertical)
        prompts = _read_jsonl(ROOT / summary["prompts_path"])
        gold = _read_jsonl(ROOT / summary["gold_path"])
        kb = _read_jsonl(ROOT / summary["kb_path"])
        assert 150 <= len(kb) <= 250
        assert summary["kb_count"] == len(kb)
        assert module.validate_evidence_coverage(gold, kb) == []
        assert module.validate_no_private_hygiene_terms(prompts + gold + kb) == []


def test_airline_healthcare_1000_linguistic_variation() -> None:
    for vertical in ["airline", "healthcare_admin"]:
        summary = _generate_vertical(vertical)
        report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))
        assert report["linguistic_variation_rate"] >= 0.60
        assert report["most_common_question_template_share"] <= 0.40
        assert report["warning_count"] == 0


def test_large_generation_still_blocks_unimplemented_verticals() -> None:
    for vertical in ["finance", "retail", "research_ai"]:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--generate-vertical",
                "--vertical",
                vertical,
                "--target-per-vertical",
                "1000",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert f"Generation for {vertical} at 1000 requires explicit implementation" in (
            result.stderr
        )


def test_docs_include_1000_generation_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-13A" in docs
    assert "--generate-vertical --vertical airline --target-per-vertical 1000" in docs
    assert "--generate-vertical --vertical healthcare_admin --target-per-vertical 1000" in docs
    assert "Retail, Finance, and Research AI 1,000 generation come later" in docs
    assert "no RAG" in docs
