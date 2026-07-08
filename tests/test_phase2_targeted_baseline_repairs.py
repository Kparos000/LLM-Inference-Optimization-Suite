from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = Path("scripts/phase4/run_phase2_targeted_baseline_repairs.py")
    spec = importlib.util.spec_from_file_location("phase2_targeted_repairs", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repairs = _load_module()


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "config_id": "cfg",
        "prompt_id": "p1",
        "vertical": "research_ai",
        "memory_mode": "mm2_hybrid_top5",
        "prompt": "SYSTEM\n\nOUTPUT CONTRACT:\nReturn JSON.",
        "citation_id_aliases": json.dumps({"E1": ["doc1"], "E2": ["doc2"], "E3": ["doc3"]}),
        "b5_required_labels": "E1,E2",
        "contract_repair_tags": "",
        "success": True,
        "generated_text": json.dumps(
            {
                "answer": "Do not provide a medical diagnosis.",
                "evidence_ids": ["doc1", "E9"],
                "confidence": 0.8,
                "insufficient_evidence": False,
                "citation_notes": "Uses doc1.",
            }
        ),
        "output_text": "",
    }
    row.update(overrides)
    return row


def test_research_ai_answer_skeleton_applied_to_prompt() -> None:
    repaired = repairs.apply_phase2_prompt_repairs(_row())

    assert "Research AI answer_skeleton" in str(repaired["prompt"])
    assert "phase2_research_ai_answer_skeleton_strengthened" in str(
        repaired["contract_repair_tags"]
    )


def test_citation_whitelist_and_evidence_normalization() -> None:
    normalized = repairs.phase2_normalize_result(_row())
    payload = json.loads(str(normalized["generated_text"]))

    assert payload["evidence_ids"] == ["E1", "E2"]
    assert normalized["phase2_evidence_selector_repair_applied"] is True
    assert json.loads(str(normalized["citations"])) == ["E1", "E2"]


def test_healthcare_safety_wording_cleanup() -> None:
    normalized = repairs.phase2_normalize_result(
        _row(
            vertical="healthcare_admin",
            generated_text=json.dumps(
                {
                    "answer": "Do not provide a medical diagnosis.",
                    "evidence_ids": ["E1"],
                    "confidence": 0.8,
                    "insufficient_evidence": False,
                    "citation_notes": "Avoid treatment advice.",
                }
            ),
        )
    )
    payload = json.loads(str(normalized["generated_text"]))

    assert "medical diagnosis" not in payload["answer"].lower()
    assert "treatment advice" not in payload["citation_notes"].lower()
    assert normalized["phase2_safety_cleanup_applied"] is True


def test_row_specific_forbidden_terms_cleaned_only_in_boundary_language() -> None:
    normalized = repairs.phase2_normalize_result(
        _row(
            vertical="healthcare_admin",
            phase2_forbidden_terms=["share patient identity"],
            generated_text=json.dumps(
                {
                    "answer": "Do not share patient identity in the response.",
                    "evidence_ids": ["E1"],
                    "confidence": 0.8,
                    "insufficient_evidence": False,
                    "citation_notes": "Avoid share patient identity wording.",
                }
            ),
        )
    )
    payload = json.loads(str(normalized["generated_text"]))

    assert "share patient identity" not in payload["answer"].lower()
    assert "share patient identity" not in payload["citation_notes"].lower()
    assert normalized["phase2_forbidden_boundary_cleanup_applied"] is True

    unsafe = repairs.phase2_normalize_result(
        _row(
            vertical="healthcare_admin",
            phase2_forbidden_terms=["share patient identity"],
            generated_text=json.dumps(
                {
                    "answer": "Share patient identity with the caller.",
                    "evidence_ids": ["E1"],
                    "confidence": 0.8,
                    "insufficient_evidence": False,
                    "citation_notes": "E1 supports the action.",
                }
            ),
        )
    )
    unsafe_payload = json.loads(str(unsafe["generated_text"]))
    assert "share patient identity" in unsafe_payload["answer"].lower()


def test_mm4_final_answer_guard_applies_same_contract() -> None:
    normalized = repairs.phase2_normalize_result(
        _row(memory_mode="mm4_bounded_agentic", vertical="healthcare_admin")
    )
    payload = json.loads(str(normalized["generated_text"]))

    assert set(payload) == {
        "answer",
        "evidence_ids",
        "confidence",
        "insufficient_evidence",
        "citation_notes",
    }
    assert normalized["contract_normalization_valid"] is True


def test_before_after_comparison_generation() -> None:
    before = [
        {
            "config_id": "cfg",
            "vertical": "research_ai",
            "json_valid_rate": 1.0,
            "generation_contract_valid_rate": 0.2,
            "format_valid_rate": 0.2,
            "evidence_match_rate": 0.1,
            "grounded_rate": 0.1,
            "safety_violation_count": 0,
            "mean_e2e_latency_ms": 100.0,
            "total_cost_usd": 0.0,
        }
    ]
    after = [{**before[0], "generation_contract_valid_rate": 1.0, "grounded_rate": 0.8}]

    comparison = repairs._comparison_rows(before, after)

    assert comparison[0]["delta_generation_contract_valid_rate"] == 0.8
    assert comparison[0]["delta_grounded_rate"] == 0.7000000000000001


def test_final_10k_blocked_unless_targeted_repair_improves_quality_and_safety() -> None:
    before_summary = {
        "safety_violation_count": 10,
        "json_valid_rate": 1.0,
        "generation_contract_valid_rate": 0.8,
        "mean_e2e_latency_ms": 1000.0,
    }
    after_summary = {
        "safety_violation_count": 10,
        "json_valid_rate": 1.0,
        "generation_contract_valid_rate": 1.0,
        "mean_e2e_latency_ms": 1100.0,
    }
    before_groups = [
        {
            "vertical": "research_ai",
            "generation_contract_valid_rate": 0.1,
            "evidence_match_rate": 0.1,
            "grounded_rate": 0.1,
            "safety_violation_count": 0,
        },
        {
            "vertical": "healthcare_admin",
            "generation_contract_valid_rate": 1.0,
            "evidence_match_rate": 0.7,
            "grounded_rate": 0.7,
            "safety_violation_count": 10,
        },
    ]
    after_groups = [
        {
            "vertical": "research_ai",
            "generation_contract_valid_rate": 0.9,
            "evidence_match_rate": 0.8,
            "grounded_rate": 0.8,
            "safety_violation_count": 0,
        },
        {
            "vertical": "healthcare_admin",
            "generation_contract_valid_rate": 1.0,
            "evidence_match_rate": 0.7,
            "grounded_rate": 0.7,
            "safety_violation_count": 10,
        },
    ]

    gate = repairs._success_gate(before_summary, after_summary, before_groups, after_groups)

    assert gate["research_ai_contract_improved_materially"] is True
    assert gate["healthcare_safety_reduced_materially"] is False
    assert not all(bool(value) for value in gate.values())
