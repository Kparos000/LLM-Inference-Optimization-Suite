from __future__ import annotations

from inference_bench.b6_quality_gate import (
    classify_b6_quality_gate,
    preflight_inference_allowed,
    preflight_status,
)


def _summary(**overrides: float) -> dict[str, float]:
    summary = {
        "json_valid_rate": 0.98,
        "generation_contract_valid_rate": 0.98,
        "evidence_match_rate": 0.91,
        "grounded_rate": 0.91,
        "safety_violation_count": 0.0,
        "truncation_rate": 0.01,
    }
    summary.update(overrides)
    return summary


def _verticals(evidence: float = 0.90, grounded: float = 0.90) -> list[dict[str, float | str]]:
    return [
        {"vertical": vertical, "evidence_match_rate": evidence, "grounded_rate": grounded}
        for vertical in ("airline", "healthcare_admin", "retail", "finance", "research_ai")
    ]


def test_b6_quality_gate_ready_when_all_thresholds_pass() -> None:
    gate = classify_b6_quality_gate(summary=_summary(), per_vertical_quality=_verticals())

    assert gate["status"] == "B6_QUALITY_READY"
    assert gate["passed"] is True
    assert gate["failed_metrics"] == []


def test_b6_quality_gate_improved_but_blocked_when_quality_above_80_below_gate() -> None:
    gate = classify_b6_quality_gate(
        summary=_summary(json_valid_rate=0.95, truncation_rate=0.04),
        per_vertical_quality=_verticals(evidence=0.84, grounded=0.84),
    )

    assert gate["status"] == "B6_QUALITY_IMPROVED_BUT_BLOCKED"
    assert "json_valid_rate" in gate["failed_metrics"]
    assert "truncation_rate" in gate["failed_metrics"]
    assert "vertical_evidence_match_rate_min" in gate["failed_metrics"]


def test_b6_quality_gate_blocked_when_evidence_and_grounding_remain_low() -> None:
    gate = classify_b6_quality_gate(
        summary=_summary(evidence_match_rate=0.79, grounded_rate=0.79),
        per_vertical_quality=_verticals(evidence=0.79, grounded=0.79),
    )

    assert gate["status"] == "B6_QUALITY_BLOCKED"
    assert gate["passed"] is False


def test_b6_preflight_blocks_absent_or_partial_or_leaked_context() -> None:
    passing = {
        "row_count": 500,
        "all_required_evidence_present_count": 500,
        "partial_present_count": 0,
        "absent_count": 0,
        "unrecoverable_row_count": 0,
        "leakage_guard_passed": True,
        "canonical_ids_exposed_to_model": False,
    }
    assert preflight_inference_allowed(passing)
    assert preflight_status(passing) == "PREFLIGHT_PASSED_B6_CONTEXT_ALIGNMENT"

    for override in (
        {"all_required_evidence_present_count": 499, "partial_present_count": 1},
        {"all_required_evidence_present_count": 499, "absent_count": 1},
        {"unrecoverable_row_count": 1},
        {"leakage_guard_passed": False},
        {"canonical_ids_exposed_to_model": True},
    ):
        blocked = {**passing, **override}
        assert not preflight_inference_allowed(blocked)
        assert preflight_status(blocked) == "PREFLIGHT_BLOCKED_B6_CONTEXT_ALIGNMENT"


def test_b6_gate_uses_unchanged_evaluator_metric_names() -> None:
    gate = classify_b6_quality_gate(
        summary=_summary(generation_contract_valid_rate=0.96),
        per_vertical_quality=_verticals(),
    )

    assert "generation_contract_valid_rate" in gate["failed_metrics"]
    assert "format_valid_rate" not in gate["failed_metrics"]
