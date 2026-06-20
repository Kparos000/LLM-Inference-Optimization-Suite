from __future__ import annotations

from inference_bench.b6r4_qwen3b_validation import (
    COMPARISON_BLOCKS,
    build_model_capacity_comparison,
    classify_b6r4_targeted_gate,
)


def _summary(*, grounded: float, evidence: float | None = None) -> dict[str, object]:
    return {
        "json_valid_rate": 1.0,
        "generation_contract_valid_rate": 1.0,
        "evidence_match_rate": evidence if evidence is not None else grounded,
        "grounded_rate": grounded,
        "safety_violation_count": 0,
        "truncation_rate": 0.0,
    }


def test_comparison_includes_b6_b6r2_b6r3_and_b6r4() -> None:
    gate = classify_b6r4_targeted_gate(_summary(grounded=0.9))

    comparison = build_model_capacity_comparison(
        b6_research_ai=_summary(grounded=0.74),
        b6r2_best=_summary(grounded=0.8077),
        b6r3_model6=_summary(grounded=0.9615),
        b6r4_summary=_summary(grounded=0.9),
        b6r4_gate=gate,
        full_500_triggered=True,
    )

    assert tuple(comparison["comparison_blocks"]) == COMPARISON_BLOCKS
    assert comparison["qwen3b_materially_improves_research_ai"] is True
    assert comparison["qwen3b_targeted_gate_passed"] is True
    assert comparison["full_500_can_proceed_on_model2_3b"] is True


def test_full_500_is_blocked_if_targeted_gate_fails() -> None:
    gate = classify_b6r4_targeted_gate(_summary(grounded=0.8, evidence=0.8))

    comparison = build_model_capacity_comparison(
        b6_research_ai=_summary(grounded=0.74),
        b6r2_best=_summary(grounded=0.8077),
        b6r3_model6=_summary(grounded=0.9615),
        b6r4_summary=_summary(grounded=0.8, evidence=0.8),
        b6r4_gate=gate,
        full_500_triggered=False,
    )

    assert comparison["qwen3b_targeted_gate_passed"] is False
    assert comparison["full_500_can_proceed_on_model2_3b"] is False
    assert comparison["larger_models_remain_necessary_for_research_ai_quality"] is True


def test_qwen3b_not_marked_materially_improved_without_b6r2_delta() -> None:
    gate = classify_b6r4_targeted_gate(_summary(grounded=0.8, evidence=0.8))

    comparison = build_model_capacity_comparison(
        b6_research_ai=_summary(grounded=0.74),
        b6r2_best=_summary(grounded=0.8077),
        b6r3_model6=_summary(grounded=0.9615),
        b6r4_summary=_summary(grounded=0.79, evidence=0.79),
        b6r4_gate=gate,
        full_500_triggered=False,
    )

    assert comparison["qwen3b_materially_improves_research_ai"] is False
