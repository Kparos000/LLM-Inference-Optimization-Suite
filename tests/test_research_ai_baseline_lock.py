from __future__ import annotations

from inference_bench.b6r6_research_ai_recovery import (
    B6R6_RESEARCH_AI_FULL_FLOOR,
    build_research_ai_baseline_lock,
    classify_b6r6_targeted_strategy,
)


def _report(evidence: float = 0.8, grounded: float = 0.8) -> dict[str, object]:
    return {
        "per_vertical_quality": [
            {
                "vertical": "research_ai",
                "row_count": 100,
                "evidence_match_rate": evidence,
                "grounded_rate": grounded,
            }
        ]
    }


def _row(evidence: bool = False, grounded: bool = False) -> dict[str, object]:
    return {
        "original_b6r4_evaluation": {
            "evidence_match": evidence,
            "groundedness": grounded,
        }
    }


def test_baseline_lock_uses_b6r4_research_ai_full_floor() -> None:
    lock = build_research_ai_baseline_lock(
        b6r4_report=_report(),
        replay_rows=[_row(), _row()],
    )

    assert lock["full_vertical_evidence_floor"] == B6R6_RESEARCH_AI_FULL_FLOOR
    assert lock["full_vertical_grounded_floor"] == B6R6_RESEARCH_AI_FULL_FLOOR
    assert lock["failed_row_evidence_floor"] == 0.0
    assert lock["effective_targeted_evidence_floor"] == B6R6_RESEARCH_AI_FULL_FLOOR


def test_baseline_lock_rejects_strategy_below_b6r4_floor() -> None:
    lock = build_research_ai_baseline_lock(
        b6r4_report=_report(),
        replay_rows=[_row(), _row()],
    )
    gate = classify_b6r6_targeted_strategy(
        summary={
            "json_valid_rate": 1.0,
            "generation_contract_valid_rate": 1.0,
            "evidence_match_rate": 0.79,
            "grounded_rate": 0.8,
            "safety_violation_count": 0,
            "truncation_rate": 0.0,
        },
        baseline_lock=lock,
    )

    assert gate["status"] == "B6R6_TARGETED_BLOCKED"
    assert gate["baseline_rejected"] is True
    assert "baseline_evidence_floor" in gate["failed_metrics"]


def test_strategy_at_80_restores_floor_but_remains_caveated() -> None:
    lock = build_research_ai_baseline_lock(
        b6r4_report=_report(),
        replay_rows=[_row(), _row()],
    )
    gate = classify_b6r6_targeted_strategy(
        summary={
            "json_valid_rate": 1.0,
            "generation_contract_valid_rate": 1.0,
            "evidence_match_rate": 0.8,
            "grounded_rate": 0.8,
            "safety_violation_count": 0,
            "truncation_rate": 0.0,
        },
        baseline_lock=lock,
    )

    assert gate["status"] == "B6R6_TARGETED_QUALITY_CAVEATED"
    assert gate["passed"] is True
    assert gate["preferred"] is False
