from __future__ import annotations

import inspect

import pytest

from inference_bench.b6r4_qwen3b_validation import (
    B6R4_FROZEN_REPLAY_INPUT,
    B6R4_MODEL_ALIAS,
    B6R4_MODEL_ID,
    classify_b6r4_full_500_gate,
    classify_b6r4_targeted_gate,
    targeted_replay_allows_full_500,
    validate_b6r4_model_selection,
    validate_b6r4_replay_input,
)


def _targeted_summary(
    *,
    json_rate: float = 1.0,
    contract_rate: float = 1.0,
    evidence_rate: float = 0.9,
    grounded_rate: float = 0.9,
    safety_count: int = 0,
    truncation_rate: float = 0.0,
) -> dict[str, object]:
    return {
        "row_count": 26,
        "json_valid_rate": json_rate,
        "generation_contract_valid_rate": contract_rate,
        "evidence_match_rate": evidence_rate,
        "grounded_rate": grounded_rate,
        "safety_violation_count": safety_count,
        "truncation_rate": truncation_rate,
    }


def test_model2_3b_is_used_not_deprecated_model2_1_5b() -> None:
    resolved = validate_b6r4_model_selection()

    assert resolved == {"model_alias": B6R4_MODEL_ALIAS, "model_id": B6R4_MODEL_ID}
    with pytest.raises(ValueError, match="deprecated model2_1_5b"):
        validate_b6r4_model_selection(model_alias="model2_1_5b")


def test_targeted_replay_uses_b6r1_frozen_rows() -> None:
    validate_b6r4_replay_input(B6R4_FROZEN_REPLAY_INPUT)

    with pytest.raises(ValueError, match="B6R1 Research AI failed-row replay"):
        validate_b6r4_replay_input(
            "data/generated/phase4/b6r2_research_ai_failed_replay_input.jsonl"
        )


def test_targeted_gate_passes_only_when_all_thresholds_pass() -> None:
    passing = classify_b6r4_targeted_gate(_targeted_summary())
    blocked = classify_b6r4_targeted_gate(_targeted_summary(evidence_rate=0.84))

    assert passing["status"] == "B6R4_TARGETED_MODEL2_3B_PASSED"
    assert targeted_replay_allows_full_500(passing) is True
    assert blocked["status"] == "B6R4_TARGETED_MODEL2_3B_BLOCKED"
    assert "evidence_match_rate" in blocked["failed_metrics"]
    assert targeted_replay_allows_full_500(blocked) is False


def test_targeted_gate_blocks_json_contract_safety_and_truncation_failures() -> None:
    gate = classify_b6r4_targeted_gate(
        _targeted_summary(
            json_rate=0.96,
            contract_rate=0.96,
            safety_count=1,
            truncation_rate=0.03,
        )
    )

    assert gate["status"] == "B6R4_TARGETED_MODEL2_3B_BLOCKED"
    assert set(gate["failed_metrics"]) == {
        "json_valid_rate",
        "generation_contract_valid_rate",
        "safety_violation_count",
        "truncation_rate",
    }


def test_full_500_gate_requires_overall_and_vertical_quality() -> None:
    summary = _targeted_summary(evidence_rate=0.91, grounded_rate=0.91)
    per_vertical = [
        {"vertical": "airline", "evidence_match_rate": 0.95, "grounded_rate": 0.95},
        {"vertical": "research_ai", "evidence_match_rate": 0.84, "grounded_rate": 0.84},
    ]

    gate = classify_b6r4_full_500_gate(summary=summary, per_vertical_quality=per_vertical)

    assert gate["status"] == "B6R4_MODEL2_3B_500_BLOCKED"
    assert "vertical_evidence_match_rate_min" in gate["failed_metrics"]
    assert "vertical_grounded_rate_min" in gate["failed_metrics"]


def test_no_workload_specific_routing_policy_is_introduced() -> None:
    import inference_bench.b6r4_qwen3b_validation as module

    source = inspect.getsource(module).lower()

    assert 'workload_specific_routing_introduced": false' in source
    assert "best model" not in source
