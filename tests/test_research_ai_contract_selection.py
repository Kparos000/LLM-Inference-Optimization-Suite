from __future__ import annotations

from inference_bench.generation_contract_registry import (
    RESEARCH_AI_FINDINGS,
    RESEARCH_AI_MINIMAL_ANSWER,
)
from inference_bench.research_ai_contract_selection import (
    select_research_ai_contract,
    summarize_contract_candidate,
    targeted_contract_passes,
)


def _summary(
    contract_id: str,
    *,
    max_new_tokens: int = 224,
    output_tokens: float = 40.0,
    latency_ms: float = 900.0,
    json_rate: float = 1.0,
    contract_rate: float = 1.0,
    evidence_rate: float = 0.9,
    grounded_rate: float = 0.9,
    truncation_rate: float = 0.0,
    safety_count: int = 0,
) -> dict[str, object]:
    return {
        "contract_id": contract_id,
        "max_new_tokens": max_new_tokens,
        "row_count": 26,
        "json_valid_rate": json_rate,
        "generation_contract_valid_rate": contract_rate,
        "evidence_match_rate": evidence_rate,
        "grounded_rate": grounded_rate,
        "truncation_rate": truncation_rate,
        "safety_violation_count": safety_count,
        "mean_output_tokens": output_tokens,
        "mean_e2e_latency_ms": latency_ms,
    }


def test_targeted_contract_pass_requires_all_quality_thresholds() -> None:
    assert targeted_contract_passes(_summary(RESEARCH_AI_MINIMAL_ANSWER)) is True
    assert targeted_contract_passes(_summary(RESEARCH_AI_MINIMAL_ANSWER, json_rate=0.96)) is False
    assert (
        targeted_contract_passes(
            _summary(RESEARCH_AI_MINIMAL_ANSWER, evidence_rate=0.84),
        )
        is False
    )
    assert (
        targeted_contract_passes(
            _summary(RESEARCH_AI_MINIMAL_ANSWER, truncation_rate=0.03),
        )
        is False
    )
    assert (
        targeted_contract_passes(
            _summary(RESEARCH_AI_MINIMAL_ANSWER, safety_count=1),
        )
        is False
    )


def test_selection_prefers_passing_quality_before_lower_cost() -> None:
    selection = select_research_ai_contract(
        [
            _summary(RESEARCH_AI_MINIMAL_ANSWER, output_tokens=10.0, evidence_rate=0.84),
            _summary(RESEARCH_AI_FINDINGS, output_tokens=75.0, latency_ms=1200.0),
        ]
    )

    assert selection["selection_status"] == "SELECTED_PASSING_CONTRACT"
    assert selection["selected_contract_id"] == RESEARCH_AI_FINDINGS


def test_selection_uses_tokens_then_latency_among_passing_candidates() -> None:
    selection = select_research_ai_contract(
        [
            _summary(RESEARCH_AI_MINIMAL_ANSWER, output_tokens=45.0, latency_ms=700.0),
            _summary(RESEARCH_AI_FINDINGS, output_tokens=42.0, latency_ms=900.0),
        ]
    )

    assert selection["selected_contract_id"] == RESEARCH_AI_FINDINGS


def test_selection_blocks_when_no_candidate_passes() -> None:
    selection = select_research_ai_contract(
        [
            _summary(RESEARCH_AI_MINIMAL_ANSWER, json_rate=0.96),
            _summary(RESEARCH_AI_FINDINGS, grounded_rate=0.84),
        ]
    )

    assert selection["selection_status"] == "NO_CONTRACT_PASSED"
    assert selection["selected_contract_id"] is None


def test_candidate_summary_uses_unchanged_evaluation_rows() -> None:
    evaluation_rows = [
        {
            "json_validity": True,
            "generation_contract_valid": True,
            "evidence_match": True,
            "groundedness": True,
            "safety_violation": False,
        },
        {
            "json_validity": True,
            "generation_contract_valid": True,
            "evidence_match": False,
            "groundedness": False,
            "safety_violation": False,
        },
    ]
    result_rows = [
        {
            "output_tokens": 40,
            "end_to_end_latency_ms": 1000,
            "truncation_detected": False,
            "retry_attempt_count": 1,
            "b6r2_effective_research_ai_contract": RESEARCH_AI_FINDINGS,
        },
        {
            "output_tokens": 30,
            "end_to_end_latency_ms": 800,
            "truncation_detected": True,
            "retry_attempt_count": 0,
            "b6r2_effective_research_ai_contract": RESEARCH_AI_FINDINGS,
        },
    ]

    summary = summarize_contract_candidate(
        contract_id=RESEARCH_AI_FINDINGS,
        max_new_tokens=224,
        evaluation_rows=evaluation_rows,
        result_rows=result_rows,
    )

    assert summary["evidence_match_rate"] == 0.5
    assert summary["grounded_rate"] == 0.5
    assert summary["truncation_rate"] == 0.5
    assert summary["retry_count"] == 1
