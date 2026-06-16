from __future__ import annotations

from inference_bench.research_ai_contract_repair import (
    RESEARCH_AI_BUDGET_STRATEGY,
    RESEARCH_AI_CONCISE_STRATEGY,
    select_research_ai_strategy,
    targeted_strategy_passes,
)


def _summary(
    strategy: str,
    *,
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
        "strategy": strategy,
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


def test_targeted_strategy_pass_requires_all_thresholds() -> None:
    assert targeted_strategy_passes(_summary(RESEARCH_AI_CONCISE_STRATEGY)) is True
    assert (
        targeted_strategy_passes(
            _summary(RESEARCH_AI_CONCISE_STRATEGY, contract_rate=0.96),
        )
        is False
    )
    assert (
        targeted_strategy_passes(
            _summary(RESEARCH_AI_CONCISE_STRATEGY, truncation_rate=0.03),
        )
        is False
    )
    assert (
        targeted_strategy_passes(
            _summary(RESEARCH_AI_CONCISE_STRATEGY, safety_count=1),
        )
        is False
    )


def test_strategy_selection_prefers_lower_tokens_then_latency() -> None:
    selection = select_research_ai_strategy(
        [
            _summary(RESEARCH_AI_BUDGET_STRATEGY, output_tokens=75.0, latency_ms=700.0),
            _summary(RESEARCH_AI_CONCISE_STRATEGY, output_tokens=42.0, latency_ms=850.0),
        ]
    )

    assert selection["selection_status"] == "SELECTED_PASSING_STRATEGY"
    assert selection["selected_strategy"] == RESEARCH_AI_CONCISE_STRATEGY


def test_strategy_selection_uses_latency_tiebreaker() -> None:
    selection = select_research_ai_strategy(
        [
            _summary(RESEARCH_AI_BUDGET_STRATEGY, output_tokens=42.0, latency_ms=700.0),
            _summary(RESEARCH_AI_CONCISE_STRATEGY, output_tokens=42.0, latency_ms=850.0),
        ]
    )

    assert selection["selected_strategy"] == RESEARCH_AI_BUDGET_STRATEGY


def test_strategy_selection_blocks_when_neither_passes() -> None:
    selection = select_research_ai_strategy(
        [
            _summary(RESEARCH_AI_CONCISE_STRATEGY, evidence_rate=0.84),
            _summary(RESEARCH_AI_BUDGET_STRATEGY, grounded_rate=0.84),
        ]
    )

    assert selection["selection_status"] == "NO_STRATEGY_PASSED"
    assert selection["selected_strategy"] is None
