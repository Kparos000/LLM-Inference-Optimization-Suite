import pytest

from inference_bench.metrics.cost import estimate_token_cost_usd


def test_estimates_token_cost_usd() -> None:
    cost = estimate_token_cost_usd(
        input_tokens=1_000,
        output_tokens=2_000,
        input_cost_per_million_tokens=0.50,
        output_cost_per_million_tokens=1.50,
    )

    assert cost == pytest.approx(0.0035)


def test_rejects_negative_token_count() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        estimate_token_cost_usd(
            input_tokens=-1,
            output_tokens=2_000,
            input_cost_per_million_tokens=0.50,
            output_cost_per_million_tokens=1.50,
        )
