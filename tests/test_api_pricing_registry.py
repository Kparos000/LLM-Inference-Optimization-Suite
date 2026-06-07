import pytest

from inference_bench.api_pricing import (
    estimate_api_cost_from_pricing,
    load_api_pricing_registry,
    resolve_api_pricing,
)


def test_ministral_pricing_is_loaded_from_verified_openrouter_snapshot() -> None:
    registry = load_api_pricing_registry()
    model5 = registry["model5_gated"]

    assert model5.model_id == "mistralai/ministral-3b-2512"
    assert model5.provider == "openrouter"
    assert model5.pricing_status == "detected_or_manual_verified"
    assert model5.input_usd_per_1m_tokens == pytest.approx(0.10)
    assert model5.output_usd_per_1m_tokens == pytest.approx(0.10)


def test_ministral_cost_calculation_uses_registered_token_rates() -> None:
    pricing = resolve_api_pricing("model5_gated")
    cost = estimate_api_cost_from_pricing(
        input_tokens=1_000_000,
        output_tokens=500_000,
        pricing=pricing,
    )

    assert cost["input_cost_usd"] == pytest.approx(0.10)
    assert cost["output_cost_usd"] == pytest.approx(0.05)
    assert cost["total_api_cost_usd"] == pytest.approx(0.15)


def test_model6_pricing_remains_available() -> None:
    pricing = resolve_api_pricing("model6_gated")

    assert pricing.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert pricing.provider == "novita"
