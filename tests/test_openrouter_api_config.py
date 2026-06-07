import os

import pytest

from inference_bench.api_pricing import resolve_api_pricing
from inference_bench.api_routes import api_key_for_route, resolve_api_provider_route
from inference_bench.config import load_project_config
from inference_bench.openrouter_api import parse_openrouter_model_metadata


def test_openrouter_model_route_loads() -> None:
    model = load_project_config().resolve_model_config("model5_gated")
    pricing = resolve_api_pricing("model5_gated")
    route = resolve_api_provider_route(model=model, pricing=pricing)

    assert route.base_url == "https://openrouter.ai/api/v1"
    assert route.chat_completions_url.endswith("/chat/completions")
    assert route.api_key_env == "OPENROUTER_API_KEY"
    assert route.provider_model_id == "mistralai/ministral-3b-2512"
    assert route.supports_streaming is True


def test_missing_openrouter_key_does_not_break_config_loading() -> None:
    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    model = load_project_config().resolve_model_config("model5_gated")
    route = resolve_api_provider_route(
        model=model,
        pricing=resolve_api_pricing("model5_gated"),
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        api_key_for_route(route, environment)


def test_openrouter_public_metadata_parses_pricing_and_capabilities() -> None:
    metadata = parse_openrouter_model_metadata(
        {
            "data": [
                {
                    "id": "mistralai/ministral-3b-2512",
                    "pricing": {"prompt": "0.0000001", "completion": "0.0000001"},
                    "context_length": 131072,
                    "supported_parameters": ["response_format", "structured_outputs"],
                }
            ]
        },
        "mistralai/ministral-3b-2512",
    )

    assert metadata.input_usd_per_1m_tokens == pytest.approx(0.10)
    assert metadata.output_usd_per_1m_tokens == pytest.approx(0.10)
    assert metadata.supports_streaming is True
    assert metadata.supports_structured_output is True
