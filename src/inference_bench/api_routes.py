"""Provider-aware routing for API-priced model execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from inference_bench.api_pricing import ApiPricingEntry
from inference_bench.config import ModelConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


@dataclass(frozen=True)
class ApiProviderRoute:
    """Secret-free API route metadata for one configured model."""

    provider: str
    backend: str
    base_url: str
    chat_completions_url: str
    api_key_env: str
    provider_model_id: str
    supports_streaming: bool


def resolve_api_provider_route(
    *,
    model: ModelConfig,
    pricing: ApiPricingEntry,
) -> ApiProviderRoute:
    """Resolve an API route without reading or returning secret values."""

    if model.execution_target == "openrouter_api" or model.provider == "openrouter":
        if pricing.provider != "openrouter":
            msg = (
                f"OpenRouter model {model.model_id} requires OpenRouter pricing, "
                f"not provider '{pricing.provider}'"
            )
            raise ValueError(msg)
        return ApiProviderRoute(
            provider="openrouter",
            backend="openrouter",
            base_url=OPENROUTER_BASE_URL,
            chat_completions_url=f"{OPENROUTER_BASE_URL}/chat/completions",
            api_key_env="OPENROUTER_API_KEY",
            provider_model_id=model.model_id,
            supports_streaming=True,
        )

    if model.execution_target == "hf_inference_provider_api":
        return ApiProviderRoute(
            provider=pricing.provider,
            backend="hf_inference_provider",
            base_url=HF_ROUTER_BASE_URL,
            chat_completions_url=f"{HF_ROUTER_BASE_URL}/chat/completions",
            api_key_env="HF_TOKEN",
            provider_model_id=f"{model.model_id}:{pricing.provider}",
            supports_streaming=True,
        )

    msg = (
        f"Model {model.model_id} has unsupported API execution target "
        f"'{model.execution_target or 'unset'}'"
    )
    raise ValueError(msg)


def api_key_for_route(
    route: ApiProviderRoute,
    environment: Mapping[str, str],
) -> str:
    """Return the configured credential or fail without exposing its value."""

    api_key = environment.get(route.api_key_env, "").strip()
    if not api_key:
        msg = f"{route.api_key_env} is required for provider '{route.provider}'"
        raise ValueError(msg)
    return api_key
