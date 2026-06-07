"""OpenRouter public model metadata helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class OpenRouterModelMetadata:
    """Relevant public metadata for one OpenRouter model."""

    model_id: str
    input_usd_per_1m_tokens: float
    output_usd_per_1m_tokens: float
    context_length: int | None
    supports_streaming: bool
    supports_structured_output: bool


def parse_openrouter_model_metadata(
    payload: dict[str, Any],
    model_id: str,
) -> OpenRouterModelMetadata:
    """Parse one model from the public OpenRouter models response."""

    rows = payload.get("data")
    if not isinstance(rows, list):
        msg = "OpenRouter models response is missing a data list"
        raise ValueError(msg)
    for row in rows:
        if not isinstance(row, dict) or row.get("id") != model_id:
            continue
        pricing = row.get("pricing")
        if not isinstance(pricing, dict):
            msg = f"OpenRouter model {model_id} is missing pricing"
            raise ValueError(msg)
        try:
            input_price = float(pricing["prompt"]) * 1_000_000
            output_price = float(pricing["completion"]) * 1_000_000
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"OpenRouter model {model_id} has incomplete token pricing"
            raise ValueError(msg) from exc
        context_length = row.get("context_length")
        if not isinstance(context_length, int):
            context_length = None
        supported = row.get("supported_parameters")
        supported_parameters = (
            {str(value) for value in supported} if isinstance(supported, list) else set()
        )
        return OpenRouterModelMetadata(
            model_id=model_id,
            input_usd_per_1m_tokens=input_price,
            output_usd_per_1m_tokens=output_price,
            context_length=context_length,
            supports_streaming=True,
            supports_structured_output=bool(
                {"response_format", "structured_outputs"}.intersection(supported_parameters)
            ),
        )
    msg = f"OpenRouter model {model_id} was not found"
    raise ValueError(msg)


def fetch_openrouter_model_metadata(
    model_id: str,
    *,
    timeout_seconds: float = 30.0,
) -> OpenRouterModelMetadata:
    """Fetch public metadata without an API key or paid generation call."""

    request = Request(
        OPENROUTER_MODELS_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        msg = f"OpenRouter models metadata returned HTTP {exc.code}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"OpenRouter models metadata request failed: {exc.reason}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = "OpenRouter models metadata response must be an object"
        raise RuntimeError(msg)
    return parse_openrouter_model_metadata(payload, model_id)
