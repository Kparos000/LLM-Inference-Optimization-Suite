"""API-priced model pricing and cost helpers.

Pricing snapshots are loaded from checked-in YAML, but live prices are captured
by scripts. This module intentionally refuses missing pricing instead of using
fake defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from inference_bench.config import load_yaml_file
from inference_bench.metrics.cost import estimate_api_token_cost_usd

DEFAULT_API_PRICING_PATH = "configs/api_pricing.yaml"


@dataclass(frozen=True)
class ApiPricingEntry:
    """Pricing snapshot for one API-priced model alias."""

    model_alias: str
    model_id: str
    provider: str
    provider_status: str
    input_cost_per_1m_tokens_usd: float
    output_cost_per_1m_tokens_usd: float
    pricing_snapshot_timestamp_utc: str
    pricing_source_url: str
    selected_for_experiment: bool = True
    context_length: int | None = None
    latency_seconds_if_available: float | None = None
    throughput_tokens_per_second_if_available: float | None = None
    supports_tools_if_available: bool | None = None
    supports_structured_output_if_available: bool | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "model_alias",
            "model_id",
            "provider",
            "provider_status",
            "pricing_snapshot_timestamp_utc",
            "pricing_source_url",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                msg = f"{field_name} must be a non-empty string"
                raise ValueError(msg)
        for field_name in (
            "input_cost_per_1m_tokens_usd",
            "output_cost_per_1m_tokens_usd",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
                msg = f"{field_name} must be >= 0"
                raise ValueError(msg)
        if self.context_length is not None and self.context_length <= 0:
            msg = "context_length must be > 0 when provided"
            raise ValueError(msg)


def load_api_pricing_config(
    path: str | Path = DEFAULT_API_PRICING_PATH,
) -> dict[str, ApiPricingEntry]:
    """Load API pricing entries keyed by model alias."""

    raw_config = load_yaml_file(path)
    raw_models = raw_config.get("models", {})
    if raw_models is None:
        return {}
    if not isinstance(raw_models, dict):
        msg = f"Config entry 'models' in {path} must be a mapping"
        raise ValueError(msg)

    entries: dict[str, ApiPricingEntry] = {}
    for alias, raw_entry in raw_models.items():
        if not isinstance(alias, str) or not isinstance(raw_entry, dict):
            msg = f"Pricing entries in {path} must be mappings keyed by model alias"
            raise ValueError(msg)
        payload = dict(cast(dict[str, Any], raw_entry))
        payload.setdefault("model_alias", alias)
        try:
            entries[alias] = ApiPricingEntry(**payload)
        except TypeError as exc:
            msg = f"Invalid API pricing entry '{alias}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid API pricing entry '{alias}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return entries


def resolve_api_pricing(
    model_alias: str,
    path: str | Path = DEFAULT_API_PRICING_PATH,
) -> ApiPricingEntry:
    """Return one API pricing entry or fail clearly."""

    entries = load_api_pricing_config(path)
    if model_alias not in entries:
        msg = (
            f"Missing API pricing for '{model_alias}' in {path}. Run "
            "scripts/phase3/snapshot_hf_inference_pricing.py before paid API smoke tests."
        )
        raise ValueError(msg)
    return entries[model_alias]


def estimate_api_cost_from_pricing(
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: ApiPricingEntry,
) -> dict[str, float]:
    """Estimate API input, output, and total cost from a pricing snapshot."""

    return estimate_api_token_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_per_1m_tokens_usd=pricing.input_cost_per_1m_tokens_usd,
        output_cost_per_1m_tokens_usd=pricing.output_cost_per_1m_tokens_usd,
    )
