"""API-priced model pricing and cost helpers.

Pricing snapshots are loaded from checked-in YAML, but live prices are captured
by scripts. This module intentionally refuses missing pricing instead of using
fake defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import load_yaml_file
from inference_bench.metrics.cost import estimate_api_token_cost_usd

DEFAULT_API_PRICING_PATH = "configs/api_pricing.yaml"
PricingStatus = Literal[
    "detected",
    "detected_or_manual_verified",
    "manual_override",
    "unavailable",
]


@dataclass(frozen=True)
class ApiPricingRegistryEntry:
    """Auditable pricing registry row, including unavailable models."""

    model_alias: str
    model_id: str
    provider: str | None
    input_usd_per_1m_tokens: float | None
    output_usd_per_1m_tokens: float | None
    pricing_source: str
    pricing_source_url: str
    pricing_last_checked: str
    pricing_status: PricingStatus
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.model_alias.strip() or not self.model_id.strip():
            msg = "model_alias and model_id must be non-empty"
            raise ValueError(msg)
        valid_statuses = {
            "detected",
            "detected_or_manual_verified",
            "manual_override",
            "unavailable",
        }
        if self.pricing_status not in valid_statuses:
            msg = (
                "pricing_status must be detected, detected_or_manual_verified, "
                "manual_override, or unavailable"
            )
            raise ValueError(msg)
        for field_name in ("input_usd_per_1m_tokens", "output_usd_per_1m_tokens"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, int | float) or isinstance(value, bool) or value < 0
            ):
                msg = f"{field_name} must be >= 0 when available"
                raise ValueError(msg)
        complete = (
            self.input_usd_per_1m_tokens is not None and self.output_usd_per_1m_tokens is not None
        )
        if (
            self.pricing_status
            in {
                "detected",
                "detected_or_manual_verified",
                "manual_override",
            }
            and not complete
        ):
            msg = f"{self.pricing_status} pricing requires complete input/output rates"
            raise ValueError(msg)
        if self.pricing_status != "unavailable" and not (self.provider or "").strip():
            msg = f"{self.pricing_status} pricing requires provider"
            raise ValueError(msg)


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
    pricing_source: str = "hugging_face_router_metadata"
    pricing_status: PricingStatus = "detected"
    notes: str = ""

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

    @property
    def input_usd_per_1m_tokens(self) -> float:
        """Return the public registry field name for input pricing."""

        return self.input_cost_per_1m_tokens_usd

    @property
    def output_usd_per_1m_tokens(self) -> float:
        """Return the public registry field name for output pricing."""

        return self.output_cost_per_1m_tokens_usd

    @property
    def pricing_last_checked(self) -> str:
        """Return the public registry timestamp field."""

        return self.pricing_snapshot_timestamp_utc


@dataclass(frozen=True)
class ManualPricingOverrideState:
    """Presence and activation state for one audited manual override."""

    model_alias: str
    present: bool
    enabled: bool
    provider: str | None = None
    pricing_source_url: str | None = None
    notes: str = ""


def _optional_price(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = "Pricing values must be numeric or null"
        raise ValueError(msg)
    return float(value)


def _registry_entry(alias: str, raw_entry: dict[str, Any]) -> ApiPricingRegistryEntry:
    """Normalize current and legacy pricing schemas."""

    input_price = raw_entry.get(
        "input_usd_per_1m_tokens",
        raw_entry.get("input_cost_per_1m_tokens_usd"),
    )
    output_price = raw_entry.get(
        "output_usd_per_1m_tokens",
        raw_entry.get("output_cost_per_1m_tokens_usd"),
    )
    provider_status = str(raw_entry.get("provider_status") or "")
    default_status = (
        "detected"
        if input_price is not None and output_price is not None and provider_status != "unavailable"
        else "unavailable"
    )
    status = str(raw_entry.get("pricing_status") or default_status)
    if status not in {
        "detected",
        "detected_or_manual_verified",
        "manual_override",
        "unavailable",
    }:
        msg = f"Invalid pricing_status '{status}' for {alias}"
        raise ValueError(msg)
    return ApiPricingRegistryEntry(
        model_alias=str(raw_entry.get("model_alias") or alias),
        model_id=str(raw_entry.get("model_id") or ""),
        provider=str(raw_entry.get("provider") or "") or None,
        input_usd_per_1m_tokens=_optional_price(input_price),
        output_usd_per_1m_tokens=_optional_price(output_price),
        pricing_source=str(
            raw_entry.get("pricing_source")
            or (
                "manual_registry_override"
                if status == "manual_override"
                else "hugging_face_router_metadata"
            )
        ),
        pricing_source_url=str(raw_entry.get("pricing_source_url") or ""),
        pricing_last_checked=str(
            raw_entry.get(
                "pricing_last_checked",
                raw_entry.get(
                    "last_checked",
                    raw_entry.get("pricing_snapshot_timestamp_utc") or "",
                ),
            )
        ),
        pricing_status=cast(PricingStatus, status),
        notes=str(raw_entry.get("notes") or ""),
    )


def load_api_pricing_registry(
    path: str | Path = DEFAULT_API_PRICING_PATH,
) -> dict[str, ApiPricingRegistryEntry]:
    """Load all detected, overridden, and unavailable registry rows."""

    raw_config = load_yaml_file(path)
    raw_models = raw_config.get("models", {})
    if raw_models is None:
        raw_models = {}
    if not isinstance(raw_models, dict):
        msg = f"Config entry 'models' in {path} must be a mapping"
        raise ValueError(msg)
    raw_overrides = raw_config.get("manual_overrides", {})
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, dict):
        msg = f"Config entry 'manual_overrides' in {path} must be a mapping"
        raise ValueError(msg)

    registry: dict[str, ApiPricingRegistryEntry] = {}
    aliases = set(raw_models) | set(raw_overrides)
    for alias in aliases:
        if not isinstance(alias, str):
            msg = "Pricing registry aliases must be strings"
            raise ValueError(msg)
        raw_model = raw_models.get(alias, {})
        raw_override = raw_overrides.get(alias, {})
        if not isinstance(raw_model, dict) or not isinstance(raw_override, dict):
            msg = f"Pricing registry entry '{alias}' must be a mapping"
            raise ValueError(msg)
        override_enabled = raw_override.get("enabled", True)
        if not isinstance(override_enabled, bool):
            msg = f"Pricing registry override '{alias}.enabled' must be boolean"
            raise ValueError(msg)
        if not override_enabled:
            raw_override = {}
        model_entry = _registry_entry(alias, cast(dict[str, Any], raw_model))
        detected_complete = (
            model_entry.pricing_status in {"detected", "detected_or_manual_verified"}
            and model_entry.input_usd_per_1m_tokens is not None
            and model_entry.output_usd_per_1m_tokens is not None
        )
        if detected_complete or not raw_override:
            registry[alias] = model_entry
            continue
        override_payload = dict(cast(dict[str, Any], raw_override))
        override_payload.setdefault("model_alias", alias)
        override_payload.setdefault("model_id", model_entry.model_id)
        override_payload["pricing_status"] = "manual_override"
        registry[alias] = _registry_entry(alias, override_payload)
    return registry


def load_manual_pricing_override_state(
    model_alias: str,
    path: str | Path = DEFAULT_API_PRICING_PATH,
) -> ManualPricingOverrideState:
    """Inspect a manual override without treating a disabled template as pricing."""

    raw_config = load_yaml_file(path)
    raw_overrides = raw_config.get("manual_overrides", {})
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, dict):
        msg = f"Config entry 'manual_overrides' in {path} must be a mapping"
        raise ValueError(msg)
    raw_override = raw_overrides.get(model_alias)
    if raw_override is None:
        return ManualPricingOverrideState(
            model_alias=model_alias,
            present=False,
            enabled=False,
        )
    if not isinstance(raw_override, dict):
        msg = f"Pricing registry override '{model_alias}' must be a mapping"
        raise ValueError(msg)
    enabled = raw_override.get("enabled", True)
    if not isinstance(enabled, bool):
        msg = f"Pricing registry override '{model_alias}.enabled' must be boolean"
        raise ValueError(msg)
    return ManualPricingOverrideState(
        model_alias=model_alias,
        present=True,
        enabled=enabled,
        provider=str(raw_override.get("provider") or "") or None,
        pricing_source_url=str(raw_override.get("pricing_source_url") or "") or None,
        notes=str(raw_override.get("notes") or ""),
    )


def load_api_pricing_config(
    path: str | Path = DEFAULT_API_PRICING_PATH,
) -> dict[str, ApiPricingEntry]:
    """Load API pricing entries keyed by model alias."""

    entries: dict[str, ApiPricingEntry] = {}
    for alias, registry_entry in load_api_pricing_registry(path).items():
        if registry_entry.pricing_status == "unavailable":
            continue
        if (
            registry_entry.input_usd_per_1m_tokens is None
            or registry_entry.output_usd_per_1m_tokens is None
            or registry_entry.provider is None
        ):
            continue
        entries[alias] = ApiPricingEntry(
            model_alias=alias,
            model_id=registry_entry.model_id,
            provider=registry_entry.provider,
            provider_status="live",
            input_cost_per_1m_tokens_usd=registry_entry.input_usd_per_1m_tokens,
            output_cost_per_1m_tokens_usd=registry_entry.output_usd_per_1m_tokens,
            pricing_snapshot_timestamp_utc=registry_entry.pricing_last_checked,
            pricing_source_url=registry_entry.pricing_source_url,
            pricing_source=registry_entry.pricing_source,
            pricing_status=registry_entry.pricing_status,
            notes=registry_entry.notes,
        )
    return entries


def resolve_api_pricing(
    model_alias: str,
    path: str | Path = DEFAULT_API_PRICING_PATH,
) -> ApiPricingEntry:
    """Return one API pricing entry or fail clearly."""

    entries = load_api_pricing_config(path)
    if model_alias not in entries:
        msg = (
            f"Missing API pricing for '{model_alias}' in {path}. Capture complete live "
            "pricing or enable a complete audited manual override before paid API smoke tests."
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
