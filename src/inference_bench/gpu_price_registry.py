"""RunPod GPU price registry and self-hosted GPU cost estimates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import load_yaml_file

DEFAULT_GPU_PRICE_REGISTRY_PATH = "configs/gpu_prices.yaml"
API_BACKEND_TYPES = {"api_provider"}
API_PROVIDERS = {"openrouter", "hf_inference_provider"}

BackendType = Literal["local_compute", "api_provider", "self_hosted_gpu"]


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _optional_float(value: object, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        msg = f"{field_name} must be numeric or null"
        raise ValueError(msg)
    parsed = float(str(value))
    if parsed < 0:
        msg = f"{field_name} must be >= 0 when provided"
        raise ValueError(msg)
    return parsed


def _optional_int(value: object, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be an integer >= 0 or null"
        raise ValueError(msg)
    return value


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class GpuPriceRecord:
    """One reviewed or placeholder GPU price registry entry."""

    key: str
    gpu_name: str
    provider: str
    hourly_price: float | None
    vram_gb: float | None
    system_ram_gb: float | None
    vcpus: int | None
    generation: str
    recommended_use: str

    def __post_init__(self) -> None:
        _non_empty(self.key, "key")
        _non_empty(self.gpu_name, "gpu_name")
        _non_empty(self.provider, "provider")
        if self.provider != "runpod":
            msg = "GPU price registry currently supports provider 'runpod'"
            raise ValueError(msg)
        if self.hourly_price is not None and self.hourly_price < 0:
            msg = "hourly_price must be >= 0 when provided"
            raise ValueError(msg)
        for field_name in ("vram_gb", "system_ram_gb"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                msg = f"{field_name} must be >= 0 when provided"
                raise ValueError(msg)
        if self.vcpus is not None and self.vcpus < 0:
            msg = "vcpus must be >= 0 when provided"
            raise ValueError(msg)
        _non_empty(self.generation, "generation")
        _non_empty(self.recommended_use, "recommended_use")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable registry row."""

        return asdict(self)


def load_gpu_price_registry(
    path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
) -> dict[str, GpuPriceRecord]:
    """Load and validate the RunPod GPU price registry."""

    payload = load_yaml_file(path)
    registry: dict[str, GpuPriceRecord] = {}
    normalized_names: set[str] = set()
    for key, value in payload.items():
        if not isinstance(value, dict):
            msg = f"GPU registry entry '{key}' must be a mapping"
            raise ValueError(msg)
        raw = cast(dict[str, Any], value)
        record = GpuPriceRecord(
            key=key,
            gpu_name=_non_empty(raw.get("gpu_name"), f"{key}.gpu_name"),
            provider=_non_empty(raw.get("provider"), f"{key}.provider"),
            hourly_price=_optional_float(raw.get("hourly_price"), f"{key}.hourly_price"),
            vram_gb=_optional_float(raw.get("vram_gb"), f"{key}.vram_gb"),
            system_ram_gb=_optional_float(raw.get("system_ram_gb"), f"{key}.system_ram_gb"),
            vcpus=_optional_int(raw.get("vcpus"), f"{key}.vcpus"),
            generation=_non_empty(raw.get("generation"), f"{key}.generation"),
            recommended_use=_non_empty(raw.get("recommended_use"), f"{key}.recommended_use"),
        )
        normalized = _normalize_name(record.gpu_name)
        if normalized in normalized_names:
            msg = f"Duplicate GPU name in registry: {record.gpu_name}"
            raise ValueError(msg)
        normalized_names.add(normalized)
        registry[key] = record
    return registry


def _resolve_record(
    gpu_name: str,
    *,
    registry_path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
) -> GpuPriceRecord:
    registry = load_gpu_price_registry(registry_path)
    if gpu_name in registry:
        return registry[gpu_name]
    requested = _normalize_name(gpu_name)
    for record in registry.values():
        if _normalize_name(record.gpu_name) == requested:
            return record
    msg = f"Unknown GPU '{gpu_name}'"
    raise KeyError(msg)


def list_supported_gpus(
    registry_path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
) -> list[str]:
    """Return display names for all GPUs known to the registry."""

    return [record.gpu_name for record in load_gpu_price_registry(registry_path).values()]


def get_gpu_metadata(
    gpu_name: str,
    *,
    registry_path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
) -> dict[str, object]:
    """Return metadata for one GPU by key or display name."""

    return _resolve_record(gpu_name, registry_path=registry_path).to_dict()


def get_gpu_price(
    gpu_name: str,
    *,
    registry_path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
    require_price: bool = False,
) -> float | None:
    """Return a reviewed hourly price, or ``None`` for an unfilled placeholder."""

    price = _resolve_record(gpu_name, registry_path=registry_path).hourly_price
    if price is None and require_price:
        msg = f"GPU hourly price is not registered for '{gpu_name}'"
        raise ValueError(msg)
    return price


def _hours_from_inputs(
    *,
    elapsed_seconds: float | None,
    elapsed_hours: float | None,
) -> float | None:
    if elapsed_seconds is not None and elapsed_hours is not None:
        msg = "Provide elapsed_seconds or elapsed_hours, not both"
        raise ValueError(msg)
    if elapsed_seconds is not None:
        if elapsed_seconds < 0:
            msg = "elapsed_seconds must be >= 0"
            raise ValueError(msg)
        return elapsed_seconds / 3600.0
    if elapsed_hours is not None:
        if elapsed_hours < 0:
            msg = "elapsed_hours must be >= 0"
            raise ValueError(msg)
        return elapsed_hours
    return None


def estimate_gpu_cost(
    *,
    gpu_name: str | None,
    elapsed_seconds: float | None = None,
    elapsed_hours: float | None = None,
    projected_seconds_by_prompt_count: dict[int, float] | None = None,
    backend_type: str = "self_hosted_gpu",
    provider: str = "runpod",
    registry_path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
) -> dict[str, object]:
    """Estimate GPU cost fields without applying API-provider token pricing.

    Missing reviewed prices intentionally produce ``None`` costs and a blocking
    reason. This keeps reports cost-aware without fabricating infrastructure
    spend.
    """

    if backend_type in API_BACKEND_TYPES or provider in API_PROVIDERS:
        return {
            "cost_applicable": False,
            "cost_blocked_reason": "api_provider_track",
            "gpu_name": gpu_name,
            "gpu_hourly_cost": None,
            "estimated_run_cost": None,
            "projected_1000_cost": None,
            "projected_10000_cost": None,
            "projected_40000_cost": None,
        }
    if gpu_name in (None, ""):
        return {
            "cost_applicable": True,
            "cost_blocked_reason": "gpu_name_missing",
            "gpu_name": gpu_name,
            "gpu_hourly_cost": None,
            "estimated_run_cost": None,
            "projected_1000_cost": None,
            "projected_10000_cost": None,
            "projected_40000_cost": None,
        }
    resolved_gpu_name = str(gpu_name)
    record = _resolve_record(resolved_gpu_name, registry_path=registry_path)
    hours = _hours_from_inputs(elapsed_seconds=elapsed_seconds, elapsed_hours=elapsed_hours)
    price = record.hourly_price
    projection_seconds = projected_seconds_by_prompt_count or {}
    if price is None:
        return {
            "cost_applicable": True,
            "cost_blocked_reason": "gpu_hourly_price_missing",
            "gpu_name": record.gpu_name,
            "gpu_provider": record.provider,
            "gpu_hourly_cost": None,
            "estimated_run_cost": None,
            "projected_1000_cost": None,
            "projected_10000_cost": None,
            "projected_40000_cost": None,
        }

    def projected_cost(prompt_count: int) -> float | None:
        seconds = projection_seconds.get(prompt_count)
        if seconds is None:
            return None
        return seconds / 3600.0 * price

    return {
        "cost_applicable": True,
        "cost_blocked_reason": None,
        "gpu_name": record.gpu_name,
        "gpu_provider": record.provider,
        "gpu_hourly_cost": price,
        "estimated_run_cost": None if hours is None else hours * price,
        "projected_1000_cost": projected_cost(1000),
        "projected_10000_cost": projected_cost(10000),
        "projected_40000_cost": projected_cost(40000),
    }
