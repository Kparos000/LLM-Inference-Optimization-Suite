"""Modular SLO profile selection and applicability rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import yaml

SLO_GROUPS = (
    "retrieval",
    "quality",
    "latency",
    "throughput",
    "resource",
    "api_cost",
    "gpu_cost",
    "retrieval_ablation",
    "compression",
    "agentic_trace",
)
PRIORITY_MODES = (
    "quality_first",
    "latency_first",
    "throughput_first",
    "cost_first",
    "balanced",
)
GROUP_TO_TARGET_FAMILY = {
    "retrieval": "retrieval_slo",
    "quality": "quality_slo",
    "latency": "latency_slo",
    "throughput": "throughput_slo",
    "resource": "resource_slo",
    "api_cost": "api_cost_slo",
    "gpu_cost": "gpu_cost_slo",
}
RETRIEVAL_ABLATION_TARGETS = {
    "overall_prompt_plus_metadata_hybrid_recall_at_5": (
        "prompt_plus_metadata_hybrid_recall_at_5_min"
    ),
    "finance_prompt_plus_metadata_hybrid_recall_at_5": (
        "finance_prompt_plus_metadata_hybrid_recall_at_5_min"
    ),
    "overall_prompt_text_only_hybrid_recall_at_5": ("prompt_text_only_hybrid_recall_at_5_min"),
    "finance_prompt_text_only_hybrid_recall_at_5": (
        "finance_prompt_text_only_hybrid_recall_at_5_min"
    ),
    "source_hint_assisted_hybrid_recall_at_5": ("source_hint_assisted_hybrid_recall_at_5_min"),
}
COMPRESSION_TARGETS = {
    "mm3_compression_token_reduction_pct": "compression_token_reduction_pct_min",
    "mm3_compression_recall_loss": "compression_recall_loss_max",
}


@dataclass(frozen=True)
class SloProfile:
    """Resolved profile ready for metric selection."""

    name: str
    description: str
    targets_path: str
    enabled_groups: tuple[str, ...]
    disabled_groups: tuple[str, ...]
    priority_mode: str
    group_weights: dict[str, float]
    vertical_overrides: dict[str, dict[str, dict[str, float | bool]]]
    group_applicability: dict[str, dict[str, Any]]
    supplemental_targets: dict[str, dict[str, float | bool]]


@dataclass(frozen=True)
class SelectedSlo:
    """One selected metric and its run-specific applicability."""

    id: str
    vertical: str
    group: str
    metric_name: str
    target: float | bool
    direction: str
    priority_weight: float
    applicable: bool
    applicability_reason: str | None = None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = f"Expected mapping in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def profile_metric_direction(metric_name: str) -> str:
    """Infer direction for both suffix and unit-suffixed SLO metric names."""

    if metric_name.endswith("_required"):
        return "required"
    if metric_name.endswith("_min") or "_min_" in metric_name:
        return "min"
    if metric_name.endswith("_max") or "_max_" in metric_name:
        return "max"
    msg = f"Cannot infer SLO direction for metric '{metric_name}'"
    raise ValueError(msg)


def load_slo_profiles(
    path: str | Path = "configs/slo_profiles.yaml",
) -> dict[str, Any]:
    """Load and validate the modular profile registry."""

    payload = _load_yaml(path)
    profiles = payload.get("profiles")
    priority_modes = payload.get("priority_modes")
    if not isinstance(profiles, dict) or not profiles:
        msg = "SLO profiles config must define profiles"
        raise ValueError(msg)
    if not isinstance(priority_modes, dict):
        msg = "SLO profiles config must define priority_modes"
        raise ValueError(msg)
    for mode in PRIORITY_MODES:
        if mode not in priority_modes:
            msg = f"SLO profiles config missing priority mode '{mode}'"
            raise ValueError(msg)
    return payload


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        msg = f"{field_name} must be a list of strings"
        raise ValueError(msg)
    return cast(list[str], value)


def resolve_slo_profile(
    profile_name: str = "default_enterprise",
    *,
    enabled_groups: list[str] | None = None,
    disabled_groups: list[str] | None = None,
    priority_mode: str | None = None,
    vertical_overrides: dict[str, dict[str, dict[str, float | bool]]] | None = None,
    path: str | Path = "configs/slo_profiles.yaml",
) -> SloProfile:
    """Resolve a named profile with optional user selections."""

    registry = load_slo_profiles(path)
    raw_profiles = cast(dict[str, Any], registry["profiles"])
    raw_profile = raw_profiles.get(profile_name)
    if not isinstance(raw_profile, dict):
        msg = f"Unknown SLO profile '{profile_name}'"
        raise KeyError(msg)
    active_enabled = (
        enabled_groups
        if enabled_groups is not None
        else _string_list(raw_profile.get("enabled_groups", []), "enabled_groups")
    )
    active_disabled = (
        disabled_groups
        if disabled_groups is not None
        else _string_list(raw_profile.get("disabled_groups", []), "disabled_groups")
    )
    unknown = sorted((set(active_enabled) | set(active_disabled)) - set(SLO_GROUPS))
    if unknown:
        msg = f"Unknown SLO groups: {', '.join(unknown)}"
        raise ValueError(msg)
    active_mode = priority_mode or str(raw_profile.get("priority_mode", "balanced"))
    if active_mode not in PRIORITY_MODES:
        msg = f"Unknown priority mode '{active_mode}'"
        raise ValueError(msg)
    raw_modes = cast(dict[str, Any], registry["priority_modes"])
    raw_weights = raw_modes[active_mode]
    if not isinstance(raw_weights, dict):
        msg = f"Priority mode '{active_mode}' must be a mapping"
        raise ValueError(msg)
    base_overrides = raw_profile.get("vertical_overrides", {})
    if not isinstance(base_overrides, dict):
        msg = "vertical_overrides must be a mapping"
        raise ValueError(msg)
    merged_overrides = cast(dict[str, dict[str, dict[str, float | bool]]], base_overrides).copy()
    if vertical_overrides:
        merged_overrides.update(vertical_overrides)
    group_applicability = raw_profile.get("group_applicability", {})
    supplemental_targets = raw_profile.get("supplemental_targets", {})
    if not isinstance(group_applicability, dict) or not isinstance(supplemental_targets, dict):
        msg = "group_applicability and supplemental_targets must be mappings"
        raise ValueError(msg)
    return SloProfile(
        name=profile_name,
        description=str(raw_profile.get("description", "")),
        targets_path=str(raw_profile.get("targets_path", "configs/slo_targets.yaml")),
        enabled_groups=tuple(group for group in active_enabled if group not in active_disabled),
        disabled_groups=tuple(active_disabled),
        priority_mode=active_mode,
        group_weights={str(key): float(value) for key, value in raw_weights.items()},
        vertical_overrides=merged_overrides,
        group_applicability=cast(dict[str, dict[str, Any]], group_applicability),
        supplemental_targets=cast(dict[str, dict[str, float | bool]], supplemental_targets),
    )


def _backend_type(engine: str, explicit_backend_type: str | None) -> str:
    if explicit_backend_type:
        return explicit_backend_type
    normalized = engine.lower()
    if any(token in normalized for token in ("api", "provider", "openrouter", "novita")):
        return "provider"
    return "self_hosted"


def _group_applicability(
    *,
    group: str,
    memory_mode: str,
    engine: str,
    hardware_name: str | None,
    telemetry_available: bool,
    gpu_hourly_price: float | None,
    backend_type: str,
    rules: dict[str, Any],
) -> tuple[bool, str | None]:
    if memory_mode == "mm0_no_context" and group in {
        "retrieval",
        "retrieval_ablation",
        "compression",
        "agentic_trace",
    }:
        return False, f"{group} does not apply to mm0_no_context"
    if group == "compression" and memory_mode != "mm3_compressed_hybrid_top5":
        return False, "compression SLOs apply only to mm3_compressed_hybrid_top5"
    if group == "agentic_trace" and memory_mode != "mm4_bounded_agentic":
        return False, "agentic trace SLOs apply only to mm4_bounded_agentic"
    if group == "retrieval_ablation" and memory_mode not in {
        "mm1_dense_top5",
        "mm2_hybrid_top5",
        "mm3_compressed_hybrid_top5",
    }:
        return False, "retrieval ablation SLOs apply to mm1/mm2/mm3"
    if group == "api_cost" and backend_type not in {"api", "provider"}:
        return False, "API cost SLOs require an API/provider backend"
    if group == "gpu_cost" and gpu_hourly_price is None:
        return False, "GPU cost SLOs require a registered hourly GPU price"
    if group == "resource" and not telemetry_available:
        return False, "resource SLOs require hardware telemetry"

    allowed_engines = rules.get("allowed_engines")
    if isinstance(allowed_engines, list) and engine not in allowed_engines:
        return False, f"{group} is not enabled for engine {engine}"
    excluded_engines = rules.get("excluded_engines")
    if isinstance(excluded_engines, list) and engine in excluded_engines:
        return False, f"{group} is excluded for engine {engine}"
    allowed_hardware = rules.get("allowed_hardware")
    if (
        isinstance(allowed_hardware, list)
        and hardware_name is not None
        and hardware_name not in allowed_hardware
    ):
        return False, f"{group} is not enabled for hardware {hardware_name}"
    excluded_hardware = rules.get("excluded_hardware")
    if (
        isinstance(excluded_hardware, list)
        and hardware_name is not None
        and hardware_name in excluded_hardware
    ):
        return False, f"{group} is excluded for hardware {hardware_name}"
    return True, None


def _targets_for_group(
    targets: dict[str, Any],
    profile: SloProfile,
    *,
    vertical: str,
    group: str,
) -> dict[str, float | bool]:
    if group in GROUP_TO_TARGET_FAMILY:
        raw_verticals = targets.get("verticals", {})
        if not isinstance(raw_verticals, dict) or vertical not in raw_verticals:
            msg = f"Unknown SLO vertical '{vertical}'"
            raise KeyError(msg)
        raw_vertical = raw_verticals[vertical]
        if not isinstance(raw_vertical, dict):
            msg = f"Invalid SLO vertical '{vertical}'"
            raise ValueError(msg)
        raw_family = raw_vertical.get(GROUP_TO_TARGET_FAMILY[group], {})
        if not isinstance(raw_family, dict):
            msg = f"Missing target family for group '{group}'"
            raise ValueError(msg)
        group_targets = {
            str(metric): cast(float | bool, target) for metric, target in raw_family.items()
        }
    elif group == "retrieval_ablation":
        raw_retrieval = targets.get("retrieval", {})
        if not isinstance(raw_retrieval, dict):
            raw_retrieval = {}
        group_targets = {
            output_name: cast(float | bool, raw_retrieval[input_name])
            for input_name, output_name in RETRIEVAL_ABLATION_TARGETS.items()
            if input_name in raw_retrieval
        }
    elif group == "compression":
        raw_compression = targets.get("compression", {})
        if not isinstance(raw_compression, dict):
            raw_compression = {}
        group_targets = {
            output_name: cast(float | bool, raw_compression[input_name])
            for input_name, output_name in COMPRESSION_TARGETS.items()
            if input_name in raw_compression
        }
    else:
        group_targets = dict(profile.supplemental_targets.get(group, {}))

    vertical_group_overrides = profile.vertical_overrides.get(vertical, {}).get(group, {})
    group_targets.update(vertical_group_overrides)
    return group_targets


def select_slos(
    profile: SloProfile,
    *,
    vertical: str,
    memory_mode: str,
    engine: str,
    hardware_name: str | None = None,
    telemetry_available: bool = False,
    gpu_hourly_price: float | None = None,
    backend_type: str | None = None,
) -> list[SelectedSlo]:
    """Materialize selected targets and run-specific applicability."""

    targets = _load_yaml(profile.targets_path)
    active_backend_type = _backend_type(engine, backend_type)
    selected: list[SelectedSlo] = []
    for group in profile.enabled_groups:
        rules = profile.group_applicability.get(group, {})
        applicable, reason = _group_applicability(
            group=group,
            memory_mode=memory_mode,
            engine=engine,
            hardware_name=hardware_name,
            telemetry_available=telemetry_available,
            gpu_hourly_price=gpu_hourly_price,
            backend_type=active_backend_type,
            rules=rules,
        )
        for metric_name, target in _targets_for_group(
            targets, profile, vertical=vertical, group=group
        ).items():
            selected.append(
                SelectedSlo(
                    id=f"{vertical}.{group}.{metric_name}",
                    vertical=vertical,
                    group=group,
                    metric_name=metric_name,
                    target=target,
                    direction=profile_metric_direction(metric_name),
                    priority_weight=profile.group_weights.get(group, 1.0),
                    applicable=applicable,
                    applicability_reason=reason,
                )
            )
    return selected


def with_priority_mode(profile: SloProfile, mode: str) -> SloProfile:
    """Return a profile with a different validated priority mode."""

    if mode not in PRIORITY_MODES:
        msg = f"Unknown priority mode '{mode}'"
        raise ValueError(msg)
    registry = load_slo_profiles()
    weights = cast(dict[str, Any], cast(dict[str, Any], registry["priority_modes"])[mode])
    return replace(
        profile,
        priority_mode=mode,
        group_weights={str(key): float(value) for key, value in weights.items()},
    )
