"""Structured optimization catalog loading and compatibility validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

OPTIMIZATION_FIELDS = (
    "id",
    "category",
    "description",
    "improves",
    "may_hurt",
    "required_engines",
    "required_hardware",
    "compatible_memory_modes",
    "incompatible_memory_modes",
    "compatible_bottlenecks",
    "required_metrics",
    "implementation_status",
    "application_method",
    "current_project_support",
    "experiment_safety_notes",
    "expected_gain_range",
    "quality_risk",
    "cost_risk",
)
IMPLEMENTATION_STATUSES = {"implemented", "engine_builtin", "config_only", "planned"}
APPLICATION_METHODS = {
    "config_toggle",
    "engine_switch",
    "model_switch",
    "workload_change",
    "hardware_change",
    "code_change",
    "agent_mode_change",
}


@dataclass(frozen=True)
class OptimizationDefinition:
    """Declarative definition for one optimization candidate."""

    id: str
    category: str
    description: str
    improves: tuple[str, ...]
    may_hurt: tuple[str, ...]
    required_engines: tuple[str, ...]
    required_hardware: tuple[str, ...]
    compatible_memory_modes: tuple[str, ...]
    incompatible_memory_modes: tuple[str, ...]
    compatible_bottlenecks: tuple[str, ...]
    required_metrics: tuple[str, ...]
    implementation_status: str
    application_method: str
    current_project_support: str
    experiment_safety_notes: tuple[str, ...]
    expected_gain_range: dict[str, Any]
    quality_risk: str
    cost_risk: str


def load_optimization_catalog(
    path: str | Path = "configs/optimization_catalog.yaml",
) -> dict[str, OptimizationDefinition]:
    """Load optimizations keyed by stable ID."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("optimizations"), list):
        msg = "Optimization catalog must define an optimizations list"
        raise ValueError(msg)
    definitions: dict[str, OptimizationDefinition] = {}
    for raw in cast(list[Any], payload["optimizations"]):
        if not isinstance(raw, dict):
            msg = "Every optimization entry must be a mapping"
            raise ValueError(msg)
        missing = [field for field in OPTIMIZATION_FIELDS if field not in raw]
        if missing:
            msg = f"Optimization entry missing fields: {', '.join(missing)}"
            raise ValueError(msg)
        status = str(raw["implementation_status"])
        method = str(raw["application_method"])
        if status not in IMPLEMENTATION_STATUSES:
            msg = f"Invalid implementation status '{status}'"
            raise ValueError(msg)
        if method not in APPLICATION_METHODS:
            msg = f"Invalid application method '{method}'"
            raise ValueError(msg)
        definition = OptimizationDefinition(
            id=str(raw["id"]),
            category=str(raw["category"]),
            description=str(raw["description"]),
            improves=tuple(str(item) for item in raw["improves"]),
            may_hurt=tuple(str(item) for item in raw["may_hurt"]),
            required_engines=tuple(str(item) for item in raw["required_engines"]),
            required_hardware=tuple(str(item) for item in raw["required_hardware"]),
            compatible_memory_modes=tuple(str(item) for item in raw["compatible_memory_modes"]),
            incompatible_memory_modes=tuple(str(item) for item in raw["incompatible_memory_modes"]),
            compatible_bottlenecks=tuple(str(item) for item in raw["compatible_bottlenecks"]),
            required_metrics=tuple(str(item) for item in raw["required_metrics"]),
            implementation_status=status,
            application_method=method,
            current_project_support=str(raw["current_project_support"]),
            experiment_safety_notes=tuple(str(item) for item in raw["experiment_safety_notes"]),
            expected_gain_range=cast(dict[str, Any], raw["expected_gain_range"]),
            quality_risk=str(raw["quality_risk"]),
            cost_risk=str(raw["cost_risk"]),
        )
        if not definition.id or definition.id in definitions:
            msg = f"Duplicate or empty optimization id '{definition.id}'"
            raise ValueError(msg)
        definitions[definition.id] = definition
    return definitions
