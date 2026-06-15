"""Structured bottleneck catalog loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

BOTTLENECK_FIELDS = (
    "id",
    "category",
    "description",
    "required_metrics",
    "trigger_conditions",
    "possible_causes",
    "compatible_optimizations",
    "severity_logic",
    "confidence_logic",
    "evidence_fields",
)


@dataclass(frozen=True)
class BottleneckDefinition:
    """Declarative definition for one measurable bottleneck."""

    id: str
    category: str
    description: str
    required_metrics: tuple[str, ...]
    trigger_conditions: tuple[dict[str, Any], ...]
    possible_causes: tuple[str, ...]
    compatible_optimizations: tuple[str, ...]
    severity_logic: dict[str, Any]
    confidence_logic: dict[str, Any]
    evidence_fields: tuple[str, ...]


def load_bottleneck_catalog(
    path: str | Path = "configs/bottleneck_catalog.yaml",
) -> dict[str, BottleneckDefinition]:
    """Load bottlenecks keyed by stable ID."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("bottlenecks"), list):
        msg = "Bottleneck catalog must define a bottlenecks list"
        raise ValueError(msg)
    definitions: dict[str, BottleneckDefinition] = {}
    for raw in cast(list[Any], payload["bottlenecks"]):
        if not isinstance(raw, dict):
            msg = "Every bottleneck entry must be a mapping"
            raise ValueError(msg)
        missing = [field for field in BOTTLENECK_FIELDS if field not in raw]
        if missing:
            msg = f"Bottleneck entry missing fields: {', '.join(missing)}"
            raise ValueError(msg)
        definition = BottleneckDefinition(
            id=str(raw["id"]),
            category=str(raw["category"]),
            description=str(raw["description"]),
            required_metrics=tuple(str(item) for item in raw["required_metrics"]),
            trigger_conditions=tuple(
                cast(dict[str, Any], item) for item in raw["trigger_conditions"]
            ),
            possible_causes=tuple(str(item) for item in raw["possible_causes"]),
            compatible_optimizations=tuple(str(item) for item in raw["compatible_optimizations"]),
            severity_logic=cast(dict[str, Any], raw["severity_logic"]),
            confidence_logic=cast(dict[str, Any], raw["confidence_logic"]),
            evidence_fields=tuple(str(item) for item in raw["evidence_fields"]),
        )
        if not definition.id or definition.id in definitions:
            msg = f"Duplicate or empty bottleneck id '{definition.id}'"
            raise ValueError(msg)
        definitions[definition.id] = definition
    return definitions
