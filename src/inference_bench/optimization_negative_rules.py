"""Negative optimization rules for post-SLO diagnosis decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_bench.config import load_yaml_file

DEFAULT_NEGATIVE_RULES_PATH = "configs/optimization_negative_rules.yaml"


@dataclass(frozen=True)
class OptimizationNegativeRule:
    """When-not-to-use rule group for optimization families."""

    id: str
    optimization_ids: tuple[str, ...]
    when_not_to_use: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id.strip():
            msg = "id must not be empty"
            raise ValueError(msg)
        if not self.optimization_ids:
            msg = "optimization_ids must not be empty"
            raise ValueError(msg)
        if not self.when_not_to_use:
            msg = "when_not_to_use must not be empty"
            raise ValueError(msg)


def load_optimization_negative_rules(
    path: str | Path = DEFAULT_NEGATIVE_RULES_PATH,
) -> dict[str, OptimizationNegativeRule]:
    """Load negative optimization rule groups."""

    payload = load_yaml_file(path)
    raw = payload.get("rules")
    if not isinstance(raw, dict):
        msg = "negative optimization rules must define a rules mapping"
        raise ValueError(msg)
    rules: dict[str, OptimizationNegativeRule] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            msg = f"Rule '{key}' must be a mapping"
            raise ValueError(msg)
        optimization_ids = value.get("optimization_ids")
        when_not_to_use = value.get("when_not_to_use")
        if not isinstance(optimization_ids, list) or not isinstance(when_not_to_use, list):
            msg = f"Rule '{key}' must define optimization_ids and when_not_to_use lists"
            raise ValueError(msg)
        rules[key] = OptimizationNegativeRule(
            id=key,
            optimization_ids=tuple(str(item) for item in optimization_ids),
            when_not_to_use=tuple(str(item) for item in when_not_to_use),
        )
    return rules


def negative_rules_for_optimization(
    optimization_id: str,
    *,
    path: str | Path = DEFAULT_NEGATIVE_RULES_PATH,
) -> list[OptimizationNegativeRule]:
    """Return rule groups that mention one optimization id."""

    return [
        rule
        for rule in load_optimization_negative_rules(path).values()
        if optimization_id in set(rule.optimization_ids)
    ]


def build_negative_rule_report(
    *,
    path: str | Path = DEFAULT_NEGATIVE_RULES_PATH,
) -> dict[str, Any]:
    """Return a compact report proving optimizations are post-SLO gated."""

    payload = load_yaml_file(path)
    rules = load_optimization_negative_rules(path)
    return {
        "principle": str(payload.get("principle") or ""),
        "rule_count": len(rules),
        "rules": {
            rule_id: {
                "optimization_ids": list(rule.optimization_ids),
                "when_not_to_use": list(rule.when_not_to_use),
            }
            for rule_id, rule in rules.items()
        },
        "baseline_matrix_config": False,
        "post_slo_diagnosis_only": True,
    }
