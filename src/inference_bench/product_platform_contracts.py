"""Pydantic contracts for the interactive product platform API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ResultType = Literal["measured", "modeled", "planned"]
ApplicabilityState = Literal[
    "applicable_measured",
    "applicable_planned",
    "not_applicable",
    "blocked_by_negative_rule",
    "future_architecture",
]


class PlatformResponse(BaseModel):
    """Common read-only API envelope used by the product backend."""

    status: str = "ok"
    result_type: ResultType = "measured"
    source_artifacts: list[str] = Field(default_factory=list)
    data: dict[str, Any]


class RecipeValidationRequest(BaseModel):
    """Optimization recipe selected by the browser demo session."""

    mandatory_repair_ids: list[str] = Field(default_factory=list)
    core_optimization_ids: list[str] = Field(default_factory=list)
    selected_slo_ids: list[str] = Field(default_factory=list)


class RecipeValidationResponse(BaseModel):
    """Validated plan-only recipe response."""

    status: str
    result_type: ResultType = "planned"
    valid: bool
    selected_optimization_ids: list[str]
    conflicts: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    blocked: list[dict[str, Any]] = Field(default_factory=list)
    plan: dict[str, Any]
