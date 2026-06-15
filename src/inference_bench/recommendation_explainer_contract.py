"""Future LLM explanation boundary for deterministic recommendation JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_TRUSTED_EXPLAINER_MODEL = "model6_gated"


@dataclass(frozen=True)
class RecommendationExplanationRequest:
    """Input contract for a future natural-language explanation service."""

    deterministic_recommendation: dict[str, Any]
    explainer_model_alias: str = DEFAULT_TRUSTED_EXPLAINER_MODEL
    preserve_metric_values: bool = True
    allow_new_recommendations: bool = False
    allow_target_changes: bool = False

    def validate(self) -> None:
        """Reject requests that would let an explainer become the source of truth."""

        if not self.deterministic_recommendation:
            msg = "deterministic_recommendation must not be empty"
            raise ValueError(msg)
        if self.allow_new_recommendations:
            msg = "The explainer may not invent recommendations"
            raise ValueError(msg)
        if self.allow_target_changes:
            msg = "The explainer may not change SLO targets"
            raise ValueError(msg)
        if not self.preserve_metric_values:
            msg = "The explainer must preserve measured metric values"
            raise ValueError(msg)


@dataclass(frozen=True)
class RecommendationExplanationResponse:
    """Future explanation output that retains deterministic provenance."""

    explanation: str
    deterministic_recommendation_hash: str
    explainer_model_alias: str
    source_of_truth: str = "deterministic_recommendation_json"
    diagnosis_modified: bool = False
