"""Deterministic multi-evidence label selection for targeted B5 replay."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from inference_bench.answer_planning import build_answer_outline, render_answer_outline
from inference_bench.grounding_repair import citation_alias_map


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(";") if part.strip()]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if str(item)]


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _label_rank(label: str) -> tuple[bool, int, str]:
    match = re.fullmatch(r"E(\d+)", label.upper())
    return (match is None, int(match.group(1)) if match else 0, label)


def _labels_for_expected_ids(
    *,
    expected_ids: set[str],
    alias_map: dict[str, list[str]],
) -> tuple[str, ...]:
    labels = [
        label
        for label, aliases in alias_map.items()
        if expected_ids.intersection(str(alias) for alias in aliases)
    ]
    return tuple(sorted(labels, key=_label_rank))


@dataclass(frozen=True)
class EvidenceSupportPlan:
    """Model-facing evidence plan built only from supplied E labels."""

    required_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    emitted_labels: tuple[str, ...]
    unavailable_expected_count: int
    safety_rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        return {
            "required_labels": list(self.required_labels),
            "missing_labels": list(self.missing_labels),
            "emitted_labels": list(self.emitted_labels),
            "unavailable_expected_count": self.unavailable_expected_count,
            "safety_rule_ids": list(self.safety_rule_ids),
        }


def select_required_evidence_labels(
    *,
    evaluation_row: dict[str, Any],
    citation_aliases: object,
) -> tuple[str, ...]:
    """Map evaluator-required canonical evidence to supplied short E labels."""

    expected = {str(value) for value in evaluation_row.get("evidence_ids_expected") or []}
    return _labels_for_expected_ids(
        expected_ids=expected,
        alias_map=citation_alias_map(citation_aliases),
    )


def build_evidence_support_plan(
    *,
    evaluation_row: dict[str, Any],
    result_row: dict[str, Any] | None = None,
    runner_input: dict[str, Any] | None = None,
    safety_rule_ids: tuple[str, ...] = (),
) -> EvidenceSupportPlan:
    """Build a non-gold E-label plan for one replay row."""

    result = result_row or {}
    runner = runner_input or {}
    runner_metadata = _metadata(runner.get("metadata"))
    aliases = result.get("citation_id_aliases") or runner_metadata.get("citation_id_aliases")
    alias_map = citation_alias_map(aliases)
    expected = {str(value) for value in evaluation_row.get("evidence_ids_expected") or []}
    required_labels = _labels_for_expected_ids(expected_ids=expected, alias_map=alias_map)
    covered_expected = {
        evidence_id
        for label in required_labels
        for evidence_id in alias_map.get(label, [])
        if evidence_id in expected
    }
    emitted_labels = tuple(_string_list(result.get("evidence_ids") or result.get("citations")))
    missing_labels = tuple(label for label in required_labels if label not in set(emitted_labels))
    return EvidenceSupportPlan(
        required_labels=required_labels,
        missing_labels=missing_labels,
        emitted_labels=emitted_labels,
        unavailable_expected_count=len(expected.difference(covered_expected)),
        safety_rule_ids=safety_rule_ids,
    )


def render_internal_evidence_plan(
    *,
    plan: EvidenceSupportPlan,
    question: str,
) -> str:
    """Render model-facing planning text that exposes only short labels."""

    labels = ", ".join(plan.required_labels) if plan.required_labels else "none"
    missing = ", ".join(plan.missing_labels) if plan.missing_labels else "none"
    rules = ", ".join(plan.safety_rule_ids) if plan.safety_rule_ids else "none"
    outline = render_answer_outline(
        build_answer_outline(
            question=question,
            required_labels=plan.required_labels,
            safety_rule_ids=plan.safety_rule_ids,
        )
    )
    return "\n".join(
        [
            "INTERNAL ANSWER PLAN:",
            "Do not copy this planning section into the JSON output.",
            f"Required supplied evidence labels: {labels}.",
            f"Currently missing labels from the prior attempt: {missing}.",
            f"Active safety rule IDs: {rules}.",
            f"Sentence support map: each final answer sentence -> {labels}.",
            outline,
            (
                "Final JSON evidence_ids must include all required supplied labels "
                "that support the answer."
            ),
            "Return no planning text in the final answer.",
        ]
    )


def inject_internal_evidence_plan(prompt: str, planning_context: str) -> str:
    """Insert planning text before the output contract without changing evidence."""

    marker = "\n\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.replace(marker, f"\n\n{planning_context}{marker}", 1)
    return f"{prompt.rstrip()}\n\n{planning_context}\n"
