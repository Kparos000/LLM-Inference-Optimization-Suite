"""Bounded multi-evidence citation repair helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from inference_bench.evaluator_contract import evaluate_generated_answer


def citation_alias_map(value: object) -> dict[str, list[str]]:
    """Normalize a citation alias mapping from metadata or JSON."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for label, aliases in value.items():
        if not isinstance(aliases, list):
            continue
        normalized[str(label)] = [str(alias) for alias in aliases if alias]
    return normalized


def evaluate_result_row(
    row: dict[str, Any],
    gold_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate one runner row with the unchanged evaluator contract."""

    return evaluate_generated_answer(
        {
            "prompt_id": str(row.get("prompt_id") or ""),
            "generated_text": str(row.get("generated_text") or ""),
            "final_status": str(row.get("final_status") or "answer"),
            "expected_output_format": row.get("expected_output_format"),
            "citation_id_aliases": citation_alias_map(row.get("citation_id_aliases")),
        },
        gold_record,
    )


@dataclass(frozen=True)
class CitationRepairDecision:
    """Decision for one optional citation-only retry."""

    should_retry: bool
    reason: str
    missing_expected_ids: tuple[str, ...]
    missing_evidence_labels: tuple[str, ...]
    missing_ids_available_in_context: tuple[str, ...]
    missing_ids_absent_from_context: tuple[str, ...]


def citation_repair_decision(
    *,
    evaluation: dict[str, Any],
    citation_aliases: object,
) -> CitationRepairDecision:
    """Retry only when missing required support is available in supplied context."""

    expected = {str(value) for value in evaluation.get("evidence_ids_expected") or []}
    found = {str(value) for value in evaluation.get("evidence_ids_found") or []}
    missing = expected.difference(found)
    alias_map = citation_alias_map(citation_aliases)
    available_aliases = {alias for aliases in alias_map.values() for alias in aliases}
    available = missing.intersection(available_aliases)
    absent = missing.difference(available_aliases)
    missing_labels = {
        label for label, aliases in alias_map.items() if missing.intersection(aliases)
    }

    if evaluation.get("evidence_match"):
        reason = "evidence_match_already_passed"
        should_retry = False
    elif not evaluation.get("generation_contract_valid"):
        reason = "generation_contract_invalid"
        should_retry = False
    elif absent:
        reason = "required_evidence_not_in_supplied_context"
        should_retry = False
    elif available:
        reason = "missing_required_evidence_available_in_supplied_context"
        should_retry = True
    else:
        reason = "no_actionable_missing_evidence"
        should_retry = False

    return CitationRepairDecision(
        should_retry=should_retry,
        reason=reason,
        missing_expected_ids=tuple(sorted(missing)),
        missing_evidence_labels=tuple(sorted(missing_labels)),
        missing_ids_available_in_context=tuple(sorted(available)),
        missing_ids_absent_from_context=tuple(sorted(absent)),
    )
