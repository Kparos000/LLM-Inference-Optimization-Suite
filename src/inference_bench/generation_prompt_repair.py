"""Bounded B4 generation-repair decisions and prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference_bench.generation_contract import (
    allowed_evidence_ids_from_aliases,
    render_citation_repair_prompt,
    render_contract_retry_prompt,
)
from inference_bench.grounding_repair import citation_alias_map


@dataclass(frozen=True)
class GenerationRepairDecision:
    """One optional generation repair decision."""

    should_retry: bool
    trigger: str
    missing_labels: tuple[str, ...] = ()


def decide_generation_repair(
    *,
    evaluation: dict[str, Any],
    result_row: dict[str, Any],
) -> GenerationRepairDecision:
    """Retry only the B4-approved failure classes."""

    if bool(evaluation.get("safety_violation")):
        return GenerationRepairDecision(True, "safety_violation")
    if not bool(evaluation.get("json_validity")):
        return GenerationRepairDecision(True, "invalid_json")
    if not bool(evaluation.get("generation_contract_valid")):
        return GenerationRepairDecision(True, "invalid_contract")

    expected = {str(value) for value in evaluation.get("evidence_ids_expected") or []}
    found = {str(value) for value in evaluation.get("evidence_ids_found") or []}
    missing = expected.difference(found)
    aliases = citation_alias_map(result_row.get("citation_id_aliases"))
    missing_labels = sorted(
        label for label, values in aliases.items() if missing.intersection(values)
    )
    available = {value for label in missing_labels for value in aliases[label]}
    if missing and missing.issubset(available):
        return GenerationRepairDecision(
            True,
            "missing_evidence_label",
            tuple(missing_labels),
        )
    return GenerationRepairDecision(False, "no_approved_repair")


def render_safety_repair_prompt(
    *,
    original_prompt: str,
    previous_output: str,
    allowed_evidence_ids: list[str],
) -> str:
    """Render one safety correction without revealing prohibited terms."""

    allowed = ", ".join(allowed_evidence_ids) or "none"
    return "\n\n".join(
        [
            original_prompt,
            "SAFETY CORRECTION TASK:",
            f"Previous output:\n{previous_output}",
            (
                "The previous output failed a deterministic safety check. Rewrite only "
                "the answer wording needed to remove unsupported advice, guarantees, "
                "projections, verification-bypass language, or unsafe instructions."
            ),
            f"Allowed evidence_id labels remain exactly: {allowed}",
            "Use only facts in the supplied evidence and preserve supported citations.",
            "Do not quote or repeat unsafe wording from the previous output.",
            "Do not invent facts, evidence, or evidence labels.",
            "Return exactly one compact five-field JSON object and nothing else.",
        ]
    )


def build_generation_repair_prompt(
    *,
    decision: GenerationRepairDecision,
    result_row: dict[str, Any],
) -> str:
    """Build the single approved repair prompt for a failed row."""

    if not decision.should_retry:
        raise ValueError("repair prompt requested for a non-retry decision")
    allowed = allowed_evidence_ids_from_aliases(result_row.get("citation_id_aliases"))
    previous_output = str(result_row.get("generated_text") or "")
    if decision.trigger == "missing_evidence_label":
        return render_citation_repair_prompt(
            original_prompt=str(result_row.get("prompt") or ""),
            previous_output=previous_output,
            allowed_evidence_ids=allowed,
            missing_evidence_labels=decision.missing_labels,
        )
    if decision.trigger == "safety_violation":
        return render_safety_repair_prompt(
            original_prompt=str(result_row.get("prompt") or ""),
            previous_output=previous_output,
            allowed_evidence_ids=allowed,
        )
    return render_contract_retry_prompt(
        bad_output=previous_output,
        violation=decision.trigger,
        allowed_evidence_ids=allowed,
    )
