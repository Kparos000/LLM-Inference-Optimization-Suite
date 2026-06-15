"""Lightweight answer-outline helpers for targeted generation replay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerOutlineSection:
    """One concise answer-planning section rendered before final JSON generation."""

    section: str
    evidence_labels: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        return {
            "section": self.section,
            "evidence_labels": list(self.evidence_labels),
            "summary": self.summary,
        }


def build_answer_outline(
    *,
    question: str,
    required_labels: tuple[str, ...],
    safety_rule_ids: tuple[str, ...] = (),
) -> tuple[AnswerOutlineSection, ...]:
    """Build a deterministic, non-gold outline for the supplied evidence labels."""

    question_lower = question.lower()
    if "compare" in question_lower or "versus" in question_lower or " vs " in question_lower:
        section = "comparison"
        summary = "Compare only the entities supported by the listed supplied evidence labels."
    elif "why" in question_lower or "explain" in question_lower:
        section = "reasoning"
        summary = "Explain the answer using every listed supplied evidence label."
    elif "route" in question_lower or "escalat" in question_lower:
        section = "action"
        summary = "State the supported action or routing decision from the listed labels."
    else:
        section = "answer"
        summary = "Answer directly using every listed supplied evidence label."

    if safety_rule_ids:
        rules = ", ".join(safety_rule_ids)
        summary = (
            f"{summary} Comply with safety rule IDs {rules} without quoting restricted wording."
        )

    return (
        AnswerOutlineSection(
            section=section,
            evidence_labels=required_labels,
            summary=summary,
        ),
    )


def render_answer_outline(outline: tuple[AnswerOutlineSection, ...]) -> str:
    """Render an outline as compact model-facing planning text."""

    if not outline:
        return "\n".join(
            [
                "Answer Outline:",
                "- Section: answer",
                "  Evidence: none",
                "  Summary: No evidence labels were selected.",
            ]
        )
    lines = ["Answer Outline:"]
    for item in outline:
        labels = ", ".join(item.evidence_labels) if item.evidence_labels else "none"
        lines.extend(
            [
                f"- Section: {item.section}",
                f"  Evidence: {labels}",
                f"  Summary: {item.summary}",
            ]
        )
    return "\n".join(lines)
