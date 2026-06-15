from __future__ import annotations

from inference_bench.answer_planning import build_answer_outline, render_answer_outline


def test_answer_outline_mentions_all_required_labels() -> None:
    outline = build_answer_outline(
        question="Explain what applies.",
        required_labels=("E1", "E3"),
    )
    rendered = render_answer_outline(outline)

    assert outline[0].section == "reasoning"
    assert outline[0].evidence_labels == ("E1", "E3")
    assert "Evidence: E1, E3" in rendered


def test_answer_outline_adds_safety_rule_ids_without_restricted_text() -> None:
    outline = build_answer_outline(
        question="What action should the agent take?",
        required_labels=("E2",),
        safety_rule_ids=("AIR-FRAUD-1",),
    )
    rendered = render_answer_outline(outline)

    assert "AIR-FRAUD-1" in rendered
    assert "restricted wording" in rendered
    assert "verification bypass" not in rendered.lower()
