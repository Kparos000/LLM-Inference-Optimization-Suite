from __future__ import annotations

from inference_bench.multi_evidence_selector import (
    build_evidence_support_plan,
    inject_internal_evidence_plan,
    render_internal_evidence_plan,
    select_required_evidence_labels,
)


def test_required_evidence_labels_are_selected_from_alias_map() -> None:
    labels = select_required_evidence_labels(
        evaluation_row={"evidence_ids_expected": ["gold-1", "gold-3"]},
        citation_aliases={
            "E3": ["gold-3", "family-3"],
            "E1": ["gold-1"],
            "E2": ["other"],
        },
    )

    assert labels == ("E1", "E3")


def test_multi_evidence_plan_tracks_missing_labels_without_gold_ids() -> None:
    plan = build_evidence_support_plan(
        evaluation_row={"evidence_ids_expected": ["gold-1", "gold-2"]},
        result_row={
            "citation_id_aliases": {"E1": ["gold-1"], "E2": ["gold-2"]},
            "evidence_ids": ["E1"],
        },
    )
    rendered = render_internal_evidence_plan(plan=plan, question="Explain the policy.")

    assert plan.required_labels == ("E1", "E2")
    assert plan.missing_labels == ("E2",)
    assert "Required supplied evidence labels: E1, E2." in rendered
    assert "gold-2" not in rendered
    assert "Answer Outline:" in rendered


def test_internal_plan_injection_keeps_evidence_and_contract_prompt() -> None:
    prompt = "RETRIEVED EVIDENCE:\nE1 text\n\nOUTPUT CONTRACT:\nReturn JSON"
    updated = inject_internal_evidence_plan(prompt, "INTERNAL ANSWER PLAN:\nE1")

    assert updated.index("RETRIEVED EVIDENCE") < updated.index("INTERNAL ANSWER PLAN")
    assert updated.index("INTERNAL ANSWER PLAN") < updated.index("OUTPUT CONTRACT")
    assert "E1 text" in updated
