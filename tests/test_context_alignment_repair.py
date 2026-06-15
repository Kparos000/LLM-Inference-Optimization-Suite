from __future__ import annotations

from inference_bench.context_alignment_repair import (
    alignment_status,
    repair_context_selection,
    represented_expected_ids,
)
from inference_bench.context_schema import ContextRecord


def _context(
    context_id: str,
    *,
    chunk_id: str,
    parent_id: str | None = None,
) -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical="airline",
        source_id="source",
        parent_id=parent_id or chunk_id,
        chunk_id=chunk_id,
        chunk_strategy="atomic",
        source_type="policy",
        title=f"Title {chunk_id}",
        text=f"Evidence text for {chunk_id}.",
        metadata={"original_doc_id": chunk_id},
        token_estimate=5,
        provenance="fixture",
        is_gold_linked=True,
    )


def test_detects_gold_evidence_absent_from_e1_e5() -> None:
    contexts = [_context("airline:other", chunk_id="other")]

    represented = represented_expected_ids(["gold-1"], contexts)

    assert represented == set()
    assert alignment_status(["gold-1"], represented) == "absent"


def test_repairs_final_e1_e5_when_required_evidence_is_available() -> None:
    current = [_context("airline:other", chunk_id="other")]
    required = _context("airline:gold-1", chunk_id="gold-1")

    repaired = repair_context_selection(
        current_contexts=current,
        candidate_contexts=[required, *current],
        expected_ids=["gold-1"],
        promoted_valid_evidence_ids=["gold-1"],
    )

    assert repaired.status == "all"
    assert repaired.contexts[0].context_id == "airline:gold-1"
    assert repaired.private_alias_map["E1"] == ["gold-1", "airline:gold-1"]
    assert repaired.changed is True


def test_marks_unrecoverable_when_evidence_is_unavailable() -> None:
    current = [_context("airline:other", chunk_id="other")]

    repaired = repair_context_selection(
        current_contexts=current,
        candidate_contexts=current,
        expected_ids=["gold-1"],
        promoted_valid_evidence_ids=["gold-1"],
    )

    assert repaired.status == "absent"
    assert repaired.missing_ids == ("gold-1",)


def test_family_mapping_is_private_and_preserved_for_evaluation() -> None:
    family = _context("airline:family", chunk_id="family")

    repaired = repair_context_selection(
        current_contexts=[],
        candidate_contexts=[family],
        expected_ids=["gold-1"],
        promoted_valid_evidence_ids=["family"],
    )

    assert repaired.status == "all"
    assert repaired.family_alias_bindings == {"gold-1": "airline:family"}
    assert "gold-1" in repaired.private_alias_map["E1"]


def test_repaired_ordering_is_deterministic() -> None:
    first = _context("airline:gold-1", chunk_id="gold-1")
    second = _context("airline:gold-2", chunk_id="gold-2")
    filler = _context("airline:filler", chunk_id="filler")

    one = repair_context_selection(
        current_contexts=[filler],
        candidate_contexts=[second, first, filler],
        expected_ids=["gold-1", "gold-2"],
        promoted_valid_evidence_ids=["gold-1", "gold-2"],
    )
    two = repair_context_selection(
        current_contexts=[filler],
        candidate_contexts=[second, first, filler],
        expected_ids=["gold-1", "gold-2"],
        promoted_valid_evidence_ids=["gold-1", "gold-2"],
    )

    assert [context.context_id for context in one.contexts] == [
        "airline:gold-1",
        "airline:gold-2",
        "airline:filler",
    ]
    assert one == two
