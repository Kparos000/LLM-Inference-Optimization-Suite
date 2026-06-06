import json

from inference_bench.context_schema import ContextRecord
from inference_bench.generation_contract import (
    render_citation_repair_prompt,
    render_generation_contract_prompt,
)
from inference_bench.grounding_repair import citation_repair_decision


def _context(index: int) -> ContextRecord:
    return ContextRecord(
        context_id=f"airline:policy-{index}",
        vertical="airline",
        source_id="airline_policy",
        parent_id=f"policy-{index}",
        chunk_id=f"policy-{index}",
        chunk_strategy="policy-section",
        source_type="policy",
        title=f"Policy {index}",
        text=f"Policy support {index}.",
        metadata={"original_doc_id": f"POL-{index}"},
        token_estimate=4,
        provenance="fixture",
        is_gold_linked=True,
    )


def test_generation_prompt_contains_evidence_checklist_and_multi_citation_rule() -> None:
    prompt = render_generation_contract_prompt(
        question="What policies apply?",
        context_records=[_context(1), _context(2)],
        memory_mode="mm2_hybrid_top5",
    )

    assert "silently check every supplied evidence block as relevant or not relevant" in prompt
    assert "Some answers require multiple evidence_ids" in prompt
    assert "citation_notes must name each emitted evidence_id" in prompt


def test_citation_repair_prompt_does_not_expose_canonical_gold_ids() -> None:
    previous = json.dumps(
        {
            "answer": "Both policies apply.",
            "evidence_ids": ["E1"],
            "confidence": 0.9,
            "insufficient_evidence": False,
            "citation_notes": "E1: first policy.",
        }
    )
    prompt = render_citation_repair_prompt(
        original_prompt="Evidence E1 and E2 are supplied.",
        previous_output=previous,
        allowed_evidence_ids=["E1", "E2"],
        missing_evidence_labels=["E2"],
    )

    assert "Correct only evidence_ids and citation_notes" in prompt
    assert "Allowed evidence_id labels remain exactly: E1, E2" in prompt
    assert "remains uncovered: E2" in prompt
    assert "POL-2" not in prompt
    assert "Do not invent citations" in prompt


def test_repair_runs_only_when_missing_evidence_is_supplied() -> None:
    evaluation = {
        "generation_contract_valid": True,
        "evidence_match": False,
        "evidence_ids_expected": ["POL-1", "POL-2"],
        "evidence_ids_found": ["POL-1"],
    }
    available = citation_repair_decision(
        evaluation=evaluation,
        citation_aliases={"E1": ["POL-1"], "E2": ["POL-2"]},
    )
    absent = citation_repair_decision(
        evaluation=evaluation,
        citation_aliases={"E1": ["POL-1"], "E2": ["POL-3"]},
    )

    assert available.should_retry is True
    assert available.reason == "missing_required_evidence_available_in_supplied_context"
    assert available.missing_evidence_labels == ("E2",)
    assert absent.should_retry is False
    assert absent.reason == "required_evidence_not_in_supplied_context"
