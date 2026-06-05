import json

import pytest

from inference_bench.context_schema import ContextRecord
from inference_bench.evaluator_contract import evaluate_generated_answer
from inference_bench.generation_contract import (
    GENERATION_CONTRACT_FORMAT,
    GenerationContract,
    generation_contract_result_fields,
    parse_generation_contract,
    render_generation_contract_prompt,
)


def context_record() -> ContextRecord:
    return ContextRecord(
        context_id="finance:finance_doc_001",
        vertical="finance",
        source_id="finance_sec_edgar_xbrl",
        parent_id="finance_doc_001",
        chunk_id="finance_doc_001",
        chunk_strategy="filing-section",
        source_type="sec_filing",
        title="AAPL Revenue Filing Section",
        text="Apple reported net sales for the fiscal period.",
        metadata={
            "original_doc_id": "finance_doc_001",
            "source_manifest_record_id": "finance_source_001",
            "ticker": "AAPL",
        },
        token_estimate=10,
        provenance="sec",
        is_gold_linked=True,
    )


def test_generation_contract_validates_required_fields() -> None:
    contract = GenerationContract(
        answer="Apple reported net sales.",
        evidence_ids=["finance_doc_001"],
        confidence=0.9,
        insufficient_evidence=False,
        citation_notes="The filing section directly states the metric.",
    )

    assert contract.to_dict()["evidence_ids"] == ["finance_doc_001"]


def test_generation_contract_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        GenerationContract(
            answer="Answer",
            evidence_ids=["doc-1"],
            confidence=1.5,
            insufficient_evidence=False,
            citation_notes="note",
        )


def test_generation_contract_parser_recovers_json_from_model_text() -> None:
    text = (
        "Here is the result:\n"
        '{"answer":"Grounded","evidence_ids":["doc-1"],"confidence":0.75,'
        '"insufficient_evidence":false,"citation_notes":"Supported by doc-1."}'
    )

    parsed = parse_generation_contract(text)

    assert parsed.json_valid is True
    assert parsed.contract_valid is True
    assert parsed.contract is not None
    assert parsed.contract.evidence_ids == ["doc-1"]


def test_generation_contract_result_fields_normalize_invalid_output() -> None:
    fields = generation_contract_result_fields("not json")

    assert fields["generation_contract_valid"] is False
    assert fields["answer"] == ""
    assert fields["evidence_ids"] == []
    assert fields["insufficient_evidence"] is None


def test_prompt_renders_stable_evidence_labels_and_json_contract() -> None:
    prompt = render_generation_contract_prompt(
        question="What did Apple report?",
        context_records=[context_record()],
        memory_mode="mm2_hybrid_top5",
    )

    assert "[EVIDENCE 1]" in prompt
    assert "evidence_id: E1" in prompt
    assert "finance_source_001" in prompt
    assert "insufficient_evidence boolean" in prompt
    assert "Return exactly one compact, single-line JSON object." in prompt
    assert "Do not use markdown" in prompt
    assert "at least one supporting provided label" in prompt
    assert "do not default to E1 only" in prompt
    assert "at or below 40 words" in prompt
    assert "Do not copy wording from these instructions" in prompt


def test_evaluator_scores_structured_evidence_and_groundedness() -> None:
    generated_text = json.dumps(
        {
            "answer": "Apple reported net sales.",
            "evidence_ids": ["finance_doc_001"],
            "confidence": 0.9,
            "insufficient_evidence": False,
            "citation_notes": "The cited filing section supports the answer.",
        }
    )
    result = evaluate_generated_answer(
        {
            "prompt_id": "finance_prompt_001",
            "generated_text": generated_text,
            "expected_output_format": GENERATION_CONTRACT_FORMAT,
            "citation_id_aliases": {"finance_doc_001": ["finance_doc_001", "finance_source_001"]},
        },
        {
            "prompt_id": "finance_prompt_001",
            "expected_status": "answer",
            "required_doc_ids": ["finance_source_001"],
            "must_include": [],
            "must_not_include": [],
        },
    )

    assert result["json_validity"] is True
    assert result["generation_contract_valid"] is True
    assert result["evidence_id_presence"] is True
    assert result["evidence_match"] is True
    assert result["groundedness"] is True


def test_evaluator_expands_short_evidence_labels() -> None:
    generated_text = json.dumps(
        {
            "answer": "Apple reported net sales.",
            "evidence_ids": ["E1"],
            "confidence": 0.9,
            "insufficient_evidence": False,
            "citation_notes": "Direct support.",
        }
    )
    result = evaluate_generated_answer(
        {
            "prompt_id": "finance_prompt_001",
            "generated_text": generated_text,
            "expected_output_format": GENERATION_CONTRACT_FORMAT,
            "citation_id_aliases": {"E1": ["finance_doc_001", "finance_source_001"]},
        },
        {
            "prompt_id": "finance_prompt_001",
            "expected_status": "answer",
            "required_doc_ids": ["finance_source_001"],
            "must_include": [],
            "must_not_include": [],
        },
    )

    assert result["evidence_ids_found"] == ["finance_source_001"]
    assert result["evidence_match"] is True
    assert result["groundedness"] is True


def test_invalid_contract_cannot_be_marked_grounded() -> None:
    generated_text = json.dumps(
        {
            "answer": "",
            "evidence_ids": ["E1"],
            "confidence": 0.9,
            "insufficient_evidence": False,
            "citation_notes": "Direct support.",
        }
    )
    result = evaluate_generated_answer(
        {
            "prompt_id": "finance_prompt_001",
            "generated_text": generated_text,
            "expected_output_format": GENERATION_CONTRACT_FORMAT,
            "citation_id_aliases": {"E1": ["finance_source_001"]},
        },
        {
            "prompt_id": "finance_prompt_001",
            "expected_status": "answer",
            "required_doc_ids": ["finance_source_001"],
            "must_include": [],
            "must_not_include": [],
        },
    )

    assert result["json_validity"] is True
    assert result["generation_contract_valid"] is False
    assert result["evidence_id_presence"] is True
    assert result["evidence_match"] is True
    assert result["groundedness"] is False


def test_evaluator_scores_insufficient_evidence_correctness() -> None:
    generated_text = json.dumps(
        {
            "answer": "",
            "evidence_ids": [],
            "confidence": 0.2,
            "insufficient_evidence": True,
            "citation_notes": "The supplied records do not answer the question.",
        }
    )
    result = evaluate_generated_answer(
        {
            "prompt_id": "airline_prompt_001",
            "generated_text": generated_text,
            "expected_output_format": GENERATION_CONTRACT_FORMAT,
        },
        {
            "prompt_id": "airline_prompt_001",
            "expected_status": "insufficient_evidence",
            "must_include": [],
            "must_not_include": [],
            "required_doc_ids": [],
        },
    )

    assert result["observed_status"] == "insufficient_evidence"
    assert result["insufficient_evidence_correct"] is True
    assert result["status_matches"] is True
    assert result["evidence_id_presence"] is False
