from __future__ import annotations

import json

from inference_bench.generation_contract import parse_generation_contract
from inference_bench.generation_contract_registry import (
    DEFAULT_GENERATION_CONTRACT_ID,
    RESEARCH_AI_ADAPTIVE,
    RESEARCH_AI_COMPARISON,
    RESEARCH_AI_FINDINGS,
    RESEARCH_AI_LIMITATIONS,
    RESEARCH_AI_MINIMAL_ANSWER,
    contract_for_vertical,
    route_research_ai_contract,
    validate_and_map_contract_text,
)


def _text(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True)


def test_default_contract_selected_for_non_research_ai_verticals() -> None:
    assert contract_for_vertical("airline") == DEFAULT_GENERATION_CONTRACT_ID
    assert contract_for_vertical("healthcare_admin") == DEFAULT_GENERATION_CONTRACT_ID
    assert contract_for_vertical("retail") == DEFAULT_GENERATION_CONTRACT_ID
    assert contract_for_vertical("finance") == DEFAULT_GENERATION_CONTRACT_ID


def test_research_ai_candidate_schemas_validate_and_map_to_common_contract() -> None:
    allowed = {"E1", "E2"}
    cases = [
        (
            RESEARCH_AI_MINIMAL_ANSWER,
            {
                "answer": "The paper reports a grounded finding.",
                "evidence": ["E1"],
                "insufficient_evidence": False,
                "confidence": "high",
            },
        ),
        (
            RESEARCH_AI_FINDINGS,
            {
                "summary": "The paper reports two findings.",
                "findings": [
                    {"claim": "The method improves retrieval.", "evidence": ["E1"]},
                    {"claim": "The ablation supports the result.", "evidence": ["E2"]},
                ],
                "insufficient_evidence": False,
                "confidence": "medium",
            },
        ),
        (
            RESEARCH_AI_LIMITATIONS,
            {
                "limitation": "The evaluation is narrow.",
                "why_it_matters": "It limits generalization.",
                "evidence": ["E1"],
                "insufficient_evidence": False,
                "confidence": "low",
            },
        ),
        (
            RESEARCH_AI_COMPARISON,
            {
                "comparison_summary": "The methods differ in supervision.",
                "items": [
                    {"item": "method A", "claim": "It uses labels.", "evidence": ["E1"]},
                    {"item": "method B", "claim": "It uses retrieval.", "evidence": ["E2"]},
                ],
                "insufficient_evidence": False,
                "confidence": "medium",
            },
        ),
    ]

    for contract_id, payload in cases:
        result = validate_and_map_contract_text(
            text=_text(payload),
            contract_id=contract_id,
            allowed_evidence_ids=allowed,
        )
        assert result.json_valid is True
        assert result.contract_valid is True
        common = parse_generation_contract(result.common_text, allowed_evidence_ids=allowed)
        assert common.contract_valid is True
        assert common.contract is not None
        assert common.contract.evidence_ids


def test_invalid_json_and_invalid_contract_are_not_mapped_as_valid() -> None:
    invalid_json = validate_and_map_contract_text(
        text='{"answer": "unfinished"',
        contract_id=RESEARCH_AI_MINIMAL_ANSWER,
        allowed_evidence_ids={"E1"},
    )
    assert invalid_json.json_valid is False
    assert invalid_json.contract_valid is False
    assert invalid_json.common_text == ""

    invalid_contract = validate_and_map_contract_text(
        text=_text(
            {
                "answer": "Has no evidence.",
                "insufficient_evidence": False,
                "confidence": "high",
            }
        ),
        contract_id=RESEARCH_AI_MINIMAL_ANSWER,
        allowed_evidence_ids={"E1"},
    )
    assert invalid_contract.json_valid is True
    assert invalid_contract.contract_valid is False
    assert "evidence" in invalid_contract.missing_fields


def test_adaptive_router_selects_direct_contract_deterministically() -> None:
    assert (
        route_research_ai_contract(prompt_text="Compare the two paper methods.", metadata={})
        == RESEARCH_AI_COMPARISON
    )
    assert (
        route_research_ai_contract(
            prompt_text="What limitation does the paper report?",
            metadata={},
        )
        == RESEARCH_AI_LIMITATIONS
    )
    assert (
        route_research_ai_contract(prompt_text="List the main findings.", metadata={})
        == RESEARCH_AI_FINDINGS
    )
    assert (
        route_research_ai_contract(prompt_text="What does the paper conclude?", metadata={})
        == RESEARCH_AI_MINIMAL_ANSWER
    )


def test_adaptive_contract_maps_through_effective_candidate() -> None:
    result = validate_and_map_contract_text(
        text=_text(
            {
                "summary": "The paper reports two findings.",
                "findings": [{"claim": "Retrieval improves accuracy.", "evidence": ["E1"]}],
                "insufficient_evidence": False,
                "confidence": "medium",
            }
        ),
        contract_id=RESEARCH_AI_ADAPTIVE,
        allowed_evidence_ids={"E1"},
        prompt_text="Summarize the main findings from the paper.",
    )

    assert result.effective_contract_id == RESEARCH_AI_FINDINGS
    assert result.contract_valid is True
    assert parse_generation_contract(result.common_text, allowed_evidence_ids={"E1"}).contract_valid


def test_adaptive_router_ignores_generic_output_contract_wording() -> None:
    prompt = "\n\n".join(
        [
            "SYSTEM:\nUse supplied evidence only.",
            "USER QUESTION:\nSummarize the paper contribution. INTERNAL ANSWER PLAN: hidden plan.",
            "OUTPUT CONTRACT:\nFor comparisons, cite every entity discussed.",
        ]
    )

    assert route_research_ai_contract(prompt_text=prompt, metadata={}) == RESEARCH_AI_MINIMAL_ANSWER
