from __future__ import annotations

import pytest

from inference_bench.agents.tools import (
    APPROVED_TOOLS,
    assemble_context,
    ensure_approved_tool,
    repair_generation_once,
    retrieve_context,
    validate_evidence,
    validate_generation_contract,
    validate_safety,
)


def _context() -> dict[str, object]:
    return {
        "context_id": "ctx-finance-001",
        "vertical": "finance",
        "source_id": "SEC-001",
        "parent_id": "SEC-001",
        "chunk_id": "SEC-001",
        "chunk_strategy": "atomic_fact",
        "source_type": "sec_xbrl_fact",
        "title": "Revenue fact",
        "text": "Reported revenue was 10 million dollars.",
        "metadata": {"ticker": "TEST"},
        "token_estimate": 8,
        "provenance": "test_fixture",
        "is_gold_linked": True,
    }


def test_only_the_approved_tools_are_exposed() -> None:
    assert APPROVED_TOOLS == {
        "retrieve_context",
        "assemble_context",
        "validate_generation_contract",
        "validate_evidence",
        "validate_safety",
        "repair_generation_once",
        "escalate",
    }
    assert "web_search" not in APPROVED_TOOLS
    assert "shell" not in APPROVED_TOOLS


def test_unauthorized_tool_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unauthorized mm4 tool"):
        ensure_approved_tool("web_search")


def test_retrieval_round_limit_is_enforced() -> None:
    contexts, rounds = retrieve_context(
        context_pool=[_context()],
        retrieval_rounds=1,
    )

    assert rounds == 2
    assert len(contexts) == 1
    with pytest.raises(RuntimeError, match="max_retrieval_rounds"):
        retrieve_context(context_pool=[_context()], retrieval_rounds=2)


def test_repair_limit_is_enforced() -> None:
    assert repair_generation_once(repair_attempts=0) == 1
    with pytest.raises(RuntimeError, match="max_repair_attempts"):
        repair_generation_once(repair_attempts=1)


def test_contract_evidence_and_safety_validation_are_local() -> None:
    _, labels, _ = assemble_context(
        question="What revenue was reported?",
        retrieved_context=[_context()],
    )
    parsed = validate_generation_contract(
        generated_text=(
            '{"answer":"Revenue was 10 million dollars.","evidence_ids":["E1"],'
            '"confidence":0.9,"insufficient_evidence":false,'
            '"citation_notes":"E1 supports the revenue amount."}'
        ),
        allowed_evidence_ids=labels,
    )

    assert (
        validate_evidence(
            contract_parse=parsed,
            allowed_evidence_ids=labels,
        )["valid"]
        is True
    )
    assert (
        validate_safety(
            generated_text="Revenue was reported.",
            vertical="finance",
        )["valid"]
        is True
    )
    assert (
        validate_safety(
            generated_text="The filing gives a guaranteed return.",
            vertical="finance",
        )["valid"]
        is False
    )


def test_evidence_validation_matches_existing_contract_semantics() -> None:
    parsed = validate_generation_contract(
        generated_text=(
            '{"answer":"Revenue was reported.","evidence_ids":["E1"],'
            '"confidence":0.8,"insufficient_evidence":false,"citation_notes":""}'
        ),
        allowed_evidence_ids=["E1"],
    )

    assert (
        validate_evidence(
            contract_parse=parsed,
            allowed_evidence_ids=["E1"],
        )["valid"]
        is True
    )
