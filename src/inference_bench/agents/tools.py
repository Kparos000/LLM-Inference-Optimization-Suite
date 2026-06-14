"""Approved deterministic tools for the bounded mm4 graph."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any

from inference_bench.agentic_contract import (
    APPROVED_AGENTIC_TOOLS,
    MM4_BOUNDED_AGENTIC_CONTRACT,
)
from inference_bench.context_schema import ContextRecord
from inference_bench.generation_contract import (
    GenerationContractParse,
    citation_aliases,
    citation_label,
    parse_generation_contract,
    render_generation_contract_prompt,
)

APPROVED_TOOLS = frozenset(APPROVED_AGENTIC_TOOLS)
ACTION_BUDGET_TOOLS = frozenset(
    {
        "retrieve_context",
        "assemble_context",
        "repair_generation_once",
    }
)

VERTICAL_SAFETY_PATTERNS: dict[str, tuple[str, ...]] = {
    "airline": (
        r"\bguaranteed compensation\b",
        r"\bguaranteed refund\b",
        r"\bbypass verification\b",
    ),
    "healthcare_admin": (
        r"\bdiagnos(?:e|is)\b",
        r"\bdosage\b",
        r"\bprescribe\b",
        r"\bbypass identity verification\b",
    ),
    "retail": (
        r"\braw user id\b",
        r"\bamazon guarantees\b",
    ),
    "finance": (
        r"\bprice target\b",
        r"\bguaranteed return\b",
        r"\b(?:buy|sell) recommendation\b",
    ),
    "research_ai": (
        r"\bthe paper proves\b",
        r"\bguaranteed result\b",
    ),
}


def ensure_approved_tool(tool_name: str) -> None:
    """Reject any tool outside the static mm4 allowlist."""

    if tool_name not in APPROVED_TOOLS:
        msg = f"Unauthorized mm4 tool: {tool_name}"
        raise ValueError(msg)


def consume_action_tool(tool_name: str, current_count: int) -> int:
    """Consume one externally meaningful action from the three-call budget."""

    ensure_approved_tool(tool_name)
    if tool_name not in ACTION_BUDGET_TOOLS:
        return current_count
    updated = current_count + 1
    if updated > MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits.max_tool_calls:
        msg = "tool_call_count exceeds max_tool_calls"
        raise RuntimeError(msg)
    return updated


def retrieve_context(
    *,
    context_pool: list[dict[str, Any]],
    retrieval_rounds: int,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], int]:
    """Select bounded context from the frozen promoted workload snapshot."""

    ensure_approved_tool("retrieve_context")
    if retrieval_rounds >= MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits.max_retrieval_rounds:
        msg = "max_retrieval_rounds reached"
        raise RuntimeError(msg)
    if top_k <= 0:
        msg = "top_k must be > 0"
        raise ValueError(msg)
    return [dict(context) for context in context_pool[:top_k]], retrieval_rounds + 1


def assemble_context(
    *,
    question: str,
    retrieved_context: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, list[str]]]:
    """Render mm4 evidence with stable short labels."""

    ensure_approved_tool("assemble_context")
    contexts = [ContextRecord(**context) for context in retrieved_context]
    aliases = {
        citation_label(index): citation_aliases(context)
        for index, context in enumerate(contexts, start=1)
    }
    prompt = render_generation_contract_prompt(
        question=question,
        context_records=contexts,
        memory_mode="mm4_bounded_agentic",
    )
    return prompt, list(aliases), aliases


def validate_generation_contract(
    *,
    generated_text: str,
    allowed_evidence_ids: Collection[str],
) -> GenerationContractParse:
    """Apply the unchanged strict generation-contract parser."""

    ensure_approved_tool("validate_generation_contract")
    return parse_generation_contract(
        generated_text,
        allowed_evidence_ids=allowed_evidence_ids,
    )


def validate_evidence(
    *,
    contract_parse: GenerationContractParse,
    allowed_evidence_ids: Collection[str],
) -> dict[str, Any]:
    """Check emitted labels without consulting evaluator-only gold data."""

    ensure_approved_tool("validate_evidence")
    contract = contract_parse.contract
    if not contract_parse.contract_valid or contract is None:
        return {
            "valid": False,
            "reason": "generation_contract_invalid",
            "evidence_ids": [],
        }
    if contract.insufficient_evidence:
        return {
            "valid": True,
            "reason": "insufficient_evidence_contract",
            "evidence_ids": [],
        }
    allowed = set(allowed_evidence_ids)
    emitted = list(dict.fromkeys(contract.evidence_ids))
    invalid = [evidence_id for evidence_id in emitted if evidence_id not in allowed]
    valid = bool(emitted) and not invalid
    return {
        "valid": valid,
        "reason": "evidence_valid" if valid else "invalid_evidence_labels",
        "evidence_ids": emitted,
        "invalid_evidence_ids": invalid,
    }


def validate_safety(*, generated_text: str, vertical: str) -> dict[str, Any]:
    """Apply a bounded vertical safety screen independent of gold answers."""

    ensure_approved_tool("validate_safety")
    patterns = VERTICAL_SAFETY_PATTERNS.get(vertical, ())
    matches = [
        pattern for pattern in patterns if re.search(pattern, generated_text, flags=re.IGNORECASE)
    ]
    return {
        "valid": not matches,
        "matched_patterns": matches,
    }


def repair_generation_once(*, repair_attempts: int) -> int:
    """Consume the single allowed repair attempt."""

    ensure_approved_tool("repair_generation_once")
    if repair_attempts >= MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits.max_repair_attempts:
        msg = "max_repair_attempts reached"
        raise RuntimeError(msg)
    return repair_attempts + 1


def escalate(*, reason: str) -> dict[str, str]:
    """Create a deterministic escalation state transition."""

    ensure_approved_tool("escalate")
    return {
        "final_status": "escalate",
        "escalation_reason": reason or "mm4_validation_failed",
    }
