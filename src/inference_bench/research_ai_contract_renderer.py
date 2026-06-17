"""Research AI vertical-specific generation contract rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from inference_bench.generation_contract import allowed_evidence_ids_from_aliases
from inference_bench.generation_contract_registry import (
    B6R2_CONTRACT_MAX_NEW_TOKENS,
    RESEARCH_AI_ADAPTIVE,
    RESEARCH_AI_COMPARISON,
    RESEARCH_AI_FINDINGS,
    RESEARCH_AI_LIMITATIONS,
    RESEARCH_AI_MINIMAL_ANSWER,
    effective_contract_id,
)
from inference_bench.schema import WorkloadItem


@dataclass(frozen=True)
class RenderedResearchAiContract:
    """A runner item with a Research AI contract-specific renderer applied."""

    item: WorkloadItem
    requested_contract_id: str
    effective_contract_id: str
    max_new_tokens: int


def _labels_from_item(item: WorkloadItem) -> list[str]:
    return allowed_evidence_ids_from_aliases(item.metadata.get("citation_id_aliases"))


def _example_schema(contract_id: str) -> str:
    if contract_id == RESEARCH_AI_MINIMAL_ANSWER:
        payload: dict[str, Any] = {
            "answer": "one to three sentences",
            "evidence": ["E1"],
            "insufficient_evidence": False,
            "confidence": "medium",
        }
    elif contract_id == RESEARCH_AI_FINDINGS:
        payload = {
            "summary": "one sentence",
            "findings": [{"claim": "short claim", "evidence": ["E1"]}],
            "insufficient_evidence": False,
            "confidence": "medium",
        }
    elif contract_id == RESEARCH_AI_LIMITATIONS:
        payload = {
            "limitation": "short limitation",
            "why_it_matters": "short explanation",
            "evidence": ["E1"],
            "insufficient_evidence": False,
            "confidence": "medium",
        }
    elif contract_id == RESEARCH_AI_COMPARISON:
        payload = {
            "comparison_summary": "one sentence",
            "items": [{"item": "method/paper/result", "claim": "short claim", "evidence": ["E1"]}],
            "insufficient_evidence": False,
            "confidence": "medium",
        }
    else:
        raise ValueError(f"Unsupported Research AI direct contract: {contract_id}")
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def render_research_ai_contract_instruction(
    *,
    requested_contract_id: str,
    effective_contract_id: str,
    allowed_labels: list[str],
) -> str:
    """Render compact Research AI JSON-only instructions for one direct contract."""

    labels = ", ".join(allowed_labels) if allowed_labels else "none"
    route_line = (
        f"Requested contract: {requested_contract_id}; effective contract: {effective_contract_id}."
        if requested_contract_id == RESEARCH_AI_ADAPTIVE
        else f"Contract: {effective_contract_id}."
    )
    return "\n".join(
        [
            "OUTPUT CONTRACT:",
            "Research AI vertical contract.",
            route_line,
            "Return exactly one compact single-line JSON object.",
            "Do not use markdown, code fences, headings, or prose outside JSON.",
            "Use only supplied evidence labels; allowed labels: " + labels + ".",
            "Cite every supplied label used to support a claim.",
            "If evidence is insufficient, set insufficient_evidence to true.",
            "Do not answer from model memory.",
            "Do not include reasoning traces or hidden planning text.",
            "Keep every field concise to avoid truncation.",
            "confidence must be exactly one of: low, medium, high.",
            "Short labels such as E1 remain private aliases for evaluator expansion.",
            "Schema:",
            _example_schema(effective_contract_id),
        ]
    )


def _prompt_without_output_contract(prompt: str) -> str:
    marker = "\n\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.split(marker, maxsplit=1)[0].rstrip()
    marker = "\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.split(marker, maxsplit=1)[0].rstrip()
    return prompt.rstrip()


def _gold_evidence_ids(item: WorkloadItem) -> list[str]:
    raw = item.metadata.get("gold_evidence_ids")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(value) for value in payload if str(value)]


def leaked_gold_ids(item: WorkloadItem) -> list[str]:
    """Return canonical gold IDs that appear in the model-facing prompt."""

    prompt = item.prompt.lower()
    return [gold_id for gold_id in _gold_evidence_ids(item) if gold_id.lower() in prompt]


def assert_no_canonical_gold_id_leakage(item: WorkloadItem) -> None:
    """Fail if the rendered prompt exposes canonical gold evidence IDs."""

    leaked = leaked_gold_ids(item)
    if leaked:
        raise ValueError(f"Canonical gold evidence IDs leaked into prompt: {', '.join(leaked)}")


def render_research_ai_contract_item(
    item: WorkloadItem,
    *,
    requested_contract_id: str,
    max_new_tokens: int,
) -> RenderedResearchAiContract:
    """Apply a Research AI contract renderer without mutating aliases or gold data."""

    if max_new_tokens not in B6R2_CONTRACT_MAX_NEW_TOKENS:
        raise ValueError("B6R2 Research AI max_new_tokens must be 224 or 320")
    if item.metadata.get("vertical") != "research_ai":
        raise ValueError("Research AI contract renderer received a non-Research-AI item")
    effective = effective_contract_id(
        requested_contract_id,
        prompt_text=item.prompt,
        metadata=item.metadata,
    )
    allowed_labels = _labels_from_item(item)
    prompt = "\n\n".join(
        [
            _prompt_without_output_contract(item.prompt),
            render_research_ai_contract_instruction(
                requested_contract_id=requested_contract_id,
                effective_contract_id=effective,
                allowed_labels=allowed_labels,
            ),
        ]
    )
    metadata = {
        **item.metadata,
        "b6r2_requested_research_ai_contract": requested_contract_id,
        "b6r2_effective_research_ai_contract": effective,
        "b6r2_research_ai_contract_version": "v1",
        "b6r2_max_new_tokens": str(max_new_tokens),
    }
    rendered = WorkloadItem(
        prompt_id=item.prompt_id,
        workload_name=item.workload_name,
        prompt=prompt,
        expected_output=item.expected_output,
        metadata=metadata,
    )
    assert_no_canonical_gold_id_leakage(rendered)
    return RenderedResearchAiContract(
        item=rendered,
        requested_contract_id=requested_contract_id,
        effective_contract_id=effective,
        max_new_tokens=max_new_tokens,
    )


def render_research_ai_retry_prompt(
    *,
    rendered_item: WorkloadItem,
    requested_contract_id: str,
    effective_contract_id: str,
    previous_output: str,
    issue: str,
    missing_labels: list[str] | tuple[str, ...] = (),
) -> str:
    """Render one bounded Research AI contract repair prompt."""

    allowed_labels = _labels_from_item(rendered_item)
    missing = ", ".join(missing_labels) if missing_labels else "none"
    return "\n\n".join(
        [
            rendered_item.prompt,
            "RESEARCH AI CONTRACT REPAIR:",
            "Correct only JSON structure, evidence labels, and concise wording.",
            "Previous output:",
            previous_output,
            f"Issue: {issue}",
            f"Missing supplied evidence labels to reconsider: {missing}",
            render_research_ai_contract_instruction(
                requested_contract_id=requested_contract_id,
                effective_contract_id=effective_contract_id,
                allowed_labels=allowed_labels,
            ),
            "Return only the corrected JSON object.",
        ]
    )
