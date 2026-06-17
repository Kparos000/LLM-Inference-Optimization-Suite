from __future__ import annotations

import inspect
import json

import pytest

from inference_bench.generation_contract_registry import (
    RESEARCH_AI_ADAPTIVE,
    RESEARCH_AI_COMPARISON,
    RESEARCH_AI_MINIMAL_ANSWER,
)
from inference_bench.research_ai_contract_renderer import (
    render_research_ai_contract_item,
)
from inference_bench.schema import WorkloadItem


def _item(prompt: str = "SYSTEM:\nUse evidence.\n\nOUTPUT CONTRACT:\nold") -> WorkloadItem:
    return WorkloadItem(
        prompt_id="research_ai_001",
        workload_name="b6",
        prompt=prompt,
        expected_output=None,
        metadata={
            "vertical": "research_ai",
            "citation_id_aliases": json.dumps({"E1": ["paper-a-section-1"], "E2": ["paper-b"]}),
            "gold_evidence_ids": json.dumps(["canonical-gold-id"]),
            "task_type": "direct_answer",
        },
    )


def test_renderer_replaces_output_contract_and_preserves_alias_metadata() -> None:
    rendered = render_research_ai_contract_item(
        _item(),
        requested_contract_id=RESEARCH_AI_MINIMAL_ANSWER,
        max_new_tokens=224,
    )

    assert "OUTPUT CONTRACT:" in rendered.item.prompt
    assert "old" not in rendered.item.prompt
    assert "Research AI vertical contract" in rendered.item.prompt
    assert '"answer"' in rendered.item.prompt
    assert rendered.item.metadata["citation_id_aliases"] == json.dumps(
        {"E1": ["paper-a-section-1"], "E2": ["paper-b"]}
    )
    assert rendered.max_new_tokens == 224


def test_renderer_adaptive_contract_routes_without_llm() -> None:
    rendered = render_research_ai_contract_item(
        _item("USER QUESTION:\nCompare the papers.\n\nOUTPUT CONTRACT:\nold"),
        requested_contract_id=RESEARCH_AI_ADAPTIVE,
        max_new_tokens=320,
    )

    assert rendered.effective_contract_id == RESEARCH_AI_COMPARISON
    assert "comparison_summary" in rendered.item.prompt


def test_renderer_blocks_canonical_gold_leakage() -> None:
    with pytest.raises(ValueError, match="Canonical gold evidence IDs leaked"):
        render_research_ai_contract_item(
            _item("USER QUESTION:\ncanonical-gold-id\n\nOUTPUT CONTRACT:\nold"),
            requested_contract_id=RESEARCH_AI_MINIMAL_ANSWER,
            max_new_tokens=224,
        )


def test_renderer_rejects_unapproved_token_budget() -> None:
    with pytest.raises(ValueError, match="224 or 320"):
        render_research_ai_contract_item(
            _item(),
            requested_contract_id=RESEARCH_AI_MINIMAL_ANSWER,
            max_new_tokens=999,
        )


def test_renderer_does_not_touch_mm4_or_langgraph() -> None:
    source = inspect.getsource(render_research_ai_contract_item).lower()

    assert "mm4" not in source
    assert "langgraph" not in source
