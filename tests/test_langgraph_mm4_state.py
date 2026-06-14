from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_bench.agents.state import AgentState


def _state(**overrides: object) -> AgentState:
    payload: dict[str, object] = {
        "prompt_id": "finance_prompt_001",
        "workload_id": "smoke_500:mm4_bounded_agentic:finance_prompt_001",
        "vertical": "finance",
        "user_question": "What revenue was reported?",
        "task_type": "evidence_lookup",
        "backend": "test",
        "model_name": "test-model",
    }
    payload.update(overrides)
    return AgentState.model_validate(payload)


def test_agent_state_validates_required_schema() -> None:
    state = _state()

    assert state.prompt_id == "finance_prompt_001"
    assert state.memory_mode == "mm4_bounded_agentic"
    assert state.retrieval_rounds == 0
    assert state.trace_events == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("retrieval_rounds", 3, "max_retrieval_rounds"),
        ("generation_attempts", 3, "max_generation_attempts"),
        ("repair_attempts", 2, "max_repair_attempts"),
        ("tool_call_count", 4, "max_tool_calls"),
    ],
)
def test_agent_state_enforces_hard_limits(
    field: str,
    value: int,
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        _state(**{field: value})


def test_agent_state_rejects_unknown_vertical() -> None:
    with pytest.raises(ValidationError, match="Unsupported vertical"):
        _state(vertical="internet")
