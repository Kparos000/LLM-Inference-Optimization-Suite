"""Benchmarkable trace schema for executable mm4 LangGraph runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference_bench.agentic_contract import MM4_BOUNDED_AGENTIC_CONTRACT
from inference_bench.agents.state import AgentState
from inference_bench.context_schema import VALID_VERTICALS


class AgentTraceEvent(BaseModel):
    """One public, non-chain-of-thought graph event."""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    node: str
    event_type: str
    latency_ms: float
    tool_name: str | None = None
    status: str

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: int) -> int:
        if value <= 0:
            msg = "sequence must be > 0"
            raise ValueError(msg)
        return value

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(cls, value: float) -> float:
        if value < 0:
            msg = "latency_ms must be >= 0"
            raise ValueError(msg)
        return value


class AgentTraceRecord(BaseModel):
    """One complete agent invocation trace suitable for JSONL output."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    run_id: str
    prompt_id: str
    workload_id: str
    vertical: str
    memory_mode: str
    backend: str
    model_name: str
    task_type: str
    risk_level: str
    retrieval_plan: dict[str, Any]
    retrieval_rounds: int
    selected_evidence: list[str]
    generation_attempts: int
    repair_attempts: int
    tool_call_count: int
    validation_result: dict[str, Any]
    final_status: str
    escalation_reason: str
    node_latencies: dict[str, float]
    token_usage: dict[str, float | int]
    error_type: str
    trace_events: list[AgentTraceEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_record(self) -> AgentTraceRecord:
        limits = MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits
        if self.vertical not in VALID_VERTICALS:
            msg = f"Unsupported vertical: {self.vertical}"
            raise ValueError(msg)
        if self.memory_mode != "mm4_bounded_agentic":
            msg = "memory_mode must be mm4_bounded_agentic"
            raise ValueError(msg)
        if self.retrieval_rounds > limits.max_retrieval_rounds:
            msg = "retrieval_rounds exceeds hard limit"
            raise ValueError(msg)
        if self.generation_attempts > limits.max_generation_attempts:
            msg = "generation_attempts exceeds hard limit"
            raise ValueError(msg)
        if self.repair_attempts > limits.max_repair_attempts:
            msg = "repair_attempts exceeds hard limit"
            raise ValueError(msg)
        if self.tool_call_count > limits.max_tool_calls:
            msg = "tool_call_count exceeds hard limit"
            raise ValueError(msg)
        return self

    @classmethod
    def from_state(
        cls,
        *,
        state: AgentState,
        trace_id: str,
        run_id: str,
    ) -> AgentTraceRecord:
        """Build a trace from final validated graph state."""

        return cls(
            trace_id=trace_id,
            run_id=run_id,
            prompt_id=state.prompt_id,
            workload_id=state.workload_id,
            vertical=state.vertical,
            memory_mode=state.memory_mode,
            backend=state.backend,
            model_name=state.model_name,
            task_type=state.task_type,
            risk_level=state.risk_level,
            retrieval_plan=state.retrieval_plan,
            retrieval_rounds=state.retrieval_rounds,
            selected_evidence=state.selected_evidence,
            generation_attempts=state.generation_attempts,
            repair_attempts=state.repair_attempts,
            tool_call_count=state.tool_call_count,
            validation_result=state.validation_result,
            final_status=state.final_status,
            escalation_reason=state.escalation_reason,
            node_latencies=state.node_latencies,
            token_usage=state.token_usage,
            error_type=state.error_type,
            trace_events=[AgentTraceEvent.model_validate(event) for event in state.trace_events],
        )
