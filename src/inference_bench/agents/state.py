"""Validated state for the bounded LangGraph mm4 workflow."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inference_bench.agentic_contract import (
    AGENTIC_FINAL_STATUSES,
    MM4_BOUNDED_AGENTIC_CONTRACT,
)
from inference_bench.context_schema import VALID_VERTICALS


class AgentState(BaseModel):
    """Complete state carried through one mm4 graph invocation."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    workload_id: str
    vertical: str
    user_question: str
    task_type: str
    risk_level: str = "unclassified"
    retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    retrieval_rounds: int = 0
    retrieved_context: list[dict[str, Any]] = Field(default_factory=list)
    selected_evidence: list[str] = Field(default_factory=list)
    generated_answer: str = ""
    validation_result: dict[str, Any] = Field(default_factory=dict)
    repair_attempts: int = 0
    final_status: str = "running"
    escalation_reason: str = ""
    node_latencies: dict[str, float] = Field(default_factory=dict)
    token_usage: dict[str, float | int] = Field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    )
    tool_call_count: int = 0
    error_type: str = ""
    trace_events: list[dict[str, Any]] = Field(default_factory=list)
    context_pool: list[dict[str, Any]] = Field(default_factory=list)
    assembled_prompt: str = ""
    repair_prompt: str = ""
    allowed_evidence_ids: list[str] = Field(default_factory=list)
    citation_id_aliases: dict[str, list[str]] = Field(default_factory=dict)
    generation_attempts: int = 0
    generation_metrics: dict[str, float | int | None] = Field(default_factory=dict)
    backend: str = "unconfigured"
    model_name: str = "unconfigured"
    memory_mode: str = "mm4_bounded_agentic"
    ablation_mode: str = "prompt_plus_metadata"
    expected_output_format: str = "generation_contract_json"
    source_prompt_record: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "prompt_id",
        "workload_id",
        "vertical",
        "user_question",
        "task_type",
        "backend",
        "model_name",
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Require non-empty identity and execution strings."""

        if not value.strip():
            msg = "Agent state string fields must not be empty"
            raise ValueError(msg)
        return value

    @field_validator("vertical")
    @classmethod
    def validate_vertical(cls, value: str) -> str:
        """Require a supported benchmark vertical."""

        if value not in VALID_VERTICALS:
            msg = f"Unsupported vertical: {value}"
            raise ValueError(msg)
        return value

    @field_validator(
        "retrieval_rounds",
        "repair_attempts",
        "tool_call_count",
        "generation_attempts",
    )
    @classmethod
    def validate_non_negative_count(cls, value: int) -> int:
        """Require non-negative execution counters."""

        if value < 0:
            msg = "Agent execution counters must be non-negative"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_hard_limits(self) -> AgentState:
        """Enforce the global mm4 limits on every state transition."""

        limits = MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits
        if self.retrieval_rounds > limits.max_retrieval_rounds:
            msg = "retrieval_rounds exceeds max_retrieval_rounds"
            raise ValueError(msg)
        if self.generation_attempts > limits.max_generation_attempts:
            msg = "generation_attempts exceeds max_generation_attempts"
            raise ValueError(msg)
        if self.repair_attempts > limits.max_repair_attempts:
            msg = "repair_attempts exceeds max_repair_attempts"
            raise ValueError(msg)
        if self.tool_call_count > limits.max_tool_calls:
            msg = "tool_call_count exceeds max_tool_calls"
            raise ValueError(msg)
        if self.final_status not in {"running", *AGENTIC_FINAL_STATUSES}:
            msg = f"Unsupported final_status: {self.final_status}"
            raise ValueError(msg)
        return self
