"""Bounded agentic memory-mode contract for Phase 3.

This module defines validation-only schemas for the future
``mm4_bounded_agentic`` mode. It does not call models, tools, APIs, or
retrievers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from inference_bench.context_schema import VALID_VERTICALS

MM4_MEMORY_MODE = "mm4_bounded_agentic"

AGENTIC_WORKFLOW_STEPS = (
    "classify_task_risk",
    "select_retrieval_strategy",
    "retrieve_evidence",
    "assemble_context",
    "generate_answer",
    "validate_citations_format_safety",
    "repair_once_if_needed",
    "escalate_if_evidence_insufficient",
)

APPROVED_AGENTIC_TOOLS = (
    "retrieve_context",
    "assemble_context",
    "validate_citations",
    "validate_format",
    "validate_safety",
    "repair_once",
    "escalate",
)

AGENTIC_FINAL_STATUSES = {
    "answer",
    "escalate",
    "insufficient_evidence",
    "failed_validation",
}


@dataclass(frozen=True)
class AgenticHardLimits:
    """Hard execution limits for the bounded agentic contract."""

    max_tool_calls: int = 3
    max_retrieval_rounds: int = 2
    max_generation_attempts: int = 2
    max_repair_attempts: int = 1
    no_internet: bool = True
    no_arbitrary_tools: bool = True
    corpus_scope: str = "project_corpus_only"

    def to_dict(self) -> dict[str, int | bool | str]:
        """Return JSON-safe hard limits."""

        return asdict(self)


@dataclass(frozen=True)
class BoundedAgenticContract:
    """Configuration contract for ``mm4_bounded_agentic``."""

    memory_mode: str
    workflow_steps: tuple[str, ...]
    approved_tools: tuple[str, ...]
    hard_limits: AgenticHardLimits
    contract_stage: str = "phase3_contract_only"
    no_model_inference_triggered: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe contract payload."""

        return {
            "memory_mode": self.memory_mode,
            "workflow_steps": list(self.workflow_steps),
            "approved_tools": list(self.approved_tools),
            "hard_limits": self.hard_limits.to_dict(),
            "contract_stage": self.contract_stage,
            "no_model_inference_triggered": self.no_model_inference_triggered,
        }


MM4_BOUNDED_AGENTIC_CONTRACT = BoundedAgenticContract(
    memory_mode=MM4_MEMORY_MODE,
    workflow_steps=AGENTIC_WORKFLOW_STEPS,
    approved_tools=APPROVED_AGENTIC_TOOLS,
    hard_limits=AgenticHardLimits(),
)


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be an integer >= 0"
        raise ValueError(msg)


def _validate_dict(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        msg = f"{field_name} must be an object/dict"
        raise ValueError(msg)


def _validate_agentic_step(step: dict[str, Any], index: int) -> None:
    _validate_dict(step, f"steps[{index}]")
    _validate_non_empty_string(str(step.get("step_name") or ""), f"steps[{index}].step_name")
    if bool(step.get("uses_internet", False)):
        msg = f"steps[{index}] may not use internet retrieval"
        raise ValueError(msg)
    tool_name = step.get("tool_name")
    if tool_name is None or tool_name == "":
        return
    if not isinstance(tool_name, str):
        msg = f"steps[{index}].tool_name must be a string when provided"
        raise ValueError(msg)
    if tool_name not in APPROVED_AGENTIC_TOOLS:
        msg = f"steps[{index}] uses unapproved tool '{tool_name}'"
        raise ValueError(msg)


@dataclass(frozen=True)
class AgenticTrace:
    """Future trace schema for bounded agentic memory-mode runs."""

    trace_id: str
    workload_id: str
    prompt_id: str
    memory_mode: str
    vertical: str
    steps: list[dict[str, Any]]
    selected_retrieval_strategy: str
    retrieval_rounds: int
    generation_attempts: int
    repair_attempts: int
    validation_results: dict[str, Any]
    escalated: bool
    escalation_reason: str
    final_status: str
    token_estimates: dict[str, int]
    latency_placeholders: dict[str, float | None]
    backend_placeholder: str
    model_placeholder: str

    def __post_init__(self) -> None:
        limits = MM4_BOUNDED_AGENTIC_CONTRACT.hard_limits
        _validate_non_empty_string(self.trace_id, "trace_id")
        _validate_non_empty_string(self.workload_id, "workload_id")
        _validate_non_empty_string(self.prompt_id, "prompt_id")
        if self.memory_mode != MM4_MEMORY_MODE:
            msg = f"memory_mode must be {MM4_MEMORY_MODE}"
            raise ValueError(msg)
        if self.vertical not in VALID_VERTICALS:
            msg = f"vertical must be one of: {', '.join(sorted(VALID_VERTICALS))}"
            raise ValueError(msg)
        if not isinstance(self.steps, list) or not self.steps:
            msg = "steps must be a non-empty list"
            raise ValueError(msg)
        for index, step in enumerate(self.steps):
            _validate_agentic_step(step, index)
        _validate_non_empty_string(
            self.selected_retrieval_strategy,
            "selected_retrieval_strategy",
        )
        _validate_non_negative_int(self.retrieval_rounds, "retrieval_rounds")
        _validate_non_negative_int(self.generation_attempts, "generation_attempts")
        _validate_non_negative_int(self.repair_attempts, "repair_attempts")
        if self.retrieval_rounds > limits.max_retrieval_rounds:
            msg = "retrieval_rounds exceeds max_retrieval_rounds"
            raise ValueError(msg)
        if self.generation_attempts > limits.max_generation_attempts:
            msg = "generation_attempts exceeds max_generation_attempts"
            raise ValueError(msg)
        if self.repair_attempts > limits.max_repair_attempts:
            msg = "repair_attempts exceeds max_repair_attempts"
            raise ValueError(msg)
        _validate_dict(self.validation_results, "validation_results")
        if not isinstance(self.escalated, bool):
            msg = "escalated must be boolean"
            raise ValueError(msg)
        if self.escalated:
            _validate_non_empty_string(self.escalation_reason, "escalation_reason")
        if self.final_status not in AGENTIC_FINAL_STATUSES:
            msg = f"final_status must be one of: {', '.join(sorted(AGENTIC_FINAL_STATUSES))}"
            raise ValueError(msg)
        _validate_dict(self.token_estimates, "token_estimates")
        for key, token_value in self.token_estimates.items():
            _validate_non_negative_int(token_value, f"token_estimates.{key}")
        _validate_dict(self.latency_placeholders, "latency_placeholders")
        for key, latency_value in self.latency_placeholders.items():
            if latency_value is not None and (
                not isinstance(latency_value, int | float) or latency_value < 0
            ):
                msg = f"latency_placeholders.{key} must be None or a number >= 0"
                raise ValueError(msg)
        _validate_non_empty_string(self.backend_placeholder, "backend_placeholder")
        _validate_non_empty_string(self.model_placeholder, "model_placeholder")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe trace payload."""

        return asdict(self)


def agentic_trace_format() -> dict[str, Any]:
    """Return the expected trace format and hard limits."""

    return {
        "schema_name": "AgenticTrace",
        "required_fields": [
            "trace_id",
            "workload_id",
            "prompt_id",
            "memory_mode",
            "vertical",
            "steps",
            "selected_retrieval_strategy",
            "retrieval_rounds",
            "generation_attempts",
            "repair_attempts",
            "validation_results",
            "escalated",
            "escalation_reason",
            "final_status",
            "token_estimates",
            "latency_placeholders",
            "backend_placeholder",
            "model_placeholder",
        ],
        "final_status_values": sorted(AGENTIC_FINAL_STATUSES),
        "contract": MM4_BOUNDED_AGENTIC_CONTRACT.to_dict(),
    }


def valid_agentic_trace_fixture() -> AgenticTrace:
    """Return a contract-only valid trace fixture."""

    return AgenticTrace(
        trace_id="trace_fixture_001",
        workload_id="test_fixture:mm4_bounded_agentic:prompt_fixture_001",
        prompt_id="prompt_fixture_001",
        memory_mode=MM4_MEMORY_MODE,
        vertical="finance",
        steps=[
            {"step_name": "classify_task_risk", "tool_name": None, "uses_internet": False},
            {
                "step_name": "select_retrieval_strategy",
                "tool_name": None,
                "uses_internet": False,
            },
            {
                "step_name": "retrieve_evidence",
                "tool_name": "retrieve_context",
                "uses_internet": False,
            },
            {
                "step_name": "assemble_context",
                "tool_name": "assemble_context",
                "uses_internet": False,
            },
            {
                "step_name": "validate_citations_format_safety",
                "tool_name": "validate_citations",
                "uses_internet": False,
            },
        ],
        selected_retrieval_strategy="hybrid_top5",
        retrieval_rounds=1,
        generation_attempts=1,
        repair_attempts=0,
        validation_results={
            "citations_valid": True,
            "format_valid": True,
            "safety_valid": True,
        },
        escalated=False,
        escalation_reason="",
        final_status="answer",
        token_estimates={
            "prompt_tokens": 32,
            "context_tokens": 512,
            "output_tokens": 96,
        },
        latency_placeholders={
            "retrieval_latency_ms": None,
            "generation_latency_ms": None,
            "validation_latency_ms": None,
        },
        backend_placeholder="not_run_contract_only",
        model_placeholder="not_run_contract_only",
    )
