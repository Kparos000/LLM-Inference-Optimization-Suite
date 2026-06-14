"""Executable bounded LangGraph workflow for mm4 agentic benchmarking."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from inference_bench.agents.prompts import (
    render_agent_answer_prompt,
    render_agent_repair_prompt,
)
from inference_bench.agents.state import AgentState
from inference_bench.agents.tools import (
    assemble_context,
    consume_action_tool,
    escalate,
    repair_generation_once,
    retrieve_context,
    validate_evidence,
    validate_generation_contract,
    validate_safety,
)
from inference_bench.runners.mock_runner import count_whitespace_tokens


@dataclass(frozen=True)
class ModelGeneration:
    """One generation attempt returned by an injected backend adapter."""

    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    e2e_latency_ms: float
    tpot_ms: float | None = None
    cost_usd: float = 0.0


class GeneratorProtocol(Protocol):
    """Backend-neutral model callable used by graph generation nodes."""

    def __call__(self, prompt: str) -> ModelGeneration:
        """Generate one bounded response."""


NodeUpdate = dict[str, Any]
NodeFunction = Callable[[AgentState], NodeUpdate]


def _event(
    *,
    state: AgentState,
    node: str,
    latency_ms: float,
    tool_name: str | None = None,
    status: str = "completed",
) -> list[dict[str, Any]]:
    events = list(state.trace_events)
    events.append(
        {
            "sequence": len(events) + 1,
            "node": node,
            "event_type": "node_complete",
            "latency_ms": latency_ms,
            "tool_name": tool_name,
            "status": status,
        }
    )
    return events


def _with_timing(
    node_name: str,
    function: NodeFunction,
    *,
    tool_name: str | None = None,
) -> NodeFunction:
    def timed(state: AgentState) -> NodeUpdate:
        started = time.perf_counter()
        try:
            update = function(state)
            status = "completed"
        except Exception as exc:
            update = {
                "error_type": type(exc).__name__,
                "final_status": "failed_validation",
                "escalation_reason": str(exc),
            }
            status = "failed"
        latency_ms = (time.perf_counter() - started) * 1000
        latencies = dict(state.node_latencies)
        latencies[node_name] = latencies.get(node_name, 0.0) + latency_ms
        update["node_latencies"] = latencies
        update["trace_events"] = _event(
            state=state,
            node=node_name,
            latency_ms=latency_ms,
            tool_name=tool_name,
            status=status,
        )
        return update

    return timed


def _classify_task(state: AgentState) -> NodeUpdate:
    high_risk = state.vertical in {"finance", "healthcare_admin"}
    if any(
        term in state.user_question.lower()
        for term in ("urgent", "diagnosis", "investment", "price target", "privacy")
    ):
        high_risk = True
    return {
        "risk_level": "high" if high_risk else "standard",
    }


def _plan_retrieval(state: AgentState) -> NodeUpdate:
    return {
        "retrieval_plan": {
            "strategy": "promoted_hybrid_top5",
            "top_k": 5,
            "max_rounds": 2,
            "project_corpus_only": True,
            "compression_allowed": True,
            "risk_level": state.risk_level,
        }
    }


def _retrieve_context(state: AgentState) -> NodeUpdate:
    contexts, rounds = retrieve_context(
        context_pool=state.context_pool,
        retrieval_rounds=state.retrieval_rounds,
        top_k=int(state.retrieval_plan.get("top_k", 5)),
    )
    return {
        "retrieved_context": contexts,
        "retrieval_rounds": rounds,
        "tool_call_count": consume_action_tool(
            "retrieve_context",
            state.tool_call_count,
        ),
    }


def _assemble_context(state: AgentState) -> NodeUpdate:
    prompt, labels, aliases = assemble_context(
        question=state.user_question,
        retrieved_context=state.retrieved_context,
    )
    return {
        "assembled_prompt": render_agent_answer_prompt(evidence_prompt=prompt),
        "allowed_evidence_ids": labels,
        "citation_id_aliases": aliases,
        "tool_call_count": consume_action_tool(
            "assemble_context",
            state.tool_call_count,
        ),
    }


def _merge_generation(
    state: AgentState,
    generation: ModelGeneration,
    *,
    prompt: str,
) -> NodeUpdate:
    token_usage = dict(state.token_usage)
    token_usage["input_tokens"] = int(token_usage.get("input_tokens", 0)) + max(
        0, generation.input_tokens
    )
    token_usage["output_tokens"] = int(token_usage.get("output_tokens", 0)) + max(
        0, generation.output_tokens
    )
    token_usage["total_tokens"] = int(token_usage["input_tokens"]) + int(
        token_usage["output_tokens"]
    )
    token_usage["cost_usd"] = float(token_usage.get("cost_usd", 0.0)) + max(
        0.0, generation.cost_usd
    )
    token_usage["comparison_input_tokens"] = int(
        token_usage.get("comparison_input_tokens", 0)
    ) + count_whitespace_tokens(prompt)
    token_usage["comparison_output_tokens"] = int(
        token_usage.get("comparison_output_tokens", 0)
    ) + count_whitespace_tokens(generation.text)
    metrics = dict(state.generation_metrics)
    if "first_ttft_ms" not in metrics:
        metrics["first_ttft_ms"] = generation.ttft_ms
    metrics.update(
        {
            "ttft_ms": generation.ttft_ms,
            "tpot_ms": generation.tpot_ms,
            "last_generation_e2e_latency_ms": generation.e2e_latency_ms,
            "generation_e2e_latency_ms": float(metrics.get("generation_e2e_latency_ms") or 0.0)
            + generation.e2e_latency_ms,
        }
    )
    return {
        "generated_answer": generation.text,
        "generation_attempts": state.generation_attempts + 1,
        "token_usage": token_usage,
        "generation_metrics": metrics,
    }


def _generate_answer(
    state: AgentState,
    *,
    generator: GeneratorProtocol,
) -> NodeUpdate:
    if state.generation_attempts >= 2:
        msg = "max_generation_attempts reached"
        raise RuntimeError(msg)
    return _merge_generation(
        state,
        generator(state.assembled_prompt),
        prompt=state.assembled_prompt,
    )


def _validate_output(state: AgentState) -> NodeUpdate:
    contract_parse = validate_generation_contract(
        generated_text=state.generated_answer,
        allowed_evidence_ids=state.allowed_evidence_ids,
    )
    evidence = validate_evidence(
        contract_parse=contract_parse,
        allowed_evidence_ids=state.allowed_evidence_ids,
    )
    safety = validate_safety(
        generated_text=state.generated_answer,
        vertical=state.vertical,
    )
    valid = contract_parse.contract_valid and evidence["valid"] and safety["valid"]
    contract = contract_parse.contract
    return {
        "selected_evidence": list(evidence.get("evidence_ids") or []),
        "validation_result": {
            "valid": valid,
            "generation_contract_valid": contract_parse.contract_valid,
            "generation_contract_error": contract_parse.error,
            "parse_error_type": contract_parse.parse_error_type,
            "parse_repair_applied": contract_parse.parse_repair_applied,
            "truncation_detected": contract_parse.truncation_detected,
            "evidence": evidence,
            "safety": safety,
            "insufficient_evidence": (
                contract.insufficient_evidence if contract is not None else None
            ),
        },
    }


def _repair_once(
    state: AgentState,
    *,
    generator: GeneratorProtocol,
) -> NodeUpdate:
    repair_attempts = repair_generation_once(repair_attempts=state.repair_attempts)
    tool_call_count = consume_action_tool(
        "repair_generation_once",
        state.tool_call_count,
    )
    violation = str(
        state.validation_result.get("generation_contract_error")
        or state.validation_result.get("evidence", {}).get("reason")
        or state.validation_result.get("safety", {}).get("matched_patterns")
        or "validation_failed"
    )
    repair_prompt = render_agent_repair_prompt(
        original_prompt=state.assembled_prompt,
        previous_output=state.generated_answer,
        violation=violation,
        allowed_evidence_ids=state.allowed_evidence_ids,
    )
    generation_update = _merge_generation(
        state,
        generator(repair_prompt),
        prompt=repair_prompt,
    )
    generation_update.update(
        {
            "repair_attempts": repair_attempts,
            "repair_prompt": repair_prompt,
            "tool_call_count": tool_call_count,
        }
    )
    return generation_update


def _finalize_or_escalate(state: AgentState) -> NodeUpdate:
    if state.validation_result.get("valid"):
        if state.validation_result.get("insufficient_evidence"):
            return {
                "final_status": "insufficient_evidence",
                "escalation_reason": "model_reported_insufficient_evidence",
            }
        return {
            "final_status": "answer",
            "escalation_reason": "",
        }
    return escalate(
        reason=str(
            state.validation_result.get("generation_contract_error")
            or state.validation_result.get("evidence", {}).get("reason")
            or "mm4_validation_failed"
        )
    )


def _after_validation(state: AgentState) -> str:
    if state.final_status == "failed_validation" or state.error_type:
        return "finalize_or_escalate"
    if state.validation_result.get("valid"):
        return "finalize_or_escalate"
    if state.repair_attempts < 1 and state.generation_attempts < 2 and state.tool_call_count < 3:
        return "repair_once"
    return "finalize_or_escalate"


CompiledMm4Graph = CompiledStateGraph[
    AgentState,
    None,
    AgentState,
    AgentState,
]


def compile_mm4_graph(*, generator: GeneratorProtocol) -> CompiledMm4Graph:
    """Compile the bounded mm4 graph with an injected model generator."""

    graph = StateGraph(AgentState)
    graph.add_node(
        "classify_task",
        cast(Any, _with_timing("classify_task", _classify_task)),
    )
    graph.add_node(
        "plan_retrieval",
        cast(Any, _with_timing("plan_retrieval", _plan_retrieval)),
    )
    graph.add_node(
        "retrieve_context",
        cast(
            Any,
            _with_timing(
                "retrieve_context",
                _retrieve_context,
                tool_name="retrieve_context",
            ),
        ),
    )
    graph.add_node(
        "assemble_context",
        cast(
            Any,
            _with_timing(
                "assemble_context",
                _assemble_context,
                tool_name="assemble_context",
            ),
        ),
    )
    graph.add_node(
        "generate_answer",
        cast(
            Any,
            _with_timing(
                "generate_answer",
                lambda state: _generate_answer(state, generator=generator),
            ),
        ),
    )
    graph.add_node(
        "validate_output",
        cast(Any, _with_timing("validate_output", _validate_output)),
    )
    graph.add_node(
        "repair_once",
        cast(
            Any,
            _with_timing(
                "repair_once",
                lambda state: _repair_once(state, generator=generator),
                tool_name="repair_generation_once",
            ),
        ),
    )
    graph.add_node(
        "finalize_or_escalate",
        cast(
            Any,
            _with_timing("finalize_or_escalate", _finalize_or_escalate),
        ),
    )

    graph.add_edge(START, "classify_task")
    graph.add_edge("classify_task", "plan_retrieval")
    graph.add_edge("plan_retrieval", "retrieve_context")
    graph.add_edge("retrieve_context", "assemble_context")
    graph.add_edge("assemble_context", "generate_answer")
    graph.add_edge("generate_answer", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        _after_validation,
        {
            "repair_once": "repair_once",
            "finalize_or_escalate": "finalize_or_escalate",
        },
    )
    graph.add_edge("repair_once", "validate_output")
    graph.add_edge("finalize_or_escalate", END)
    return graph.compile()


def run_mm4_graph(
    *,
    graph: CompiledMm4Graph,
    initial_state: AgentState,
) -> AgentState:
    """Invoke a compiled graph and validate the final state."""

    result = graph.invoke(initial_state)
    return AgentState.model_validate(result)
