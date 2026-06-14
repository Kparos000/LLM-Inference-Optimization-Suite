from __future__ import annotations

from collections.abc import Iterator

from inference_bench.agents.langgraph_mm4 import (
    ModelGeneration,
    compile_mm4_graph,
    run_mm4_graph,
)
from inference_bench.agents.state import AgentState

VALID_OUTPUT = (
    '{"answer":"Revenue was 10 million dollars.","evidence_ids":["E1"],'
    '"confidence":0.9,"insufficient_evidence":false,'
    '"citation_notes":"E1 supports the revenue amount."}'
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


def _state() -> AgentState:
    return AgentState(
        prompt_id="finance_prompt_001",
        workload_id="smoke_500:mm4_bounded_agentic:finance_prompt_001",
        vertical="finance",
        user_question="What revenue was reported?",
        task_type="evidence_lookup",
        context_pool=[_context()],
        backend="test",
        model_name="test-model",
    )


class SequenceGenerator:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs: Iterator[str] = iter(outputs)
        self.call_count = 0

    def __call__(self, prompt: str) -> ModelGeneration:
        assert "hidden reasoning" in prompt
        self.call_count += 1
        return ModelGeneration(
            text=next(self.outputs),
            input_tokens=20,
            output_tokens=10,
            ttft_ms=5.0,
            tpot_ms=2.0,
            e2e_latency_ms=25.0,
        )


def test_graph_compiles_and_records_all_success_path_nodes() -> None:
    generator = SequenceGenerator([VALID_OUTPUT])
    graph = compile_mm4_graph(generator=generator)

    result = run_mm4_graph(graph=graph, initial_state=_state())
    traced_nodes = [event["node"] for event in result.trace_events]

    assert result.final_status == "answer"
    assert result.retrieval_rounds == 1
    assert result.tool_call_count == 2
    assert result.generation_attempts == 1
    assert generator.call_count == 1
    assert traced_nodes == [
        "classify_task",
        "plan_retrieval",
        "retrieve_context",
        "assemble_context",
        "generate_answer",
        "validate_output",
        "finalize_or_escalate",
    ]
    assert all(latency >= 0 for latency in result.node_latencies.values())


def test_graph_repairs_once_then_succeeds() -> None:
    generator = SequenceGenerator(["not json", VALID_OUTPUT])

    result = run_mm4_graph(
        graph=compile_mm4_graph(generator=generator),
        initial_state=_state(),
    )

    assert result.final_status == "answer"
    assert result.repair_attempts == 1
    assert result.generation_attempts == 2
    assert result.tool_call_count == 3
    assert generator.call_count == 2


def test_graph_escalates_after_the_single_failed_repair() -> None:
    generator = SequenceGenerator(["not json", "still not json"])

    result = run_mm4_graph(
        graph=compile_mm4_graph(generator=generator),
        initial_state=_state(),
    )

    assert result.final_status == "escalate"
    assert result.repair_attempts == 1
    assert result.generation_attempts == 2
    assert result.tool_call_count == 3
    assert result.validation_result["valid"] is False
