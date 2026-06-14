from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from inference_bench.agentic_comparison import build_memory_mode_row
from inference_bench.agents.langgraph_mm4 import ModelGeneration
from inference_bench.config import load_yaml_file
from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.evaluator_contract import evaluate_generated_answers

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/phase4/run_mm4_agentic_smoke.py"
EXPERIMENT_PATH = REPO_ROOT / "configs/experiments/a5_mm4_bounded_agentic_smoke.yaml"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_mm4_agentic_smoke", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeGenerator:
    def __call__(self, prompt: str) -> ModelGeneration:
        assert "E1" in prompt
        return ModelGeneration(
            text=(
                '{"answer":"Revenue was 10 million dollars.","evidence_ids":["E1"],'
                '"confidence":0.9,"insufficient_evidence":false,'
                '"citation_notes":"E1 supports the revenue amount."}'
            ),
            input_tokens=20,
            output_tokens=10,
            ttft_ms=5.0,
            tpot_ms=2.0,
            e2e_latency_ms=25.0,
        )


def _record() -> WorkloadRecord:
    context = ContextRecord(
        context_id="ctx-finance-001",
        vertical="finance",
        source_id="SEC-001",
        parent_id="SEC-001",
        chunk_id="SEC-001",
        chunk_strategy="atomic_fact",
        source_type="sec_xbrl_fact",
        title="Revenue fact",
        text="Reported revenue was 10 million dollars.",
        metadata={"ticker": "TEST"},
        token_estimate=8,
        provenance="test_fixture",
        is_gold_linked=True,
    )
    return WorkloadRecord(
        workload_id="smoke_500:mm2_hybrid_top5:finance_prompt_001",
        prompt_id="finance_prompt_001",
        vertical="finance",
        memory_mode="mm2_hybrid_top5",
        messages=[{"role": "user", "content": "What revenue was reported?"}],
        context_records=[context],
        context_token_estimate=8,
        retrieval_metadata={"retrieval_type": "hybrid"},
        expected_output_format="generation_contract_json",
        gold_evidence_ids=["SEC-001"],
        dataset_split="test_fixture",
        source_prompt_record={
            "prompt_id": "finance_prompt_001",
            "question": "What revenue was reported?",
            "task_type": "evidence_lookup",
        },
    )


def test_mm4_experiment_config_is_frozen_and_bounded() -> None:
    config = load_yaml_file(EXPERIMENT_PATH)

    assert config["memory_mode"] == "mm4_bounded_agentic"
    assert config["framework"] == "langgraph"
    assert config["total_prompts"] == 50
    assert config["prompts_per_vertical"] == 10
    assert config["max_tool_calls"] == 3
    assert config["max_retrieval_rounds"] == 2
    assert config["max_generation_attempts"] == 2
    assert config["max_repair_attempts"] == 1
    assert config["internet_tools"] is False
    assert config["arbitrary_tools"] is False


def test_smoke_rows_are_traceable_and_scoreable_without_live_inference() -> None:
    module = _load_script()
    rows, traces = module._run_agentic_records(
        records=[_record()],
        generator=FakeGenerator(),
        backend="test",
        model_name="test-model",
        run_id="test-mm4",
    )
    evaluated = evaluate_generated_answers(
        rows,
        [
            {
                "prompt_id": "finance_prompt_001",
                "expected_status": "answer",
                "expected_output_format": "generation_contract_json",
                "must_include": [],
                "must_not_include": ["price target"],
                "required_evidence_ids": ["SEC-001"],
            }
        ],
    )

    assert len(rows) == 1
    assert len(traces) == 1
    assert rows[0]["memory_mode"] == "mm4_bounded_agentic"
    assert rows[0]["comparison_token_count_source"] == "whitespace_normalized"
    assert rows[0]["comparison_input_tokens"] > 0
    assert traces[0]["trace_events"]
    assert evaluated[0]["generation_contract_valid"] is True
    assert evaluated[0]["evidence_match"] is True
    assert evaluated[0]["groundedness"] is True


def test_comparison_keeps_unavailable_cost_explicit() -> None:
    row = build_memory_mode_row(
        memory_mode="mm4_bounded_agentic",
        result_rows=[
            {
                "success": True,
                "input_tokens": 10,
                "output_tokens": 5,
                "retrieval_rounds": 1,
                "repair_attempts": 0,
                "final_status": "answer",
                "node_latencies": {"generate_answer": 10.0},
            }
        ],
        evaluation_summary={"grounded_rate": 1.0},
        latency_summary={"mean_e2e_latency_ms": 20.0},
    )

    assert row["total_cost_usd"] is None
    assert row["token_count_source"] == "whitespace_normalized"
    assert row["cost_status"] == "unavailable_no_gpu_hourly_price"
    assert "total_cost_usd" in row["missing_metrics"]
