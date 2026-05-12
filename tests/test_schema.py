import pytest

from inference_bench.schema import BenchmarkResult, WorkloadItem


def test_valid_workload_item() -> None:
    item = WorkloadItem(
        prompt_id="prompt-1",
        workload_name="smoke",
        prompt="Summarize this request.",
        metadata={"category": "chat"},
    )

    assert item.prompt_id == "prompt-1"
    assert item.metadata == {"category": "chat"}


def test_workload_item_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt must not be empty"):
        WorkloadItem(prompt_id="prompt-1", workload_name="smoke", prompt="")


def test_valid_benchmark_result() -> None:
    result = BenchmarkResult(
        run_id="run-1",
        timestamp_utc="2026-05-12T04:00:00Z",
        backend="dry-run",
        model_name="placeholder-model",
        optimization="none",
        workload_name="smoke",
        prompt_id="prompt-1",
        input_tokens=10,
        output_tokens=5,
        ttft_ms=12.5,
        tpot_ms=3.0,
        end_to_end_latency_ms=27.5,
        throughput_tokens_per_second=181.8,
        peak_memory_mb=None,
        estimated_cost_usd=None,
        success=True,
    )

    assert result.to_dict()["run_id"] == "run-1"
    assert BenchmarkResult.csv_fieldnames()[0] == "run_id"


def test_benchmark_result_rejects_negative_token_count() -> None:
    with pytest.raises(ValueError, match="input_tokens must be >= 0"):
        BenchmarkResult(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            backend="dry-run",
            model_name="placeholder-model",
            optimization="none",
            workload_name="smoke",
            prompt_id="prompt-1",
            input_tokens=-1,
            output_tokens=5,
            ttft_ms=None,
            tpot_ms=None,
            end_to_end_latency_ms=20.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=None,
            success=False,
            error_message="failed",
        )
