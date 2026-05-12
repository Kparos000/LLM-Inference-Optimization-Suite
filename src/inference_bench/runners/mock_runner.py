"""Deterministic mock benchmark runner."""

from __future__ import annotations

from pathlib import Path

from inference_bench.metrics import (
    calculate_end_to_end_latency_ms,
    calculate_tokens_per_second,
    calculate_tpot_ms,
    calculate_ttft_ms,
    estimate_token_cost_usd,
)
from inference_bench.results import write_results_csv
from inference_bench.schema import BenchmarkResult
from inference_bench.workloads.loader import load_jsonl_workload


def count_whitespace_tokens(text: str) -> int:
    """Count tokens with a deterministic whitespace approximation."""

    return len(text.split())


def run_mock_benchmark(
    workload_path: str | Path,
    output_path: str | Path,
    run_id: str = "mock-run",
    backend: str = "mock",
    model_name: str = "mock-model",
    optimization: str = "none",
) -> list[BenchmarkResult]:
    """Run a deterministic benchmark pass without model execution."""

    workload_items = load_jsonl_workload(workload_path)
    results: list[BenchmarkResult] = []

    for index, item in enumerate(workload_items):
        input_tokens = count_whitespace_tokens(item.prompt)
        output_tokens = max(8, min(128, input_tokens // 2 + 8))

        request_start_s = index * 0.1
        first_token_s = request_start_s + 0.05 + input_tokens * 0.001
        request_end_s = first_token_s + output_tokens * 0.01
        elapsed_seconds = request_end_s - request_start_s

        results.append(
            BenchmarkResult(
                run_id=run_id,
                timestamp_utc="1970-01-01T00:00:00Z",
                backend=backend,
                model_name=model_name,
                optimization=optimization,
                workload_name=item.workload_name,
                prompt_id=item.prompt_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=calculate_ttft_ms(request_start_s, first_token_s),
                tpot_ms=calculate_tpot_ms(first_token_s, request_end_s, output_tokens),
                end_to_end_latency_ms=calculate_end_to_end_latency_ms(
                    request_start_s,
                    request_end_s,
                ),
                throughput_tokens_per_second=calculate_tokens_per_second(
                    input_tokens + output_tokens,
                    elapsed_seconds,
                ),
                peak_memory_mb=None,
                estimated_cost_usd=estimate_token_cost_usd(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    input_cost_per_million_tokens=0.0,
                    output_cost_per_million_tokens=0.0,
                ),
                success=True,
                error_message=None,
            )
        )

    write_results_csv(results, output_path)
    return results
