from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.reporting.summary import summarize_results
from inference_bench.results import write_results_csv
from inference_bench.schema import BenchmarkResult


def _result(
    *,
    prompt_id: str = "prompt-1",
    optimization: str = "none",
    latency_ms: float = 100.0,
    ttft_ms: float | None = 20.0,
    tpot_ms: float | None = 10.0,
    throughput: float | None = 120.0,
    cost: float | None = 0.01,
    success: bool = True,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id="run-1",
        timestamp_utc="2026-05-12T04:00:00Z",
        backend="mock",
        model_name="mock-model",
        optimization=optimization,
        workload_name="smoke",
        prompt_id=prompt_id,
        input_tokens=10,
        output_tokens=8,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        end_to_end_latency_ms=latency_ms,
        throughput_tokens_per_second=throughput,
        peak_memory_mb=None,
        estimated_cost_usd=cost,
        success=success,
    )


def test_summarizes_results_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    write_results_csv(
        [
            _result(prompt_id="prompt-1", latency_ms=100.0, cost=0.01),
            _result(prompt_id="prompt-2", optimization="baseline", latency_ms=200.0, cost=0.02),
        ],
        csv_path,
    )

    summary = summarize_results(csv_path)

    assert summary["row_count"] == 2
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 0
    assert summary["backends"] == ["mock"]
    assert summary["models"] == ["mock-model"]
    assert summary["optimizations"] == ["baseline", "none"]
    assert summary["workloads"] == ["smoke"]
    assert summary["avg_end_to_end_latency_ms"] == 150.0
    assert summary["avg_ttft_ms"] == 20.0
    assert summary["avg_tpot_ms"] == 10.0
    assert summary["avg_throughput_tokens_per_second"] == 120.0
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.03)


def test_summarizes_empty_results_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    write_results_csv([], csv_path)

    summary = summarize_results(csv_path)

    assert summary["row_count"] == 0
    assert summary["success_count"] == 0
    assert summary["failure_count"] == 0
    assert summary["backends"] == []
    assert summary["avg_end_to_end_latency_ms"] is None
    assert summary["total_estimated_cost_usd"] is None


def test_missing_csv_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize_results(tmp_path / "missing.csv")


def test_cli_report_summary_succeeds_with_temp_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    write_results_csv([_result()], csv_path)

    result = CliRunner().invoke(
        app,
        ["report-summary", "--input-csv", str(csv_path)],
    )

    assert result.exit_code == 0
    assert "Benchmark Result Summary" in result.output
    assert "row_count" in result.output
