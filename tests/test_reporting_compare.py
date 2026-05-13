import csv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.reporting.compare import compare_result_files, write_comparison_csv
from inference_bench.results import write_results_csv
from inference_bench.schema import BenchmarkResult


def _result(
    *,
    run_id: str,
    workload_name: str,
    latency_ms: float,
    success: bool = True,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=run_id,
        timestamp_utc="2026-05-12T04:00:00Z",
        backend="mock",
        model_name="mock-model",
        optimization="none",
        workload_name=workload_name,
        prompt_id=f"{workload_name}-001",
        input_tokens=10,
        output_tokens=8,
        ttft_ms=20.0,
        tpot_ms=10.0,
        end_to_end_latency_ms=latency_ms,
        throughput_tokens_per_second=120.0,
        peak_memory_mb=None,
        estimated_cost_usd=0.0,
        success=success,
        error_message=None if success else "failed",
    )


def test_compare_result_files_returns_one_summary_per_csv(tmp_path: Path) -> None:
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    write_results_csv(
        [_result(run_id="run-1", workload_name="short_chat", latency_ms=100.0)], first_csv
    )
    write_results_csv(
        [_result(run_id="run-2", workload_name="code_helpdesk", latency_ms=200.0)],
        second_csv,
    )

    rows = compare_result_files([first_csv, second_csv])

    assert len(rows) == 2
    assert rows[0]["source_file"] == str(first_csv)
    assert rows[0]["row_count"] == 1
    assert rows[0]["workloads"] == ["short_chat"]
    assert rows[1]["source_file"] == str(second_csv)
    assert rows[1]["avg_end_to_end_latency_ms"] == 200.0


def test_write_comparison_csv_writes_output(tmp_path: Path) -> None:
    output_path = tmp_path / "comparison.csv"
    rows: list[dict[str, object]] = [
        {
            "source_file": "first.csv",
            "row_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "backends": ["mock"],
            "models": ["mock-model"],
            "optimizations": ["none"],
            "workloads": ["short_chat"],
            "avg_end_to_end_latency_ms": 100.0,
            "avg_ttft_ms": 20.0,
            "avg_tpot_ms": 10.0,
            "avg_throughput_tokens_per_second": 120.0,
            "total_estimated_cost_usd": 0.0,
        }
    ]

    written_path = write_comparison_csv(rows, output_path)

    assert written_path == output_path
    with output_path.open(encoding="utf-8", newline="") as file:
        [row] = list(csv.DictReader(file))
    assert row["source_file"] == "first.csv"
    assert row["backends"] == "mock"
    assert row["avg_end_to_end_latency_ms"] == "100.0"


def test_empty_comparison_csv_writes_standard_header(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.csv"

    write_comparison_csv([], output_path)

    with output_path.open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        remaining_rows = list(reader)

    assert header == [
        "source_file",
        "row_count",
        "success_count",
        "failure_count",
        "backends",
        "models",
        "optimizations",
        "workloads",
        "avg_end_to_end_latency_ms",
        "avg_ttft_ms",
        "avg_tpot_ms",
        "avg_throughput_tokens_per_second",
        "total_estimated_cost_usd",
    ]
    assert remaining_rows == []


def test_missing_input_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compare_result_files([tmp_path / "missing.csv"])


def test_cli_compare_results_succeeds_with_two_temp_csv_files(tmp_path: Path) -> None:
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    output_csv = tmp_path / "comparison.csv"
    write_results_csv(
        [_result(run_id="run-1", workload_name="short_chat", latency_ms=100.0)], first_csv
    )
    write_results_csv(
        [_result(run_id="run-2", workload_name="code_helpdesk", latency_ms=200.0)],
        second_csv,
    )

    result = CliRunner().invoke(
        app,
        [
            "compare-results",
            "--input-csv",
            str(first_csv),
            "--input-csv",
            str(second_csv),
            "--output-csv",
            str(output_csv),
        ],
    )

    assert result.exit_code == 0
    assert output_csv.exists()
    assert "Benchmark Result Comparison" in result.output
    assert "Output path:" in result.output
