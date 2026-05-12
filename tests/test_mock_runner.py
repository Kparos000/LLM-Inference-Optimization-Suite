import csv
from pathlib import Path

from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.runners.mock_runner import count_whitespace_tokens, run_mock_benchmark
from inference_bench.workloads.loader import load_jsonl_workload


def test_count_whitespace_tokens() -> None:
    assert count_whitespace_tokens("one two  three") == 3
    assert count_whitespace_tokens(" \n\t ") == 0


def test_run_mock_benchmark_returns_one_result_per_workload_item(tmp_path: Path) -> None:
    workload_path = Path("data/prompts/smoke_workload.jsonl")
    output_path = tmp_path / "mock_results.csv"

    results = run_mock_benchmark(workload_path, output_path)

    assert len(results) == len(load_jsonl_workload(workload_path))


def test_run_mock_benchmark_writes_csv_file(tmp_path: Path) -> None:
    output_path = tmp_path / "mock_results.csv"

    results = run_mock_benchmark("data/prompts/smoke_workload.jsonl", output_path)

    assert output_path.exists()
    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == len(results)


def test_mock_results_have_expected_values(tmp_path: Path) -> None:
    results = run_mock_benchmark("data/prompts/smoke_workload.jsonl", tmp_path / "results.csv")

    assert all(result.backend == "mock" for result in results)
    assert all(result.success for result in results)
    assert all(result.end_to_end_latency_ms >= 0 for result in results)
    assert all(result.output_tokens > 0 for result in results)


def test_cli_mock_run_succeeds_with_tmp_output_file(tmp_path: Path) -> None:
    runner = CliRunner()
    output_path = tmp_path / "mock_results.csv"

    result = runner.invoke(
        app,
        [
            "mock-run",
            "--workload-path",
            "data/prompts/smoke_workload.jsonl",
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "Benchmark rows written: 3" in result.output
    assert str(output_path) in result.output
    assert output_path.exists()
