import csv
from pathlib import Path

from inference_bench.results import write_results_csv
from inference_bench.schema import BenchmarkResult


def _result() -> BenchmarkResult:
    return BenchmarkResult(
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


def test_writes_csv_with_one_result(tmp_path: Path) -> None:
    output_path = write_results_csv([_result()], tmp_path / "nested" / "results.csv")

    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["success"] == "True"


def test_writes_header_for_empty_results(tmp_path: Path) -> None:
    output_path = write_results_csv([], tmp_path / "results.csv")

    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    assert rows == [BenchmarkResult.csv_fieldnames()]
