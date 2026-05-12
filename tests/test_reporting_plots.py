from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.reporting.plots import (
    plot_cost_by_optimization,
    plot_latency_by_optimization,
    plot_throughput_by_optimization,
)
from inference_bench.results import write_results_csv
from inference_bench.schema import BenchmarkResult


def _result(
    *,
    prompt_id: str = "prompt-1",
    optimization: str = "none",
    latency_ms: float = 100.0,
    throughput: float | None = 120.0,
    cost: float | None = 0.01,
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
        ttft_ms=20.0,
        tpot_ms=10.0,
        end_to_end_latency_ms=latency_ms,
        throughput_tokens_per_second=throughput,
        peak_memory_mb=None,
        estimated_cost_usd=cost,
        success=True,
    )


def test_plot_functions_create_png_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    write_results_csv(
        [
            _result(prompt_id="prompt-1", optimization="none"),
            _result(prompt_id="prompt-2", optimization="baseline", latency_ms=200.0),
        ],
        csv_path,
    )

    output_paths = [
        plot_latency_by_optimization(csv_path, tmp_path / "figures" / "latency.png"),
        plot_throughput_by_optimization(csv_path, tmp_path / "figures" / "throughput.png"),
        plot_cost_by_optimization(csv_path, tmp_path / "figures" / "cost.png"),
    ]

    for output_path in output_paths:
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert output_path.stat().st_size > 0


def test_plot_functions_raise_value_error_without_usable_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    write_results_csv([], csv_path)

    with pytest.raises(ValueError, match="No usable data"):
        plot_latency_by_optimization(csv_path, tmp_path / "latency.png")

    with pytest.raises(ValueError, match="No usable data"):
        plot_throughput_by_optimization(csv_path, tmp_path / "throughput.png")

    with pytest.raises(ValueError, match="No usable data"):
        plot_cost_by_optimization(csv_path, tmp_path / "cost.png")


def test_cli_make_plots_succeeds_with_temp_csv_and_output_dir(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    output_dir = tmp_path / "figures"
    write_results_csv([_result()], csv_path)

    result = CliRunner().invoke(
        app,
        [
            "make-plots",
            "--input-csv",
            str(csv_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "latency_by_optimization.png" in result.output
    assert (output_dir / "latency_by_optimization.png").exists()
    assert (output_dir / "throughput_by_optimization.png").exists()
    assert (output_dir / "cost_by_optimization.png").exists()
