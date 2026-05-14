import csv
import json
from pathlib import Path

from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.reporting.phase1_plots import generate_phase1_plots


def _write_comparison_csv(path: Path, include_optional_columns: bool = True) -> None:
    fieldnames = [
        "source_file",
        "row_count",
        "success_count",
        "failure_count",
        "workloads",
        "avg_end_to_end_latency_ms",
        "p95_end_to_end_latency_ms",
        "p99_end_to_end_latency_ms",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
    ]
    if include_optional_columns:
        fieldnames.extend(["avg_tpot_ms", "p95_tpot_ms", "p99_tpot_ms"])

    rows: list[dict[str, str]] = []
    for workload_index, workload in enumerate(["short_chat", "long_context"], start=1):
        for concurrency in [8, 32]:
            source_file = (
                f"results/raw/vllm_qwen0_5b_{workload}_5000_conc{concurrency}_chunked_results.csv"
            )
            row = {
                "source_file": source_file,
                "row_count": "5000",
                "success_count": "5000",
                "failure_count": "0",
                "workloads": workload,
                "avg_end_to_end_latency_ms": str(180 + workload_index + concurrency),
                "p95_end_to_end_latency_ms": str(220 + workload_index + concurrency),
                "p99_end_to_end_latency_ms": str(260 + workload_index + concurrency),
                "avg_ttft_ms": str(15 + workload_index + concurrency),
                "p95_ttft_ms": str(25 + workload_index + concurrency),
                "p99_ttft_ms": str(35 + workload_index + concurrency),
            }
            if include_optional_columns:
                row.update(
                    {
                        "avg_tpot_ms": "3.1",
                        "p95_tpot_ms": "3.5",
                        "p99_tpot_ms": "3.8",
                    }
                )
            rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metadata_files(base_dir: Path) -> None:
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for workload in ["short_chat", "long_context"]:
        for concurrency in [8, 32]:
            metadata_path = (
                raw_dir / f"vllm_qwen0_5b_{workload}_5000_conc{concurrency}_chunked_metadata.json"
            )
            metadata = {
                "aggregate_requests_per_second": 25.0 + concurrency,
                "aggregate_output_tokens_per_second": 1800.0 + (concurrency * 100),
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_generate_phase1_plots_creates_pngs_and_manifest(tmp_path: Path) -> None:
    input_csv = tmp_path / "processed" / "comparison.csv"
    output_dir = tmp_path / "figures"
    _write_comparison_csv(input_csv)
    _write_metadata_files(tmp_path)

    manifest = generate_phase1_plots(input_csv=input_csv, output_dir=output_dir)

    expected_plots = [
        "aggregate_requests_per_second_by_concurrency.png",
        "aggregate_output_tokens_per_second_by_concurrency.png",
        "avg_latency_by_concurrency.png",
        "p99_latency_by_concurrency.png",
        "workload_avg_latency_at_conc32.png",
    ]
    for file_name in expected_plots:
        plot_path = output_dir / file_name
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    assert (output_dir / "plot_manifest.json").exists()
    assert len(manifest["generated_plots"]) >= 5


def test_make_phase1_plots_cli_creates_manifest(tmp_path: Path) -> None:
    input_csv = tmp_path / "processed" / "comparison.csv"
    output_dir = tmp_path / "figures"
    _write_comparison_csv(input_csv)
    _write_metadata_files(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "make-phase1-plots",
            "--input-csv",
            str(input_csv),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "plot_manifest.json" in result.output
    assert (output_dir / "avg_latency_by_concurrency.png").exists()


def test_generate_phase1_plots_skips_missing_optional_columns(tmp_path: Path) -> None:
    input_csv = tmp_path / "processed" / "comparison.csv"
    output_dir = tmp_path / "figures"
    _write_comparison_csv(input_csv, include_optional_columns=False)
    _write_metadata_files(tmp_path)

    manifest = generate_phase1_plots(input_csv=input_csv, output_dir=output_dir)

    assert (output_dir / "avg_latency_by_concurrency.png").exists()
    assert (output_dir / "plot_manifest.json").exists()
    assert any("avg_tpot_by_concurrency.png" in item for item in manifest["skipped_plots"])
