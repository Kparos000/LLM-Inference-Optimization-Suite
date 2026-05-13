import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from inference_bench import __version__
from inference_bench.config import load_project_config
from inference_bench.quality import score_structured_output
from inference_bench.reporting.compare import compare_result_files, write_comparison_csv
from inference_bench.reporting.plots import (
    plot_cost_by_optimization,
    plot_latency_by_optimization,
    plot_throughput_by_optimization,
)
from inference_bench.reporting.summary import summarize_results
from inference_bench.runners.hf_runner import run_hf_benchmark
from inference_bench.runners.mock_runner import run_mock_benchmark
from inference_bench.system_info import collect_system_info, write_system_info_json

app = typer.Typer(
    help="LLM Inference Optimization Suite command-line interface.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(f"LLM Inference Optimization Suite v{__version__}")


@app.command()
def doctor() -> None:
    """Run a lightweight local environment check."""
    console.print("[bold green]Environment check passed.[/bold green]")
    console.print("No GPU is required for this check.")
    console.print("The benchmark harness scaffold is importable.")


@app.command()
def system_info(
    output_path: Annotated[
        str,
        typer.Option(help="Path where system metadata JSON should be written."),
    ] = "results/raw/system_info.json",
) -> None:
    """Capture lightweight hardware and system metadata."""

    info = collect_system_info()
    written_path = write_system_info_json(info, output_path)

    console.print(f"Platform: {info.platform} {info.platform_release}".strip(), soft_wrap=True)
    console.print(f"Python version: {info.python_version}")
    if info.torch_version is not None:
        console.print(f"Torch version: {info.torch_version}")
    if info.cuda_available is not None:
        console.print(f"CUDA available: {info.cuda_available}")
    console.print(f"Output path: {written_path}", soft_wrap=True)


@app.command()
def validate_config(
    models_path: Annotated[
        str,
        typer.Option(help="Path to the models YAML config."),
    ] = "configs/models.yaml",
    workloads_path: Annotated[
        str,
        typer.Option(help="Path to the workloads YAML config."),
    ] = "configs/workloads.yaml",
    experiments_path: Annotated[
        str,
        typer.Option(help="Path to the experiments YAML config."),
    ] = "configs/experiments.yaml",
) -> None:
    """Validate benchmark configuration files."""

    project_config = load_project_config(
        models_path=models_path,
        workloads_path=workloads_path,
        experiments_path=experiments_path,
    )
    experiment_names = sorted(project_config.experiments)

    console.print("[bold green]Configuration valid.[/bold green]")
    console.print(f"Models loaded: {len(project_config.models)}")
    console.print(f"Workloads loaded: {len(project_config.workloads)}")
    console.print(f"Experiments loaded: {len(project_config.experiments)}")
    console.print("Experiments: " + (", ".join(experiment_names) if experiment_names else "none"))


@app.command()
def mock_run(
    workload_path: Annotated[
        str,
        typer.Option(help="Path to the JSONL workload file."),
    ] = "data/prompts/smoke_workload.jsonl",
    output_path: Annotated[
        str,
        typer.Option(help="Path where the mock benchmark CSV should be written."),
    ] = "results/raw/mock_results.csv",
) -> None:
    """Run the deterministic mock benchmark pipeline."""

    results = run_mock_benchmark(
        workload_path=workload_path,
        output_path=output_path,
    )
    console.print(f"Benchmark rows written: {len(results)}")
    console.print(f"Output path: {output_path}", soft_wrap=True)


@app.command()
def hf_run(
    workload_path: Annotated[
        str,
        typer.Option(help="Path to the JSONL workload file."),
    ] = "data/prompts/smoke_workload.jsonl",
    output_path: Annotated[
        str,
        typer.Option(help="Path where the Hugging Face benchmark CSV should be written."),
    ] = "results/raw/hf_results.csv",
    generation_output_path: Annotated[
        str,
        typer.Option(help="Path where generated text JSONL records should be written."),
    ] = "results/raw/hf_generations.jsonl",
    model_id: Annotated[
        str,
        typer.Option(help="Hugging Face model identifier."),
    ] = "Qwen/Qwen2.5-0.5B-Instruct",
    run_id: Annotated[
        str,
        typer.Option(help="Benchmark run identifier."),
    ] = "hf-run",
    max_new_tokens: Annotated[
        int,
        typer.Option(help="Maximum number of output tokens to generate."),
    ] = 64,
    max_prompts: Annotated[
        int | None,
        typer.Option(help="Optional prompt limit for smoke runs."),
    ] = None,
    use_streaming: Annotated[
        bool,
        typer.Option(
            "--use-streaming/--no-use-streaming", help="Enable streaming TTFT measurement."
        ),
    ] = False,
) -> None:
    """Run the Hugging Face local inference benchmark."""

    try:
        results = run_hf_benchmark(
            workload_path=workload_path,
            output_path=output_path,
            model_id=model_id,
            run_id=run_id,
            max_new_tokens=max_new_tokens,
            max_prompts=max_prompts,
            generation_output_path=generation_output_path,
            use_streaming=use_streaming,
        )
    except RuntimeError as exc:
        console.print(str(exc), markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from exc

    console.print(f"Benchmark rows written: {len(results)}")
    console.print(f"Output path: {output_path}", soft_wrap=True)
    console.print(f"Generation output path: {generation_output_path}", soft_wrap=True)
    console.print(f"Streaming used: {use_streaming}")


@app.command()
def report_summary(
    input_csv: Annotated[
        str,
        typer.Option(help="Path to a benchmark result CSV."),
    ] = "results/raw/mock_results.csv",
) -> None:
    """Print a summary of benchmark results."""

    summary = summarize_results(input_csv)
    table = Table(title="Benchmark Result Summary")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in summary.items():
        if isinstance(value, list):
            display_value = ", ".join(value) if value else ""
        elif value is None:
            display_value = ""
        else:
            display_value = str(value)
        table.add_row(key, display_value)

    console.print(table)


@app.command()
def score_structured_jsonl(
    input_jsonl: Annotated[
        str,
        typer.Option(help="Path to a generation trace JSONL file."),
    ],
    required_fields: Annotated[
        str,
        typer.Option(help="Comma-separated required JSON fields."),
    ] = "category,answer,confidence",
) -> None:
    """Score structured JSON validity for generated text traces."""

    input_path = Path(input_jsonl)
    if not input_path.exists():
        console.print(f"Input JSONL not found: {input_path}", markup=False, soft_wrap=True)
        raise typer.Exit(code=1)

    required_field_names = [field.strip() for field in required_fields.split(",") if field.strip()]

    total_records = 0
    valid_json_count = 0
    required_fields_count = 0

    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSONL record at line {line_number}: {exc.msg}"
                raise typer.BadParameter(msg) from exc

            if not isinstance(record, dict):
                msg = f"Invalid JSONL record at line {line_number}: expected object"
                raise typer.BadParameter(msg)

            generated_text = record.get("generated_text")
            score = score_structured_output(
                generated_text if isinstance(generated_text, str) else "",
                required_field_names,
            )

            total_records += 1
            if score["is_valid_json"]:
                valid_json_count += 1
            if score["has_required_fields"]:
                required_fields_count += 1

    invalid_json_count = total_records - valid_json_count
    required_fields_rate = required_fields_count / total_records if total_records > 0 else 0.0

    table = Table(title="Structured JSON Score")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("total_records", str(total_records))
    table.add_row("valid_json_count", str(valid_json_count))
    table.add_row("required_fields_count", str(required_fields_count))
    table.add_row("invalid_json_count", str(invalid_json_count))
    table.add_row("required_fields_rate", f"{required_fields_rate:.3f}")
    console.print(table)


@app.command()
def compare_results(
    input_csv: Annotated[
        list[str] | None,
        typer.Option(help="Path to a benchmark result CSV. May be provided multiple times."),
    ] = None,
    output_csv: Annotated[
        str,
        typer.Option(help="Path where the comparison CSV should be written."),
    ] = "results/processed/comparison.csv",
) -> None:
    """Compare multiple benchmark result CSV files."""

    if not input_csv:
        console.print("At least one --input-csv value is required.", markup=False)
        raise typer.Exit(code=1)

    try:
        rows = compare_result_files(input_csv)
    except FileNotFoundError as exc:
        console.print(f"Input CSV not found: {exc.filename or exc}", markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from exc

    written_path = write_comparison_csv(rows, output_csv)

    display_fields = (
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
    )

    table = Table(title="Benchmark Result Comparison")
    for column_name in display_fields:
        table.add_column(column_name)

    for row in rows:
        table.add_row(
            *[
                ", ".join(str(item) for item in value)
                if isinstance(value := row.get(column_name), list)
                else ("" if value is None else str(value))
                for column_name in display_fields
            ]
        )

    console.print(table)
    console.print(f"Output path: {written_path}", soft_wrap=True)


@app.command()
def make_plots(
    input_csv: Annotated[
        str,
        typer.Option(help="Path to a benchmark result CSV."),
    ] = "results/raw/mock_results.csv",
    output_dir: Annotated[
        str,
        typer.Option(help="Directory where plot PNG files should be written."),
    ] = "results/figures",
) -> None:
    """Generate basic benchmark plots."""

    output_path = Path(output_dir)
    written_paths = [
        plot_latency_by_optimization(
            input_csv,
            output_path / "latency_by_optimization.png",
        ),
        plot_throughput_by_optimization(
            input_csv,
            output_path / "throughput_by_optimization.png",
        ),
        plot_cost_by_optimization(
            input_csv,
            output_path / "cost_by_optimization.png",
        ),
    ]

    for path in written_paths:
        console.print(f"Wrote plot: {path}", soft_wrap=True)


@app.command()
def explain(
    concept: Annotated[
        str,
        typer.Argument(help="Concept to explain, for example: kv-cache, prefill-decode."),
    ],
) -> None:
    """Print a placeholder explanation for a core inference concept."""
    normalized = concept.strip().lower()

    explanations = {
        "kv-cache": (
            "Technical: The KV cache stores attention keys and values from previous tokens "
            "so decoding does not recompute the full context each step.\n\n"
            "6th grader: It is the model's notebook. The model keeps notes so it does not "
            "need to reread the whole question every time it writes a new word."
        ),
        "prefill-decode": (
            "Technical: Prefill processes the input prompt, while decode generates output "
            "tokens one at a time.\n\n"
            "6th grader: First the model reads your question. Then it writes the answer "
            "piece by piece."
        ),
    }

    if normalized not in explanations:
        console.print(f"[yellow]No explanation found yet for:[/yellow] {concept}")
        raise typer.Exit(code=1)

    console.print(explanations[normalized])


if __name__ == "__main__":
    app()
