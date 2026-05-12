from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from inference_bench import __version__
from inference_bench.config import load_project_config
from inference_bench.reporting.plots import (
    plot_cost_by_optimization,
    plot_latency_by_optimization,
    plot_throughput_by_optimization,
)
from inference_bench.reporting.summary import summarize_results
from inference_bench.runners.mock_runner import run_mock_benchmark

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
