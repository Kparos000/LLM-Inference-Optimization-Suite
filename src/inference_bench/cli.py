from typing import Annotated

import typer
from rich.console import Console

from inference_bench import __version__
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
