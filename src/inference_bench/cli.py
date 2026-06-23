import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from inference_bench import __version__
from inference_bench.config import load_project_config
from inference_bench.load_profiles import load_sequence_buckets, load_traffic_profiles
from inference_bench.optimization_negative_rules import load_optimization_negative_rules
from inference_bench.quality import score_structured_output
from inference_bench.reporting.compare import compare_result_files, write_comparison_csv
from inference_bench.reporting.phase1_plots import (
    DEFAULT_INPUT_CSV,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TITLE_PREFIX,
    generate_phase1_plots,
)
from inference_bench.reporting.plots import (
    plot_cost_by_optimization,
    plot_latency_by_optimization,
    plot_throughput_by_optimization,
)
from inference_bench.reporting.summary import summarize_results
from inference_bench.result_track_schema import RESULT_TRACK_JOIN_KEYS, validate_result_track_row
from inference_bench.runners.hf_runner import run_hf_benchmark
from inference_bench.runners.mock_runner import run_mock_benchmark
from inference_bench.runners.openai_compatible_runner import run_openai_compatible_benchmark
from inference_bench.runners.openai_load_runner import run_openai_compatible_load_benchmark
from inference_bench.runtime_registry import load_runtime_registry
from inference_bench.serving_profiles import load_serving_profiles
from inference_bench.slo import SLO_METRIC_FAMILIES, SLO_VERTICALS, load_slo_config
from inference_bench.slo_profiles import load_slo_profiles
from inference_bench.system_info import collect_system_info, write_system_info_json
from inference_bench.workloads.scaled_generator import generate_scaled_workloads

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
    runtime_registry_path: Annotated[
        str,
        typer.Option(help="Path to the runtime/engine registry YAML config."),
    ] = "configs/runtime_engines.yaml",
    serving_profiles_path: Annotated[
        str,
        typer.Option(help="Path to the serving-profile YAML config."),
    ] = "configs/serving_profiles.yaml",
    load_profiles_path: Annotated[
        str,
        typer.Option(help="Path to the production load profiles YAML config."),
    ] = "configs/load_profiles.yaml",
    optimization_negative_rules_path: Annotated[
        str,
        typer.Option(help="Path to the optimization negative-rules YAML config."),
    ] = "configs/optimization_negative_rules.yaml",
    slo_targets_path: Annotated[
        str,
        typer.Option(help="Path to the production SLO target YAML config."),
    ] = "configs/slo_targets.yaml",
    slo_profiles_path: Annotated[
        str,
        typer.Option(help="Path to the modular SLO profiles YAML config."),
    ] = "configs/slo_profiles.yaml",
) -> None:
    """Validate benchmark configuration files."""

    project_config = load_project_config(
        models_path=models_path,
        workloads_path=workloads_path,
        experiments_path=experiments_path,
    )
    runtime_registry = load_runtime_registry(runtime_registry_path)
    serving_profiles = load_serving_profiles(serving_profiles_path)
    sequence_buckets = load_sequence_buckets(load_profiles_path)
    traffic_profiles = load_traffic_profiles(load_profiles_path)
    negative_rules = load_optimization_negative_rules(optimization_negative_rules_path)
    slo_config = load_slo_config(slo_targets_path)
    slo_profiles = load_slo_profiles(slo_profiles_path)
    result_track_errors = validate_result_track_row(
        {
            "run_id": "config-validation",
            "config_id": "schema-smoke",
            "prompt_id": "prompt-0",
            "vertical": "airline",
            "model_alias": "model2_3b",
            "memory_mode": "mm2_hybrid_top5",
            "runtime": "vllm",
            "backend_type": "self_hosted_gpu",
            "engine": "vllm",
            "hardware": "remote_rtx3070",
            "provider": "huggingface",
            "concurrency": 1,
            "api_cost_usd": None,
            "gpu_cost_usd": None,
            "gpu_hourly_price_usd": None,
        }
    )
    if result_track_errors:
        msg = "Result-track schema validation failed: " + ", ".join(result_track_errors)
        raise typer.BadParameter(msg)
    experiment_names = sorted(project_config.experiments)
    slo_verticals = slo_config.get("verticals", {})
    profiles = slo_profiles.get("profiles", {})

    console.print("[bold green]Configuration valid.[/bold green]")
    console.print(f"Models loaded: {len(project_config.models)}")
    console.print(f"Model aliases loaded: {len(project_config.model_aliases)}")
    console.print(f"Runtime engines loaded: {len(runtime_registry)}")
    console.print(f"Serving profiles loaded: {len(serving_profiles)}")
    console.print(
        "Sequence length buckets loaded: "
        f"{len(sequence_buckets['input'])} input, {len(sequence_buckets['output'])} output"
    )
    console.print(f"Traffic profiles loaded: {len(traffic_profiles)}")
    console.print(f"Optimization negative-rule groups loaded: {len(negative_rules)}")
    console.print(
        f"SLO targets loaded: {len(slo_verticals)} verticals, "
        f"{len(SLO_METRIC_FAMILIES)} metric families"
    )
    console.print(f"SLO profiles loaded: {len(profiles)}")
    console.print(f"Result track schema join keys loaded: {len(RESULT_TRACK_JOIN_KEYS)}")
    console.print(f"Workloads loaded: {len(project_config.workloads)}")
    console.print(f"Experiments loaded: {len(project_config.experiments)}")
    console.print("SLO verticals: " + ", ".join(SLO_VERTICALS))
    console.print("Experiments: " + (", ".join(experiment_names) if experiment_names else "none"))


@app.command("generate-workloads")
def generate_workloads_command(
    output_dir: Annotated[
        str,
        typer.Option(help="Directory where scaled workload JSONL files should be written."),
    ] = "data/prompts/scaled",
    count: Annotated[
        int,
        typer.Option(help="Number of prompts to generate per workload."),
    ] = 100,
    seed: Annotated[
        int,
        typer.Option(help="Deterministic generation seed."),
    ] = 42,
    workloads: Annotated[
        list[str] | None,
        typer.Option(help="Workload names to generate. May be provided multiple times."),
    ] = None,
) -> None:
    """Generate deterministic synthetic scaled workload files."""

    written_paths = generate_scaled_workloads(
        output_dir=output_dir,
        count=count,
        seed=seed,
        workloads=workloads,
    )

    for path in written_paths:
        console.print(f"Wrote {path}", soft_wrap=True)


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
def openai_compatible_run(
    workload_path: Annotated[
        str,
        typer.Option(help="Path to the JSONL workload file."),
    ] = "data/prompts/smoke_workload.jsonl",
    output_path: Annotated[
        str,
        typer.Option(help="Path where the benchmark CSV should be written."),
    ] = "results/raw/openai_compatible_results.csv",
    generation_output_path: Annotated[
        str,
        typer.Option(help="Path where generated text JSONL records should be written."),
    ] = "results/raw/openai_compatible_generations.jsonl",
    model: Annotated[
        str,
        typer.Option(help="Model name served by the OpenAI-compatible endpoint."),
    ] = "Qwen/Qwen2.5-0.5B-Instruct",
    base_url: Annotated[
        str,
        typer.Option(help="OpenAI-compatible API base URL."),
    ] = "http://localhost:8000/v1",
    api_key: Annotated[
        str,
        typer.Option(help="API key for the OpenAI-compatible endpoint."),
    ] = "EMPTY",
    run_id: Annotated[
        str,
        typer.Option(help="Benchmark run identifier."),
    ] = "openai-compatible-run",
    backend: Annotated[
        str,
        typer.Option(help="Backend label to store in result rows."),
    ] = "openai_compatible",
    optimization: Annotated[
        str,
        typer.Option(help="Optimization label to store in result rows."),
    ] = "vllm_baseline",
    max_new_tokens: Annotated[
        int,
        typer.Option(help="Maximum number of output tokens to request."),
    ] = 64,
    max_prompts: Annotated[
        int | None,
        typer.Option(help="Optional prompt limit for smoke runs."),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Enable streaming TTFT measurement."),
    ] = True,
    timeout_seconds: Annotated[
        float,
        typer.Option(help="Request timeout in seconds."),
    ] = 120.0,
) -> None:
    """Run a benchmark against an OpenAI-compatible endpoint."""

    try:
        results = run_openai_compatible_benchmark(
            workload_path=workload_path,
            output_path=output_path,
            generation_output_path=generation_output_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            run_id=run_id,
            backend=backend,
            optimization=optimization,
            max_new_tokens=max_new_tokens,
            max_prompts=max_prompts,
            stream=stream,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError as exc:
        console.print(str(exc), markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from exc

    console.print(f"Benchmark rows written: {len(results)}")
    console.print(f"Output path: {output_path}", soft_wrap=True)
    console.print(f"Generation output path: {generation_output_path}", soft_wrap=True)
    console.print(f"Base URL: {base_url}", soft_wrap=True)
    console.print(f"Streaming used: {stream}")


@app.command()
def openai_load_run(
    workload_path: Annotated[
        str,
        typer.Option(help="Path to the JSONL workload file."),
    ] = "data/prompts/smoke_workload.jsonl",
    output_path: Annotated[
        str,
        typer.Option(help="Path where the benchmark CSV should be written."),
    ] = "results/raw/openai_load_results.csv",
    generation_output_path: Annotated[
        str,
        typer.Option(help="Path where generated text JSONL records should be written."),
    ] = "results/raw/openai_load_generations.jsonl",
    run_metadata_path: Annotated[
        str | None,
        typer.Option(help="Optional path where run-level metadata JSON should be written."),
    ] = None,
    model: Annotated[
        str,
        typer.Option(help="Model name served by the OpenAI-compatible endpoint."),
    ] = "Qwen/Qwen2.5-0.5B-Instruct",
    base_url: Annotated[
        str,
        typer.Option(help="OpenAI-compatible API base URL."),
    ] = "http://localhost:8000/v1",
    api_key: Annotated[
        str,
        typer.Option(help="API key for the OpenAI-compatible endpoint."),
    ] = "EMPTY",
    run_id: Annotated[
        str,
        typer.Option(help="Benchmark run identifier."),
    ] = "openai-load-run",
    backend: Annotated[
        str,
        typer.Option(help="Backend label to store in result rows."),
    ] = "vllm",
    optimization: Annotated[
        str,
        typer.Option(help="Optimization label to store in result rows."),
    ] = "vllm_baseline",
    concurrency: Annotated[
        int,
        typer.Option(help="Maximum number of concurrent requests."),
    ] = 1,
    max_new_tokens: Annotated[
        int,
        typer.Option(help="Maximum number of output tokens to request."),
    ] = 64,
    max_prompts: Annotated[
        int | None,
        typer.Option(help="Optional prompt limit for calibration runs."),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="Enable streaming TTFT measurement."),
    ] = True,
    timeout_seconds: Annotated[
        float,
        typer.Option(help="Request timeout in seconds."),
    ] = 120.0,
    chunk_size: Annotated[
        int | None,
        typer.Option(help="Optional number of prompts to process before flushing outputs."),
    ] = None,
    checkpoint_path: Annotated[
        str | None,
        typer.Option(help="Optional checkpoint JSON path for chunked runs."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Resume from checkpoint when available."),
    ] = False,
    log_path: Annotated[
        str | None,
        typer.Option(help="Optional progress log path for chunked runs."),
    ] = None,
    progress_interval: Annotated[
        int,
        typer.Option(help="Progress reporting interval in prompts."),
    ] = 100,
) -> None:
    """Run a concurrent benchmark against an OpenAI-compatible endpoint."""

    try:
        results = run_openai_compatible_load_benchmark(
            workload_path=workload_path,
            output_path=output_path,
            generation_output_path=generation_output_path,
            run_metadata_path=run_metadata_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            run_id=run_id,
            backend=backend,
            optimization=optimization,
            concurrency=concurrency,
            max_new_tokens=max_new_tokens,
            max_prompts=max_prompts,
            stream=stream,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            resume=resume,
            log_path=log_path,
            progress_interval=progress_interval,
        )
    except RuntimeError as exc:
        console.print(str(exc), markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from exc

    console.print(f"Benchmark rows written: {len(results)}")
    console.print(f"Output path: {output_path}", soft_wrap=True)
    console.print(f"Generation output path: {generation_output_path}", soft_wrap=True)
    if run_metadata_path is not None:
        console.print(f"Run metadata path: {run_metadata_path}", soft_wrap=True)
    if checkpoint_path is not None:
        console.print(f"Checkpoint path: {checkpoint_path}", soft_wrap=True)
    if log_path is not None:
        console.print(f"Log path: {log_path}", soft_wrap=True)
    console.print(f"Base URL: {base_url}", soft_wrap=True)
    console.print(f"Concurrency: {concurrency}")
    if chunk_size is not None:
        console.print(f"Chunk size: {chunk_size}")
    console.print(f"Resume used: {resume}")
    console.print(f"Streaming used: {stream}")


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
        "p50_end_to_end_latency_ms",
        "p95_end_to_end_latency_ms",
        "p99_end_to_end_latency_ms",
        "avg_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "avg_tpot_ms",
        "p50_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
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


@app.command("make-phase1-plots")
def make_phase1_plots(
    input_csv: Annotated[
        str,
        typer.Option(help="Path to the Phase 1 comparison CSV."),
    ] = DEFAULT_INPUT_CSV,
    output_dir: Annotated[
        str,
        typer.Option(help="Directory where Phase 1 plot PNG files should be written."),
    ] = DEFAULT_OUTPUT_DIR,
    title_prefix: Annotated[
        str,
        typer.Option(help="Title prefix for generated figures."),
    ] = DEFAULT_TITLE_PREFIX,
) -> None:
    """Generate report-ready Phase 1 plots from curated sample artifacts."""

    try:
        manifest = generate_phase1_plots(
            input_csv=input_csv,
            output_dir=output_dir,
            title_prefix=title_prefix,
        )
    except FileNotFoundError as exc:
        console.print(f"Input CSV not found: {exc.filename or exc}", markup=False, soft_wrap=True)
        raise typer.Exit(code=1) from exc

    for path in manifest["generated_plots"]:
        console.print(f"Wrote plot: {path}", soft_wrap=True)
    for skipped_plot in manifest["skipped_plots"]:
        console.print(f"Skipped plot: {skipped_plot}", soft_wrap=True)
    console.print(f"Manifest: {Path(output_dir) / 'plot_manifest.json'}", soft_wrap=True)


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
