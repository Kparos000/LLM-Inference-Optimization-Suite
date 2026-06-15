"""Run the frozen A1 remote RTX 3070 vLLM smoke and write reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PHASE4_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from evaluate_generation_outputs import (  # noqa: E402
    build_summary_rows,
    load_gold_records,
    load_result_rows,
    result_row_to_generated_answer,
)
from run_openai_compatible_smoke import (  # noqa: E402
    DEFAULT_API_KEY,
    check_server_readiness,
)

from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    build_runtime_projections,
    sample_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
    write_runtime_projection,
)
from inference_bench.metrics import summarize_latency_ms  # noqa: E402
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.openai_compatible_runner import (  # noqa: E402
    run_openai_compatible_benchmark,
)
from inference_bench.workload_adapter import (  # noqa: E402
    load_phase3_workload_records,
    workload_record_to_runner_item,
    write_runner_workload_jsonl,
)

DEFAULT_WORKLOAD = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_RUNNER_INPUT = "data/generated/phase4/a1_remote_rtx3070_runner_input.jsonl"
DEFAULT_RAW_OUTPUT = "results/raw/a1_remote_rtx3070_vllm_smoke_results.jsonl"
DEFAULT_METRICS_OUTPUT = "results/raw/a1_remote_rtx3070_vllm_smoke_metrics.csv"
DEFAULT_MANIFEST = "results/raw/a1_remote_rtx3070_vllm_smoke_manifest.json"
DEFAULT_EVAL_REPORT = "results/processed/a1_remote_rtx3070_vllm_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/a1_remote_rtx3070_vllm_eval_summary.csv"
DEFAULT_LATENCY_SUMMARY = "results/processed/a1_remote_rtx3070_vllm_latency_summary.csv"
DEFAULT_RUNTIME_PROJECTION = "results/processed/a1_remote_rtx3070_vllm_runtime_projection.json"
DEFAULT_GPU_CSV = "results/processed/a1_remote_rtx3070_gpu_telemetry.csv"
DEFAULT_GPU_SUMMARY = "results/processed/a1_remote_rtx3070_gpu_telemetry_summary.json"
MODEL_ALIAS = "model1_0_5b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS_PER_VERTICAL = 10
TOTAL_PROMPTS = 50
MAX_NEW_TOKENS = 128


def build_parser() -> argparse.ArgumentParser:
    """Build the A1 CLI parser."""

    parser = argparse.ArgumentParser(description="Run the frozen remote RTX 3070 vLLM smoke.")
    parser.add_argument("--workload-path", default=DEFAULT_WORKLOAD)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--output-path", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--metrics-output-path", default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--latency-summary-path", default=DEFAULT_LATENCY_SUMMARY)
    parser.add_argument("--runtime-projection-path", default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--gpu-telemetry-csv", default=DEFAULT_GPU_CSV)
    parser.add_argument("--gpu-telemetry-summary", default=DEFAULT_GPU_SUMMARY)
    parser.add_argument("--telemetry-ssh-host", default=None)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export and validate the frozen input without contacting a server.",
    )
    return parser


def select_balanced_runner_items(
    workload_path: str | Path,
    *,
    prompts_per_vertical: int = PROMPTS_PER_VERTICAL,
    max_total_prompts: int = TOTAL_PROMPTS,
) -> list[Any]:
    """Select a deterministic equal-sized sample for each promoted vertical."""

    if prompts_per_vertical <= 0:
        msg = "prompts_per_vertical must be > 0"
        raise ValueError(msg)
    if max_total_prompts <= 0:
        msg = "max_total_prompts must be > 0"
        raise ValueError(msg)
    records_by_vertical: dict[str, list[Any]] = defaultdict(list)
    for record in load_phase3_workload_records(workload_path):
        if record.vertical in VERTICALS:
            records_by_vertical[record.vertical].append(record)

    selected: list[Any] = []
    for vertical in VERTICALS:
        candidates = records_by_vertical.get(vertical, [])
        if len(candidates) < prompts_per_vertical:
            msg = (
                f"Workload has {len(candidates)} {vertical} records; "
                f"{prompts_per_vertical} are required"
            )
            raise ValueError(msg)
        selected.extend(
            workload_record_to_runner_item(record) for record in candidates[:prompts_per_vertical]
        )

    expected_count = prompts_per_vertical * len(VERTICALS)
    if len(selected) != expected_count or expected_count > max_total_prompts:
        msg = f"Balanced selection must contain at most {max_total_prompts} records"
        raise ValueError(msg)
    return selected


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write homogeneous rows to CSV."""

    if not rows:
        msg = "At least one row is required"
        raise ValueError(msg)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _float_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field_name)
        if value in (None, ""):
            continue
        values.append(float(str(value)))
    return values


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("success"))]


def latency_summary_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build aggregate and per-vertical request latency rows."""

    grouped: list[tuple[str, list[dict[str, Any]]]] = [("all", result_rows)]
    grouped.extend(
        (
            vertical,
            [row for row in result_rows if str(row.get("vertical")) == vertical],
        )
        for vertical in VERTICALS
    )
    rows: list[dict[str, Any]] = []
    for vertical, group_rows in grouped:
        success_rows = _successful_rows(group_rows)
        e2e_values = _float_values(success_rows, "end_to_end_latency_ms")
        ttft_values = _float_values(success_rows, "ttft_ms")
        tpot_values = _float_values(success_rows, "tpot_ms")
        throughput_values = _float_values(success_rows, "throughput_tokens_per_second")
        e2e = summarize_latency_ms(e2e_values) if e2e_values else {}
        ttft = summarize_latency_ms(ttft_values) if ttft_values else {}
        tpot = summarize_latency_ms(tpot_values) if tpot_values else {}
        throughput = summarize_latency_ms(throughput_values) if throughput_values else {}
        rows.append(
            {
                "vertical": vertical,
                "request_count": len(group_rows),
                "success_count": len(success_rows),
                "mean_e2e_latency_ms": e2e.get("mean"),
                "p50_e2e_latency_ms": e2e.get("p50"),
                "p95_e2e_latency_ms": e2e.get("p95"),
                "p99_e2e_latency_ms": e2e.get("p99"),
                "mean_ttft_ms": ttft.get("mean"),
                "p50_ttft_ms": ttft.get("p50"),
                "p95_ttft_ms": ttft.get("p95"),
                "p99_ttft_ms": ttft.get("p99"),
                "mean_tpot_ms": tpot.get("mean"),
                "p50_tpot_ms": tpot.get("p50"),
                "p95_tpot_ms": tpot.get("p95"),
                "p99_tpot_ms": tpot.get("p99"),
                "mean_total_tokens_per_second": throughput.get("mean"),
            }
        )
    return rows


def evaluate_result_rows(
    *,
    result_rows: list[dict[str, Any]],
    output_path: str | Path,
    eval_report_path: str | Path,
    eval_summary_path: str | Path,
    block: str = "A1",
    experiment: str = "remote_rtx3070_vllm_smoke",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the unchanged evaluator contract and write smoke reports."""

    generated_answers = [result_row_to_generated_answer(row) for row in result_rows]
    evaluation_rows = evaluate_generated_answers(
        generated_answers,
        load_gold_records("data/scaleup_2000_full"),
    )
    summary = build_summary_rows(
        results_path=output_path,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
    )[0]
    report = {
        "block": block,
        "experiment": experiment,
        "model_inference_triggered": True,
        "evaluator_modified": False,
        "row_count": len(evaluation_rows),
        "summary": summary,
        "evaluation_rows": evaluation_rows,
    }
    write_json(eval_report_path, report)
    write_csv_rows(eval_summary_path, [summary])
    return report, summary


def build_manifest(
    *,
    runner_input_path: str | Path,
    output_path: str | Path,
    result_rows: list[dict[str, Any]],
    start_time: str,
    end_time: str,
    command: str,
) -> RunManifest:
    """Build the A1 run manifest."""

    error_count = len(result_rows) - len(_successful_rows(result_rows))
    return RunManifest(
        run_id="a1-remote-rtx3070-vllm-smoke",
        timestamp_utc=end_time,
        backend="vllm",
        model_alias=MODEL_ALIAS,
        model_id=MODEL_ID,
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(runner_input_path),
        output_path=str(output_path),
        max_records=TOTAL_PROMPTS,
        git_commit=current_git_commit(REPO_ROOT),
        command=command,
        status="completed",
        start_time=start_time,
        end_time=end_time,
        error_count=error_count,
    )


def sanitized_command(argv: list[str]) -> str:
    """Return a command string with any API key value redacted."""

    sanitized = list(argv)
    for index, argument in enumerate(sanitized[:-1]):
        if argument == "--api-key":
            sanitized[index + 1] = "***"
    return " ".join([Path(sys.executable).name, *sanitized])


def run_a1(args: argparse.Namespace) -> dict[str, Any]:
    """Run the A1 smoke or perform its dry-run input validation."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        msg = f"{MODEL_ALIAS} must resolve to {MODEL_ID}, received {model.model_id}"
        raise RuntimeError(msg)

    items = select_balanced_runner_items(args.workload_path)
    runner_input = write_runner_workload_jsonl(items, args.runner_input_path)
    if args.dry_run:
        return {
            "status": "dry_run",
            "runner_input_path": str(runner_input),
            "record_count": len(items),
            "vertical_counts": {
                vertical: sum(1 for item in items if str(item.metadata.get("vertical")) == vertical)
                for vertical in VERTICALS
            },
        }

    readiness = check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    telemetry_samples: list[GpuTelemetrySample] = []
    telemetry_errors: list[str] = []
    stop_event = threading.Event()

    def collect_telemetry() -> None:
        try:
            telemetry_samples.extend(
                sample_gpu_telemetry(
                    duration_seconds=3600.0,
                    interval_seconds=args.telemetry_interval_seconds,
                    ssh_host=args.telemetry_ssh_host,
                    stop_requested=stop_event.is_set,
                )
            )
        except Exception as exc:  # noqa: BLE001
            telemetry_errors.append(f"{type(exc).__name__}: {exc}")

    telemetry_thread = threading.Thread(
        target=collect_telemetry,
        name="a1-gpu-telemetry",
        daemon=True,
    )
    telemetry_thread.start()
    start_time = utc_now()
    wall_start = time.perf_counter()
    try:
        run_openai_compatible_benchmark(
            workload_path=runner_input,
            output_path=args.metrics_output_path,
            generation_output_path=args.output_path,
            model=MODEL_ID,
            base_url=args.base_url,
            api_key=args.api_key,
            run_id="a1-remote-rtx3070-vllm-smoke",
            backend="vllm",
            optimization="a1_remote_rtx3070_vllm_smoke",
            max_new_tokens=MAX_NEW_TOKENS,
            max_prompts=TOTAL_PROMPTS,
            stream=True,
            timeout_seconds=args.timeout_seconds,
        )
    finally:
        wall_seconds = time.perf_counter() - wall_start
        stop_event.set()
        telemetry_thread.join(timeout=max(5.0, args.telemetry_interval_seconds + 3.0))
    end_time = utc_now()

    result_rows = load_result_rows(args.output_path)
    eval_report, eval_summary = evaluate_result_rows(
        result_rows=result_rows,
        output_path=args.output_path,
        eval_report_path=args.eval_report_path,
        eval_summary_path=args.eval_summary_path,
    )
    latency_rows = latency_summary_rows(result_rows)
    write_csv_rows(args.latency_summary_path, latency_rows)
    all_latency = latency_rows[0]
    projection = build_runtime_projections(
        measured_prompt_count=len(result_rows),
        measured_wall_seconds=wall_seconds,
        mean_latency_ms=float(all_latency["mean_e2e_latency_ms"] or 0.0),
        p50_latency_ms=float(all_latency["p50_e2e_latency_ms"] or 0.0),
        p95_latency_ms=float(all_latency["p95_e2e_latency_ms"] or 0.0),
    )
    projection.update(
        {
            "source_run": "a1-remote-rtx3070-vllm-smoke",
            "model_id": MODEL_ID,
            "hardware": "remote_rtx3070",
            "telemetry_errors": telemetry_errors,
        }
    )
    write_runtime_projection(args.runtime_projection_path, projection)
    write_gpu_telemetry_csv(args.gpu_telemetry_csv, telemetry_samples)
    write_gpu_telemetry_summary(args.gpu_telemetry_summary, telemetry_samples)
    manifest = build_manifest(
        runner_input_path=runner_input,
        output_path=args.output_path,
        result_rows=result_rows,
        start_time=start_time,
        end_time=end_time,
        command=sanitized_command(sys.argv),
    )
    write_run_manifest(manifest, args.manifest_path)
    return {
        "status": "completed",
        "server_readiness": readiness.to_dict(),
        "row_count": len(result_rows),
        "success_count": len(_successful_rows(result_rows)),
        "wall_seconds": wall_seconds,
        "evaluation_summary": eval_summary,
        "latency_summary": all_latency,
        "telemetry_sample_count": len(telemetry_samples),
        "telemetry_errors": telemetry_errors,
        "projection": projection,
        "evaluation_report": eval_report,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the A1 CLI."""

    args = build_parser().parse_args(argv)
    try:
        result = run_a1(args)
    except Exception as exc:  # noqa: BLE001
        print(f"A1 remote vLLM smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
