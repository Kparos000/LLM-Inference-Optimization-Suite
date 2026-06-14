"""Run the frozen A2 remote RTX 3070 SGLang smoke and write reports."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PHASE4_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from evaluate_generation_outputs import load_result_rows  # noqa: E402
from run_openai_compatible_smoke import (  # noqa: E402
    DEFAULT_API_KEY,
    check_server_readiness,
)
from run_remote_vllm_smoke import (  # noqa: E402
    evaluate_result_rows,
    latency_summary_rows,
    sanitized_command,
    select_balanced_runner_items,
    write_csv_rows,
)

from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    sample_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.openai_compatible_runner import (  # noqa: E402
    run_openai_compatible_benchmark,
)
from inference_bench.workload_adapter import write_runner_workload_jsonl  # noqa: E402

DEFAULT_WORKLOAD = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_RUNNER_INPUT = "data/generated/phase4/a2_remote_rtx3070_runner_input.jsonl"
DEFAULT_RAW_OUTPUT = "results/raw/a2_remote_rtx3070_sglang_smoke_results.jsonl"
DEFAULT_METRICS_OUTPUT = "results/raw/a2_remote_rtx3070_sglang_smoke_metrics.csv"
DEFAULT_MANIFEST = "results/raw/a2_remote_rtx3070_sglang_smoke_manifest.json"
DEFAULT_EVAL_REPORT = "results/processed/a2_remote_rtx3070_sglang_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/a2_remote_rtx3070_sglang_eval_summary.csv"
DEFAULT_LATENCY_SUMMARY = "results/processed/a2_remote_rtx3070_sglang_latency_summary.csv"
DEFAULT_GPU_CSV = "results/processed/a2_remote_rtx3070_sglang_gpu_telemetry.csv"
DEFAULT_GPU_SUMMARY = "results/processed/a2_remote_rtx3070_sglang_gpu_telemetry_summary.json"
MODEL_ALIAS = "model1_0_5b"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TOTAL_PROMPTS = 50
MAX_NEW_TOKENS = 128


def build_parser() -> argparse.ArgumentParser:
    """Build the A2 CLI parser."""

    parser = argparse.ArgumentParser(description="Run the frozen remote RTX 3070 SGLang smoke.")
    parser.add_argument("--workload-path", default=DEFAULT_WORKLOAD)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--output-path", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--metrics-output-path", default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--latency-summary-path", default=DEFAULT_LATENCY_SUMMARY)
    parser.add_argument("--gpu-telemetry-csv", default=DEFAULT_GPU_CSV)
    parser.add_argument("--gpu-telemetry-summary", default=DEFAULT_GPU_SUMMARY)
    parser.add_argument("--telemetry-ssh-host", default=None)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--telemetry-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export and validate the frozen input without contacting a server.",
    )
    return parser


def _successful_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if bool(row.get("success"))]


def build_manifest(
    *,
    runner_input_path: str | Path,
    output_path: str | Path,
    telemetry_path: str | Path,
    telemetry_summary_path: str | Path,
    result_rows: list[dict[str, Any]],
    start_time: str,
    end_time: str,
    command: str,
) -> RunManifest:
    """Build the A2 run manifest with telemetry attachments."""

    return RunManifest(
        run_id="a2-remote-rtx3070-sglang-smoke",
        timestamp_utc=end_time,
        backend="sglang",
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
        error_count=len(result_rows) - len(_successful_rows(result_rows)),
        telemetry_path=str(telemetry_path),
        telemetry_summary_path=str(telemetry_summary_path),
    )


def run_a2(args: argparse.Namespace) -> dict[str, Any]:
    """Run the A2 smoke or perform its dry-run input validation."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        msg = f"{MODEL_ALIAS} must resolve to {MODEL_ID}, received {model.model_id}"
        raise RuntimeError(msg)

    items = select_balanced_runner_items(args.workload_path)
    if len(items) != TOTAL_PROMPTS:
        msg = f"A2 requires exactly {TOTAL_PROMPTS} records, received {len(items)}"
        raise RuntimeError(msg)
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
                    duration_seconds=args.telemetry_duration_seconds,
                    interval_seconds=args.telemetry_interval_seconds,
                    ssh_host=args.telemetry_ssh_host,
                    stop_requested=stop_event.is_set,
                )
            )
        except Exception as exc:  # noqa: BLE001
            telemetry_errors.append(f"{type(exc).__name__}: {exc}")

    telemetry_thread = threading.Thread(
        target=collect_telemetry,
        name="a2-gpu-telemetry",
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
            run_id="a2-remote-rtx3070-sglang-smoke",
            backend="sglang",
            optimization="a2_remote_rtx3070_sglang_smoke",
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
    if len(result_rows) > TOTAL_PROMPTS:
        msg = f"A2 produced {len(result_rows)} rows; the limit is {TOTAL_PROMPTS}"
        raise RuntimeError(msg)
    _, eval_summary = evaluate_result_rows(
        result_rows=result_rows,
        output_path=args.output_path,
        eval_report_path=args.eval_report_path,
        eval_summary_path=args.eval_summary_path,
        block="A2",
        experiment="remote_rtx3070_sglang_smoke",
    )
    latency_rows = latency_summary_rows(result_rows)
    write_csv_rows(args.latency_summary_path, latency_rows)
    write_gpu_telemetry_csv(args.gpu_telemetry_csv, telemetry_samples)
    write_gpu_telemetry_summary(
        args.gpu_telemetry_summary,
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    manifest = build_manifest(
        runner_input_path=runner_input,
        output_path=args.output_path,
        telemetry_path=args.gpu_telemetry_csv,
        telemetry_summary_path=args.gpu_telemetry_summary,
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
        "latency_summary": latency_rows[0],
        "telemetry_sample_count": len(telemetry_samples),
        "telemetry_errors": telemetry_errors,
        "manifest_path": str(args.manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the A2 CLI."""

    args = build_parser().parse_args(argv)
    try:
        result = run_a2(args)
    except Exception as exc:  # noqa: BLE001
        print(
            f"A2 remote SGLang smoke failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
