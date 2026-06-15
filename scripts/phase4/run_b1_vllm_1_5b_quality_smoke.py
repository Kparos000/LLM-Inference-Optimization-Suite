"""Run the Phase B1 Qwen2.5-1.5B vLLM quality-gated smoke."""

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

from evaluate_generation_outputs import build_summary_rows, load_result_rows  # noqa: E402
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
    write_json,
)

from inference_bench.b1_quality import (  # noqa: E402
    build_b1_comparison,
    build_b1_runtime_projection,
    build_per_vertical_quality,
    build_quality_gate,
    build_root_cause_analysis,
    mean_or_none,
)
from inference_bench.config import load_project_config, load_yaml_file  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    sample_gpu_telemetry,
    summarize_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.streaming_metrics import (  # noqa: E402
    StreamingMetrics,
    request_streaming_chat_completion,
)
from inference_bench.workload_adapter import write_runner_workload_jsonl  # noqa: E402

DEFAULT_WORKLOAD = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_RUNNER_INPUT = "data/generated/phase4/b1_remote_rtx3070_runner_input.jsonl"
DEFAULT_RAW_OUTPUT = "results/raw/b1_remote_rtx3070_vllm_1_5b_results.jsonl"
DEFAULT_MANIFEST = "results/raw/b1_remote_rtx3070_vllm_1_5b_manifest.json"
DEFAULT_QUALITY_REPORT = "results/processed/b1_vllm_1_5b_quality_report.json"
DEFAULT_QUALITY_SUMMARY = "results/processed/b1_vllm_1_5b_quality_summary.csv"
DEFAULT_COMPARISON = "results/processed/b1_vllm_1_5b_vs_0_5b_comparison.json"
DEFAULT_RUNTIME_PROJECTION = "results/processed/b1_vllm_1_5b_runtime_projection.json"
DEFAULT_LATENCY_SUMMARY = "results/processed/b1_vllm_1_5b_latency_summary.csv"
DEFAULT_GPU_CSV = "results/processed/b1_vllm_1_5b_gpu_telemetry.csv"
DEFAULT_GPU_SUMMARY = "results/processed/b1_vllm_1_5b_gpu_telemetry_summary.json"
DEFAULT_RUNPOD_CONFIG = "configs/runpod_projection_prices.yaml"
A1_RAW_OUTPUT = "results/raw/a1_remote_rtx3070_vllm_smoke_results.jsonl"
A1_EVAL_SUMMARY = "results/processed/a1_remote_rtx3070_vllm_eval_summary.csv"
A1_LATENCY_SUMMARY = "results/processed/a1_remote_rtx3070_vllm_latency_summary.csv"
A1_GPU_SUMMARY = "results/processed/a1_remote_rtx3070_gpu_telemetry_summary.json"
MODEL_ALIAS = "model2_1_5b"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS_PER_VERTICAL = 20
TOTAL_PROMPTS = 100
MAX_NEW_TOKENS = 128


def build_parser() -> argparse.ArgumentParser:
    """Build the B1 CLI parser."""

    parser = argparse.ArgumentParser(description="Run the Qwen2.5-1.5B quality-gated vLLM smoke.")
    parser.add_argument("--workload-path", default=DEFAULT_WORKLOAD)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--output-path", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--quality-report-path", default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--quality-summary-path", default=DEFAULT_QUALITY_SUMMARY)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON)
    parser.add_argument("--runtime-projection-path", default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--latency-summary-path", default=DEFAULT_LATENCY_SUMMARY)
    parser.add_argument("--gpu-telemetry-csv", default=DEFAULT_GPU_CSV)
    parser.add_argument("--gpu-telemetry-summary", default=DEFAULT_GPU_SUMMARY)
    parser.add_argument("--runpod-projection-config", default=DEFAULT_RUNPOD_CONFIG)
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


def _throughput(metrics: StreamingMetrics) -> float | None:
    if metrics.e2e_latency_ms <= 0:
        return None
    return metrics.total_tokens / (metrics.e2e_latency_ms / 1000.0)


def _result_row(
    *,
    item: Any,
    metrics: StreamingMetrics,
) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    contract = generation_contract_result_fields(
        metrics.generated_text,
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
    )
    return {
        "run_id": "b1-remote-rtx3070-vllm-1-5b-quality-smoke",
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": "b1_remote_rtx3070_vllm_1_5b_quality_smoke",
        "prompt": item.prompt,
        "generated_text": metrics.generated_text,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
        "token_count_source": metrics.token_count_source,
        "content_chunk_count": metrics.content_chunk_count,
        "streaming_available": metrics.streaming_available,
        "ttft_ms": metrics.ttft_ms,
        "itl_p50_ms": metrics.itl_p50_ms,
        "itl_p95_ms": metrics.itl_p95_ms,
        "itl_p99_ms": metrics.itl_p99_ms,
        "tpot_ms": metrics.tpot_ms,
        "end_to_end_latency_ms": metrics.e2e_latency_ms,
        "throughput_tokens_per_second": _throughput(metrics),
        "peak_memory_mb": None,
        "estimated_cost_usd": None,
        "success": True,
        "error_message": None,
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.metadata.get("vertical"),
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": aliases,
        **contract,
    }


def _failure_row(*, item: Any, exc: Exception, elapsed_ms: float) -> dict[str, Any]:
    return {
        "run_id": "b1-remote-rtx3070-vllm-1-5b-quality-smoke",
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": "b1_remote_rtx3070_vllm_1_5b_quality_smoke",
        "prompt": item.prompt,
        "generated_text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "token_count_source": "unavailable",
        "content_chunk_count": 0,
        "streaming_available": False,
        "ttft_ms": None,
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "itl_p99_ms": None,
        "tpot_ms": None,
        "end_to_end_latency_ms": elapsed_ms,
        "throughput_tokens_per_second": None,
        "peak_memory_mb": None,
        "estimated_cost_usd": None,
        "success": False,
        "error_message": f"{type(exc).__name__}: {exc}",
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.metadata.get("vertical"),
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": item.metadata.get("citation_id_aliases"),
        "generation_contract_valid": False,
        "generation_contract_error": "request_failed",
        "generation_contract_missing_fields": [],
        "parse_error_type": "request_failed",
        "parse_repair_applied": False,
        "truncation_detected": False,
        "answer": "",
        "evidence_ids": [],
        "citations": [],
        "confidence": None,
        "insufficient_evidence": None,
        "citation_notes": "",
    }


def run_streaming_requests(
    *,
    items: list[Any],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Run at most 100 sequential streaming requests and persist every row."""

    if len(items) != TOTAL_PROMPTS:
        msg = f"B1 requires exactly {TOTAL_PROMPTS} items, received {len(items)}"
        raise ValueError(msg)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    api_route = f"{base_url.rstrip('/')}/chat/completions"
    for item in items:
        request_start = time.perf_counter()
        try:
            metrics = request_streaming_chat_completion(
                api_key=api_key,
                model_id=MODEL_ID,
                prompt=item.prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                api_route=api_route,
                timeout_seconds=timeout_seconds,
            )
            row = _result_row(item=item, metrics=metrics)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - request_start) * 1000.0
            row = _failure_row(item=item, exc=exc, elapsed_ms=elapsed_ms)
        rows.append(row)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def _float_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if bool(row.get("success")) and row.get(field) not in (None, "")
    ]


def build_latency_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extend the established latency summary with measured ITL fields."""

    rows = latency_summary_rows(result_rows)
    for row in rows:
        vertical = str(row["vertical"])
        group = (
            result_rows
            if vertical == "all"
            else [result for result in result_rows if str(result.get("vertical")) == vertical]
        )
        row["mean_itl_p50_ms"] = mean_or_none(_float_values(group, "itl_p50_ms"))
        row["mean_itl_p95_ms"] = mean_or_none(_float_values(group, "itl_p95_ms"))
        row["mean_itl_p99_ms"] = mean_or_none(_float_values(group, "itl_p99_ms"))
        row["provider_usage_row_count"] = sum(
            1 for result in group if result.get("token_count_source") == "provider_usage"
        )
        row["whitespace_fallback_row_count"] = sum(
            1 for result in group if result.get("token_count_source") == "whitespace_fallback"
        )
    return rows


def _quality_summary_rows(
    *,
    overall: dict[str, Any],
    per_vertical: list[dict[str, object]],
    latency_rows: list[dict[str, Any]],
    gate_status: str,
) -> list[dict[str, Any]]:
    latency_by_vertical = {str(row["vertical"]): row for row in latency_rows}
    quality_rows: list[dict[str, object]] = [
        {
            "vertical": "all",
            "row_count": overall.get("row_count"),
            "json_valid_rate": overall.get("json_valid_rate"),
            "generation_contract_valid_rate": overall.get("generation_contract_valid_rate"),
            "evidence_id_presence_rate": overall.get("evidence_id_presence_rate"),
            "evidence_match_rate": overall.get("evidence_match_rate"),
            "grounded_rate": overall.get("grounded_rate"),
            "safety_violation_count": overall.get("safety_violation_count"),
            "safety_violation_rate": overall.get("safety_violation_rate"),
            "truncation_count": overall.get("truncation_count"),
            "truncation_rate": overall.get("truncation_rate"),
        },
        *per_vertical,
    ]
    output: list[dict[str, Any]] = []
    for quality in quality_rows:
        vertical = str(quality["vertical"])
        latency = latency_by_vertical[vertical]
        output.append(
            {
                "vertical": vertical,
                "row_count": quality.get("row_count"),
                "json_valid_rate": quality.get("json_valid_rate"),
                "generation_contract_valid_rate": quality.get("generation_contract_valid_rate"),
                "evidence_id_presence_rate": quality.get("evidence_id_presence_rate"),
                "evidence_match_rate": quality.get("evidence_match_rate"),
                "grounded_rate": quality.get("grounded_rate"),
                "safety_violation_count": quality.get("safety_violation_count"),
                "safety_violation_rate": quality.get("safety_violation_rate"),
                "truncation_count": quality.get("truncation_count"),
                "truncation_rate": quality.get("truncation_rate"),
                "mean_ttft_ms": latency.get("mean_ttft_ms"),
                "mean_tpot_ms": latency.get("mean_tpot_ms"),
                "mean_itl_p50_ms": latency.get("mean_itl_p50_ms"),
                "mean_itl_p95_ms": latency.get("mean_itl_p95_ms"),
                "mean_itl_p99_ms": latency.get("mean_itl_p99_ms"),
                "mean_e2e_latency_ms": latency.get("mean_e2e_latency_ms"),
                "mean_total_tokens_per_second": latency.get("mean_total_tokens_per_second"),
                "quality_gate_status": gate_status if vertical == "all" else "",
            }
        )
    return output


def _load_csv_first(path: str | Path) -> dict[str, Any]:
    rows = load_result_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _build_manifest(
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
    return RunManifest(
        run_id="b1-remote-rtx3070-vllm-1-5b-quality-smoke",
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
        error_count=sum(1 for row in result_rows if not bool(row.get("success"))),
        telemetry_path=str(telemetry_path),
        telemetry_summary_path=str(telemetry_summary_path),
    )


def run_b1(args: argparse.Namespace) -> dict[str, Any]:
    """Run B1 or validate its frozen 100-prompt input."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        msg = f"{MODEL_ALIAS} must resolve to {MODEL_ID}, received {model.model_id}"
        raise RuntimeError(msg)

    items = select_balanced_runner_items(
        args.workload_path,
        prompts_per_vertical=PROMPTS_PER_VERTICAL,
        max_total_prompts=TOTAL_PROMPTS,
    )
    if len(items) != TOTAL_PROMPTS:
        msg = f"B1 requires exactly {TOTAL_PROMPTS} records, received {len(items)}"
        raise RuntimeError(msg)
    runner_input = write_runner_workload_jsonl(items, args.runner_input_path)
    vertical_counts = {
        vertical: sum(1 for item in items if str(item.metadata.get("vertical")) == vertical)
        for vertical in VERTICALS
    }
    if any(count != PROMPTS_PER_VERTICAL for count in vertical_counts.values()):
        raise RuntimeError(f"B1 vertical balance is invalid: {vertical_counts}")
    if args.dry_run:
        return {
            "status": "dry_run",
            "runner_input_path": str(runner_input),
            "record_count": len(items),
            "vertical_counts": vertical_counts,
            "model_id": MODEL_ID,
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
        name="b1-gpu-telemetry",
        daemon=True,
    )
    telemetry_thread.start()
    start_time = utc_now()
    wall_start = time.perf_counter()
    try:
        result_rows = run_streaming_requests(
            items=items,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            output_path=args.output_path,
        )
    finally:
        wall_seconds = time.perf_counter() - wall_start
        stop_event.set()
        telemetry_thread.join(timeout=max(5.0, args.telemetry_interval_seconds + 3.0))
    end_time = utc_now()

    if len(result_rows) != TOTAL_PROMPTS:
        msg = f"B1 produced {len(result_rows)} rows; exactly {TOTAL_PROMPTS} are required"
        raise RuntimeError(msg)
    eval_report, eval_summary = evaluate_result_rows(
        result_rows=result_rows,
        output_path=args.output_path,
        eval_report_path=args.quality_report_path,
        eval_summary_path=args.quality_summary_path,
        block="B1",
        experiment="remote_rtx3070_vllm_1_5b_quality_smoke",
    )
    evaluation_rows = eval_report["evaluation_rows"]
    if not isinstance(evaluation_rows, list):
        raise RuntimeError("B1 evaluator report did not contain evaluation rows")
    per_vertical = build_per_vertical_quality(
        evaluation_rows,
        result_rows,
        verticals=VERTICALS,
    )
    gate = build_quality_gate(eval_summary)
    root_cause = build_root_cause_analysis(
        gate=gate,
        per_vertical_quality=per_vertical,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
    )
    latency_rows = build_latency_rows(result_rows)
    write_csv_rows(args.latency_summary_path, latency_rows)
    write_csv_rows(
        args.quality_summary_path,
        _quality_summary_rows(
            overall=eval_summary,
            per_vertical=per_vertical,
            latency_rows=latency_rows,
            gate_status=str(gate["status"]),
        ),
    )

    write_gpu_telemetry_csv(args.gpu_telemetry_csv, telemetry_samples)
    write_gpu_telemetry_summary(
        args.gpu_telemetry_summary,
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    telemetry_summary = summarize_gpu_telemetry(
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    all_latency = latency_rows[0]
    runpod_config = load_yaml_file(args.runpod_projection_config)
    runpod_targets = {key: value for key, value in runpod_config.items() if isinstance(value, dict)}
    projection = build_b1_runtime_projection(
        measured_prompt_count=len(result_rows),
        measured_wall_seconds=wall_seconds,
        mean_latency_ms=float(all_latency["mean_e2e_latency_ms"] or 0.0),
        p50_latency_ms=float(all_latency["p50_e2e_latency_ms"] or 0.0),
        p95_latency_ms=float(all_latency["p95_e2e_latency_ms"] or 0.0),
        runpod_targets=runpod_targets,
    )
    projection.update(
        {
            "source_run": "b1-remote-rtx3070-vllm-1-5b-quality-smoke",
            "model_id": MODEL_ID,
            "hardware": "remote_rtx3070",
        }
    )
    write_json(args.runtime_projection_path, projection)

    a1_rows = load_result_rows(A1_RAW_OUTPUT)
    a1_prompt_ids = {str(row.get("prompt_id") or "") for row in a1_rows}
    matched_result_rows = [
        row for row in result_rows if str(row.get("prompt_id") or "") in a1_prompt_ids
    ]
    matched_evaluation_rows = [
        row for row in evaluation_rows if str(row.get("prompt_id") or "") in a1_prompt_ids
    ]
    matched_quality = build_summary_rows(
        results_path=args.output_path,
        result_rows=matched_result_rows,
        evaluation_rows=matched_evaluation_rows,
    )[0]
    matched_latency = build_latency_rows(matched_result_rows)[0]
    a1_quality = _load_csv_first(A1_EVAL_SUMMARY)
    a1_latency = _load_csv_first(A1_LATENCY_SUMMARY)
    comparison = build_b1_comparison(
        baseline_quality=a1_quality,
        candidate_quality=eval_summary,
        baseline_latency=a1_latency,
        candidate_latency=all_latency,
        baseline_telemetry=_load_json(A1_GPU_SUMMARY),
        candidate_telemetry=telemetry_summary,
        baseline_prompt_ids=a1_prompt_ids,
        candidate_prompt_ids={str(row.get("prompt_id") or "") for row in result_rows},
    )
    comparison["quality_gate"] = gate
    comparison["throughput_token_count_comparability"] = {
        "status": "NOT_TOKENIZER_MATCHED",
        "a1_source": "legacy_runner_whitespace_estimate",
        "b1_source": "provider_usage",
        "interpretation": (
            "Observed throughput deltas are retained, but token-rate claims are not "
            "fully comparable until both runs use the same tokenizer provenance."
        ),
    }
    comparison["matched_overlap_comparison"] = build_b1_comparison(
        baseline_quality=a1_quality,
        candidate_quality=matched_quality,
        baseline_latency=a1_latency,
        candidate_latency=matched_latency,
        baseline_telemetry=None,
        candidate_telemetry=None,
        baseline_prompt_ids=a1_prompt_ids,
        candidate_prompt_ids={str(row.get("prompt_id") or "") for row in matched_result_rows},
    )
    comparison["matched_overlap_comparison"]["gpu_telemetry_note"] = (
        "Board telemetry covers the full 100-prompt B1 run and cannot be isolated "
        "honestly to the 50-prompt overlap."
    )
    comparison["matched_overlap_comparison"]["throughput_token_count_comparability"] = comparison[
        "throughput_token_count_comparability"
    ]
    write_json(args.comparison_path, comparison)

    quality_report = {
        **eval_report,
        "status": gate["status"],
        "quality_gate": gate,
        "per_vertical_quality": per_vertical,
        "latency_summary": all_latency,
        "gpu_telemetry_summary": telemetry_summary,
        "root_cause_analysis": root_cause,
        "request_success_count": sum(1 for row in result_rows if bool(row.get("success"))),
        "request_failure_count": sum(1 for row in result_rows if not bool(row.get("success"))),
        "wall_seconds": wall_seconds,
        "telemetry_errors": telemetry_errors,
        "concurrency": 1,
        "optional_concurrency_sweep_executed": False,
        "optional_concurrency_sweep_reason": (
            "B1 freezes the quality decision at concurrency 1. "
            "Concurrency 2/4 is deferred unless a separate bounded follow-up is approved."
        ),
    }
    write_json(args.quality_report_path, quality_report)
    manifest = _build_manifest(
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
        "status": gate["status"],
        "server_readiness": readiness.to_dict(),
        "row_count": len(result_rows),
        "success_count": sum(1 for row in result_rows if bool(row.get("success"))),
        "wall_seconds": wall_seconds,
        "quality_gate": gate,
        "evaluation_summary": eval_summary,
        "per_vertical_quality": per_vertical,
        "latency_summary": all_latency,
        "gpu_telemetry_summary": telemetry_summary,
        "telemetry_errors": telemetry_errors,
        "runtime_projection": projection,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the B1 CLI."""

    args = build_parser().parse_args(argv)
    try:
        result = run_b1(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B1 vLLM quality smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
