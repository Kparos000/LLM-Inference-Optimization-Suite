"""Run the frozen B4 context-aligned Qwen2.5-1.5B vLLM smoke."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from statistics import fmean
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import load_gold_records, load_result_rows  # noqa: E402
from run_openai_compatible_smoke import (  # noqa: E402
    DEFAULT_API_KEY,
    check_server_readiness,
)
from run_remote_vllm_smoke import (  # noqa: E402
    evaluate_result_rows,
    latency_summary_rows,
    sanitized_command,
    write_csv_rows,
    write_json,
)

from inference_bench.b1_quality import build_per_vertical_quality  # noqa: E402
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.generation_prompt_repair import (  # noqa: E402
    build_generation_repair_prompt,
    decide_generation_repair,
)
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    sample_gpu_telemetry,
    summarize_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.streaming_metrics import (  # noqa: E402
    StreamingMetrics,
    request_streaming_chat_completion,
)

MODEL_ALIAS = "model2_1_5b"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOTAL_PROMPTS = 100
PROMPTS_PER_VERTICAL = 20
MAX_NEW_TOKENS = 160

DEFAULT_RUNNER_INPUT = "data/generated/phase4/b4_context_aligned_runner_input.jsonl"
DEFAULT_PREFLIGHT = "results/processed/b4_context_alignment_report.json"
DEFAULT_RAW = "results/raw/b4_vllm_1_5b_context_aligned_results.jsonl"
DEFAULT_MANIFEST = "results/raw/b4_vllm_1_5b_context_aligned_manifest.json"
DEFAULT_EVAL_REPORT = "results/processed/b4_vllm_1_5b_context_aligned_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/b4_vllm_1_5b_context_aligned_eval_summary.csv"
DEFAULT_COMPARISON_JSON = "results/processed/b4_b1_vs_b4_comparison.json"
DEFAULT_COMPARISON_CSV = "results/processed/b4_b1_vs_b4_comparison.csv"
DEFAULT_LATENCY = "results/processed/b4_vllm_1_5b_context_aligned_latency_summary.csv"
DEFAULT_GPU_CSV = "results/processed/b4_vllm_1_5b_context_aligned_gpu_telemetry.csv"
DEFAULT_GPU_SUMMARY = "results/processed/b4_vllm_1_5b_context_aligned_gpu_telemetry_summary.json"
B1_REPORT = "results/processed/b1_vllm_1_5b_quality_report.json"
B1_LATENCY = "results/processed/b1_vllm_1_5b_latency_summary.csv"
B1_GPU = "results/processed/b1_vllm_1_5b_gpu_telemetry_summary.json"
B1_RAW = "results/raw/b1_remote_rtx3070_vllm_1_5b_results.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Build the B4 runner CLI."""

    parser = argparse.ArgumentParser(
        description="Run the 100-prompt B4 context-aligned vLLM quality smoke."
    )
    parser.add_argument("--runner-input", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--preflight-report", default=DEFAULT_PREFLIGHT)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--output", default=DEFAULT_RAW)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--comparison-json", default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--comparison-csv", default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--latency-summary", default=DEFAULT_LATENCY)
    parser.add_argument("--gpu-telemetry-csv", default=DEFAULT_GPU_CSV)
    parser.add_argument("--gpu-telemetry-summary", default=DEFAULT_GPU_SUMMARY)
    parser.add_argument("--telemetry-ssh-host", default=None)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--telemetry-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _read_runner_items(path: str | Path) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            items.append(WorkloadItem(**payload))
    return items


def _throughput(total_tokens: int, latency_ms: float) -> float | None:
    return total_tokens / (latency_ms / 1000.0) if latency_ms > 0 else None


def _metric_fields(metrics: StreamingMetrics) -> dict[str, Any]:
    return {
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
        "throughput_tokens_per_second": _throughput(
            metrics.total_tokens,
            metrics.e2e_latency_ms,
        ),
    }


def _result_row(item: WorkloadItem, metrics: StreamingMetrics) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    return {
        "run_id": "b4-vllm-1-5b-context-aligned",
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": "b4_context_alignment_and_bounded_generation_repair",
        "prompt": item.prompt,
        "generated_text": metrics.generated_text,
        **_metric_fields(metrics),
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
        "context_alignment_status": item.metadata.get("context_alignment_status"),
        **generation_contract_result_fields(
            metrics.generated_text,
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        ),
    }


def _failure_row(item: WorkloadItem, exc: Exception, elapsed_ms: float) -> dict[str, Any]:
    return {
        "run_id": "b4-vllm-1-5b-context-aligned",
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": "b4_context_alignment_and_bounded_generation_repair",
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
        "context_alignment_status": item.metadata.get("context_alignment_status"),
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
        "repair_attempted": False,
        "repair_trigger": "request_failed",
    }


def _merge_repair(
    *,
    initial: dict[str, Any],
    repair_metrics: StreamingMetrics,
    trigger: str,
) -> dict[str, Any]:
    aliases = initial.get("citation_id_aliases")
    initial_latency = float(initial.get("end_to_end_latency_ms") or 0.0)
    total_latency = initial_latency + repair_metrics.e2e_latency_ms
    total_input = int(initial.get("input_tokens") or 0) + repair_metrics.input_tokens
    total_output = int(initial.get("output_tokens") or 0) + repair_metrics.output_tokens
    total_tokens = total_input + total_output
    final = {
        **initial,
        "generated_text": repair_metrics.generated_text,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "content_chunk_count": int(initial.get("content_chunk_count") or 0)
        + repair_metrics.content_chunk_count,
        "end_to_end_latency_ms": total_latency,
        "throughput_tokens_per_second": _throughput(total_tokens, total_latency),
        "initial_generated_text": initial.get("generated_text"),
        "initial_input_tokens": initial.get("input_tokens"),
        "initial_output_tokens": initial.get("output_tokens"),
        "initial_end_to_end_latency_ms": initial.get("end_to_end_latency_ms"),
        "initial_generation_contract_valid": initial.get("generation_contract_valid"),
        "initial_parse_error_type": initial.get("parse_error_type"),
        "repair_attempted": True,
        "repair_trigger": trigger,
        "repair_generated_text": repair_metrics.generated_text,
        "repair_input_tokens": repair_metrics.input_tokens,
        "repair_output_tokens": repair_metrics.output_tokens,
        "repair_ttft_ms": repair_metrics.ttft_ms,
        "repair_itl_p50_ms": repair_metrics.itl_p50_ms,
        "repair_itl_p95_ms": repair_metrics.itl_p95_ms,
        "repair_itl_p99_ms": repair_metrics.itl_p99_ms,
        "repair_tpot_ms": repair_metrics.tpot_ms,
        "repair_e2e_latency_ms": repair_metrics.e2e_latency_ms,
        **generation_contract_result_fields(
            repair_metrics.generated_text,
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        ),
    }
    return final


def run_requests(
    *,
    items: list[WorkloadItem],
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Run exactly 100 prompts with at most one bounded repair each."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    route = f"{base_url.rstrip('/')}/chat/completions"
    rows: list[dict[str, Any]] = []
    for item in items:
        started = time.perf_counter()
        try:
            metrics = request_streaming_chat_completion(
                api_key=api_key,
                model_id=MODEL_ID,
                prompt=item.prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                api_route=route,
                timeout_seconds=timeout_seconds,
            )
            row = _result_row(item, metrics)
            initial_evaluation = evaluate_result_row(
                row,
                gold_by_prompt.get(item.prompt_id),
            )
            row["initial_json_validity"] = initial_evaluation.get("json_validity")
            row["initial_evidence_match"] = initial_evaluation.get("evidence_match")
            row["initial_groundedness"] = initial_evaluation.get("groundedness")
            row["initial_safety_violation"] = initial_evaluation.get("safety_violation")
            row["initial_safety_violation_terms"] = initial_evaluation.get("safety_violation_terms")
            decision = decide_generation_repair(
                evaluation=initial_evaluation,
                result_row=row,
            )
            row["repair_attempted"] = False
            row["repair_trigger"] = decision.trigger
            if decision.should_retry:
                repair_prompt = build_generation_repair_prompt(
                    decision=decision,
                    result_row=row,
                )
                repair_metrics = request_streaming_chat_completion(
                    api_key=api_key,
                    model_id=MODEL_ID,
                    prompt=repair_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    api_route=route,
                    timeout_seconds=timeout_seconds,
                )
                row = _merge_repair(
                    initial=row,
                    repair_metrics=repair_metrics,
                    trigger=decision.trigger,
                )
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(
                item,
                exc,
                (time.perf_counter() - started) * 1000.0,
            )
        rows.append(row)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def _b4_gate(
    *,
    summary: dict[str, Any],
    b1_summary: dict[str, Any],
    context_alignment_improved: bool,
) -> dict[str, Any]:
    thresholds = {
        "json_valid_rate": 0.95,
        "generation_contract_valid_rate": 0.95,
        "evidence_match_rate": 0.70,
        "grounded_rate": 0.70,
        "safety_violation_count": 0,
    }
    checks: dict[str, dict[str, Any]] = {}
    for metric, threshold in thresholds.items():
        observed = float(summary[metric])
        passed = (
            observed == threshold if metric == "safety_violation_count" else observed >= threshold
        )
        checks[metric] = {
            "observed": observed,
            "threshold": threshold,
            "passed": passed,
        }
    evidence_delta = float(summary["evidence_match_rate"]) - float(
        b1_summary["evidence_match_rate"]
    )
    grounded_delta = float(summary["grounded_rate"]) - float(b1_summary["grounded_rate"])
    ready = all(bool(check["passed"]) for check in checks.values())
    if ready and context_alignment_improved:
        status = "QUALITY_READY"
    elif (
        context_alignment_improved
        and evidence_delta >= 0.20
        and grounded_delta >= 0.20
        and float(summary["safety_violation_count"]) == 0
    ):
        status = "QUALITY_IMPROVED_BUT_BLOCKED"
    else:
        status = "QUALITY_BLOCKED"
    return {
        "status": status,
        "checks": checks,
        "context_alignment_improved": context_alignment_improved,
        "evidence_match_percentage_point_delta": evidence_delta * 100.0,
        "groundedness_percentage_point_delta": grounded_delta * 100.0,
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return fmean(values) if values else None


def build_latency_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build B4 latency rows including streamed ITL fields."""

    rows = latency_summary_rows(result_rows)
    for row in rows:
        vertical = str(row["vertical"])
        group = (
            result_rows
            if vertical == "all"
            else [result for result in result_rows if result.get("vertical") == vertical]
        )
        row["mean_itl_p50_ms"] = _mean(group, "itl_p50_ms")
        row["mean_itl_p95_ms"] = _mean(group, "itl_p95_ms")
        row["mean_itl_p99_ms"] = _mean(group, "itl_p99_ms")
    return rows


def _delta(baseline: Any, candidate: Any) -> dict[str, float | None]:
    if baseline in (None, "") or candidate in (None, ""):
        return {"baseline": None, "candidate": None, "absolute_delta": None}
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "absolute_delta": candidate_value - baseline_value,
    }


def _telemetry_value(payload: dict[str, Any], group: str, metric: str) -> Any:
    group_payload = payload.get(group)
    return group_payload.get(metric) if isinstance(group_payload, dict) else None


def _build_comparison(
    *,
    b1_report: dict[str, Any],
    b4_summary: dict[str, Any],
    b1_latency: dict[str, Any],
    b4_latency: dict[str, Any],
    b1_gpu: dict[str, Any],
    b4_gpu: dict[str, Any],
    b1_rows: list[dict[str, Any]],
    b4_rows: list[dict[str, Any]],
    context_report: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    b1_summary = cast(dict[str, Any], b1_report["summary"])
    quality_metrics = (
        "json_valid_rate",
        "generation_contract_valid_rate",
        "evidence_id_presence_rate",
        "evidence_match_rate",
        "grounded_rate",
        "safety_violation_count",
        "truncation_rate",
    )
    latency_metrics = (
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_p50_ms",
        "mean_itl_p95_ms",
        "mean_itl_p99_ms",
        "mean_e2e_latency_ms",
        "mean_total_tokens_per_second",
    )
    telemetry_sources = {
        "mean_gpu_utilization_percent": ("utilization_gpu_percent", "mean"),
        "max_gpu_memory_used_mb": ("memory_used_mb", "max"),
        "mean_power_draw_w": ("power_draw_w", "mean"),
        "max_temperature_c": ("temperature_c", "max"),
    }
    return {
        "baseline": "B1_Qwen2.5-1.5B_vLLM",
        "candidate": "B4_Qwen2.5-1.5B_vLLM_context_aligned",
        "prompt_matched": True,
        "prompt_count": 100,
        "quality_deltas": {
            metric: _delta(b1_summary.get(metric), b4_summary.get(metric))
            for metric in quality_metrics
        },
        "latency_throughput_deltas": {
            metric: _delta(b1_latency.get(metric), b4_latency.get(metric))
            for metric in latency_metrics
        },
        "token_deltas": {
            "mean_input_tokens": _delta(
                _mean(b1_rows, "input_tokens"),
                _mean(b4_rows, "input_tokens"),
            ),
            "mean_output_tokens": _delta(
                _mean(b1_rows, "output_tokens"),
                _mean(b4_rows, "output_tokens"),
            ),
            "mean_total_tokens": _delta(
                _mean(b1_rows, "total_tokens"),
                _mean(b4_rows, "total_tokens"),
            ),
        },
        "gpu_telemetry_deltas": {
            metric: _delta(
                _telemetry_value(b1_gpu, *source),
                _telemetry_value(b4_gpu, *source),
            )
            for metric, source in telemetry_sources.items()
        },
        "context_alignment": context_report["summary_rows"],
        "quality_gate": gate,
        "repair_attempt_count": sum(bool(row.get("repair_attempted")) for row in b4_rows),
        "initial_safety_violation_count": sum(
            bool(row.get("initial_safety_violation")) for row in b4_rows
        ),
    }


def _comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name in (
        "quality_deltas",
        "latency_throughput_deltas",
        "token_deltas",
        "gpu_telemetry_deltas",
    ):
        group = cast(dict[str, dict[str, Any]], comparison[group_name])
        for metric, values in group.items():
            rows.append({"metric_group": group_name, "metric": metric, **values})
    return rows


def _write_quality_summary(
    *,
    path: str | Path,
    overall: dict[str, Any],
    per_vertical: list[dict[str, object]],
    latency_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    latency_by_vertical = {str(row["vertical"]): row for row in latency_rows}
    rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = [
        {"vertical": "all", **overall},
        *[dict(row) for row in per_vertical],
    ]
    for quality in quality_rows:
        vertical = str(quality["vertical"])
        latency = latency_by_vertical[vertical]
        rows.append(
            {
                "vertical": vertical,
                "row_count": quality.get("row_count"),
                "json_valid_rate": quality.get("json_valid_rate"),
                "generation_contract_valid_rate": quality.get("generation_contract_valid_rate"),
                "evidence_id_presence_rate": quality.get("evidence_id_presence_rate"),
                "evidence_match_rate": quality.get("evidence_match_rate"),
                "grounded_rate": quality.get("grounded_rate"),
                "safety_violation_count": quality.get("safety_violation_count"),
                "truncation_count": quality.get("truncation_count"),
                "mean_ttft_ms": latency.get("mean_ttft_ms"),
                "mean_tpot_ms": latency.get("mean_tpot_ms"),
                "mean_itl_p50_ms": latency.get("mean_itl_p50_ms"),
                "mean_itl_p95_ms": latency.get("mean_itl_p95_ms"),
                "mean_itl_p99_ms": latency.get("mean_itl_p99_ms"),
                "mean_e2e_latency_ms": latency.get("mean_e2e_latency_ms"),
                "mean_total_tokens_per_second": latency.get("mean_total_tokens_per_second"),
                "quality_gate_status": gate["status"] if vertical == "all" else "",
            }
        )
    write_csv_rows(path, rows)


def run_b4(args: argparse.Namespace) -> dict[str, Any]:
    """Run B4 after enforcing the offline preflight gate."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        raise RuntimeError(f"{MODEL_ALIAS} resolved to unexpected model {model.model_id}")
    preflight = _read_json(args.preflight_report)
    if not bool(preflight.get("inference_allowed")):
        raise RuntimeError("B4 inference is blocked because context preflight did not improve")
    items = _read_runner_items(args.runner_input)
    if len(items) != TOTAL_PROMPTS:
        raise RuntimeError(f"B4 requires exactly {TOTAL_PROMPTS} rows")
    counts = {
        vertical: sum(item.metadata.get("vertical") == vertical for item in items)
        for vertical in VERTICALS
    }
    if any(count != PROMPTS_PER_VERTICAL for count in counts.values()):
        raise RuntimeError(f"B4 vertical balance is invalid: {counts}")
    if any(item.metadata.get("context_alignment_status") != "all" for item in items):
        raise RuntimeError("B4 runner input contains unresolved context alignment")
    if args.dry_run:
        return {
            "status": "dry_run",
            "record_count": len(items),
            "vertical_counts": counts,
            "max_new_tokens": MAX_NEW_TOKENS,
            "inference_allowed": True,
        }

    readiness = check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold = {str(row.get("prompt_id") or ""): row for row in gold_rows}
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

    thread = threading.Thread(target=collect_telemetry, name="b4-gpu-telemetry", daemon=True)
    thread.start()
    start_time = utc_now()
    wall_start = time.perf_counter()
    try:
        result_rows = run_requests(
            items=items,
            gold_by_prompt=gold,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            output_path=args.output,
        )
    finally:
        wall_seconds = time.perf_counter() - wall_start
        stop_event.set()
        thread.join(timeout=max(5.0, args.telemetry_interval_seconds + 3.0))
    end_time = utc_now()

    eval_report, eval_summary = evaluate_result_rows(
        result_rows=result_rows,
        output_path=args.output,
        eval_report_path=args.eval_report,
        eval_summary_path=args.eval_summary,
        block="B4",
        experiment="vllm_1_5b_context_aligned_quality_repair",
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    per_vertical = build_per_vertical_quality(
        evaluation_rows,
        result_rows,
        verticals=VERTICALS,
    )
    latency_rows = build_latency_rows(result_rows)
    write_csv_rows(args.latency_summary, latency_rows)
    write_gpu_telemetry_csv(args.gpu_telemetry_csv, telemetry_samples)
    write_gpu_telemetry_summary(
        args.gpu_telemetry_summary,
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    telemetry = summarize_gpu_telemetry(
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    b1_report = _read_json(B1_REPORT)
    gate = _b4_gate(
        summary=eval_summary,
        b1_summary=cast(dict[str, Any], b1_report["summary"]),
        context_alignment_improved=bool(preflight["preflight_improved"]),
    )
    _write_quality_summary(
        path=args.eval_summary,
        overall=eval_summary,
        per_vertical=per_vertical,
        latency_rows=latency_rows,
        gate=gate,
    )
    b1_latency = next(row for row in load_result_rows(B1_LATENCY) if row.get("vertical") == "all")
    comparison = _build_comparison(
        b1_report=b1_report,
        b4_summary=eval_summary,
        b1_latency=b1_latency,
        b4_latency=latency_rows[0],
        b1_gpu=_read_json(B1_GPU),
        b4_gpu=telemetry,
        b1_rows=load_result_rows(B1_RAW),
        b4_rows=result_rows,
        context_report=preflight,
        gate=gate,
    )
    write_json(args.comparison_json, comparison)
    write_csv_rows(args.comparison_csv, _comparison_rows(comparison))
    quality_report = {
        **eval_report,
        "status": gate["status"],
        "quality_gate": gate,
        "per_vertical_quality": per_vertical,
        "latency_summary": latency_rows[0],
        "gpu_telemetry_summary": telemetry,
        "context_alignment_preflight": preflight["summary_rows"],
        "request_success_count": sum(bool(row.get("success")) for row in result_rows),
        "request_failure_count": sum(not bool(row.get("success")) for row in result_rows),
        "repair_attempt_count": sum(bool(row.get("repair_attempted")) for row in result_rows),
        "repair_trigger_counts": {
            trigger: sum(row.get("repair_trigger") == trigger for row in result_rows)
            for trigger in (
                "invalid_json",
                "invalid_contract",
                "missing_evidence_label",
                "safety_violation",
            )
        },
        "initial_safety_violation_count": sum(
            bool(row.get("initial_safety_violation")) for row in result_rows
        ),
        "final_safety_violation_count": eval_summary["safety_violation_count"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_new_tokens_justification": (
            "B3 observed six 128-token truncations; B4 uses the allowed bounded increase to 160."
        ),
        "wall_seconds": wall_seconds,
        "telemetry_errors": telemetry_errors,
        "concurrency": 1,
    }
    write_json(args.eval_report, quality_report)
    manifest = RunManifest(
        run_id="b4-vllm-1-5b-context-aligned",
        timestamp_utc=end_time,
        backend="vllm",
        model_alias=MODEL_ALIAS,
        model_id=MODEL_ID,
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(args.runner_input),
        output_path=str(args.output),
        max_records=TOTAL_PROMPTS,
        git_commit=current_git_commit(ROOT),
        command=sanitized_command(sys.argv),
        status="completed",
        start_time=start_time,
        end_time=end_time,
        error_count=sum(not bool(row.get("success")) for row in result_rows),
        telemetry_path=str(args.gpu_telemetry_csv),
        telemetry_summary_path=str(args.gpu_telemetry_summary),
    )
    write_run_manifest(manifest, args.manifest)
    return {
        "status": gate["status"],
        "server_readiness": readiness.to_dict(),
        "row_count": len(result_rows),
        "success_count": sum(bool(row.get("success")) for row in result_rows),
        "repair_attempt_count": quality_report["repair_attempt_count"],
        "evaluation_summary": eval_summary,
        "quality_gate": gate,
        "latency_summary": latency_rows[0],
        "gpu_telemetry_summary": telemetry,
        "wall_seconds": wall_seconds,
    }


def main() -> int:
    """Run the B4 CLI."""

    args = build_parser().parse_args()
    try:
        result = run_b4(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B4 vLLM smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
