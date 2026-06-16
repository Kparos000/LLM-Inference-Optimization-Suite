"""Run the B6 500-prompt Qwen2.5-1.5B vLLM quality scale gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import load_gold_records, load_result_rows  # noqa: E402
from run_b5_targeted_generation_repair import (  # noqa: E402
    _apply_lexical_guard,
    _merge_retry,
    _missing_labels,
    _repair_prompt,
)
from run_openai_compatible_smoke import DEFAULT_API_KEY, check_server_readiness  # noqa: E402
from run_remote_vllm_smoke import (  # noqa: E402
    evaluate_result_rows,
    latency_summary_rows,
    sanitized_command,
    write_csv_rows,
    write_json,
)

from inference_bench.b1_quality import build_per_vertical_quality  # noqa: E402
from inference_bench.b6_quality_gate import (  # noqa: E402
    classify_b6_quality_gate,
    preflight_inference_allowed,
)
from inference_bench.config import load_project_config  # noqa: E402
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
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runtime_projection import (  # noqa: E402
    build_runtime_projection_report,
    write_runtime_projection_artifacts,
)
from inference_bench.safety_generation_repair import decide_targeted_retry  # noqa: E402
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.streaming_metrics import (  # noqa: E402
    StreamingMetrics,
    request_streaming_chat_completion,
)

MODEL_ALIAS = "model2_1_5b"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOTAL_PROMPTS = 500
PROMPTS_PER_VERTICAL = 100
MAX_NEW_TOKENS = 160
RUN_ID = "b6-vllm-1-5b-500-quality-gate"
OPTIMIZATION = "b6_context_aligned_b5_repairs_500_quality_gate"

DEFAULT_RUNNER_INPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
DEFAULT_PREFLIGHT = "results/processed/b6_context_alignment_preflight_report.json"
DEFAULT_RAW = "results/raw/b6_vllm_1_5b_500_results.jsonl"
DEFAULT_MANIFEST = "results/raw/b6_vllm_1_5b_500_manifest.json"
DEFAULT_EVAL_REPORT = "results/processed/b6_vllm_1_5b_500_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/b6_vllm_1_5b_500_eval_summary.csv"
DEFAULT_LATENCY = "results/processed/b6_vllm_1_5b_500_latency_summary.csv"
DEFAULT_GPU_CSV = "results/processed/b6_vllm_1_5b_500_gpu_telemetry.csv"
DEFAULT_GPU_SUMMARY = "results/processed/b6_vllm_1_5b_500_gpu_telemetry_summary.json"
DEFAULT_COMPARISON_JSON = "results/processed/b6_b5_vs_b6_comparison.json"
DEFAULT_COMPARISON_CSV = "results/processed/b6_b5_vs_b6_comparison.csv"
DEFAULT_RUNTIME_REPORT = "results/processed/b6_runtime_projection_report.json"
DEFAULT_RUNTIME_SUMMARY = "results/processed/b6_runtime_projection_summary.csv"
DEFAULT_RUNPOD_PROFILES = "configs/runpod_projection_prices.yaml"
DEFAULT_B5_REPORT = "results/processed/b5_full_frozen_100_report.json"
DEFAULT_B5_LATENCY = "results/processed/b5_full_frozen_100_latency_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the B6 runner CLI."""

    parser = argparse.ArgumentParser(description="Run the B6 500-prompt vLLM quality scale gate.")
    parser.add_argument("--runner-input", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--preflight-report", default=DEFAULT_PREFLIGHT)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--output", default=DEFAULT_RAW)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--latency-summary", default=DEFAULT_LATENCY)
    parser.add_argument("--gpu-telemetry-csv", default=DEFAULT_GPU_CSV)
    parser.add_argument("--gpu-telemetry-summary", default=DEFAULT_GPU_SUMMARY)
    parser.add_argument("--comparison-json", default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--comparison-csv", default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--runtime-report", default=DEFAULT_RUNTIME_REPORT)
    parser.add_argument("--runtime-summary", default=DEFAULT_RUNTIME_SUMMARY)
    parser.add_argument("--runpod-profiles", default=DEFAULT_RUNPOD_PROFILES)
    parser.add_argument("--b5-report", default=DEFAULT_B5_REPORT)
    parser.add_argument("--b5-latency-summary", default=DEFAULT_B5_LATENCY)
    parser.add_argument("--telemetry-ssh-host", default=None)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--telemetry-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads((ROOT / Path(path)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / Path(path)).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], payload)


def _read_runner_items(path: str | Path) -> list[WorkloadItem]:
    items: list[WorkloadItem] = []
    with (ROOT / Path(path)).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object row: {path}")
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
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": OPTIMIZATION,
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
        "b5_required_labels": item.metadata.get("b5_required_labels"),
        "b5_safety_rule_ids": item.metadata.get("b5_safety_rule_ids"),
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
        **generation_contract_result_fields(
            metrics.generated_text,
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        ),
    }


def _failure_row(item: WorkloadItem, exc: Exception, elapsed_ms: float) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    return {
        "run_id": RUN_ID,
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": OPTIMIZATION,
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
        "citation_id_aliases": aliases,
        "context_alignment_status": item.metadata.get("context_alignment_status"),
        "b5_required_labels": item.metadata.get("b5_required_labels"),
        "b5_safety_rule_ids": item.metadata.get("b5_safety_rule_ids"),
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
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


def _run_items(
    *,
    items: list[WorkloadItem],
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    output = ROOT / Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    route = f"{base_url.rstrip('/')}/chat/completions"
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
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
            initial = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
            row["initial_json_validity"] = initial.get("json_validity")
            row["initial_contract_validity"] = initial.get("generation_contract_valid")
            row["initial_evidence_match"] = initial.get("evidence_match")
            row["initial_groundedness"] = initial.get("groundedness")
            row["initial_safety_violation"] = initial.get("safety_violation")
            row["initial_safety_violation_terms"] = initial.get("safety_violation_terms")
            evaluation = initial
            if bool(evaluation.get("safety_violation")):
                row = _apply_lexical_guard(row=row, evaluation=evaluation)
                evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))

            while True:
                missing = _missing_labels(evaluation=evaluation, row=row)
                decision = decide_targeted_retry(
                    evaluation=evaluation,
                    missing_labels=missing,
                    attempt_count=int(row.get("retry_attempt_count") or 0),
                    max_attempts=2,
                )
                row["last_retry_decision"] = decision.trigger
                if not decision.should_retry:
                    break
                repair_metrics = request_streaming_chat_completion(
                    api_key=api_key,
                    model_id=MODEL_ID,
                    prompt=_repair_prompt(
                        row=row,
                        trigger=decision.trigger,
                        missing_labels=decision.missing_labels,
                    ),
                    max_new_tokens=MAX_NEW_TOKENS,
                    api_route=route,
                    timeout_seconds=timeout_seconds,
                )
                row = _merge_retry(
                    current=row,
                    metrics=repair_metrics,
                    trigger=decision.trigger,
                )
                evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
                if bool(evaluation.get("safety_violation")):
                    row = _apply_lexical_guard(row=row, evaluation=evaluation)
                    evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
            row["final_evaluation_snapshot"] = evaluation
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(item, exc, (time.perf_counter() - started) * 1000.0)
        row["sequence_index"] = index
        rows.append(row)
        with output.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return fmean(values) if values else None


def _latency_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    for quality in [{"vertical": "all", **overall}, *[dict(row) for row in per_vertical]]:
        vertical = str(quality["vertical"])
        latency = latency_by_vertical.get(vertical, {})
        rows.append(
            {
                "vertical": vertical,
                "row_count": quality.get("row_count"),
                "json_valid_rate": quality.get("json_valid_rate"),
                "generation_contract_valid_rate": quality.get("generation_contract_valid_rate"),
                "evidence_match_rate": quality.get("evidence_match_rate"),
                "grounded_rate": quality.get("grounded_rate"),
                "safety_violation_count": quality.get("safety_violation_count"),
                "truncation_count": quality.get("truncation_count"),
                "truncation_rate": quality.get("truncation_rate"),
                "mean_ttft_ms": latency.get("mean_ttft_ms"),
                "p50_ttft_ms": latency.get("p50_ttft_ms"),
                "p95_ttft_ms": latency.get("p95_ttft_ms"),
                "p99_ttft_ms": latency.get("p99_ttft_ms"),
                "mean_tpot_ms": latency.get("mean_tpot_ms"),
                "mean_itl_p50_ms": latency.get("mean_itl_p50_ms"),
                "mean_itl_p95_ms": latency.get("mean_itl_p95_ms"),
                "mean_itl_p99_ms": latency.get("mean_itl_p99_ms"),
                "mean_e2e_latency_ms": latency.get("mean_e2e_latency_ms"),
                "p50_e2e_latency_ms": latency.get("p50_e2e_latency_ms"),
                "p95_e2e_latency_ms": latency.get("p95_e2e_latency_ms"),
                "p99_e2e_latency_ms": latency.get("p99_e2e_latency_ms"),
                "mean_total_tokens_per_second": latency.get("mean_total_tokens_per_second"),
                "quality_gate_status": gate["status"] if vertical == "all" else "",
            }
        )
    write_csv_rows(ROOT / Path(path), rows)


def _metric_delta(baseline: Any, candidate: Any) -> dict[str, float | None]:
    if baseline in (None, "") or candidate in (None, ""):
        return {"baseline": None, "candidate": None, "absolute_delta": None}
    base = float(baseline)
    cand = float(candidate)
    return {"baseline": base, "candidate": cand, "absolute_delta": cand - base}


def _csv_union(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _comparison_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name in ("quality_deltas", "latency_throughput_deltas"):
        group = cast(dict[str, dict[str, Any]], comparison[group_name])
        for metric, values in group.items():
            rows.append({"metric_group": group_name, "metric": metric, **values})
    return rows


def _load_csv_all_row(path: str | Path) -> dict[str, Any]:
    rows = load_result_rows(ROOT / Path(path))
    for row in rows:
        if row.get("vertical") == "all":
            return row
    return {}


def _build_b5_vs_b6_comparison(
    *,
    b5_report_path: str | Path,
    b5_latency_path: str | Path,
    b6_summary: dict[str, Any],
    b6_latency: dict[str, Any],
    b6_prompt_count: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    b5_report = _read_json(b5_report_path) if (ROOT / Path(b5_report_path)).exists() else {}
    b5_summary = cast(dict[str, Any], b5_report.get("summary") or {})
    b5_latency = (
        _load_csv_all_row(b5_latency_path) if (ROOT / Path(b5_latency_path)).exists() else {}
    )
    quality_metrics = (
        "json_valid_rate",
        "generation_contract_valid_rate",
        "evidence_match_rate",
        "grounded_rate",
        "safety_violation_count",
        "truncation_rate",
    )
    latency_metrics = (
        "mean_ttft_ms",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_p50_ms",
        "mean_itl_p95_ms",
        "mean_itl_p99_ms",
        "mean_e2e_latency_ms",
        "p50_e2e_latency_ms",
        "p95_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_total_tokens_per_second",
    )
    return {
        "baseline": "B5_full_frozen_100",
        "candidate": "B6_500_quality_gate",
        "prompt_matched": False,
        "baseline_prompt_count": b5_summary.get("row_count"),
        "candidate_prompt_count": b6_prompt_count,
        "comparison_scope": (
            "B5 is a frozen 100-prompt gate; B6 is a larger balanced 500-prompt gate."
        ),
        "quality_deltas": {
            metric: _metric_delta(b5_summary.get(metric), b6_summary.get(metric))
            for metric in quality_metrics
        },
        "latency_throughput_deltas": {
            metric: _metric_delta(b5_latency.get(metric), b6_latency.get(metric))
            for metric in latency_metrics
        },
        "quality_gate": gate,
    }


def _write_manifest(
    *,
    path: str | Path,
    runner_input_path: str | Path,
    output_path: str | Path,
    telemetry_path: str | Path,
    telemetry_summary_path: str | Path,
    start_time: str,
    end_time: str | None,
    status: str,
    error_count: int,
    command: list[str],
) -> None:
    manifest = RunManifest(
        run_id=RUN_ID,
        timestamp_utc=end_time or start_time,
        backend="vllm",
        model_alias=MODEL_ALIAS,
        model_id=MODEL_ID,
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(runner_input_path),
        output_path=str(output_path),
        max_records=TOTAL_PROMPTS,
        git_commit=current_git_commit(ROOT),
        command=sanitized_command(command),
        status=status,
        start_time=start_time,
        end_time=end_time,
        error_count=error_count,
        telemetry_path=str(telemetry_path),
        telemetry_summary_path=str(telemetry_summary_path),
    )
    write_run_manifest(manifest, ROOT / Path(path))


def run_b6(args: argparse.Namespace) -> dict[str, Any]:
    """Run B6 after enforcing the offline preflight gate."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        raise RuntimeError(f"{MODEL_ALIAS} resolved to unexpected model {model.model_id}")
    preflight = _read_json(args.preflight_report)
    if not preflight_inference_allowed(preflight):
        raise RuntimeError("B6 inference is blocked because context preflight failed")
    items = _read_runner_items(args.runner_input)
    if len(items) != TOTAL_PROMPTS:
        raise RuntimeError(f"B6 requires exactly {TOTAL_PROMPTS} rows")
    counts = {
        vertical: sum(item.metadata.get("vertical") == vertical for item in items)
        for vertical in VERTICALS
    }
    if any(count != PROMPTS_PER_VERTICAL for count in counts.values()):
        raise RuntimeError(f"B6 vertical balance is invalid: {counts}")
    if any(item.metadata.get("context_alignment_status") != "all" for item in items):
        raise RuntimeError("B6 runner input contains unresolved context alignment")
    if any(item.metadata.get("b5_planning_active") != "true" for item in items):
        raise RuntimeError("B6 runner input does not have B5 planning active")
    if args.dry_run:
        return {
            "status": "dry_run",
            "record_count": len(items),
            "vertical_counts": counts,
            "max_new_tokens": MAX_NEW_TOKENS,
            "concurrency": 1,
            "inference_allowed": True,
        }

    start_time = utc_now()
    _write_manifest(
        path=args.manifest,
        runner_input_path=args.runner_input,
        output_path=args.output,
        telemetry_path=args.gpu_telemetry_csv,
        telemetry_summary_path=args.gpu_telemetry_summary,
        start_time=start_time,
        end_time=None,
        status="running",
        error_count=0,
        command=sys.argv,
    )
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

    thread = threading.Thread(target=collect_telemetry, name="b6-gpu-telemetry", daemon=True)
    thread.start()
    wall_start = time.perf_counter()
    try:
        result_rows = _run_items(
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
        output_path=ROOT / Path(args.output),
        eval_report_path=ROOT / Path(args.eval_report),
        eval_summary_path=ROOT / Path(args.eval_summary),
        block="B6",
        experiment="vllm_1_5b_500_context_aligned_b5_repairs",
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    per_vertical = build_per_vertical_quality(
        evaluation_rows,
        result_rows,
        verticals=VERTICALS,
    )
    latency_rows = _latency_rows(result_rows)
    write_csv_rows(ROOT / Path(args.latency_summary), latency_rows)
    write_gpu_telemetry_csv(ROOT / Path(args.gpu_telemetry_csv), telemetry_samples)
    write_gpu_telemetry_summary(
        ROOT / Path(args.gpu_telemetry_summary),
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    telemetry = summarize_gpu_telemetry(
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    gate = classify_b6_quality_gate(
        summary=eval_summary,
        per_vertical_quality=per_vertical,
    )
    _write_quality_summary(
        path=args.eval_summary,
        overall=eval_summary,
        per_vertical=per_vertical,
        latency_rows=latency_rows,
        gate=gate,
    )
    comparison = _build_b5_vs_b6_comparison(
        b5_report_path=args.b5_report,
        b5_latency_path=args.b5_latency_summary,
        b6_summary=eval_summary,
        b6_latency=latency_rows[0],
        b6_prompt_count=len(result_rows),
        gate=gate,
    )
    write_json(ROOT / Path(args.comparison_json), comparison)
    _csv_union(args.comparison_csv, _comparison_rows(comparison))
    runtime_report = build_runtime_projection_report(
        measured_prompt_count=len(result_rows),
        measured_wall_seconds=wall_seconds,
        runpod_profiles=_read_yaml(args.runpod_profiles),
    )
    write_runtime_projection_artifacts(
        report=runtime_report,
        report_path=ROOT / Path(args.runtime_report),
        summary_path=ROOT / Path(args.runtime_summary),
    )
    retry_trigger_counts = Counter(
        trigger
        for row in result_rows
        for trigger in cast(list[str], row.get("retry_triggers") or [])
    )
    quality_report = {
        **eval_report,
        "status": gate["status"],
        "quality_gate": gate,
        "per_vertical_quality": per_vertical,
        "latency_summary": latency_rows[0],
        "gpu_telemetry_summary": telemetry,
        "wall_seconds": wall_seconds,
        "telemetry_errors": telemetry_errors,
        "request_success_count": sum(bool(row.get("success")) for row in result_rows),
        "request_failure_count": sum(not bool(row.get("success")) for row in result_rows),
        "retry_attempt_count": sum(int(row.get("retry_attempt_count") or 0) for row in result_rows),
        "retry_trigger_counts": dict(sorted(retry_trigger_counts.items())),
        "lexical_guard_count": sum(bool(row.get("lexical_guard_applied")) for row in result_rows),
        "initial_safety_violation_count": sum(
            bool(row.get("initial_safety_violation")) for row in result_rows
        ),
        "final_safety_violation_count": eval_summary["safety_violation_count"],
        "context_alignment_preflight": preflight["summary_rows"],
        "b5_vs_b6_comparison_path": args.comparison_json,
        "runtime_projection_report_path": args.runtime_report,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "model_inference_triggered": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_new_tokens_justification": (
            "B5 passed the frozen 100-prompt quality gate at 160 after B3 found "
            "128-token truncation failures."
        ),
        "concurrency": 1,
    }
    write_json(ROOT / Path(args.eval_report), quality_report)
    error_count = sum(not bool(row.get("success")) for row in result_rows)
    _write_manifest(
        path=args.manifest,
        runner_input_path=args.runner_input,
        output_path=args.output,
        telemetry_path=args.gpu_telemetry_csv,
        telemetry_summary_path=args.gpu_telemetry_summary,
        start_time=start_time,
        end_time=end_time,
        status="completed",
        error_count=error_count,
        command=sys.argv,
    )
    return {
        "status": gate["status"],
        "server_readiness": readiness.to_dict(),
        "row_count": len(result_rows),
        "success_count": sum(bool(row.get("success")) for row in result_rows),
        "quality_gate": gate,
        "evaluation_summary": eval_summary,
        "per_vertical_quality": per_vertical,
        "latency_summary": latency_rows[0],
        "gpu_telemetry_summary": telemetry,
        "wall_seconds": wall_seconds,
        "report": args.eval_report,
        "summary": args.eval_summary,
        "comparison": args.comparison_json,
        "runtime_projection": args.runtime_report,
    }


def main() -> int:
    """Run the B6 CLI."""

    args = build_parser().parse_args()
    try:
        result = run_b6(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B6 vLLM 500-prompt gate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
