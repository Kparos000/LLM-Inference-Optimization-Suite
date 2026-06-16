"""Run B6R1 Research AI truncation and contract repair."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import load_gold_records  # noqa: E402
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
    write_json,
)

from inference_bench.b1_quality import build_per_vertical_quality  # noqa: E402
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
from inference_bench.research_ai_contract_repair import (  # noqa: E402
    RESEARCH_AI_BUDGET_STRATEGY,
    RESEARCH_AI_CONCISE_STRATEGY,
    apply_research_ai_strategy,
    build_failure_audit_report,
    build_research_ai_replay_rows,
    classify_b6r1_full_gate,
    read_jsonl,
    select_research_ai_strategy,
    summarize_research_ai_strategy,
    targeted_strategy_passes,
    write_csv,
    write_failure_audit_artifacts,
    write_jsonl,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.safety_generation_repair import decide_targeted_retry  # noqa: E402
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.streaming_metrics import (  # noqa: E402
    StreamingMetrics,
    request_streaming_chat_completion,
)

MODEL_ALIAS = "model2_1_5b"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
B6_RUNNER_INPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
B6_RESULTS = "results/raw/b6_vllm_1_5b_500_results.jsonl"
B6_REPORT = "results/processed/b6_vllm_1_5b_500_eval_report.json"
REPLAY_INPUT = "data/generated/phase4/b6r1_research_ai_failed_replay_input.jsonl"
AUDIT_REPORT = "results/processed/b6r1_research_ai_failure_audit_report.json"
AUDIT_SUMMARY = "results/processed/b6r1_research_ai_failure_audit_summary.csv"
TARGETED_RAW = "results/raw/b6r1_research_ai_targeted_replay_results.jsonl"
TARGETED_REPORT = "results/processed/b6r1_research_ai_targeted_replay_report.json"
TARGETED_SUMMARY = "results/processed/b6r1_research_ai_targeted_replay_summary.csv"
STRATEGY_COMPARISON = "results/processed/b6r1_research_ai_strategy_comparison.json"
FULL_RAW = "results/raw/b6r1_vllm_1_5b_500_repaired_results.jsonl"
FULL_REPORT = "results/processed/b6r1_vllm_1_5b_500_repaired_eval_report.json"
FULL_SUMMARY = "results/processed/b6r1_vllm_1_5b_500_repaired_eval_summary.csv"
B6_COMPARISON = "results/processed/b6_vs_b6r1_comparison.json"
FULL_GPU_CSV = "results/processed/b6r1_vllm_1_5b_500_gpu_telemetry.csv"
FULL_GPU_SUMMARY = "results/processed/b6r1_vllm_1_5b_500_gpu_telemetry_summary.json"
TARGETED_MANIFEST = "results/raw/b6r1_research_ai_targeted_replay_manifest.json"
FULL_MANIFEST = "results/raw/b6r1_vllm_1_5b_500_repaired_manifest.json"

STRATEGIES = (RESEARCH_AI_CONCISE_STRATEGY, RESEARCH_AI_BUDGET_STRATEGY)


def build_parser() -> argparse.ArgumentParser:
    """Build B6R1 CLI."""

    parser = argparse.ArgumentParser(description="Repair B6 Research AI contract failures.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--telemetry-ssh-host", default=None)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--telemetry-duration-seconds", type=float, default=7200.0)
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-full-rerun", action="store_true")
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads((ROOT / Path(path)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _read_runner_items(path: str | Path) -> list[WorkloadItem]:
    return [WorkloadItem(**row) for row in read_jsonl(ROOT / Path(path))]


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


def _result_row(
    *,
    item: WorkloadItem,
    metrics: StreamingMetrics,
    run_id: str,
    optimization: str,
    strategy: str,
) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    return {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r1_research_ai_contract_repair",
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "backend_type": "self_hosted_gpu",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "concurrency": 1,
        "model_alias": MODEL_ALIAS,
        "model_name": MODEL_ID,
        "optimization": optimization,
        "b6r1_strategy": strategy,
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


def _failure_row(
    *,
    item: WorkloadItem,
    exc: Exception,
    elapsed_ms: float,
    run_id: str,
    optimization: str,
    strategy: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r1_research_ai_contract_repair",
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "backend_type": "self_hosted_gpu",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "concurrency": 1,
        "model_alias": MODEL_ALIAS,
        "model_name": MODEL_ID,
        "optimization": optimization,
        "b6r1_strategy": strategy,
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
        "parse_error_type": "request_failed",
        "parse_repair_applied": False,
        "truncation_detected": False,
        "answer": "",
        "evidence_ids": [],
        "citations": [],
        "confidence": None,
        "insufficient_evidence": None,
        "citation_notes": "",
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
    }


def _safe_item_for_strategy(
    item: WorkloadItem,
    *,
    strategy: str,
) -> tuple[WorkloadItem, int]:
    if item.metadata.get("vertical") == "research_ai":
        repaired, max_tokens = apply_research_ai_strategy(item, strategy=strategy)
    else:
        repaired, max_tokens = item, 160
    expected = json.loads(repaired.metadata.get("gold_evidence_ids", "[]"))
    leaked = [
        evidence_id
        for evidence_id in expected
        if str(evidence_id).lower() in repaired.prompt.lower()
    ]
    if leaked:
        raise RuntimeError(f"Canonical evidence ID leakage for {item.prompt_id}")
    return repaired, max_tokens


def _run_items(
    *,
    items: list[WorkloadItem],
    strategy: str,
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    output_path: str | Path,
    run_id: str,
    optimization: str,
) -> list[dict[str, Any]]:
    output = ROOT / Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    route = f"{base_url.rstrip('/')}/chat/completions"
    rows: list[dict[str, Any]] = []
    for item in items:
        item, max_tokens = _safe_item_for_strategy(item, strategy=strategy)
        started = time.perf_counter()
        try:
            metrics = request_streaming_chat_completion(
                api_key=api_key,
                model_id=MODEL_ID,
                prompt=item.prompt,
                max_new_tokens=max_tokens,
                api_route=route,
                timeout_seconds=timeout_seconds,
            )
            row = _result_row(
                item=item,
                metrics=metrics,
                run_id=run_id,
                optimization=optimization,
                strategy=strategy,
            )
            evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
            row["initial_json_validity"] = evaluation.get("json_validity")
            row["initial_contract_validity"] = evaluation.get("generation_contract_valid")
            row["initial_evidence_match"] = evaluation.get("evidence_match")
            row["initial_groundedness"] = evaluation.get("groundedness")
            row["initial_safety_violation"] = evaluation.get("safety_violation")
            row["initial_safety_violation_terms"] = evaluation.get("safety_violation_terms")
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
                retry_metrics = request_streaming_chat_completion(
                    api_key=api_key,
                    model_id=MODEL_ID,
                    prompt=_repair_prompt(
                        row=row,
                        trigger=decision.trigger,
                        missing_labels=decision.missing_labels,
                    ),
                    max_new_tokens=max_tokens,
                    api_route=route,
                    timeout_seconds=timeout_seconds,
                )
                row = _merge_retry(
                    current=row,
                    metrics=retry_metrics,
                    trigger=decision.trigger,
                )
                evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
                if bool(evaluation.get("safety_violation")):
                    row = _apply_lexical_guard(row=row, evaluation=evaluation)
                    evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
            row["final_evaluation_snapshot"] = evaluation
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(
                item=item,
                exc=exc,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                run_id=run_id,
                optimization=optimization,
                strategy=strategy,
            )
        rows.append(row)
        with output.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def _write_manifest(
    *,
    path: str | Path,
    run_id: str,
    input_path: str,
    output_path: str,
    max_records: int,
    start_time: str,
    end_time: str,
    status: str,
    error_count: int,
    command: list[str],
    telemetry_path: str | None = None,
    telemetry_summary_path: str | None = None,
) -> None:
    manifest = RunManifest(
        run_id=run_id,
        timestamp_utc=end_time,
        backend="vllm",
        model_alias=MODEL_ALIAS,
        model_id=MODEL_ID,
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=input_path,
        output_path=output_path,
        max_records=max_records,
        git_commit=current_git_commit(ROOT),
        command=sanitized_command(command),
        status=status,
        start_time=start_time,
        end_time=end_time,
        error_count=error_count,
        telemetry_path=telemetry_path,
        telemetry_summary_path=telemetry_summary_path,
    )
    write_run_manifest(manifest, ROOT / Path(path))


def _build_replay_input() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner_rows = read_jsonl(ROOT / B6_RUNNER_INPUT)
    result_rows = read_jsonl(ROOT / B6_RESULTS)
    b6_report = _read_json(B6_REPORT)
    replay_rows = build_research_ai_replay_rows(
        runner_rows=runner_rows,
        result_rows=result_rows,
        evaluation_rows=cast(list[dict[str, Any]], b6_report["evaluation_rows"]),
    )
    write_jsonl(ROOT / REPLAY_INPUT, replay_rows)
    audit = build_failure_audit_report(replay_rows)
    write_failure_audit_artifacts(
        report=audit,
        report_path=ROOT / AUDIT_REPORT,
        summary_path=ROOT / AUDIT_SUMMARY,
    )
    return replay_rows, audit


def _replay_items(replay_rows: list[dict[str, Any]]) -> list[WorkloadItem]:
    return [WorkloadItem(**cast(dict[str, Any], row["runner_input"])) for row in replay_rows]


def _evaluate_strategy(
    *,
    strategy: str,
    items: list[WorkloadItem],
    gold_by_prompt: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = f"results/raw/b6r1_research_ai_{strategy}_results.jsonl"
    report_path = f"results/processed/b6r1_research_ai_{strategy}_report.json"
    summary_path = f"results/processed/b6r1_research_ai_{strategy}_summary.csv"
    run_id = f"b6r1-research-ai-{strategy}"
    started = utc_now()
    wall_start = time.perf_counter()
    rows = _run_items(
        items=items,
        strategy=strategy,
        gold_by_prompt=gold_by_prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        output_path=output_path,
        run_id=run_id,
        optimization=f"b6r1_{strategy}",
    )
    wall_seconds = time.perf_counter() - wall_start
    ended = utc_now()
    eval_report, _eval_summary = evaluate_result_rows(
        result_rows=rows,
        output_path=ROOT / output_path,
        eval_report_path=ROOT / report_path,
        eval_summary_path=ROOT / summary_path,
        block="B6R1",
        experiment=strategy,
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    strategy_summary = summarize_research_ai_strategy(
        strategy=strategy,
        evaluation_rows=evaluation_rows,
        result_rows=rows,
    )
    strategy_summary["wall_seconds"] = wall_seconds
    strategy_summary["output_path"] = output_path
    strategy_summary["report_path"] = report_path
    strategy_summary["summary_path"] = summary_path
    write_json(
        ROOT / report_path,
        {
            **eval_report,
            "strategy_summary": strategy_summary,
            "wall_seconds": wall_seconds,
            "targeted_strategy_passed": targeted_strategy_passes(strategy_summary),
        },
    )
    _write_manifest(
        path=f"results/raw/b6r1_research_ai_{strategy}_manifest.json",
        run_id=run_id,
        input_path=REPLAY_INPUT,
        output_path=output_path,
        max_records=len(items),
        start_time=started,
        end_time=ended,
        status="completed",
        error_count=sum(not bool(row.get("success")) for row in rows),
        command=sys.argv,
    )
    return {"summary": strategy_summary, "rows": rows, "evaluation_rows": evaluation_rows}


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return sum(values) / len(values) if values else None


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


def _comparison(
    *,
    b6_summary: dict[str, Any],
    b6_research_ai: dict[str, Any],
    b6r1_summary: dict[str, Any],
    b6r1_research_ai: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    metrics = (
        "json_valid_rate",
        "generation_contract_valid_rate",
        "evidence_match_rate",
        "grounded_rate",
        "safety_violation_count",
        "truncation_rate",
    )

    def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return {
            metric: {
                "before": before.get(metric),
                "after": after.get(metric),
                "absolute_delta": float(after.get(metric) or 0) - float(before.get(metric) or 0),
            }
            for metric in metrics
        }

    return {
        "baseline": "B6_500",
        "candidate": "B6R1_500_repaired",
        "overall_deltas": delta(b6_summary, b6r1_summary),
        "research_ai_deltas": delta(b6_research_ai, b6r1_research_ai),
        "quality_gate": gate,
    }


def _run_full_rerun(
    *,
    selected_strategy: str,
    gold_by_prompt: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    items = _read_runner_items(B6_RUNNER_INPUT)
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

    thread = threading.Thread(
        target=collect_telemetry,
        name="b6r1-gpu-telemetry",
        daemon=True,
    )
    thread.start()
    started = utc_now()
    wall_start = time.perf_counter()
    try:
        rows = _run_items(
            items=items,
            strategy=selected_strategy,
            gold_by_prompt=gold_by_prompt,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            output_path=FULL_RAW,
            run_id="b6r1-vllm-1-5b-500-repaired",
            optimization=f"b6r1_{selected_strategy}_full_500",
        )
    finally:
        wall_seconds = time.perf_counter() - wall_start
        stop_event.set()
        thread.join(timeout=max(5.0, args.telemetry_interval_seconds + 3.0))
    ended = utc_now()
    eval_report, eval_summary = evaluate_result_rows(
        result_rows=rows,
        output_path=ROOT / FULL_RAW,
        eval_report_path=ROOT / FULL_REPORT,
        eval_summary_path=ROOT / FULL_SUMMARY,
        block="B6R1",
        experiment=f"{selected_strategy}_full_500",
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    per_vertical = build_per_vertical_quality(evaluation_rows, rows, verticals=VERTICALS)
    gate = classify_b6r1_full_gate(summary=eval_summary, per_vertical_quality=per_vertical)
    latency = _latency_rows(rows)
    write_csv(ROOT / FULL_SUMMARY, [{"vertical": "all", **eval_summary}, *per_vertical])
    write_gpu_telemetry_csv(ROOT / FULL_GPU_CSV, telemetry_samples)
    write_gpu_telemetry_summary(
        ROOT / FULL_GPU_SUMMARY,
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    telemetry = summarize_gpu_telemetry(
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    b6 = _read_json(B6_REPORT)
    b6_research_ai = next(
        row for row in b6["per_vertical_quality"] if row["vertical"] == "research_ai"
    )
    b6r1_research_ai = next(row for row in per_vertical if row["vertical"] == "research_ai")
    comparison = _comparison(
        b6_summary=cast(dict[str, Any], b6["summary"]),
        b6_research_ai=b6_research_ai,
        b6r1_summary=eval_summary,
        b6r1_research_ai=cast(dict[str, Any], b6r1_research_ai),
        gate=gate,
    )
    write_json(ROOT / B6_COMPARISON, comparison)
    write_json(
        ROOT / FULL_REPORT,
        {
            **eval_report,
            "status": gate["status"],
            "quality_gate": gate,
            "selected_strategy": selected_strategy,
            "per_vertical_quality": per_vertical,
            "latency_summary": latency[0],
            "gpu_telemetry_summary": telemetry,
            "telemetry_errors": telemetry_errors,
            "wall_seconds": wall_seconds,
            "b6_vs_b6r1_comparison_path": B6_COMPARISON,
            "evaluator_modified": False,
            "gold_data_modified": False,
            "promoted_retrieval_modified": False,
            "concurrency": 1,
        },
    )
    _write_manifest(
        path=FULL_MANIFEST,
        run_id="b6r1-vllm-1-5b-500-repaired",
        input_path=B6_RUNNER_INPUT,
        output_path=FULL_RAW,
        max_records=len(rows),
        start_time=started,
        end_time=ended,
        status="completed",
        error_count=sum(not bool(row.get("success")) for row in rows),
        command=sys.argv,
        telemetry_path=FULL_GPU_CSV,
        telemetry_summary_path=FULL_GPU_SUMMARY,
    )
    return {
        "summary": eval_summary,
        "per_vertical_quality": per_vertical,
        "quality_gate": gate,
        "latency_summary": latency[0],
        "gpu_telemetry_summary": telemetry,
        "comparison": comparison,
        "wall_seconds": wall_seconds,
    }


def run_b6r1(args: argparse.Namespace) -> dict[str, Any]:
    """Build the B6R1 replay set and optionally run targeted/full repair."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        raise RuntimeError(f"{MODEL_ALIAS} resolved to unexpected model {model.model_id}")
    replay_rows, audit = _build_replay_input()
    replay_items = _replay_items(replay_rows)
    if args.skip_inference:
        return {
            "status": "audit_only",
            "targeted_replay_row_count": len(replay_rows),
            "audit_report": AUDIT_REPORT,
        }
    check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    strategy_results = {
        strategy: _evaluate_strategy(
            strategy=strategy,
            items=replay_items,
            gold_by_prompt=gold_by_prompt,
            args=args,
        )
        for strategy in STRATEGIES
    }
    summaries = [result["summary"] for result in strategy_results.values()]
    selection = select_research_ai_strategy(summaries)
    selected_strategy = selection["selected_strategy"]
    targeted_status = "B6R1_TARGETED_PASSED" if selected_strategy else "B6R1_BLOCKED"
    recommendation = (
        "Run the frozen 500-row B6R1 rerun with the selected Research AI repair strategy."
        if selected_strategy
        else (
            "Do not run 1,000 prompts, concurrency, SGLang, mm4, or RunPod. "
            "Escalate Research AI to a stronger model or a dedicated mm4 Research AI "
            "path in a separate controlled block."
        )
    )
    combined_rows: list[dict[str, Any]] = []
    for result in strategy_results.values():
        for row in cast(list[dict[str, Any]], result["rows"]):
            combined_rows.append(row)
    write_jsonl(ROOT / TARGETED_RAW, combined_rows)
    comparison = {
        "block": "B6R1",
        "status": targeted_status,
        "targeted_replay_row_count": len(replay_rows),
        "strategy_summaries": summaries,
        "selection": selection,
        "recommended_next_block": recommendation,
        "audit_report": AUDIT_REPORT,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
    }
    write_json(ROOT / STRATEGY_COMPARISON, comparison)
    write_json(
        ROOT / TARGETED_REPORT,
        {
            "block": "B6R1",
            "targeted_replay_row_count": len(replay_rows),
            "strategy_summaries": summaries,
            "strategy_comparison_path": STRATEGY_COMPARISON,
            "selected_strategy": selected_strategy,
            "full_500_rerun_triggered": bool(selected_strategy and not args.skip_full_rerun),
            "status": targeted_status,
            "recommended_next_block": recommendation,
            "audit": audit,
        },
    )
    write_csv(ROOT / TARGETED_SUMMARY, summaries)
    _write_manifest(
        path=TARGETED_MANIFEST,
        run_id="b6r1-research-ai-targeted-replay",
        input_path=REPLAY_INPUT,
        output_path=TARGETED_RAW,
        max_records=len(combined_rows),
        start_time=utc_now(),
        end_time=utc_now(),
        status="completed",
        error_count=sum(not bool(row.get("success")) for row in combined_rows),
        command=sys.argv,
    )
    full_result: dict[str, Any] | None = None
    if selected_strategy and not args.skip_full_rerun:
        full_result = _run_full_rerun(
            selected_strategy=str(selected_strategy),
            gold_by_prompt=gold_by_prompt,
            args=args,
        )
    elif selected_strategy is None:
        write_json(
            ROOT / B6_COMPARISON,
            {
                "baseline": "B6_500",
                "candidate": "B6R1_not_run_full_500",
                "status": "B6R1_BLOCKED",
                "reason": "No targeted Research AI strategy passed.",
                "recommended_next_block": recommendation,
            },
        )
    return {
        "status": targeted_status,
        "targeted_replay_row_count": len(replay_rows),
        "strategy_summaries": summaries,
        "selection": selection,
        "recommended_next_block": recommendation,
        "full_500_rerun_triggered": full_result is not None,
        "full_result": full_result,
    }


def main() -> int:
    """Run the B6R1 CLI."""

    args = build_parser().parse_args()
    try:
        result = run_b6r1(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B6R1 Research AI repair failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
