"""Run B6R2 Research AI contract selection and optional frozen 500 rerun."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
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
from inference_bench.generation_contract_registry import (  # noqa: E402
    B6R2_CONTRACT_MAX_NEW_TOKENS,
    RESEARCH_AI_CONTRACT_IDS,
    validate_and_map_contract_text,
)
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    sample_gpu_telemetry,
    summarize_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.research_ai_contract_renderer import (  # noqa: E402
    RenderedResearchAiContract,
    render_research_ai_contract_item,
    render_research_ai_retry_prompt,
)
from inference_bench.research_ai_contract_repair import (  # noqa: E402
    build_failure_audit_report,
    build_research_ai_replay_rows,
    read_jsonl,
    write_csv,
    write_failure_audit_artifacts,
    write_jsonl,
)
from inference_bench.research_ai_contract_selection import (  # noqa: E402
    classify_b6r2_full_gate,
    select_research_ai_contract,
    summarize_contract_candidate,
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
DEFAULT_MAX_NEW_TOKENS = 160
TARGETED_RETRY_LIMIT = 2

B6_RUNNER_INPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
B6_RESULTS = "results/raw/b6_vllm_1_5b_500_results.jsonl"
B6_REPORT = "results/processed/b6_vllm_1_5b_500_eval_report.json"

REPLAY_INPUT = "data/generated/phase4/b6r2_research_ai_failed_replay_input.jsonl"
AUDIT_REPORT = "results/processed/b6r2_research_ai_failure_audit_report.json"
AUDIT_SUMMARY = "results/processed/b6r2_research_ai_failure_audit_summary.csv"
TARGETED_RAW = "results/raw/b6r2_research_ai_contract_selection_results.jsonl"
TARGETED_REPORT = "results/processed/b6r2_research_ai_contract_selection_report.json"
TARGETED_SUMMARY = "results/processed/b6r2_research_ai_contract_selection_summary.csv"
TARGETED_MANIFEST = "results/raw/b6r2_research_ai_contract_selection_manifest.json"

FULL_RAW = "results/raw/b6r2_vllm_1_5b_500_results.jsonl"
FULL_MANIFEST = "results/raw/b6r2_vllm_1_5b_500_manifest.json"
FULL_REPORT = "results/processed/b6r2_vllm_1_5b_500_eval_report.json"
FULL_SUMMARY = "results/processed/b6r2_vllm_1_5b_500_eval_summary.csv"
FULL_GPU_CSV = "results/processed/b6r2_vllm_1_5b_500_gpu_telemetry.csv"
FULL_GPU_SUMMARY = "results/processed/b6r2_vllm_1_5b_500_gpu_telemetry_summary.json"
B6_COMPARISON = "results/processed/b6_vs_b6r2_comparison.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the B6R2 CLI."""

    parser = argparse.ArgumentParser(
        description="Select a Research AI vertical contract and optionally rerun B6."
    )
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


def _base_result_row(
    *,
    item: WorkloadItem,
    metrics: StreamingMetrics,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    return {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r2_research_ai_vertical_contract",
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
        "prompt": item.prompt,
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
    }


def _default_result_row(
    *,
    item: WorkloadItem,
    metrics: StreamingMetrics,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    return {
        **_base_result_row(
            item=item,
            metrics=metrics,
            run_id=run_id,
            optimization=optimization,
        ),
        "generated_text": metrics.generated_text,
        **generation_contract_result_fields(
            metrics.generated_text,
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        ),
    }


def _research_ai_result_row(
    *,
    rendered: RenderedResearchAiContract,
    metrics: StreamingMetrics,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    item = rendered.item
    aliases = item.metadata.get("citation_id_aliases")
    allowed = allowed_evidence_ids_from_aliases(aliases)
    validation = validate_and_map_contract_text(
        text=metrics.generated_text,
        contract_id=rendered.requested_contract_id,
        allowed_evidence_ids=allowed,
        prompt_text=item.prompt,
        metadata=item.metadata,
    )
    generated_text = validation.common_text or metrics.generated_text
    return {
        **_base_result_row(
            item=item,
            metrics=metrics,
            run_id=run_id,
            optimization=optimization,
        ),
        "generated_text": generated_text,
        "raw_generated_text": metrics.generated_text,
        "b6r2_requested_research_ai_contract": rendered.requested_contract_id,
        "b6r2_effective_research_ai_contract": rendered.effective_contract_id,
        "b6r2_max_new_tokens": rendered.max_new_tokens,
        "b6r2_contract_validation": validation.to_dict(),
        "b6r2_contract_json_valid": validation.json_valid,
        "b6r2_contract_valid": validation.contract_valid,
        "b6r2_contract_error": validation.error,
        **generation_contract_result_fields(
            generated_text,
            allowed_evidence_ids=allowed,
        ),
    }


def _failure_row(
    *,
    item: WorkloadItem,
    exc: Exception,
    elapsed_ms: float,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    return {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r2_research_ai_vertical_contract",
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
        "prompt": item.prompt,
        "generated_text": "",
        "raw_generated_text": "",
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
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
    }


def _merge_research_ai_retry(
    *,
    current: dict[str, Any],
    retry_row: dict[str, Any],
    metrics: StreamingMetrics,
    trigger: str,
) -> dict[str, Any]:
    current_latency = float(current.get("end_to_end_latency_ms") or 0.0)
    total_latency = current_latency + metrics.e2e_latency_ms
    total_input = int(current.get("input_tokens") or 0) + metrics.input_tokens
    total_output = int(current.get("output_tokens") or 0) + metrics.output_tokens
    total_tokens = total_input + total_output
    retry_count = int(current.get("retry_attempt_count") or 0) + 1
    triggers = [*cast(list[str], current.get("retry_triggers") or []), trigger]
    retry_history = [
        *cast(list[dict[str, Any]], current.get("retry_history") or []),
        {
            "attempt": retry_count,
            "trigger": trigger,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "ttft_ms": metrics.ttft_ms,
            "tpot_ms": metrics.tpot_ms,
            "e2e_latency_ms": metrics.e2e_latency_ms,
        },
    ]
    replace_fields = {
        key: retry_row.get(key)
        for key in (
            "generated_text",
            "raw_generated_text",
            "b6r2_contract_validation",
            "b6r2_contract_json_valid",
            "b6r2_contract_valid",
            "b6r2_contract_error",
            "generation_contract_valid",
            "generation_contract_error",
            "generation_contract_missing_fields",
            "parse_error_type",
            "parse_repair_applied",
            "truncation_detected",
            "answer",
            "evidence_ids",
            "citations",
            "confidence",
            "insufficient_evidence",
            "citation_notes",
        )
    }
    return {
        **current,
        **replace_fields,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "content_chunk_count": int(current.get("content_chunk_count") or 0)
        + metrics.content_chunk_count,
        "end_to_end_latency_ms": total_latency,
        "throughput_tokens_per_second": _throughput(total_tokens, total_latency),
        "retry_attempt_count": retry_count,
        "retry_triggers": triggers,
        "retry_history": retry_history,
        "last_retry_trigger": trigger,
        "last_retry_ttft_ms": metrics.ttft_ms,
        "last_retry_itl_p50_ms": metrics.itl_p50_ms,
        "last_retry_itl_p95_ms": metrics.itl_p95_ms,
        "last_retry_itl_p99_ms": metrics.itl_p99_ms,
        "last_retry_tpot_ms": metrics.tpot_ms,
        "last_retry_e2e_latency_ms": metrics.e2e_latency_ms,
    }


def _research_ai_repair_prompt(
    *,
    rendered: RenderedResearchAiContract,
    row: dict[str, Any],
    trigger: str,
    missing_labels: tuple[str, ...],
) -> str:
    previous = (
        "[redacted because the previous output violated a safety rule]"
        if trigger == "safety_violation"
        else str(row.get("raw_generated_text") or row.get("generated_text") or "")
    )
    return render_research_ai_retry_prompt(
        rendered_item=rendered.item,
        requested_contract_id=rendered.requested_contract_id,
        effective_contract_id=rendered.effective_contract_id,
        previous_output=previous,
        issue=trigger,
        missing_labels=missing_labels,
    )


def _run_research_ai_item(
    *,
    item: WorkloadItem,
    contract_id: str,
    max_new_tokens: int,
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    rendered = render_research_ai_contract_item(
        item,
        requested_contract_id=contract_id,
        max_new_tokens=max_new_tokens,
    )
    route = f"{base_url.rstrip('/')}/chat/completions"
    metrics = request_streaming_chat_completion(
        api_key=api_key,
        model_id=MODEL_ID,
        prompt=rendered.item.prompt,
        max_new_tokens=max_new_tokens,
        api_route=route,
        timeout_seconds=timeout_seconds,
    )
    row = _research_ai_result_row(
        rendered=rendered,
        metrics=metrics,
        run_id=run_id,
        optimization=optimization,
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
            max_attempts=TARGETED_RETRY_LIMIT,
        )
        row["last_retry_decision"] = decision.trigger
        if not decision.should_retry:
            break
        retry_prompt = _research_ai_repair_prompt(
            rendered=rendered,
            row=row,
            trigger=decision.trigger,
            missing_labels=decision.missing_labels,
        )
        retry_metrics = request_streaming_chat_completion(
            api_key=api_key,
            model_id=MODEL_ID,
            prompt=retry_prompt,
            max_new_tokens=max_new_tokens,
            api_route=route,
            timeout_seconds=timeout_seconds,
        )
        retry_row = _research_ai_result_row(
            rendered=rendered,
            metrics=retry_metrics,
            run_id=run_id,
            optimization=optimization,
        )
        row = _merge_research_ai_retry(
            current=row,
            retry_row=retry_row,
            metrics=retry_metrics,
            trigger=decision.trigger,
        )
        evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
        if bool(evaluation.get("safety_violation")):
            row = _apply_lexical_guard(row=row, evaluation=evaluation)
            evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
    row["final_evaluation_snapshot"] = evaluation
    return row


def _run_default_item(
    *,
    item: WorkloadItem,
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    route = f"{base_url.rstrip('/')}/chat/completions"
    metrics = request_streaming_chat_completion(
        api_key=api_key,
        model_id=MODEL_ID,
        prompt=item.prompt,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        api_route=route,
        timeout_seconds=timeout_seconds,
    )
    row = _default_result_row(
        item=item,
        metrics=metrics,
        run_id=run_id,
        optimization=optimization,
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
        repair_metrics = request_streaming_chat_completion(
            api_key=api_key,
            model_id=MODEL_ID,
            prompt=_repair_prompt(
                row=row,
                trigger=decision.trigger,
                missing_labels=decision.missing_labels,
            ),
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
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
    return row


def _run_items(
    *,
    items: list[WorkloadItem],
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    output_path: str | Path,
    run_id: str,
    optimization: str,
    research_ai_contract_id: str,
    research_ai_max_new_tokens: int,
) -> list[dict[str, Any]]:
    output = ROOT / Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        started = time.perf_counter()
        try:
            if item.metadata.get("vertical") == "research_ai":
                row = _run_research_ai_item(
                    item=item,
                    contract_id=research_ai_contract_id,
                    max_new_tokens=research_ai_max_new_tokens,
                    gold_by_prompt=gold_by_prompt,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                    run_id=run_id,
                    optimization=optimization,
                )
            else:
                row = _run_default_item(
                    item=item,
                    gold_by_prompt=gold_by_prompt,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                    run_id=run_id,
                    optimization=optimization,
                )
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(
                item=item,
                exc=exc,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                run_id=run_id,
                optimization=optimization,
            )
        row["sequence_index"] = index
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


def _evaluate_contract_candidate(
    *,
    contract_id: str,
    max_new_tokens: int,
    items: list[WorkloadItem],
    gold_by_prompt: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_id = f"b6r2-research-ai-{contract_id}-{max_new_tokens}"
    output_path = f"results/raw/b6r2_research_ai_{contract_id}_{max_new_tokens}_results.jsonl"
    report_path = f"results/processed/b6r2_research_ai_{contract_id}_{max_new_tokens}_report.json"
    summary_path = f"results/processed/b6r2_research_ai_{contract_id}_{max_new_tokens}_summary.csv"
    wall_start = time.perf_counter()
    rows = _run_items(
        items=items,
        gold_by_prompt=gold_by_prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        output_path=output_path,
        run_id=run_id,
        optimization=f"b6r2_{contract_id}_{max_new_tokens}",
        research_ai_contract_id=contract_id,
        research_ai_max_new_tokens=max_new_tokens,
    )
    wall_seconds = time.perf_counter() - wall_start
    eval_report, _eval_summary = evaluate_result_rows(
        result_rows=rows,
        output_path=ROOT / output_path,
        eval_report_path=ROOT / report_path,
        eval_summary_path=ROOT / summary_path,
        block="B6R2",
        experiment=f"{contract_id}_{max_new_tokens}",
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    summary = summarize_contract_candidate(
        contract_id=contract_id,
        max_new_tokens=max_new_tokens,
        evaluation_rows=evaluation_rows,
        result_rows=rows,
    )
    summary["wall_seconds"] = wall_seconds
    summary["output_path"] = output_path
    summary["report_path"] = report_path
    summary["summary_path"] = summary_path
    write_json(
        ROOT / report_path,
        {
            **eval_report,
            "candidate_summary": summary,
            "wall_seconds": wall_seconds,
            "targeted_contract_passed": summary["passed"],
        },
    )
    write_csv(ROOT / summary_path, [summary])
    return {"summary": summary, "rows": rows, "evaluation_rows": evaluation_rows}


def _latency_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = latency_summary_rows(result_rows)
    for row in rows:
        vertical = str(row["vertical"])
        group = (
            result_rows
            if vertical == "all"
            else [result for result in result_rows if result.get("vertical") == vertical]
        )
        values = [float(result["itl_p50_ms"]) for result in group if result.get("itl_p50_ms")]
        row["mean_itl_p50_ms"] = sum(values) / len(values) if values else None
    return rows


def _comparison(
    *,
    b6_summary: dict[str, Any],
    b6_research_ai: dict[str, Any],
    b6r2_summary: dict[str, Any],
    b6r2_research_ai: dict[str, Any],
    gate: dict[str, Any],
    selected_contract: str,
    selected_max_new_tokens: int,
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
        "candidate": "B6R2_500_vertical_research_ai_contract",
        "selected_research_ai_contract": selected_contract,
        "selected_research_ai_max_new_tokens": selected_max_new_tokens,
        "overall_deltas": delta(b6_summary, b6r2_summary),
        "research_ai_deltas": delta(b6_research_ai, b6r2_research_ai),
        "quality_gate": gate,
    }


def _run_full_rerun(
    *,
    selected_contract: str,
    selected_max_new_tokens: int,
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
        name="b6r2-gpu-telemetry",
        daemon=True,
    )
    thread.start()
    started = utc_now()
    wall_start = time.perf_counter()
    try:
        rows = _run_items(
            items=items,
            gold_by_prompt=gold_by_prompt,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            output_path=FULL_RAW,
            run_id="b6r2-vllm-1-5b-500",
            optimization=f"b6r2_{selected_contract}_{selected_max_new_tokens}_full_500",
            research_ai_contract_id=selected_contract,
            research_ai_max_new_tokens=selected_max_new_tokens,
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
        block="B6R2",
        experiment=f"{selected_contract}_{selected_max_new_tokens}_full_500",
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    per_vertical = build_per_vertical_quality(evaluation_rows, rows, verticals=VERTICALS)
    gate = classify_b6r2_full_gate(summary=eval_summary, per_vertical_quality=per_vertical)
    latency = _latency_rows(rows)
    write_csv_rows(ROOT / FULL_SUMMARY, [{"vertical": "all", **eval_summary}, *per_vertical])
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
    b6r2_research_ai = next(row for row in per_vertical if row["vertical"] == "research_ai")
    comparison = _comparison(
        b6_summary=cast(dict[str, Any], b6["summary"]),
        b6_research_ai=cast(dict[str, Any], b6_research_ai),
        b6r2_summary=eval_summary,
        b6r2_research_ai=cast(dict[str, Any], b6r2_research_ai),
        gate=gate,
        selected_contract=selected_contract,
        selected_max_new_tokens=selected_max_new_tokens,
    )
    write_json(ROOT / B6_COMPARISON, comparison)
    retry_trigger_counts = Counter(
        trigger for row in rows for trigger in cast(list[str], row.get("retry_triggers") or [])
    )
    write_json(
        ROOT / FULL_REPORT,
        {
            **eval_report,
            "status": gate["status"],
            "quality_gate": gate,
            "selected_research_ai_contract": selected_contract,
            "selected_research_ai_max_new_tokens": selected_max_new_tokens,
            "per_vertical_quality": per_vertical,
            "latency_summary": latency[0],
            "gpu_telemetry_summary": telemetry,
            "telemetry_errors": telemetry_errors,
            "wall_seconds": wall_seconds,
            "retry_attempt_count": sum(int(row.get("retry_attempt_count") or 0) for row in rows),
            "retry_trigger_counts": dict(sorted(retry_trigger_counts.items())),
            "b6_vs_b6r2_comparison_path": B6_COMPARISON,
            "evaluator_modified": False,
            "gold_data_modified": False,
            "promoted_retrieval_modified": False,
            "concurrency": 1,
        },
    )
    _write_manifest(
        path=FULL_MANIFEST,
        run_id="b6r2-vllm-1-5b-500",
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


def run_b6r2(args: argparse.Namespace) -> dict[str, Any]:
    """Run B6R2 Research AI contract selection and optional full rerun."""

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
    candidate_results: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    for contract_id in RESEARCH_AI_CONTRACT_IDS:
        for max_new_tokens in B6R2_CONTRACT_MAX_NEW_TOKENS:
            result = _evaluate_contract_candidate(
                contract_id=contract_id,
                max_new_tokens=max_new_tokens,
                items=replay_items,
                gold_by_prompt=gold_by_prompt,
                args=args,
            )
            candidate_results.append(result)
            combined_rows.extend(cast(list[dict[str, Any]], result["rows"]))
    summaries = [cast(dict[str, Any], result["summary"]) for result in candidate_results]
    selection = select_research_ai_contract(summaries)
    selected_contract = selection.get("selected_contract_id")
    selected_max_tokens = selection.get("selected_max_new_tokens")
    targeted_status = "B6R2_TARGETED_PASSED" if selected_contract else "B6R2_BLOCKED"
    full_triggered = bool(selected_contract and selected_max_tokens and not args.skip_full_rerun)
    write_jsonl(ROOT / TARGETED_RAW, combined_rows)
    write_csv(ROOT / TARGETED_SUMMARY, summaries)
    write_json(
        ROOT / TARGETED_REPORT,
        {
            "block": "B6R2",
            "status": targeted_status,
            "targeted_replay_row_count": len(replay_rows),
            "candidate_summaries": summaries,
            "selection": selection,
            "selected_research_ai_contract": selected_contract,
            "selected_research_ai_max_new_tokens": selected_max_tokens,
            "full_500_rerun_triggered": full_triggered,
            "audit": audit,
            "evaluator_modified": False,
            "gold_data_modified": False,
            "promoted_retrieval_modified": False,
            "concurrency": 1,
        },
    )
    _write_manifest(
        path=TARGETED_MANIFEST,
        run_id="b6r2-research-ai-contract-selection",
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
    if full_triggered:
        full_result = _run_full_rerun(
            selected_contract=str(selected_contract),
            selected_max_new_tokens=int(selected_max_tokens),
            gold_by_prompt=gold_by_prompt,
            args=args,
        )
    elif selected_contract is None:
        write_json(
            ROOT / B6_COMPARISON,
            {
                "baseline": "B6_500",
                "candidate": "B6R2_not_run_full_500",
                "status": "B6R2_BLOCKED",
                "reason": "No targeted Research AI contract candidate passed.",
                "selection": selection,
            },
        )
    return {
        "status": targeted_status,
        "targeted_replay_row_count": len(replay_rows),
        "candidate_summaries": summaries,
        "selection": selection,
        "full_500_rerun_triggered": full_result is not None,
        "full_result": full_result,
        "targeted_report": TARGETED_REPORT,
        "targeted_summary": TARGETED_SUMMARY,
        "comparison": B6_COMPARISON,
    }


def main() -> int:
    """Run the B6R2 CLI."""

    args = build_parser().parse_args()
    try:
        result = run_b6r2(args)
    except Exception as exc:  # noqa: BLE001
        print(
            f"B6R2 Research AI contract selection failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
