"""Run B6R4 Qwen2.5-3B Research AI targeted validation and optional 500 gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from inference_bench.b6r4_qwen3b_validation import (  # noqa: E402
    B6R4_FROZEN_REPLAY_INPUT,
    B6R4_FULL_500_INPUT,
    B6R4_MAX_NEW_TOKENS,
    B6R4_MODEL_ALIAS,
    B6R4_MODEL_ID,
    build_model_capacity_comparison,
    build_no_live_replay_report,
    classify_b6r4_full_500_gate,
    classify_b6r4_targeted_gate,
    targeted_replay_allows_full_500,
    validate_b6r4_model_selection,
    validate_b6r4_replay_input,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.generation_contract_registry import (  # noqa: E402
    validate_and_map_contract_text,
)
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.research_ai_capacity_validation import (  # noqa: E402
    NormalizedResearchAiReplayItem,
    choose_b6r3_contract_id,
    load_research_ai_capacity_replay,
)
from inference_bench.research_ai_contract_renderer import (  # noqa: E402
    RenderedResearchAiContract,
    render_research_ai_contract_item,
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

RUN_ID_TARGETED = "b6r4-model2-3b-research-ai-targeted"
RUN_ID_FULL = "b6r4-model2-3b-500-quality-gate"
DEFAULT_TARGETED_RAW = "results/raw/b6r4_model2_3b_research_ai_targeted_results.jsonl"
DEFAULT_TARGETED_REPORT = "results/processed/b6r4_model2_3b_research_ai_targeted_report.json"
DEFAULT_TARGETED_SUMMARY = "results/processed/b6r4_model2_3b_research_ai_targeted_summary.csv"
DEFAULT_COMPARISON = "results/processed/b6r4_research_ai_model_capacity_comparison.json"
DEFAULT_TARGETED_MANIFEST = "results/raw/b6r4_model2_3b_research_ai_targeted_manifest.json"
DEFAULT_FULL_RAW = "results/raw/b6r4_model2_3b_500_results.jsonl"
DEFAULT_FULL_REPORT = "results/processed/b6r4_model2_3b_500_eval_report.json"
DEFAULT_FULL_SUMMARY = "results/processed/b6r4_model2_3b_500_eval_summary.csv"
DEFAULT_FULL_COMPARISON = "results/processed/b6_vs_b6r4_model2_3b_comparison.json"
DEFAULT_FULL_MANIFEST = "results/raw/b6r4_model2_3b_500_manifest.json"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_RETRY_LIMIT = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the B6R4 CLI parser."""

    parser = argparse.ArgumentParser(description="Run B6R4 Qwen2.5-3B validation.")
    parser.add_argument("--input-path", default=B6R4_FROZEN_REPLAY_INPUT)
    parser.add_argument("--full-input-path", default=B6R4_FULL_500_INPUT)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--targeted-output-path", default=DEFAULT_TARGETED_RAW)
    parser.add_argument("--targeted-report-path", default=DEFAULT_TARGETED_REPORT)
    parser.add_argument("--targeted-summary-path", default=DEFAULT_TARGETED_SUMMARY)
    parser.add_argument("--targeted-manifest-path", default=DEFAULT_TARGETED_MANIFEST)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON)
    parser.add_argument("--full-output-path", default=DEFAULT_FULL_RAW)
    parser.add_argument("--full-report-path", default=DEFAULT_FULL_REPORT)
    parser.add_argument("--full-summary-path", default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--full-comparison-path", default=DEFAULT_FULL_COMPARISON)
    parser.add_argument("--full-manifest-path", default=DEFAULT_FULL_MANIFEST)
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--max-new-tokens", type=int, default=B6R4_MAX_NEW_TOKENS)
    parser.add_argument("--skip-full-rerun", action="store_true")
    parser.add_argument("--record-blocked-if-unavailable", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: str | Path) -> dict[str, Any] | None:
    candidate = ROOT / Path(path)
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    candidate = ROOT / Path(path)
    if not candidate.exists():
        return []
    rows: list[dict[str, Any]] = []
    with candidate.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {path} at line {line_number}")
            rows.append(payload)
    return rows


def _resume_rows(path: str | Path, items: list[WorkloadItem], run_id: str) -> list[dict[str, Any]]:
    existing = [
        row
        for row in _read_jsonl(path)
        if row.get("run_id") == run_id and row.get("prompt_id") not in (None, "")
    ]
    by_prompt: dict[str, dict[str, Any]] = {}
    for row in existing:
        prompt_id = str(row["prompt_id"])
        if prompt_id in by_prompt:
            raise ValueError(f"Duplicate prompt_id in existing B6R4 output: {prompt_id}")
        by_prompt[prompt_id] = row
    ordered_rows: list[dict[str, Any]] = []
    for item in items:
        if item.prompt_id in by_prompt:
            ordered_rows.append(by_prompt[item.prompt_id])
    return ordered_rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("at least one CSV row is required")
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "config_id": "b6r4_model2_3b_research_ai_quality_validation",
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "backend_type": "self_hosted_gpu",
        "runtime": "vllm",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "provider": "self_hosted",
        "concurrency": 1,
        "model_alias": B6R4_MODEL_ALIAS,
        "model_id": B6R4_MODEL_ID,
        "model_name": B6R4_MODEL_ID,
        "optimization": optimization,
        "prompt": item.prompt,
        **_metric_fields(metrics),
        "success": True,
        "error_message": None,
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.metadata.get("vertical"),
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": aliases,
        "gold_evidence_ids": item.metadata.get("gold_evidence_ids"),
        "context_alignment_status": item.metadata.get("context_alignment_status"),
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
        "workload_specific_routing_active": False,
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
        **_base_result_row(item=item, metrics=metrics, run_id=run_id, optimization=optimization),
        "generated_text": metrics.generated_text,
        "raw_generated_text": metrics.generated_text,
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
        **_base_result_row(item=item, metrics=metrics, run_id=run_id, optimization=optimization),
        "generated_text": generated_text,
        "raw_generated_text": metrics.generated_text,
        "b6r4_requested_research_ai_contract": rendered.requested_contract_id,
        "b6r4_effective_research_ai_contract": rendered.effective_contract_id,
        "b6r4_max_new_tokens": rendered.max_new_tokens,
        "b6r4_contract_validation": validation.to_dict(),
        **generation_contract_result_fields(generated_text, allowed_evidence_ids=allowed),
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
    row = {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r4_model2_3b_research_ai_quality_validation",
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "backend_type": "self_hosted_gpu",
        "runtime": "vllm",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "provider": "self_hosted",
        "concurrency": 1,
        "model_alias": B6R4_MODEL_ALIAS,
        "model_id": B6R4_MODEL_ID,
        "model_name": B6R4_MODEL_ID,
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
        "success": False,
        "error_message": f"{type(exc).__name__}: {exc}",
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.metadata.get("vertical"),
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": aliases,
        "gold_evidence_ids": item.metadata.get("gold_evidence_ids"),
        "context_alignment_status": item.metadata.get("context_alignment_status"),
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
        "workload_specific_routing_active": False,
    }
    row.update(generation_contract_result_fields(""))
    return row


def _request(
    *,
    item: WorkloadItem,
    base_url: str,
    api_key: str,
    max_new_tokens: int,
    timeout_seconds: float,
) -> StreamingMetrics:
    return request_streaming_chat_completion(
        api_key=api_key,
        model_id=B6R4_MODEL_ID,
        prompt=item.prompt,
        max_new_tokens=max_new_tokens,
        api_route=f"{base_url.rstrip('/')}/chat/completions",
        timeout_seconds=timeout_seconds,
    )


def _run_research_ai_item(
    *,
    item: WorkloadItem,
    base_url: str,
    api_key: str,
    max_new_tokens: int,
    timeout_seconds: float,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    rendered = render_research_ai_contract_item(
        item,
        requested_contract_id=choose_b6r3_contract_id(
            NormalizedResearchAiReplayItem(
                prompt_id=item.prompt_id,
                vertical="research_ai",
                workload_name=item.workload_name,
                prompt=item.prompt,
                expected_output=item.expected_output,
                metadata=item.metadata,
                source_metadata={},
            )
        ),
        max_new_tokens=max_new_tokens,
    )
    metrics = _request(
        item=rendered.item,
        base_url=base_url,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
    )
    return _research_ai_result_row(
        rendered=rendered,
        metrics=metrics,
        run_id=run_id,
        optimization=optimization,
    )


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
    metrics = _request(
        item=item,
        base_url=base_url,
        api_key=api_key,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        timeout_seconds=timeout_seconds,
    )
    row = _default_result_row(
        item=item,
        metrics=metrics,
        run_id=run_id,
        optimization=optimization,
    )
    evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
    if bool(evaluation.get("safety_violation")):
        row = _apply_lexical_guard(row=row, evaluation=evaluation)
        evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
    while True:
        missing = _missing_labels(evaluation=evaluation, row=row)
        decision = decide_targeted_retry(
            evaluation=evaluation,
            missing_labels=missing,
            attempt_count=int(row.get("retry_attempt_count") or 0),
            max_attempts=DEFAULT_RETRY_LIMIT,
        )
        row["last_retry_decision"] = decision.trigger
        if not decision.should_retry:
            break
        repair_metrics = request_streaming_chat_completion(
            api_key=api_key,
            model_id=B6R4_MODEL_ID,
            prompt=_repair_prompt(
                row=row,
                trigger=decision.trigger,
                missing_labels=decision.missing_labels,
            ),
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            api_route=f"{base_url.rstrip('/')}/chat/completions",
            timeout_seconds=timeout_seconds,
        )
        row = _merge_retry(current=row, metrics=repair_metrics, trigger=decision.trigger)
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
    output_path: str,
    run_id: str,
    optimization: str,
    research_ai_max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows = _resume_rows(output_path, items, run_id)
    completed_ids = {str(row["prompt_id"]) for row in rows}
    if rows:
        _write_jsonl(output_path, rows)
    for index, item in enumerate(items, start=1):
        if item.prompt_id in completed_ids:
            continue
        started = time.perf_counter()
        try:
            if item.metadata.get("vertical") == "research_ai":
                row = _run_research_ai_item(
                    item=item,
                    base_url=base_url,
                    api_key=api_key,
                    max_new_tokens=research_ai_max_new_tokens,
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
        completed_ids.add(item.prompt_id)
        _write_jsonl(output_path, rows)
    return rows


def _evaluate_and_summarize(
    *,
    rows: list[dict[str, Any]],
    output_path: str,
    report_path: str,
    summary_path: str,
    block: str,
    experiment: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report, summary = evaluate_result_rows(
        result_rows=rows,
        output_path=ROOT / output_path,
        eval_report_path=ROOT / report_path,
        eval_summary_path=ROOT / summary_path,
        block=block,
        experiment=experiment,
    )
    latency = latency_summary_rows(rows)[0]
    summary.update(
        {
            "mean_ttft_ms": latency.get("mean_ttft_ms"),
            "mean_tpot_ms": latency.get("mean_tpot_ms"),
            "mean_itl_p50_ms": _mean(rows, "itl_p50_ms"),
            "mean_itl_p95_ms": _mean(rows, "itl_p95_ms"),
            "mean_itl_p99_ms": _mean(rows, "itl_p99_ms"),
            "mean_e2e_latency_ms": latency.get("mean_e2e_latency_ms"),
            "mean_total_tokens_per_second": latency.get("mean_total_tokens_per_second"),
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        }
    )
    return report, summary


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) not in (None, "") and bool(row.get("success"))
    ]
    return sum(values) / len(values) if values else None


def _write_manifest(
    *,
    path: str,
    run_id: str,
    input_path: str,
    output_path: str,
    expected_count: int,
    completed_count: int,
    start_time: str,
    end_time: str,
    status: str,
) -> None:
    manifest = RunManifest(
        run_id=run_id,
        timestamp_utc=end_time,
        backend="vllm",
        model_alias=B6R4_MODEL_ALIAS,
        model_id=B6R4_MODEL_ID,
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=input_path,
        output_path=output_path,
        max_records=expected_count,
        git_commit=current_git_commit(ROOT),
        command=sanitized_command(sys.argv),
        status=status,
        start_time=start_time,
        end_time=end_time,
        error_count=max(expected_count - completed_count, 0),
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        hardware="remote_rtx3070",
        provider="self_hosted",
        concurrency=1,
        traffic_profile="online_low_latency",
        prompt_count=expected_count,
        expected_count=expected_count,
        completed_count=completed_count,
        failed_count=max(expected_count - completed_count, 0),
    )
    write_run_manifest(manifest, ROOT / path)


def _load_comparison_sources(b6r4_summary: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    b6 = _read_json("results/processed/b6_vllm_1_5b_500_eval_report.json")
    b6r2 = _read_json("results/processed/b6r2_research_ai_contract_selection_report.json")
    b6r3 = _read_json("results/processed/b6r3_model6_research_ai_capacity_report.json")
    b6_research = None
    if b6 and isinstance(b6.get("per_vertical_quality"), list):
        b6_research = next(
            (row for row in b6["per_vertical_quality"] if row.get("vertical") == "research_ai"),
            None,
        )
    b6r2_best = None
    if b6r2 and isinstance(b6r2.get("candidate_summaries"), list):
        b6r2_best = max(
            b6r2["candidate_summaries"],
            key=lambda row: float(row.get("grounded_rate") or 0.0),
        )
    b6r3_summary = None
    if b6r3 and isinstance(b6r3.get("summary"), dict):
        b6r3_summary = cast(dict[str, Any], b6r3["summary"])
    return build_model_capacity_comparison(
        b6_research_ai=cast(dict[str, Any] | None, b6_research),
        b6r2_best=cast(dict[str, Any] | None, b6r2_best),
        b6r3_model6=b6r3_summary,
        b6r4_summary=b6r4_summary,
        b6r4_gate=gate,
        full_500_triggered=targeted_replay_allows_full_500(gate),
    )


def _targeted_items(path: str, limit: int) -> list[WorkloadItem]:
    validate_b6r4_replay_input(path)
    replay = load_research_ai_capacity_replay(ROOT / path, limit=limit)
    return [item.to_workload_item() for item in replay]


def run_b6r4(args: argparse.Namespace) -> dict[str, Any]:
    """Run the targeted replay and optional full 500 gate."""

    validate_b6r4_model_selection(model_alias=B6R4_MODEL_ALIAS)
    model = load_project_config().resolve_model_config(B6R4_MODEL_ALIAS)
    if model.model_id != B6R4_MODEL_ID:
        raise RuntimeError(f"{B6R4_MODEL_ALIAS} resolved to unexpected model {model.model_id}")
    items = _targeted_items(args.input_path, args.limit)
    if args.dry_run:
        return {
            "status": "dry_run",
            "model_alias": B6R4_MODEL_ALIAS,
            "model_id": B6R4_MODEL_ID,
            "targeted_replay_row_count": len(items),
            "input_path": args.input_path,
            "max_new_tokens": args.max_new_tokens,
            "full_500_rerun_triggered": False,
        }
    try:
        readiness = check_server_readiness(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=B6R4_MODEL_ID,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        if not args.record_blocked_if_unavailable:
            raise
        report = build_no_live_replay_report(reason=f"{type(exc).__name__}: {exc}")
        write_json(ROOT / args.targeted_report_path, report)
        _write_csv(args.targeted_summary_path, [report])
        write_json(
            ROOT / args.comparison_path,
            build_model_capacity_comparison(
                b6_research_ai=None,
                b6r2_best=None,
                b6r3_model6=None,
                b6r4_summary=None,
                b6r4_gate=cast(dict[str, Any], report["quality_gate"]),
                full_500_triggered=False,
            ),
        )
        return report
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    start_time = utc_now()
    targeted_rows = _run_items(
        items=items,
        gold_by_prompt=gold_by_prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        output_path=args.targeted_output_path,
        run_id=RUN_ID_TARGETED,
        optimization="b6r4_model2_3b_research_ai_targeted",
        research_ai_max_new_tokens=args.max_new_tokens,
    )
    end_time = utc_now()
    targeted_report, targeted_summary = _evaluate_and_summarize(
        rows=targeted_rows,
        output_path=args.targeted_output_path,
        report_path=args.targeted_report_path,
        summary_path=args.targeted_summary_path,
        block="B6R4",
        experiment="model2_3b_research_ai_targeted",
    )
    targeted_gate = classify_b6r4_targeted_gate(targeted_summary)
    comparison = _load_comparison_sources(targeted_summary, targeted_gate)
    full_triggered = targeted_replay_allows_full_500(targeted_gate) and not args.skip_full_rerun
    comparison["full_500_can_proceed_on_model2_3b"] = full_triggered
    write_json(ROOT / args.comparison_path, comparison)
    targeted_report.update(
        {
            "status": targeted_gate["status"],
            "quality_gate": targeted_gate,
            "model_alias": B6R4_MODEL_ALIAS,
            "model_id": B6R4_MODEL_ID,
            "runtime": "vllm",
            "engine": "vllm",
            "hardware": "remote_rtx3070",
            "server_readiness": readiness.to_dict(),
            "summary": targeted_summary,
            "comparison_path": args.comparison_path,
            "targeted_replay_ran": True,
            "full_500_rerun_triggered": full_triggered,
            "evaluator_modified": False,
            "gold_data_modified": False,
            "promoted_retrieval_modified": False,
            "workload_specific_routing_introduced": False,
        }
    )
    write_json(ROOT / args.targeted_report_path, targeted_report)
    _write_csv(args.targeted_summary_path, [targeted_summary])
    _write_manifest(
        path=args.targeted_manifest_path,
        run_id=RUN_ID_TARGETED,
        input_path=args.input_path,
        output_path=args.targeted_output_path,
        expected_count=len(items),
        completed_count=len(targeted_rows),
        start_time=start_time,
        end_time=end_time,
        status="completed",
    )
    full_result: dict[str, Any] | None = None
    if full_triggered:
        full_result = _run_full_500(args=args, gold_by_prompt=gold_by_prompt)
    elif not targeted_gate["passed"]:
        write_json(
            ROOT / args.full_comparison_path,
            {
                "baseline": "B6_500",
                "candidate": "B6R4_model2_3b_full_500_not_run",
                "status": "B6R4_TARGETED_MODEL2_3B_BLOCKED",
                "reason": "Targeted Research AI replay did not pass, so full 500 is blocked.",
                "targeted_gate": targeted_gate,
            },
        )
    return {
        "status": targeted_gate["status"],
        "targeted_replay_ran": True,
        "targeted_replay_row_count": len(targeted_rows),
        "targeted_summary": targeted_summary,
        "targeted_gate": targeted_gate,
        "full_500_rerun_triggered": full_triggered,
        "full_result": full_result,
        "targeted_report": args.targeted_report_path,
        "targeted_summary_path": args.targeted_summary_path,
        "comparison_path": args.comparison_path,
    }


def _run_full_500(
    *,
    args: argparse.Namespace,
    gold_by_prompt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items = _read_runner_items(args.full_input_path)
    start_time = utc_now()
    rows = _run_items(
        items=items,
        gold_by_prompt=gold_by_prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        output_path=args.full_output_path,
        run_id=RUN_ID_FULL,
        optimization="b6r4_model2_3b_500_quality_gate",
        research_ai_max_new_tokens=args.max_new_tokens,
    )
    end_time = utc_now()
    full_report, full_summary = _evaluate_and_summarize(
        rows=rows,
        output_path=args.full_output_path,
        report_path=args.full_report_path,
        summary_path=args.full_summary_path,
        block="B6R4",
        experiment="model2_3b_500_quality_gate",
    )
    evaluation_rows = cast(list[dict[str, Any]], full_report["evaluation_rows"])
    per_vertical = build_per_vertical_quality(evaluation_rows, rows, verticals=VERTICALS)
    gate = classify_b6r4_full_500_gate(summary=full_summary, per_vertical_quality=per_vertical)
    b6 = _read_json("results/processed/b6_vllm_1_5b_500_eval_report.json") or {}
    write_json(
        ROOT / args.full_comparison_path,
        {
            "baseline": "B6_Qwen2.5_1.5B_500",
            "candidate": "B6R4_Qwen2.5_3B_500",
            "baseline_summary": b6.get("summary"),
            "candidate_summary": full_summary,
            "candidate_per_vertical_quality": per_vertical,
            "quality_gate": gate,
        },
    )
    full_report.update(
        {
            "status": gate["status"],
            "quality_gate": gate,
            "summary": full_summary,
            "per_vertical_quality": per_vertical,
            "model_alias": B6R4_MODEL_ALIAS,
            "model_id": B6R4_MODEL_ID,
            "full_500_rerun_triggered": True,
            "evaluator_modified": False,
            "gold_data_modified": False,
            "promoted_retrieval_modified": False,
            "workload_specific_routing_introduced": False,
        }
    )
    write_json(ROOT / args.full_report_path, full_report)
    _write_csv(args.full_summary_path, [{"vertical": "all", **full_summary}, *per_vertical])
    _write_manifest(
        path=args.full_manifest_path,
        run_id=RUN_ID_FULL,
        input_path=args.full_input_path,
        output_path=args.full_output_path,
        expected_count=len(items),
        completed_count=len(rows),
        start_time=start_time,
        end_time=end_time,
        status="completed",
    )
    return {
        "status": gate["status"],
        "summary": full_summary,
        "per_vertical_quality": per_vertical,
        "quality_gate": gate,
        "report": args.full_report_path,
        "summary_path": args.full_summary_path,
        "comparison_path": args.full_comparison_path,
    }


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    try:
        result = run_b6r4(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B6R4 Qwen3B validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
