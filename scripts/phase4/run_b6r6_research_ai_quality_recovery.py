"""Run B6R6 Research AI quality recovery for Qwen2.5-3B."""

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

from evaluate_generation_outputs import (  # noqa: E402
    build_summary_rows,
    load_gold_records,
    result_row_to_generated_answer,
)
from run_b5_targeted_generation_repair import (  # noqa: E402
    _apply_lexical_guard,
    _merge_retry,
    _missing_labels,
    _repair_prompt,
)
from run_openai_compatible_smoke import DEFAULT_API_KEY, check_server_readiness  # noqa: E402
from run_remote_vllm_smoke import latency_summary_rows, sanitized_command, write_json  # noqa: E402

from inference_bench.b1_quality import build_per_vertical_quality  # noqa: E402
from inference_bench.b6r5_finance_research_repair import (  # noqa: E402
    STRATEGY_EVIDENCE_PREPLAN,
    build_b6r4_vs_b6r5_comparison,
)
from inference_bench.b6r5_finance_research_repair import (  # noqa: E402
    apply_strategy_to_prompt as apply_b6r5_strategy_to_prompt,
)
from inference_bench.b6r6_research_ai_recovery import (  # noqa: E402
    B6R6_B6R4_EVAL_REPORT,
    B6R6_B6R4_RAW_RESULTS,
    B6R6_B6R5_TARGETED_REPORT,
    B6R6_FULL_500_INPUT,
    B6R6_MODEL_ALIAS,
    B6R6_MODEL_ID,
    B6R6_REPLAY_INPUT,
    B6R6_STRATEGIES,
    B6R6_VERTICAL,
    STRATEGY_A_ORIGINAL,
    STRATEGY_B_B6R2_BEST_CONTRACT,
    STRATEGY_D_ANSWER_SKELETON,
    apply_research_ai_strategy_prompt,
    build_failure_audit,
    build_research_ai_baseline_lock,
    build_research_ai_replay_rows,
    classify_b6r6_full_gate,
    classify_b6r6_targeted_strategy,
    finance_repair_candidate_passes,
    full_rerun_allowed,
    map_answer_skeleton_to_common_text,
    max_new_tokens_for_strategy,
    no_policy_mutation_flags,
    select_b6r6_strategy,
    summarize_strategy_rows,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.generation_contract_registry import (  # noqa: E402
    RESEARCH_AI_LIMITATIONS,
    validate_and_map_contract_text,
)
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.research_ai_capacity_validation import (  # noqa: E402
    NormalizedResearchAiReplayItem,
    choose_b6r3_contract_id,
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

RUN_ID_TARGETED = "b6r6-research-ai-targeted-replay"
RUN_ID_FULL = "b6r6-model2-3b-500-quality-gate"

DEFAULT_FAILURE_AUDIT_REPORT = "results/processed/b6r6_research_ai_failure_audit_report.json"
DEFAULT_FAILURE_AUDIT_SUMMARY = "results/processed/b6r6_research_ai_failure_audit_summary.csv"
DEFAULT_TARGETED_RAW = "results/raw/b6r6_research_ai_targeted_replay_results.jsonl"
DEFAULT_TARGETED_REPORT = "results/processed/b6r6_research_ai_targeted_replay_report.json"
DEFAULT_TARGETED_SUMMARY = "results/processed/b6r6_research_ai_targeted_replay_summary.csv"
DEFAULT_STRATEGY_COMPARISON = "results/processed/b6r6_strategy_comparison.json"
DEFAULT_FULL_RAW = "results/raw/b6r6_model2_3b_500_results.jsonl"
DEFAULT_FULL_REPORT = "results/processed/b6r6_model2_3b_500_eval_report.json"
DEFAULT_FULL_SUMMARY = "results/processed/b6r6_model2_3b_500_eval_summary.csv"
DEFAULT_FULL_COMPARISON = "results/processed/b6r5_vs_b6r6_comparison.json"
DEFAULT_TARGETED_MANIFEST = "results/raw/b6r6_research_ai_targeted_manifest.json"
DEFAULT_FULL_MANIFEST = "results/raw/b6r6_model2_3b_500_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the B6R6 CLI parser."""

    parser = argparse.ArgumentParser(description="Run B6R6 Research AI recovery.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--failure-replay-input", default=B6R6_REPLAY_INPUT)
    parser.add_argument("--full-input-path", default=B6R6_FULL_500_INPUT)
    parser.add_argument("--b6r4-raw-results", default=B6R6_B6R4_RAW_RESULTS)
    parser.add_argument("--b6r4-eval-report", default=B6R6_B6R4_EVAL_REPORT)
    parser.add_argument("--b6r5-targeted-report", default=B6R6_B6R5_TARGETED_REPORT)
    parser.add_argument("--failure-audit-report-path", default=DEFAULT_FAILURE_AUDIT_REPORT)
    parser.add_argument("--failure-audit-summary-path", default=DEFAULT_FAILURE_AUDIT_SUMMARY)
    parser.add_argument("--targeted-output-path", default=DEFAULT_TARGETED_RAW)
    parser.add_argument("--targeted-report-path", default=DEFAULT_TARGETED_REPORT)
    parser.add_argument("--targeted-summary-path", default=DEFAULT_TARGETED_SUMMARY)
    parser.add_argument("--strategy-comparison-path", default=DEFAULT_STRATEGY_COMPARISON)
    parser.add_argument("--targeted-manifest-path", default=DEFAULT_TARGETED_MANIFEST)
    parser.add_argument("--full-output-path", default=DEFAULT_FULL_RAW)
    parser.add_argument("--full-report-path", default=DEFAULT_FULL_REPORT)
    parser.add_argument("--full-summary-path", default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--full-comparison-path", default=DEFAULT_FULL_COMPARISON)
    parser.add_argument("--full-manifest-path", default=DEFAULT_FULL_MANIFEST)
    parser.add_argument("--skip-full-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads((ROOT / Path(path)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object: {path}"
        raise ValueError(msg)
    return payload


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
                msg = f"Expected JSON object in {path} at line {line_number}"
                raise ValueError(msg)
            rows.append(payload)
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        msg = "at least one CSV row is required"
        raise ValueError(msg)
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _workload_item(payload: dict[str, Any]) -> WorkloadItem:
    return WorkloadItem(
        prompt_id=str(payload["prompt_id"]),
        workload_name=str(payload["workload_name"]),
        prompt=str(payload["prompt"]),
        expected_output=str(payload["expected_output"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def _read_runner_items(path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def build_failure_set_and_audit(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Write B6R6 replay input, audit, and baseline lock."""

    raw_rows = _read_jsonl(args.b6r4_raw_results)
    b6r4_report = _read_json(args.b6r4_eval_report)
    runner_items = _read_runner_items(args.full_input_path)
    replay_rows = build_research_ai_replay_rows(
        raw_rows=raw_rows,
        evaluation_rows=cast(list[dict[str, Any]], b6r4_report["evaluation_rows"]),
        runner_items_by_prompt={str(row.get("prompt_id")): row for row in runner_items},
    )
    _write_jsonl(args.failure_replay_input, replay_rows)
    audit = build_failure_audit(replay_rows)
    baseline_lock = build_research_ai_baseline_lock(
        b6r4_report=b6r4_report,
        replay_rows=replay_rows,
    )
    audit["baseline_lock"] = baseline_lock
    write_json(ROOT / args.failure_audit_report_path, audit)
    _write_csv(
        args.failure_audit_summary_path,
        [
            {
                "prompt_id": example["prompt_id"],
                "primary_root_cause": example["primary_root_cause"],
                "root_causes": ";".join(example["root_causes"]),
                "required_evidence_labels": ";".join(example["required_evidence_labels"]),
                "original_evidence_labels": ";".join(example["original_evidence_labels"]),
            }
            for example in audit["examples"]
        ],
    )
    return replay_rows, audit, baseline_lock


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


def _request(
    *,
    prompt: str,
    base_url: str,
    api_key: str,
    max_new_tokens: int,
    timeout_seconds: float,
) -> StreamingMetrics:
    return request_streaming_chat_completion(
        api_key=api_key,
        model_id=B6R6_MODEL_ID,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        api_route=f"{base_url.rstrip('/')}/chat/completions",
        timeout_seconds=timeout_seconds,
    )


def _render_research_item(
    item: WorkloadItem,
    *,
    strategy_id: str,
) -> tuple[WorkloadItem, RenderedResearchAiContract | None, int]:
    max_new_tokens = max_new_tokens_for_strategy(strategy_id)
    if strategy_id == STRATEGY_D_ANSWER_SKELETON:
        prompt = _prompt_without_output_contract(item.prompt)
        labels = [
            label.strip()
            for label in str(item.metadata.get("b5_required_labels") or "").split(",")
            if label.strip()
        ]
        prompt = apply_research_ai_strategy_prompt(
            prompt=prompt,
            strategy_id=strategy_id,
            required_labels=labels,
        )
        return (
            WorkloadItem(
                prompt_id=item.prompt_id,
                workload_name=item.workload_name,
                prompt=prompt,
                expected_output=item.expected_output,
                metadata={
                    **item.metadata,
                    "b6r6_research_ai_strategy": strategy_id,
                    "b6r6_max_new_tokens": str(max_new_tokens),
                },
            ),
            None,
            max_new_tokens,
        )
    requested = (
        RESEARCH_AI_LIMITATIONS
        if strategy_id == STRATEGY_B_B6R2_BEST_CONTRACT
        else choose_b6r3_contract_id(
            NormalizedResearchAiReplayItem(
                prompt_id=item.prompt_id,
                vertical=B6R6_VERTICAL,
                workload_name=item.workload_name,
                prompt=item.prompt,
                expected_output=item.expected_output,
                metadata=item.metadata,
                source_metadata={},
            )
        )
    )
    render_budget = 320
    rendered = render_research_ai_contract_item(
        item,
        requested_contract_id=requested,
        max_new_tokens=render_budget,
    )
    labels = [
        label.strip()
        for label in str(item.metadata.get("b5_required_labels") or "").split(",")
        if label.strip()
    ]
    prompt = apply_research_ai_strategy_prompt(
        prompt=rendered.item.prompt,
        strategy_id=strategy_id,
        required_labels=labels,
    )
    prompt_item = WorkloadItem(
        prompt_id=rendered.item.prompt_id,
        workload_name=rendered.item.workload_name,
        prompt=prompt,
        expected_output=rendered.item.expected_output,
        metadata={
            **rendered.item.metadata,
            "b6r6_research_ai_strategy": strategy_id,
            "b6r6_max_new_tokens": str(max_new_tokens),
        },
    )
    return prompt_item, rendered, max_new_tokens


def _prompt_without_output_contract(prompt: str) -> str:
    marker = "\n\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.split(marker, maxsplit=1)[0].rstrip()
    marker = "\nOUTPUT CONTRACT:\n"
    if marker in prompt:
        return prompt.split(marker, maxsplit=1)[0].rstrip()
    return prompt.rstrip()


def _common_text_and_fields(
    *,
    item: WorkloadItem,
    text: str,
    rendered: RenderedResearchAiContract | None,
    strategy_id: str,
) -> tuple[str, dict[str, Any]]:
    aliases = item.metadata.get("citation_id_aliases")
    allowed = allowed_evidence_ids_from_aliases(aliases)
    if strategy_id == STRATEGY_D_ANSWER_SKELETON:
        common_text = map_answer_skeleton_to_common_text(text)
        return common_text, generation_contract_result_fields(
            common_text,
            allowed_evidence_ids=allowed,
        )
    if rendered is None:
        return text, generation_contract_result_fields(text, allowed_evidence_ids=allowed)
    validation = validate_and_map_contract_text(
        text=text,
        contract_id=rendered.requested_contract_id,
        allowed_evidence_ids=allowed,
        prompt_text=item.prompt,
        metadata=item.metadata,
    )
    common_text = validation.common_text or text
    fields = generation_contract_result_fields(common_text, allowed_evidence_ids=allowed)
    fields.update(
        {
            "b6r6_research_ai_contract_validation": validation.to_dict(),
            "b6r6_requested_research_ai_contract": rendered.requested_contract_id,
            "b6r6_effective_research_ai_contract": rendered.effective_contract_id,
        }
    )
    return common_text, fields


def _base_result_row(
    *,
    item: WorkloadItem,
    prompt_item: WorkloadItem,
    prompt: str,
    metrics: StreamingMetrics,
    run_id: str,
    strategy_id: str,
    optimization: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r6_research_ai_quality_recovery",
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "backend_type": "self_hosted_gpu",
        "runtime": "vllm",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "provider": "self_hosted",
        "concurrency": 1,
        "model_alias": B6R6_MODEL_ALIAS,
        "model_id": B6R6_MODEL_ID,
        "model_name": B6R6_MODEL_ID,
        "optimization": optimization,
        "b6r6_strategy_id": strategy_id,
        "prompt": prompt,
        **_metric_fields(metrics),
        "success": True,
        "error_message": None,
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.metadata.get("vertical"),
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": prompt_item.metadata.get("citation_id_aliases"),
        "gold_evidence_ids": item.metadata.get("gold_evidence_ids"),
        "context_alignment_status": item.metadata.get("context_alignment_status"),
        "retry_attempt_count": 0,
        "retry_triggers": [],
        "lexical_guard_applied": False,
        "workload_specific_routing_active": False,
        **no_policy_mutation_flags(),
    }


def _run_research_ai_item(
    *,
    item: WorkloadItem,
    strategy_id: str,
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    run_id: str,
    optimization: str,
    gold_by_prompt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prompt_item, rendered, max_new_tokens = _render_research_item(item, strategy_id=strategy_id)
    metrics = _request(
        prompt=prompt_item.prompt,
        base_url=base_url,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
    )
    generated_text, fields = _common_text_and_fields(
        item=prompt_item,
        text=metrics.generated_text,
        rendered=rendered,
        strategy_id=strategy_id,
    )
    row = _base_result_row(
        item=item,
        prompt_item=prompt_item,
        prompt=prompt_item.prompt,
        metrics=metrics,
        run_id=run_id,
        strategy_id=strategy_id,
        optimization=optimization,
    )
    row.update(
        {
            "generated_text": generated_text,
            "raw_generated_text": metrics.generated_text,
            **fields,
        }
    )
    evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
    if bool(evaluation.get("safety_violation")):
        row = _apply_lexical_guard(row=row, evaluation=evaluation)
    return row


def _run_default_item(
    *,
    item: WorkloadItem,
    strategy_id: str,
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    prompt = item.prompt
    if item.metadata.get("vertical") == "finance":
        labels = [
            label.strip()
            for label in str(item.metadata.get("b5_required_labels") or "").split(",")
            if label.strip()
        ]
        prompt = apply_b6r5_strategy_to_prompt(
            prompt=prompt,
            strategy_id=STRATEGY_EVIDENCE_PREPLAN,
            required_labels=labels,
            vertical="finance",
        )
        active_strategy = "b6r5_finance_evidence_selection_preplan"
    else:
        active_strategy = "baseline"
    metrics = _request(
        prompt=prompt,
        base_url=base_url,
        api_key=api_key,
        max_new_tokens=160,
        timeout_seconds=timeout_seconds,
    )
    row = _base_result_row(
        item=item,
        prompt_item=item,
        prompt=prompt,
        metrics=metrics,
        run_id=run_id,
        strategy_id=(
            strategy_id if item.metadata.get("vertical") == "research_ai" else active_strategy
        ),
        optimization=optimization,
    )
    fields = generation_contract_result_fields(
        metrics.generated_text,
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(
            item.metadata.get("citation_id_aliases")
        ),
    )
    row.update(
        {
            "generated_text": metrics.generated_text,
            "raw_generated_text": metrics.generated_text,
            **fields,
        }
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
            max_attempts=2,
        )
        row["last_retry_decision"] = decision.trigger
        if not decision.should_retry:
            break
        repair_metrics = request_streaming_chat_completion(
            api_key=api_key,
            model_id=B6R6_MODEL_ID,
            prompt=_repair_prompt(
                row=row,
                trigger=decision.trigger,
                missing_labels=decision.missing_labels,
            ),
            max_new_tokens=160,
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


def _control_rows_from_b6r4(
    *,
    replay_rows: list[dict[str, Any]],
    b6r4_raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_prompt = {str(row.get("prompt_id")): row for row in b6r4_raw_rows}
    rows: list[dict[str, Any]] = []
    for index, replay in enumerate(replay_rows, start=1):
        source = dict(by_prompt[str(replay["prompt_id"])])
        source.update(
            {
                "run_id": RUN_ID_TARGETED,
                "b6r6_strategy_id": STRATEGY_A_ORIGINAL,
                "optimization": "b6r6_b6r4_original_behavior_control",
                "sequence_index": index,
                **no_policy_mutation_flags(),
            }
        )
        rows.append(source)
    return rows


def _resume_rows(path: str, run_id: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in _read_jsonl(path)
        if row.get("run_id") == run_id and row.get("prompt_id") not in (None, "")
    ]
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("b6r6_strategy_id")), str(row.get("prompt_id")))
        if key in seen:
            msg = f"Duplicate B6R6 result row for {key}"
            raise ValueError(msg)
        seen.add(key)
    return rows


def _evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    results_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generated = [result_row_to_generated_answer(row) for row in rows]
    evaluation_rows = evaluate_generated_answers(
        generated,
        load_gold_records("data/scaleup_2000_full"),
    )
    summary = build_summary_rows(
        results_path=results_path,
        result_rows=rows,
        evaluation_rows=evaluation_rows,
    )[0]
    latency = latency_summary_rows(rows)[0]
    summary.update(
        {
            "mean_ttft_ms": latency.get("mean_ttft_ms"),
            "mean_tpot_ms": latency.get("mean_tpot_ms"),
            "mean_e2e_latency_ms": latency.get("mean_e2e_latency_ms"),
            "mean_total_tokens_per_second": latency.get("mean_total_tokens_per_second"),
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        }
    )
    return evaluation_rows, summary


def run_targeted_strategies(
    *,
    replay_rows: list[dict[str, Any]],
    baseline_lock: dict[str, Any],
    args: argparse.Namespace,
    gold_by_prompt: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run all B6R6 targeted Research AI strategies."""

    all_rows = _resume_rows(args.targeted_output_path, RUN_ID_TARGETED)
    completed = {(str(row.get("b6r6_strategy_id")), str(row.get("prompt_id"))) for row in all_rows}
    if not any(strategy == STRATEGY_A_ORIGINAL for strategy, _ in completed):
        control_rows = _control_rows_from_b6r4(
            replay_rows=replay_rows,
            b6r4_raw_rows=_read_jsonl(args.b6r4_raw_results),
        )
        all_rows.extend(control_rows)
        completed.update((STRATEGY_A_ORIGINAL, str(row["prompt_id"])) for row in control_rows)
        _write_jsonl(args.targeted_output_path, all_rows)
    for strategy_id in B6R6_STRATEGIES:
        if strategy_id == STRATEGY_A_ORIGINAL:
            continue
        for replay in replay_rows:
            item = _workload_item(replay)
            key = (strategy_id, item.prompt_id)
            if key in completed:
                continue
            started = time.perf_counter()
            try:
                row = _run_research_ai_item(
                    item=item,
                    strategy_id=strategy_id,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                    run_id=RUN_ID_TARGETED,
                    optimization=f"b6r6_{strategy_id}",
                    gold_by_prompt=gold_by_prompt,
                )
            except Exception as exc:  # noqa: BLE001
                row = _failure_row(
                    item=item,
                    prompt=item.prompt,
                    exc=exc,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    strategy_id=strategy_id,
                    run_id=RUN_ID_TARGETED,
                    optimization=f"b6r6_{strategy_id}",
                )
            row["sequence_index"] = len(all_rows) + 1
            all_rows.append(row)
            completed.add(key)
            _write_jsonl(args.targeted_output_path, all_rows)
    strategy_summaries: list[dict[str, Any]] = []
    strategy_reports: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for strategy_id in B6R6_STRATEGIES:
        rows = [row for row in all_rows if row.get("b6r6_strategy_id") == strategy_id]
        evaluation_rows, evaluator_summary = _evaluate_rows(
            rows=rows,
            results_path=args.targeted_output_path,
        )
        summary = summarize_strategy_rows(result_rows=rows, evaluation_rows=evaluation_rows)
        summary.update(
            {
                "strategy_id": strategy_id,
                "json_valid_rate": evaluator_summary["json_valid_rate"],
                "generation_contract_valid_rate": evaluator_summary[
                    "generation_contract_valid_rate"
                ],
                "safety_violation_count": evaluator_summary["safety_violation_count"],
                "truncation_rate": evaluator_summary["truncation_rate"],
            }
        )
        gate = classify_b6r6_targeted_strategy(summary=summary, baseline_lock=baseline_lock)
        strategy_summaries.append(summary)
        strategy_reports[strategy_id] = {
            "summary": summary,
            "quality_gate": gate,
            "evaluation_rows": evaluation_rows,
        }
        summary_rows.append(summary)
    selection = select_b6r6_strategy(
        strategy_summaries=strategy_summaries,
        baseline_lock=baseline_lock,
    )
    finance_report = _read_json(args.b6r5_targeted_report)
    comparison = {
        "status": selection["selection_status"],
        "baseline_lock": baseline_lock,
        "finance_repair_candidate_passes": finance_repair_candidate_passes(finance_report),
        "selection": selection,
        "strategy_reports": strategy_reports,
        "full_500_rerun_allowed": full_rerun_allowed(
            selection=selection,
            b6r5_report=finance_report,
        ),
        **no_policy_mutation_flags(),
    }
    write_json(ROOT / args.strategy_comparison_path, comparison)
    targeted_report = {
        "block": "B6R6",
        "status": selection["selection_status"],
        "row_count": len(replay_rows),
        "baseline_lock": baseline_lock,
        "strategy_reports": strategy_reports,
        "selection": selection,
        "full_500_rerun_triggered": full_rerun_allowed(
            selection=selection,
            b6r5_report=finance_report,
        )
        and not args.skip_full_rerun,
        **no_policy_mutation_flags(),
    }
    write_json(ROOT / args.targeted_report_path, targeted_report)
    _write_csv(args.targeted_summary_path, summary_rows)
    return all_rows, comparison


def _failure_row(
    *,
    item: WorkloadItem,
    prompt: str,
    exc: Exception,
    elapsed_ms: float,
    strategy_id: str,
    run_id: str,
    optimization: str,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "timestamp_utc": utc_now(),
        "config_id": "b6r6_research_ai_quality_recovery",
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "backend_type": "self_hosted_gpu",
        "runtime": "vllm",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "provider": "self_hosted",
        "concurrency": 1,
        "model_alias": B6R6_MODEL_ALIAS,
        "model_id": B6R6_MODEL_ID,
        "model_name": B6R6_MODEL_ID,
        "optimization": optimization,
        "b6r6_strategy_id": strategy_id,
        "prompt": prompt,
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
        "citation_id_aliases": item.metadata.get("citation_id_aliases"),
        "gold_evidence_ids": item.metadata.get("gold_evidence_ids"),
        **no_policy_mutation_flags(),
    }
    row.update(generation_contract_result_fields(""))
    return row


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
        model_alias=B6R6_MODEL_ALIAS,
        model_id=B6R6_MODEL_ID,
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


def run_full_500(
    *,
    args: argparse.Namespace,
    selected_strategy: str,
) -> dict[str, Any]:
    """Run the full frozen 500 with Finance B6R5 and Research AI B6R6 strategies."""

    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    items = [_workload_item(row) for row in _read_runner_items(args.full_input_path)]
    rows = _resume_rows(args.full_output_path, RUN_ID_FULL)
    completed_ids = {str(row.get("prompt_id")) for row in rows}
    if rows:
        _write_jsonl(args.full_output_path, rows)
    start_time = utc_now()
    for item in items:
        if item.prompt_id in completed_ids:
            continue
        started = time.perf_counter()
        try:
            if item.metadata.get("vertical") == B6R6_VERTICAL:
                row = _run_research_ai_item(
                    item=item,
                    strategy_id=selected_strategy,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                    run_id=RUN_ID_FULL,
                    optimization=f"b6r6_{selected_strategy}_500_quality_gate",
                    gold_by_prompt=gold_by_prompt,
                )
            else:
                row = _run_default_item(
                    item=item,
                    strategy_id=selected_strategy,
                    gold_by_prompt=gold_by_prompt,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    timeout_seconds=args.timeout_seconds,
                    run_id=RUN_ID_FULL,
                    optimization=f"b6r6_{selected_strategy}_500_quality_gate",
                )
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(
                item=item,
                prompt=item.prompt,
                exc=exc,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                strategy_id=selected_strategy,
                run_id=RUN_ID_FULL,
                optimization=f"b6r6_{selected_strategy}_500_quality_gate",
            )
        row["sequence_index"] = len(rows) + 1
        rows.append(row)
        completed_ids.add(item.prompt_id)
        _write_jsonl(args.full_output_path, rows)
    end_time = utc_now()
    evaluation_rows, summary = _evaluate_rows(rows=rows, results_path=args.full_output_path)
    per_vertical = build_per_vertical_quality(evaluation_rows, rows, verticals=VERTICALS)
    gate = classify_b6r6_full_gate(summary=summary, per_vertical_quality=per_vertical)
    report = {
        "block": "B6R6",
        "experiment": "model2_3b_research_ai_quality_recovery_500",
        "status": gate["status"],
        "model_alias": B6R6_MODEL_ALIAS,
        "model_id": B6R6_MODEL_ID,
        "selected_research_ai_strategy": selected_strategy,
        "finance_strategy": "b6r5_evidence_selection_preplan",
        "row_count": len(rows),
        "summary": summary,
        "per_vertical_quality": per_vertical,
        "quality_gate": gate,
        "evaluation_rows": evaluation_rows,
        "deployability_readiness": gate["deployability_readiness"],
        "benchmark_execution_readiness": gate["benchmark_execution_readiness"],
        **no_policy_mutation_flags(),
    }
    write_json(ROOT / args.full_report_path, report)
    _write_csv(args.full_summary_path, [{"vertical": "all", **summary}, *per_vertical])
    write_json(
        ROOT / args.full_comparison_path,
        {
            "baseline": "B6R4_model2_3b_500",
            "candidate": "B6R6_model2_3b_500",
            "b6r4": _read_json(args.b6r4_eval_report),
            "b6r5": build_b6r4_vs_b6r5_comparison(
                b6r4_report=_read_json(args.b6r4_eval_report),
                b6r5_report=None,
                selected_strategy="evidence_selection_preplan",
            ),
            "b6r6": report,
        },
    )
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
    return report


def run_b6r6(args: argparse.Namespace) -> dict[str, Any]:
    """Run the B6R6 failure audit, targeted strategies, and optional full gate."""

    model = load_project_config().resolve_model_config(B6R6_MODEL_ALIAS)
    if model.model_id != B6R6_MODEL_ID:
        msg = f"{B6R6_MODEL_ALIAS} resolved to unexpected model {model.model_id}"
        raise RuntimeError(msg)
    replay_rows, audit, baseline_lock = build_failure_set_and_audit(args)
    if args.dry_run:
        return {
            "status": "dry_run",
            "research_ai_failed_row_count": len(replay_rows),
            "failure_audit": audit,
            "baseline_lock": baseline_lock,
            "full_500_rerun_triggered": False,
        }
    readiness = check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=B6R6_MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    targeted_started = utc_now()
    targeted_rows, comparison = run_targeted_strategies(
        replay_rows=replay_rows,
        baseline_lock=baseline_lock,
        args=args,
        gold_by_prompt=gold_by_prompt,
    )
    targeted_ended = utc_now()
    _write_manifest(
        path=args.targeted_manifest_path,
        run_id=RUN_ID_TARGETED,
        input_path=args.failure_replay_input,
        output_path=args.targeted_output_path,
        expected_count=len(replay_rows) * len(B6R6_STRATEGIES),
        completed_count=len(targeted_rows),
        start_time=targeted_started,
        end_time=targeted_ended,
        status="completed",
    )
    selection = cast(dict[str, Any], comparison["selection"])
    selected_strategy = str(selection.get("selected_strategy") or "")
    full_triggered = (
        full_rerun_allowed(
            selection=selection,
            b6r5_report=_read_json(args.b6r5_targeted_report),
        )
        and not args.skip_full_rerun
    )
    full_result: dict[str, Any] | None = None
    if full_triggered:
        full_result = run_full_500(args=args, selected_strategy=selected_strategy)
    return {
        "status": (
            full_result["status"]
            if full_result
            else selection.get("selection_status", "B6R6_TARGETED_BLOCKED")
        ),
        "server_readiness": readiness.to_dict(),
        "research_ai_failed_row_count": len(replay_rows),
        "root_cause_breakdown": audit["root_cause_breakdown"],
        "baseline_lock": baseline_lock,
        "selection": selection,
        "selected_research_ai_strategy": selected_strategy,
        "full_500_rerun_triggered": full_triggered,
        "full_result": full_result,
        "deployability_readiness": (
            full_result.get("deployability_readiness") if full_result else "NOT_READY"
        ),
        "benchmark_execution_readiness": (
            full_result.get("benchmark_execution_readiness")
            if full_result
            else ("READY_WITH_QUALITY_CAVEAT" if selection.get("targeted_passed") else "NOT_READY")
        ),
        "targeted_report": args.targeted_report_path,
        "strategy_comparison": args.strategy_comparison_path,
        "full_report": args.full_report_path if full_result else None,
    }


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    try:
        result = run_b6r6(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B6R6 Research AI recovery failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
