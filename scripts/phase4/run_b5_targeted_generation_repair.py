"""Run B5 targeted safety and citation-selection replay over B4 failures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
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

from evaluate_generation_outputs import load_gold_records  # noqa: E402
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
    render_citation_repair_prompt,
    render_contract_retry_prompt,
)
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    sample_gpu_telemetry,
    summarize_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.multi_evidence_selector import (  # noqa: E402
    EvidenceSupportPlan,
    build_evidence_support_plan,
    inject_internal_evidence_plan,
    render_internal_evidence_plan,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.safety_generation_repair import (  # noqa: E402
    apply_lexical_guard_to_text,
    decide_targeted_retry,
    detect_safety_rule_ids,
    preserve_json_with_safe_answer,
    render_safety_rule_repair_prompt,
)
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.slo_diagnosis import diagnose_slos  # noqa: E402
from inference_bench.slo_profiles import resolve_slo_profile  # noqa: E402
from inference_bench.streaming_metrics import (  # noqa: E402
    StreamingMetrics,
    request_streaming_chat_completion,
)

MODEL_ALIAS = "model2_1_5b"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 160
TARGETED_PROMPT_COUNT = 25
TOTAL_PROMPT_COUNT = 100

DEFAULT_B4_AUDIT = "results/processed/b4_generation_quality_audit_report.json"
DEFAULT_B4_EVAL = "results/processed/b4_vllm_1_5b_context_aligned_eval_report.json"
DEFAULT_B4_RESULTS = "results/raw/b4_vllm_1_5b_context_aligned_results.jsonl"
DEFAULT_B4_RUNNER_INPUT = "data/generated/phase4/b4_context_aligned_runner_input.jsonl"
DEFAULT_B4_LATENCY = "results/processed/b4_vllm_1_5b_context_aligned_latency_summary.csv"
DEFAULT_B4_GPU = "results/processed/b4_vllm_1_5b_context_aligned_gpu_telemetry_summary.json"

DEFAULT_REPLAY = "results/processed/b5_failed_prompt_replay.jsonl"
DEFAULT_REPORT = "results/processed/b5_targeted_repair_report.json"
DEFAULT_SUMMARY = "results/processed/b5_targeted_repair_summary.csv"
DEFAULT_COMPARISON = "results/processed/b5_b4_vs_b5_comparison.json"
DEFAULT_MANIFEST = "results/raw/b5_targeted_generation_repair_manifest.json"
DEFAULT_LATENCY = "results/processed/b5_targeted_repair_latency_summary.csv"
DEFAULT_GPU_CSV = "results/processed/b5_targeted_repair_gpu_telemetry.csv"
DEFAULT_GPU_SUMMARY = "results/processed/b5_targeted_repair_gpu_telemetry_summary.json"

DEFAULT_FULL_REPLAY = "results/raw/b5_full_frozen_100_replay.jsonl"
DEFAULT_FULL_REPORT = "results/processed/b5_full_frozen_100_report.json"
DEFAULT_FULL_SUMMARY = "results/processed/b5_full_frozen_100_summary.csv"
DEFAULT_FULL_LATENCY = "results/processed/b5_full_frozen_100_latency_summary.csv"

QUALITY_GATE_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.85,
    "grounded_rate": 0.85,
    "safety_violation_count": 0,
}


@dataclass(frozen=True)
class PreparedItem:
    """B5 item with deterministic planning metadata."""

    item: WorkloadItem
    source_item: WorkloadItem
    plan: EvidenceSupportPlan


def build_parser() -> argparse.ArgumentParser:
    """Build the B5 CLI."""

    parser = argparse.ArgumentParser(
        description="Run B5 targeted generation-quality repair over B4 failed prompts."
    )
    parser.add_argument("--b4-audit-report", default=DEFAULT_B4_AUDIT)
    parser.add_argument("--b4-eval-report", default=DEFAULT_B4_EVAL)
    parser.add_argument("--b4-results", default=DEFAULT_B4_RESULTS)
    parser.add_argument("--b4-runner-input", default=DEFAULT_B4_RUNNER_INPUT)
    parser.add_argument("--b4-latency-summary", default=DEFAULT_B4_LATENCY)
    parser.add_argument("--b4-gpu-summary", default=DEFAULT_B4_GPU)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--output", default=DEFAULT_REPLAY)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-json", default=DEFAULT_COMPARISON)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--latency-summary", default=DEFAULT_LATENCY)
    parser.add_argument("--gpu-telemetry-csv", default=DEFAULT_GPU_CSV)
    parser.add_argument("--gpu-telemetry-summary", default=DEFAULT_GPU_SUMMARY)
    parser.add_argument("--full-output", default=DEFAULT_FULL_REPLAY)
    parser.add_argument("--full-report", default=DEFAULT_FULL_REPORT)
    parser.add_argument("--full-summary", default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--full-latency-summary", default=DEFAULT_FULL_LATENCY)
    parser.add_argument("--telemetry-ssh-host", default=None)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--telemetry-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--skip-full-rerun", action="store_true")
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


def _read_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / Path(path)).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object row: {path}")
                rows.append(cast(dict[str, Any], payload))
    return rows


def _read_runner_items(path: str | Path) -> list[WorkloadItem]:
    return [WorkloadItem(**row) for row in _read_jsonl_dicts(path)]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / Path(path)).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _question_from_prompt(prompt: str) -> str:
    if "\nUSER QUESTION:\n" not in prompt:
        return prompt
    tail = prompt.split("\nUSER QUESTION:\n", maxsplit=1)[1]
    return tail.split("\n\nOUTPUT CONTRACT:\n", maxsplit=1)[0].strip()


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
        "run_id": "b5-targeted-generation-repair",
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": "b5_safety_and_citation_selection_repair",
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
        "run_id": "b5-targeted-generation-repair",
        "timestamp_utc": utc_now(),
        "prompt_id": item.prompt_id,
        "workload_name": item.workload_name,
        "backend": "vllm",
        "model_name": MODEL_ID,
        "optimization": "b5_safety_and_citation_selection_repair",
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


def _refresh_contract_fields(row: dict[str, Any]) -> dict[str, Any]:
    aliases = row.get("citation_id_aliases")
    refreshed = {
        **row,
        **generation_contract_result_fields(
            str(row.get("generated_text") or ""),
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        ),
    }
    return refreshed


def _apply_lexical_guard(
    *,
    row: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    terms = tuple(str(term) for term in evaluation.get("safety_violation_terms") or [])
    repair = preserve_json_with_safe_answer(
        str(row.get("generated_text") or ""),
        allowed_evidence_ids=tuple(
            allowed_evidence_ids_from_aliases(row.get("citation_id_aliases"))
        ),
        prohibited_terms=terms,
    )
    if not repair.changed:
        return row
    refreshed = _refresh_contract_fields({**row, "generated_text": repair.repaired_text})
    refreshed["lexical_guard_applied"] = True
    refreshed["lexical_guard_rule_ids"] = list(repair.rule_ids)
    return refreshed


def _merge_retry(
    *,
    current: dict[str, Any],
    metrics: StreamingMetrics,
    trigger: str,
) -> dict[str, Any]:
    aliases = current.get("citation_id_aliases")
    current_latency = float(current.get("end_to_end_latency_ms") or 0.0)
    total_latency = current_latency + metrics.e2e_latency_ms
    total_input = int(current.get("input_tokens") or 0) + metrics.input_tokens
    total_output = int(current.get("output_tokens") or 0) + metrics.output_tokens
    total_tokens = total_input + total_output
    triggers = [*cast(list[str], current.get("retry_triggers") or []), trigger]
    retry_count = int(current.get("retry_attempt_count") or 0) + 1
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
    return {
        **current,
        "generated_text": metrics.generated_text,
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
        **generation_contract_result_fields(
            metrics.generated_text,
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        ),
    }


def _safe_bad_output(text: str) -> str:
    repair = apply_lexical_guard_to_text(text)
    return repair.repaired_text


def _csv_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(part) for part in value if str(part))
    return ()


def _repair_prompt(
    *,
    row: dict[str, Any],
    trigger: str,
    missing_labels: tuple[str, ...],
) -> str:
    allowed = allowed_evidence_ids_from_aliases(row.get("citation_id_aliases"))
    if trigger == "safety_violation":
        rule_ids = detect_safety_rule_ids(str(row.get("generated_text") or ""))
        if not rule_ids:
            rule_ids = _csv_tuple(row.get("b5_safety_rule_ids"))
        return render_safety_rule_repair_prompt(result_row=row, rule_ids=rule_ids)
    if trigger == "missing_evidence_label":
        return render_citation_repair_prompt(
            original_prompt=str(row.get("prompt") or ""),
            previous_output=_safe_bad_output(str(row.get("generated_text") or "")),
            allowed_evidence_ids=allowed,
            missing_evidence_labels=missing_labels,
        )
    return render_contract_retry_prompt(
        bad_output=_safe_bad_output(str(row.get("generated_text") or "")),
        violation=trigger,
        allowed_evidence_ids=allowed,
    )


def _missing_labels(
    *,
    evaluation: dict[str, Any],
    row: dict[str, Any],
) -> tuple[str, ...]:
    if bool(evaluation.get("evidence_match")) or not bool(
        evaluation.get("generation_contract_valid")
    ):
        return ()
    plan = build_evidence_support_plan(evaluation_row=evaluation, result_row=row)
    return plan.missing_labels


def _run_prepared_items(
    *,
    prepared_items: list[PreparedItem],
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
    for prepared in prepared_items:
        item = prepared.item
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
            initial_evaluation = evaluate_result_row(row, gold_by_prompt.get(item.prompt_id))
            row["initial_json_validity"] = initial_evaluation.get("json_validity")
            row["initial_contract_validity"] = initial_evaluation.get("generation_contract_valid")
            row["initial_evidence_match"] = initial_evaluation.get("evidence_match")
            row["initial_groundedness"] = initial_evaluation.get("groundedness")
            row["initial_safety_violation"] = initial_evaluation.get("safety_violation")
            row["initial_safety_violation_terms"] = initial_evaluation.get("safety_violation_terms")
            evaluation = initial_evaluation
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
                prompt = _repair_prompt(
                    row=row,
                    trigger=decision.trigger,
                    missing_labels=decision.missing_labels,
                )
                repair_metrics = request_streaming_chat_completion(
                    api_key=api_key,
                    model_id=MODEL_ID,
                    prompt=prompt,
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
        rows.append(row)
        with output.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return rows


def _prepare_items(
    *,
    source_items: list[WorkloadItem],
    b4_eval_rows: list[dict[str, Any]],
    b4_result_rows: list[dict[str, Any]],
    b4_runner_rows: list[dict[str, Any]],
    prompt_ids: set[str],
    b4_failure_rows: dict[str, dict[str, Any]],
) -> list[PreparedItem]:
    eval_by_prompt = {str(row.get("prompt_id") or ""): row for row in b4_eval_rows}
    result_by_prompt = {str(row.get("prompt_id") or ""): row for row in b4_result_rows}
    runner_by_prompt = {str(row.get("prompt_id") or ""): row for row in b4_runner_rows}
    prepared: list[PreparedItem] = []
    for source in source_items:
        if source.prompt_id not in prompt_ids:
            continue
        evaluation = eval_by_prompt[source.prompt_id]
        result = result_by_prompt[source.prompt_id]
        failure = b4_failure_rows.get(source.prompt_id, {})
        prohibited_terms = tuple(str(term) for term in failure.get("safety_violation_terms") or [])
        rule_ids = detect_safety_rule_ids(source.prompt, prohibited_terms=prohibited_terms)
        plan = build_evidence_support_plan(
            evaluation_row=evaluation,
            result_row=result,
            runner_input=runner_by_prompt.get(source.prompt_id),
            safety_rule_ids=rule_ids,
        )
        planning_context = render_internal_evidence_plan(
            plan=plan,
            question=_question_from_prompt(source.prompt),
        )
        metadata = {
            **source.metadata,
            "b5_required_labels": ",".join(plan.required_labels),
            "b5_missing_labels_from_b4": ",".join(plan.missing_labels),
            "b5_safety_rule_ids": ",".join(plan.safety_rule_ids),
        }
        prepared.append(
            PreparedItem(
                item=WorkloadItem(
                    prompt_id=source.prompt_id,
                    workload_name=source.workload_name,
                    prompt=inject_internal_evidence_plan(source.prompt, planning_context),
                    expected_output=source.expected_output,
                    metadata=metadata,
                ),
                source_item=source,
                plan=plan,
            )
        )
    return prepared


def _quality_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    for metric, threshold in QUALITY_GATE_THRESHOLDS.items():
        observed = float(summary.get(metric) or 0.0)
        if metric == "safety_violation_count":
            passed = observed == float(threshold)
            operator = "=="
        else:
            passed = observed >= float(threshold)
            operator = ">="
        checks[metric] = {
            "observed": observed,
            "threshold": threshold,
            "operator": operator,
            "passed": passed,
        }
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    return {
        "status": "QUALITY_READY" if not failed else "QUALITY_BLOCKED",
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


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
    write_csv_rows(ROOT / Path(path), rows)


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _quality_metrics(
    evaluation_rows: list[dict[str, Any]],
    *,
    prompt_ids: set[str],
) -> dict[str, Any]:
    rows = [row for row in evaluation_rows if str(row.get("prompt_id")) in prompt_ids]
    return {
        "grounded_rate": _rate(rows, "groundedness"),
        "evidence_match_rate": _rate(rows, "evidence_match"),
        "task_success_rate": _rate(rows, "status_matches"),
        "generation_contract_valid_rate": _rate(rows, "generation_contract_valid"),
        "safety_violation_count": sum(bool(row.get("safety_violation")) for row in rows),
    }


def _retrieval_by_vertical() -> dict[str, dict[str, float]]:
    manifest = _read_json(
        "data/generated/context_engineering/retrieval_source_of_truth_manifest.json"
    )
    raw_metrics = cast(dict[str, Any], manifest["metrics_by_vertical"])
    return {
        vertical: {
            "candidate_recall_at_20_min": float(metrics["candidate_recall_at_20"]),
            "candidate_recall_at_50_min": float(metrics["candidate_recall_at_50"]),
            "final_recall_at_5_min": float(metrics["final_recall_at_5"]),
            "mrr_min": float(metrics["mrr"]),
        }
        for vertical, metrics in raw_metrics.items()
    }


def _latency_by_vertical(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        metrics = {
            key: float(value)
            for key, value in row.items()
            if key != "vertical" and value not in (None, "")
        }
        for source, target in (
            ("mean_itl_p50_ms", "itl_p50_ms"),
            ("mean_itl_p95_ms", "itl_p95_ms"),
            ("mean_itl_p99_ms", "itl_p99_ms"),
        ):
            if source in metrics:
                metrics[target] = metrics[source]
        output[str(row["vertical"])] = metrics
    return output


def _telemetry_metrics(telemetry: dict[str, Any]) -> dict[str, float]:
    utilization = cast(dict[str, Any], telemetry.get("utilization_gpu_percent", {}))
    memory_used = cast(dict[str, Any], telemetry.get("memory_used_mb", {}))
    memory_total = cast(dict[str, Any], telemetry.get("memory_total_mb", {}))
    metrics: dict[str, float] = {}
    if utilization.get("mean") not in (None, ""):
        metrics["mean_gpu_utilization_percent"] = float(utilization["mean"])
    if memory_used.get("max") not in (None, ""):
        metrics["max_gpu_memory_used_mb"] = float(memory_used["max"])
    if memory_total.get("max") not in (None, ""):
        metrics["gpu_memory_total_mb"] = float(memory_total["max"])
    return metrics


def _throughput_metrics(
    *,
    result_rows: list[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, float]:
    successes = sum(bool(row.get("success")) for row in result_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in result_rows)
    return {
        "requests_per_second_min": len(result_rows) / wall_seconds if wall_seconds else 0.0,
        "successful_requests_per_second_min": successes / wall_seconds if wall_seconds else 0.0,
        "aggregate_tokens_per_second": output_tokens / wall_seconds if wall_seconds else 0.0,
    }


def _model_metadata(model_alias: str) -> dict[str, Any]:
    models = _read_yaml("configs/models.yaml")
    aliases = cast(dict[str, str], models["model_aliases"])
    return cast(dict[str, Any], models[aliases[model_alias]])


def _build_slo_diagnosis(
    *,
    evaluation_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    telemetry: dict[str, Any],
    wall_seconds: float,
    scope: str,
    runner_input_path: str,
) -> dict[str, Any]:
    profile = resolve_slo_profile()
    retrieval = _retrieval_by_vertical()
    latency = _latency_by_vertical(latency_rows)
    shared = {
        **_throughput_metrics(result_rows=result_rows, wall_seconds=wall_seconds),
        **_telemetry_metrics(telemetry),
    }
    hardware = _read_yaml("configs/hardware/remote_rtx3070.yaml")
    experiment_config = {
        "block": "B5",
        "engine": "vllm",
        "model_alias": MODEL_ALIAS,
        "model_id": MODEL_ID,
        "memory_mode": "mm2_hybrid_top5",
        "ablation_mode": "prompt_plus_metadata",
        "concurrency": 1,
        "max_records": len(result_rows),
        "max_new_tokens": MAX_NEW_TOKENS,
        "runner_input_path": runner_input_path,
        "diagnosis_scope": scope,
    }
    diagnoses: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        prompt_ids = {
            str(row["prompt_id"]) for row in result_rows if row.get("vertical") == vertical
        }
        if not prompt_ids:
            continue
        metrics = {
            **retrieval[vertical],
            **_quality_metrics(evaluation_rows, prompt_ids=prompt_ids),
            **latency.get(vertical, {}),
            **shared,
        }
        diagnosis = diagnose_slos(
            run_metrics=metrics,
            profile=profile,
            experiment_config=experiment_config,
            model_metadata=_model_metadata(MODEL_ALIAS),
            hardware_profile=hardware,
            engine="vllm",
            memory_mode="mm2_hybrid_top5",
            vertical=vertical,
            telemetry_available=True,
            backend_type="self_hosted",
        )
        diagnosis["block"] = "B5"
        diagnosis["targeted_scope_caveat"] = scope
        diagnoses.append(diagnosis)
    primary_counts = Counter(
        str(item.get("primary_recommendation", {}).get("optimization_id", ""))
        for item in diagnoses
        if isinstance(item.get("primary_recommendation"), dict)
    )
    return {
        "scope": scope,
        "profile": profile.name,
        "diagnoses": diagnoses,
        "aggregate": {
            "diagnosis_count": len(diagnoses),
            "failed_slo_count": sum(len(item["failed_slos"]) for item in diagnoses),
            "unavailable_metric_count": sum(len(item["unavailable_metrics"]) for item in diagnoses),
            "primary_recommendation_counts": dict(sorted(primary_counts.items())),
        },
    }


def _metric_delta(baseline: Any, candidate: Any) -> dict[str, float | None]:
    if baseline in (None, "") or candidate in (None, ""):
        return {"baseline": None, "candidate": None, "absolute_delta": None}
    base = float(baseline)
    cand = float(candidate)
    return {"baseline": base, "candidate": cand, "absolute_delta": cand - base}


def _baseline_subset_summary(
    *,
    b4_eval_rows: list[dict[str, Any]],
    b4_result_rows: list[dict[str, Any]],
    prompt_ids: set[str],
) -> dict[str, Any]:
    subset_eval = [row for row in b4_eval_rows if str(row.get("prompt_id")) in prompt_ids]
    subset_results = [row for row in b4_result_rows if str(row.get("prompt_id")) in prompt_ids]
    from evaluate_generation_outputs import build_summary_rows

    return build_summary_rows(
        results_path=DEFAULT_B4_RESULTS,
        result_rows=subset_results,
        evaluation_rows=subset_eval,
    )[0]


def _build_comparison(
    *,
    b4_summary: dict[str, Any],
    b5_summary: dict[str, Any],
    b4_latency: dict[str, Any],
    b5_latency: dict[str, Any],
    b4_gpu: dict[str, Any],
    b5_gpu: dict[str, Any],
    full_rerun_triggered: bool,
) -> dict[str, Any]:
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
        "baseline": "B4_failed_prompt_subset",
        "candidate": "B5_targeted_repair_subset",
        "prompt_matched": True,
        "prompt_count": b5_summary["row_count"],
        "full_100_rerun_triggered": full_rerun_triggered,
        "quality_deltas": {
            metric: _metric_delta(b4_summary.get(metric), b5_summary.get(metric))
            for metric in quality_metrics
        },
        "latency_throughput_deltas": {
            metric: _metric_delta(b4_latency.get(metric), b5_latency.get(metric))
            for metric in latency_metrics
        },
        "gpu_telemetry_deltas": {
            metric: _metric_delta(
                cast(dict[str, Any], b4_gpu.get(group, {})).get(stat),
                cast(dict[str, Any], b5_gpu.get(group, {})).get(stat),
            )
            for metric, (group, stat) in telemetry_sources.items()
        },
    }


def _run_with_telemetry(
    *,
    prepared_items: list[PreparedItem],
    gold_by_prompt: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    output_path: str,
) -> tuple[list[dict[str, Any]], list[GpuTelemetrySample], dict[str, Any], float, list[str]]:
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

    thread = threading.Thread(target=collect_telemetry, name="b5-gpu-telemetry", daemon=True)
    thread.start()
    wall_start = time.perf_counter()
    try:
        rows = _run_prepared_items(
            prepared_items=prepared_items,
            gold_by_prompt=gold_by_prompt,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            output_path=output_path,
        )
    finally:
        wall_seconds = time.perf_counter() - wall_start
        stop_event.set()
        thread.join(timeout=max(5.0, args.telemetry_interval_seconds + 3.0))
    telemetry = summarize_gpu_telemetry(
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    return rows, telemetry_samples, telemetry, wall_seconds, telemetry_errors


def run_b5(args: argparse.Namespace) -> dict[str, Any]:
    """Run targeted B5 replay, then full frozen rerun only if target gate passes."""

    model = load_project_config().resolve_model_config(MODEL_ALIAS)
    if model.model_id != MODEL_ID:
        raise RuntimeError(f"{MODEL_ALIAS} resolved to unexpected model {model.model_id}")

    audit = _read_json(args.b4_audit_report)
    b4_eval_report = _read_json(args.b4_eval_report)
    b4_eval_rows = cast(list[dict[str, Any]], b4_eval_report["evaluation_rows"])
    b4_result_rows = _read_jsonl_dicts(args.b4_results)
    b4_runner_rows = _read_jsonl_dicts(args.b4_runner_input)
    source_items = _read_runner_items(args.b4_runner_input)
    b4_failure_rows = {
        str(row["prompt_id"]): row for row in cast(list[dict[str, Any]], audit["failure_rows"])
    }
    targeted_prompt_ids = set(b4_failure_rows)
    if len(targeted_prompt_ids) != TARGETED_PROMPT_COUNT:
        raise RuntimeError(f"B5 expected {TARGETED_PROMPT_COUNT} B4 failures")
    targeted_items = _prepare_items(
        source_items=source_items,
        b4_eval_rows=b4_eval_rows,
        b4_result_rows=b4_result_rows,
        b4_runner_rows=b4_runner_rows,
        prompt_ids=targeted_prompt_ids,
        b4_failure_rows=b4_failure_rows,
    )
    if len(targeted_items) != TARGETED_PROMPT_COUNT:
        raise RuntimeError("B5 targeted item selection is incomplete")
    if args.dry_run:
        return {
            "status": "dry_run",
            "targeted_prompt_count": len(targeted_items),
            "targeted_prompt_ids": sorted(targeted_prompt_ids),
            "full_rerun_planned_if_gate_passes": not args.skip_full_rerun,
        }

    readiness = check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    start_time = utc_now()
    (
        targeted_rows,
        telemetry_samples,
        telemetry,
        wall_seconds,
        telemetry_errors,
    ) = _run_with_telemetry(
        prepared_items=targeted_items,
        gold_by_prompt=gold,
        args=args,
        output_path=args.output,
    )
    end_time = utc_now()

    eval_report, eval_summary = evaluate_result_rows(
        result_rows=targeted_rows,
        output_path=args.output,
        eval_report_path=args.report,
        eval_summary_path=args.summary,
        block="B5",
        experiment="targeted_safety_and_citation_selection_repair",
    )
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    latency_rows = _latency_rows(targeted_rows)
    write_csv_rows(ROOT / Path(args.latency_summary), latency_rows)
    write_gpu_telemetry_csv(ROOT / Path(args.gpu_telemetry_csv), telemetry_samples)
    write_gpu_telemetry_summary(
        ROOT / Path(args.gpu_telemetry_summary),
        telemetry_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=args.telemetry_duration_seconds,
    )
    gate = _quality_gate(eval_summary)
    per_vertical = build_per_vertical_quality(
        evaluation_rows,
        targeted_rows,
        verticals=VERTICALS,
    )
    _write_quality_summary(
        path=args.summary,
        overall=eval_summary,
        per_vertical=per_vertical,
        latency_rows=latency_rows,
        gate=gate,
    )
    b4_summary = _baseline_subset_summary(
        b4_eval_rows=b4_eval_rows,
        b4_result_rows=b4_result_rows,
        prompt_ids=targeted_prompt_ids,
    )
    b4_latency_rows = _read_csv(args.b4_latency_summary)
    b4_latency_all = next(row for row in b4_latency_rows if row.get("vertical") == "all")
    b4_gpu = _read_json(args.b4_gpu_summary)
    full_rerun_triggered = bool(gate["passed"] and not args.skip_full_rerun)
    comparison = _build_comparison(
        b4_summary=b4_summary,
        b5_summary=eval_summary,
        b4_latency=b4_latency_all,
        b5_latency=latency_rows[0],
        b4_gpu=b4_gpu,
        b5_gpu=telemetry,
        full_rerun_triggered=full_rerun_triggered,
    )
    write_json(ROOT / Path(args.comparison_json), comparison)

    slo_diagnosis = _build_slo_diagnosis(
        evaluation_rows=evaluation_rows,
        result_rows=targeted_rows,
        latency_rows=latency_rows,
        telemetry=telemetry,
        wall_seconds=wall_seconds,
        scope="B5 targeted replay over 25 B4 failed prompt IDs only",
        runner_input_path=args.b4_runner_input,
    )
    full_result: dict[str, Any] | None = None
    if full_rerun_triggered:
        all_prompt_ids = {item.prompt_id for item in source_items}
        all_items = _prepare_items(
            source_items=source_items,
            b4_eval_rows=b4_eval_rows,
            b4_result_rows=b4_result_rows,
            b4_runner_rows=b4_runner_rows,
            prompt_ids=all_prompt_ids,
            b4_failure_rows=b4_failure_rows,
        )
        full_rows = _run_prepared_items(
            prepared_items=all_items,
            gold_by_prompt=gold,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_seconds=args.timeout_seconds,
            output_path=args.full_output,
        )
        full_eval_report, full_summary = evaluate_result_rows(
            result_rows=full_rows,
            output_path=args.full_output,
            eval_report_path=args.full_report,
            eval_summary_path=args.full_summary,
            block="B5",
            experiment="full_frozen_100_safety_and_citation_selection_repair",
        )
        full_latency_rows = _latency_rows(full_rows)
        write_csv_rows(ROOT / Path(args.full_latency_summary), full_latency_rows)
        full_result = {
            "row_count": len(full_rows),
            "summary": full_summary,
            "quality_gate": _quality_gate(full_summary),
            "report_path": args.full_report,
            "summary_path": args.full_summary,
            "output_path": args.full_output,
            "latency_summary_path": args.full_latency_summary,
            "evaluation_row_count": full_eval_report["row_count"],
        }

    blocking_metrics = gate["failed_metrics"]
    recommendation = (
        "Run the full frozen 100-prompt B5 replay and then re-run SLO diagnosis."
        if full_rerun_triggered
        else (
            "Stop scaling. Repair the targeted failed metrics before any concurrency "
            f"or prompt-count increase: {', '.join(blocking_metrics)}."
        )
    )
    report = {
        **eval_report,
        "status": gate["status"],
        "quality_gate": gate,
        "targeted_prompt_ids": sorted(targeted_prompt_ids),
        "targeted_prompt_count": len(targeted_rows),
        "per_vertical_quality": per_vertical,
        "latency_summary": latency_rows[0],
        "gpu_telemetry_summary": telemetry,
        "wall_seconds": wall_seconds,
        "telemetry_errors": telemetry_errors,
        "retry_trigger_counts": dict(
            sorted(
                Counter(
                    trigger
                    for row in targeted_rows
                    for trigger in cast(list[str], row.get("retry_triggers") or [])
                ).items()
            )
        ),
        "lexical_guard_count": sum(bool(row.get("lexical_guard_applied")) for row in targeted_rows),
        "full_100_rerun_triggered": full_rerun_triggered,
        "full_100_result": full_result,
        "b4_vs_b5_comparison_path": args.comparison_json,
        "slo_diagnosis": slo_diagnosis,
        "recommendation": recommendation,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "concurrency": 1,
    }
    write_json(ROOT / Path(args.report), report)
    manifest = RunManifest(
        run_id="b5-targeted-generation-repair",
        timestamp_utc=end_time,
        backend="vllm",
        model_alias=MODEL_ALIAS,
        model_id=MODEL_ID,
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(args.b4_runner_input),
        output_path=str(args.output),
        max_records=TARGETED_PROMPT_COUNT,
        git_commit=current_git_commit(ROOT),
        command=sanitized_command(sys.argv),
        status="completed",
        start_time=start_time,
        end_time=end_time,
        error_count=sum(not bool(row.get("success")) for row in targeted_rows),
        telemetry_path=str(args.gpu_telemetry_csv),
        telemetry_summary_path=str(args.gpu_telemetry_summary),
    )
    write_run_manifest(manifest, ROOT / Path(args.manifest))
    return {
        "status": gate["status"],
        "server_readiness": readiness.to_dict(),
        "targeted_row_count": len(targeted_rows),
        "targeted_success_count": sum(bool(row.get("success")) for row in targeted_rows),
        "quality_gate": gate,
        "evaluation_summary": eval_summary,
        "latency_summary": latency_rows[0],
        "full_100_rerun_triggered": full_rerun_triggered,
        "full_100_result": full_result,
        "report": args.report,
        "summary": args.summary,
        "comparison": args.comparison_json,
        "replay": args.output,
    }


def main() -> int:
    """Run the B5 CLI."""

    args = build_parser().parse_args()
    try:
        result = run_b5(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B5 targeted repair failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
