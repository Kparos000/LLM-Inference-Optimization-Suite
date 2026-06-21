"""B7 controlled 1,000-prompt baseline helpers."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any, cast

from inference_bench.b6r5_finance_research_repair import (
    parse_alias_map,
    required_labels_from_aliases,
)
from inference_bench.cache_readiness import calculate_cache_readiness
from inference_bench.context_corpora import VERTICALS
from inference_bench.gpu_telemetry import build_runtime_projections
from inference_bench.load_profiles import ArrivalMode, build_load_profile_report

B7_MODEL_ALIAS = "model2_3b"
B7_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
B7_RUN_ID = "b7-model2-3b-1000-baseline"
B7_CONFIG_ID = "b7_model2_3b_1000_controlled_baseline"
B7_EXPECTED_PROMPTS_PER_VERTICAL = 200
B7_EXPECTED_PROMPT_COUNT = B7_EXPECTED_PROMPTS_PER_VERTICAL * len(VERTICALS)
B7_MEMORY_MODE = "mm2_hybrid_top5"
B7_TRAFFIC_PROFILE = "online_low_latency"
B7_REQUEST_ARRIVAL_MODE: ArrivalMode = "closed_loop"
B7_CONCURRENCY = 1
B7_RUNTIME = "vllm"
B7_HARDWARE = "remote_rtx3070"
B7_BACKEND_TYPE = "self_hosted_gpu"
B7_PROVIDER = "self_hosted"
B7_RESEARCH_AI_STRATEGY = "answer_skeleton"
B7_FINANCE_STRATEGY = "b6r5_evidence_selection_preplan"

B7_QUALITY_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.90,
    "grounded_rate": 0.90,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
    "vertical_evidence_match_rate_min": 0.85,
    "vertical_grounded_rate_min": 0.85,
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return dict(raw) if isinstance(raw, dict) else {}


def required_labels_for_runner_row(row: Mapping[str, Any]) -> list[str]:
    """Return supplied E-labels that cover required gold evidence for one row."""

    metadata = _metadata(row)
    explicit = metadata.get("b5_required_labels")
    if isinstance(explicit, str) and explicit.strip():
        return [label.strip() for label in explicit.split(",") if label.strip()]
    return required_labels_from_aliases(
        gold_evidence_ids=_json_list(metadata.get("gold_evidence_ids")),
        alias_map=parse_alias_map(metadata.get("citation_id_aliases")),
    )


def _runtime_selection_passes(selection: Mapping[str, Any]) -> bool:
    return (
        selection.get("model_alias") == B7_MODEL_ALIAS
        and selection.get("model_id") == B7_MODEL_ID
        and selection.get("runtime") == B7_RUNTIME
        and selection.get("engine") == B7_RUNTIME
        and selection.get("backend_type") == B7_BACKEND_TYPE
        and selection.get("hardware_type") == B7_HARDWARE
        and _as_bool(selection.get("live_run_allowed"))
    )


def preflight_b7_runner_rows(
    rows: list[dict[str, Any]],
    *,
    model_alias: str,
    model_id: str,
    runtime_selection: Mapping[str, Any],
    artifact_sync_dry_run_passed: bool,
    checkpoint_resume_enabled: bool,
    manifest_enabled: bool,
) -> dict[str, Any]:
    """Validate B7 runner rows before model inference."""

    prompt_ids = [str(row.get("prompt_id") or "") for row in rows]
    duplicate_prompt_ids = sorted(
        prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1
    )
    vertical_counts = Counter(
        str(_metadata(row).get("vertical") or row.get("vertical") or "") for row in rows
    )
    missing_required: list[dict[str, Any]] = []
    canonical_exposed: list[str] = []
    for row in rows:
        metadata = _metadata(row)
        labels = required_labels_for_runner_row(row)
        if not labels:
            missing_required.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "vertical": metadata.get("vertical"),
                    "reason": "no_required_evidence_label_in_e1_e5",
                }
            )
        if _as_bool(metadata.get("canonical_ids_exposed_to_model")) or _as_bool(
            row.get("canonical_ids_exposed_to_model")
        ):
            canonical_exposed.append(str(row.get("prompt_id") or ""))

    per_vertical_expected = {
        vertical: vertical_counts.get(vertical, 0) == B7_EXPECTED_PROMPTS_PER_VERTICAL
        for vertical in VERTICALS
    }
    checks = {
        "row_count": len(rows) == B7_EXPECTED_PROMPT_COUNT,
        "per_vertical_split": all(per_vertical_expected.values()),
        "unique_prompt_ids": not duplicate_prompt_ids and all(prompt_ids),
        "required_evidence_present_in_e1_e5": not missing_required,
        "no_canonical_id_leakage_flagged": not canonical_exposed,
        "model_alias_resolves": model_alias == B7_MODEL_ALIAS and model_id == B7_MODEL_ID,
        "runtime_registry_allows_vllm_remote_rtx3070": _runtime_selection_passes(runtime_selection),
        "artifact_sync_dry_run_passed": artifact_sync_dry_run_passed,
        "checkpoint_resume_enabled": checkpoint_resume_enabled,
        "manifest_enabled": manifest_enabled,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "block": "B7",
        "status": (
            "PREFLIGHT_PASSED_B7_CONTROLLED_1000_BASELINE"
            if not failed_checks
            else "PREFLIGHT_BLOCKED_B7_CONTROLLED_1000_BASELINE"
        ),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "row_count": len(rows),
        "expected_count": B7_EXPECTED_PROMPT_COUNT,
        "prompts_per_vertical": dict(vertical_counts),
        "expected_prompts_per_vertical": B7_EXPECTED_PROMPTS_PER_VERTICAL,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "missing_required_evidence_rows": missing_required,
        "canonical_ids_exposed_prompt_ids": canonical_exposed,
        "model_alias": model_alias,
        "model_id": model_id,
        "runtime_selection": dict(runtime_selection),
        "artifact_sync_dry_run_passed": artifact_sync_dry_run_passed,
        "checkpoint_resume_enabled": checkpoint_resume_enabled,
        "manifest_enabled": manifest_enabled,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "model_inference_triggered": False,
    }


def classify_b7_quality_gate(
    *,
    summary: Mapping[str, Any],
    per_vertical_quality: list[dict[str, Any]],
    completed_count: int,
    expected_count: int = B7_EXPECTED_PROMPT_COUNT,
    artifact_sync_verified: bool,
    telemetry_sample_count: int,
) -> dict[str, Any]:
    """Classify B7 quality and run-safety readiness."""

    vertical_rows = [row for row in per_vertical_quality if row.get("vertical") in VERTICALS]
    min_evidence = min(
        (_float(row.get("evidence_match_rate")) for row in vertical_rows),
        default=0.0,
    )
    min_grounded = min((_float(row.get("grounded_rate")) for row in vertical_rows), default=0.0)
    observed = {
        "json_valid_rate": _float(summary.get("json_valid_rate")),
        "generation_contract_valid_rate": _float(summary.get("generation_contract_valid_rate")),
        "evidence_match_rate": _float(summary.get("evidence_match_rate")),
        "grounded_rate": _float(summary.get("grounded_rate")),
        "safety_violation_count": _float(summary.get("safety_violation_count")),
        "truncation_rate": _float(summary.get("truncation_rate")),
        "vertical_evidence_match_rate_min": min_evidence,
        "vertical_grounded_rate_min": min_grounded,
        "completed_count": completed_count,
        "expected_count": expected_count,
        "telemetry_sample_count": telemetry_sample_count,
    }
    checks: dict[str, dict[str, object]] = {}
    for metric, threshold in B7_QUALITY_THRESHOLDS.items():
        value = observed[metric]
        if metric in {"safety_violation_count", "truncation_rate"}:
            passed = value <= threshold
            operator = "<="
        else:
            passed = value >= threshold
            operator = ">="
        checks[metric] = {
            "observed": value,
            "threshold": threshold,
            "operator": operator,
            "passed": passed,
        }
    checks["completed_expected_count"] = {
        "observed": completed_count,
        "threshold": expected_count,
        "operator": "==",
        "passed": completed_count == expected_count,
    }
    checks["artifact_sync_verified"] = {
        "observed": artifact_sync_verified,
        "threshold": True,
        "operator": "is",
        "passed": artifact_sync_verified,
    }
    checks["gpu_telemetry_captured"] = {
        "observed": telemetry_sample_count,
        "threshold": 1,
        "operator": ">=",
        "passed": telemetry_sample_count >= 1,
    }
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    quality_failed = [
        metric
        for metric in failed
        if metric
        not in {
            "artifact_sync_verified",
            "gpu_telemetry_captured",
            "completed_expected_count",
        }
    ]
    if not failed:
        status = "B7_CONTROLLED_1000_BASELINE_READY"
    elif not quality_failed and completed_count == expected_count:
        status = "B7_CONTROLLED_1000_RUN_SAFETY_BLOCKED"
    else:
        status = "B7_CONTROLLED_1000_BASELINE_BLOCKED"
    return {
        "status": status,
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
        "observed": observed,
        "benchmark_execution_readiness": "READY" if not failed else "NOT_READY",
        "next_api_load_probe_allowed": not failed,
        "runpod_readiness_claimed": False,
    }


def build_b7_load_and_cache_report(
    *,
    rows: list[dict[str, Any]],
    traffic_profile: str = B7_TRAFFIC_PROFILE,
    concurrency: int = B7_CONCURRENCY,
    request_arrival_mode: ArrivalMode = B7_REQUEST_ARRIVAL_MODE,
) -> dict[str, object]:
    """Build B7 ISL/OSL and cache-readiness metrics."""

    input_tokens = [int(row.get("input_tokens") or 0) for row in rows]
    output_tokens = [int(row.get("output_tokens") or 0) for row in rows]
    prompts = [str(row.get("prompt") or "") for row in rows]
    context_blocks = [_json_list(row.get("selected_context_ids")) for row in rows]
    load_report = build_load_profile_report(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        traffic_profile=traffic_profile,
        concurrency=concurrency,
        request_arrival_mode=request_arrival_mode,
    )
    cache = calculate_cache_readiness(
        prompts=prompts,
        context_blocks=context_blocks,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        context_window_tokens=4096,
    )
    return {
        "load_profile": load_report,
        "cache_readiness": cache.to_dict(),
    }


def build_b7_runtime_projection(
    *,
    measured_prompt_count: int,
    measured_wall_seconds: float,
    mean_latency_ms: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    selected_full_matrix_config_count: int = 4,
) -> dict[str, object]:
    """Project B7 runtime for larger controlled prompt counts."""

    selected_full_matrix_prompt_count = 10_000 * selected_full_matrix_config_count
    projection = build_runtime_projections(
        measured_prompt_count=measured_prompt_count,
        measured_wall_seconds=measured_wall_seconds,
        mean_latency_ms=mean_latency_ms,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        target_prompt_counts=(2_000, 10_000, selected_full_matrix_prompt_count),
    )
    rows = []
    projection_rows = cast(list[dict[str, object]], projection["projections"])
    for row in projection_rows:
        projected = dict(row)
        seconds = _float(projected.get("estimated_seconds_from_measured_throughput"))
        projected["estimated_hours_from_measured_throughput"] = seconds / 3600.0
        rows.append(projected)
    projection["projections"] = rows
    projection["requested_projection_targets"] = {
        "controlled_2000_prompt_run": 2_000,
        "final_10000_prompt_single_config": 10_000,
        "selected_full_matrix_config_count": selected_full_matrix_config_count,
        "selected_full_matrix_prompt_count": selected_full_matrix_prompt_count,
    }
    projection["runpod_readiness_claimed"] = False
    projection["runpod_cost_projection_blocked_reason"] = (
        "RunPod hourly price registry and throughput calibration are not both registered."
    )
    return projection
