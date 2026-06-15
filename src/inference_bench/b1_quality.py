"""Quality gates, comparisons, and runtime projections for Phase B1."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

QUALITY_GATE_THRESHOLDS = {
    "json_valid_rate": 0.95,
    "generation_contract_valid_rate": 0.85,
    "evidence_match_rate": 0.60,
    "grounded_rate": 0.60,
    "safety_violation_count": 0,
}

QUALITY_METRICS = (
    "json_valid_rate",
    "generation_contract_valid_rate",
    "evidence_id_presence_rate",
    "evidence_match_rate",
    "grounded_rate",
    "safety_violation_count",
    "truncation_rate",
)

LATENCY_METRICS = (
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_e2e_latency_ms",
    "mean_total_tokens_per_second",
)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        msg = "boolean is not a numeric metric"
        raise ValueError(msg)
    return float(str(value))


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(str(value))


def build_quality_gate(summary: Mapping[str, object]) -> dict[str, object]:
    """Evaluate the fixed B1 quality gate without modifying evaluator semantics."""

    checks: dict[str, dict[str, object]] = {}
    for metric, threshold in QUALITY_GATE_THRESHOLDS.items():
        raw_value = summary.get(metric)
        if raw_value is None:
            checks[metric] = {
                "observed": None,
                "threshold": threshold,
                "operator": "==" if metric == "safety_violation_count" else ">=",
                "passed": False,
                "reason": "metric_missing",
            }
            continue
        observed = _float_value(raw_value)
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
            "reason": None if passed else "threshold_not_met",
        }

    failed_metrics = [metric for metric, check in checks.items() if not bool(check["passed"])]
    return {
        "status": "PASSED" if not failed_metrics else "QUALITY_BLOCKED",
        "passed": not failed_metrics,
        "checks": checks,
        "failed_metrics": failed_metrics,
    }


def build_per_vertical_quality(
    evaluation_rows: Sequence[Mapping[str, object]],
    result_rows: Sequence[Mapping[str, object]],
    *,
    verticals: Sequence[str],
) -> list[dict[str, object]]:
    """Aggregate unchanged evaluator output by vertical."""

    result_by_prompt = {str(row.get("prompt_id") or ""): row for row in result_rows}
    output: list[dict[str, object]] = []
    for vertical in verticals:
        rows = [
            row
            for row in evaluation_rows
            if str(
                row.get("vertical")
                or result_by_prompt.get(str(row.get("prompt_id") or ""), {}).get("vertical")
                or ""
            )
            == vertical
        ]
        total = len(rows)
        json_valid = sum(1 for row in rows if _bool_value(row.get("json_validity")))
        contract_valid = sum(1 for row in rows if _bool_value(row.get("generation_contract_valid")))
        evidence_presence = sum(1 for row in rows if _bool_value(row.get("evidence_id_presence")))
        evidence_match = sum(1 for row in rows if _bool_value(row.get("evidence_match")))
        grounded = sum(1 for row in rows if _bool_value(row.get("groundedness")))
        safety = sum(1 for row in rows if _bool_value(row.get("safety_violation")))
        truncation = sum(
            1
            for row in rows
            if _bool_value(
                result_by_prompt.get(str(row.get("prompt_id") or ""), {}).get("truncation_detected")
            )
        )
        output.append(
            {
                "vertical": vertical,
                "row_count": total,
                "json_valid_count": json_valid,
                "json_valid_rate": _rate(json_valid, total),
                "generation_contract_valid_count": contract_valid,
                "generation_contract_valid_rate": _rate(contract_valid, total),
                "evidence_id_presence_count": evidence_presence,
                "evidence_id_presence_rate": _rate(evidence_presence, total),
                "evidence_match_count": evidence_match,
                "evidence_match_rate": _rate(evidence_match, total),
                "grounded_count": grounded,
                "grounded_rate": _rate(grounded, total),
                "safety_violation_count": safety,
                "safety_violation_rate": _rate(safety, total),
                "truncation_count": truncation,
                "truncation_rate": _rate(truncation, total),
            }
        )
    return output


def build_root_cause_analysis(
    *,
    gate: Mapping[str, object],
    per_vertical_quality: Sequence[Mapping[str, object]],
    result_rows: Sequence[Mapping[str, object]],
    evaluation_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe observed failure concentrations without inventing causes."""

    raw_failed_metrics = gate.get("failed_metrics")
    failed_metrics = (
        [str(value) for value in raw_failed_metrics]
        if isinstance(raw_failed_metrics, Sequence) and not isinstance(raw_failed_metrics, str)
        else []
    )
    parse_error_counts: dict[str, int] = {}
    token_sources: dict[str, int] = {}
    safety_term_counts: dict[str, int] = {}
    evidence_mismatch_with_partial_required_match = 0
    evidence_mismatch_with_no_required_match = 0
    evidence_mismatch_by_vertical: dict[str, int] = {}
    for row in result_rows:
        parse_error = str(row.get("parse_error_type") or "none")
        parse_error_counts[parse_error] = parse_error_counts.get(parse_error, 0) + 1
        token_source = str(row.get("token_count_source") or "unknown")
        token_sources[token_source] = token_sources.get(token_source, 0) + 1
    for row in per_vertical_quality:
        vertical = str(row.get("vertical") or "unknown")
        evidence_mismatch_by_vertical[vertical] = _int_value(
            row.get("row_count") or 0
        ) - _int_value(row.get("evidence_match_count") or 0)
    for row in evaluation_rows:
        if not _bool_value(row.get("evidence_match")):
            evidence_ids = row.get("evidence_ids_found")
            if isinstance(evidence_ids, Sequence) and not isinstance(evidence_ids, str):
                if evidence_ids:
                    evidence_mismatch_with_partial_required_match += 1
                else:
                    evidence_mismatch_with_no_required_match += 1
        safety_terms = row.get("safety_violation_terms")
        if isinstance(safety_terms, Sequence) and not isinstance(safety_terms, str):
            for term in safety_terms:
                key = str(term)
                safety_term_counts[key] = safety_term_counts.get(key, 0) + 1

    weakest_verticals: dict[str, dict[str, object]] = {}
    for metric in failed_metrics:
        if metric == "safety_violation_count":
            affected = [
                row
                for row in per_vertical_quality
                if _int_value(row.get("safety_violation_count") or 0) > 0
            ]
            affected.sort(
                key=lambda row: _int_value(row.get("safety_violation_count") or 0),
                reverse=True,
            )
        else:
            affected = sorted(
                per_vertical_quality,
                key=lambda row: _float_value(row.get(metric) or 0.0),
            )
        if affected:
            weakest_verticals[metric] = dict(affected[0])

    return {
        "classification": "observed_failure_concentration",
        "failed_gate_metrics": failed_metrics,
        "weakest_vertical_by_failed_metric": weakest_verticals,
        "parse_error_counts": dict(sorted(parse_error_counts.items())),
        "token_count_sources": dict(sorted(token_sources.items())),
        "evidence_mismatch_by_vertical": dict(sorted(evidence_mismatch_by_vertical.items())),
        "evidence_mismatches_with_partial_required_match": (
            evidence_mismatch_with_partial_required_match
        ),
        "evidence_mismatches_with_no_required_match": evidence_mismatch_with_no_required_match,
        "safety_violation_term_counts": dict(sorted(safety_term_counts.items())),
        "interpretation_limits": [
            (
                "The unchanged deterministic evaluator identifies outcome failures, "
                "not semantic causality."
            ),
            "The 100-prompt smoke is a gate, not a statistically complete benchmark.",
            "Retrieval and gold data were held fixed and were not modified by B1.",
        ],
    }


def _numeric_delta(baseline: object, candidate: object) -> dict[str, float | None]:
    if baseline is None or candidate is None:
        return {"baseline": None, "candidate": None, "absolute_delta": None, "percent_delta": None}
    baseline_value = _float_value(baseline)
    candidate_value = _float_value(candidate)
    percent_delta = (
        (candidate_value - baseline_value) / baseline_value * 100.0 if baseline_value != 0 else None
    )
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "absolute_delta": candidate_value - baseline_value,
        "percent_delta": percent_delta,
    }


def _nested_value(payload: Mapping[str, object], *keys: str) -> object:
    current: object = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def build_b1_comparison(
    *,
    baseline_quality: Mapping[str, object],
    candidate_quality: Mapping[str, object],
    baseline_latency: Mapping[str, object],
    candidate_latency: Mapping[str, object],
    baseline_telemetry: Mapping[str, object] | None,
    candidate_telemetry: Mapping[str, object] | None,
    baseline_prompt_ids: set[str],
    candidate_prompt_ids: set[str],
) -> dict[str, object]:
    """Compare B1 with A1 while preserving sample-size caveats."""

    quality = {
        metric: _numeric_delta(baseline_quality.get(metric), candidate_quality.get(metric))
        for metric in QUALITY_METRICS
    }
    latency = {
        metric: _numeric_delta(baseline_latency.get(metric), candidate_latency.get(metric))
        for metric in LATENCY_METRICS
    }
    telemetry_sources = {
        "mean_gpu_utilization_percent": ("utilization_gpu_percent", "mean"),
        "max_gpu_utilization_percent": ("utilization_gpu_percent", "max"),
        "mean_gpu_memory_used_mb": ("memory_used_mb", "mean"),
        "max_gpu_memory_used_mb": ("memory_used_mb", "max"),
        "mean_power_draw_w": ("power_draw_w", "mean"),
        "max_power_draw_w": ("power_draw_w", "max"),
        "mean_temperature_c": ("temperature_c", "mean"),
        "max_temperature_c": ("temperature_c", "max"),
    }
    telemetry = (
        {
            metric: _numeric_delta(
                _nested_value(baseline_telemetry, *path),
                _nested_value(candidate_telemetry, *path),
            )
            for metric, path in telemetry_sources.items()
        }
        if baseline_telemetry is not None and candidate_telemetry is not None
        else {
            metric: {
                "baseline": None,
                "candidate": None,
                "absolute_delta": None,
                "percent_delta": None,
            }
            for metric in telemetry_sources
        }
    )
    overlap = baseline_prompt_ids & candidate_prompt_ids
    return {
        "baseline": "A1_Qwen2.5-0.5B_Instruct_vLLM",
        "candidate": "B1_Qwen2.5-1.5B_Instruct_vLLM",
        "comparison_scope": "same_hardware_engine_memory_mode_and_generation_settings",
        "prompt_scope": {
            "baseline_count": len(baseline_prompt_ids),
            "candidate_count": len(candidate_prompt_ids),
            "overlap_count": len(overlap),
            "baseline_is_subset_of_candidate": baseline_prompt_ids <= candidate_prompt_ids,
            "fully_prompt_matched": baseline_prompt_ids == candidate_prompt_ids,
            "caveat": (
                "A1 used 10 prompts per vertical and B1 uses 20. "
                "The A1 prompt IDs are expected to be a deterministic subset, "
                "so aggregate deltas also reflect the added 50 prompts."
            ),
        },
        "quality_deltas": quality,
        "latency_throughput_deltas": latency,
        "gpu_telemetry_deltas": telemetry,
    }


def _validate_projection_inputs(
    measured_prompt_count: int,
    measured_wall_seconds: float,
    latency_values: Sequence[float],
) -> None:
    if measured_prompt_count <= 0:
        msg = "measured_prompt_count must be > 0"
        raise ValueError(msg)
    if measured_wall_seconds <= 0:
        msg = "measured_wall_seconds must be > 0"
        raise ValueError(msg)
    if any(value < 0 for value in latency_values):
        msg = "latency values must be >= 0"
        raise ValueError(msg)


def build_b1_runtime_projection(
    *,
    measured_prompt_count: int,
    measured_wall_seconds: float,
    mean_latency_ms: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    runpod_targets: Mapping[str, Mapping[str, object]],
    target_prompt_counts: Sequence[int] = (500, 2500, 5000, 10000),
    full_matrix_config_count: int = 8,
) -> dict[str, object]:
    """Project B1 runtimes and leave external GPU costs null until configured."""

    _validate_projection_inputs(
        measured_prompt_count,
        measured_wall_seconds,
        (mean_latency_ms, p50_latency_ms, p95_latency_ms),
    )
    if full_matrix_config_count <= 0:
        msg = "full_matrix_config_count must be > 0"
        raise ValueError(msg)
    requests_per_second = measured_prompt_count / measured_wall_seconds
    projections: list[dict[str, object]] = []
    for prompt_count in target_prompt_counts:
        if prompt_count <= 0:
            msg = "target prompt counts must be > 0"
            raise ValueError(msg)
        projections.append(
            {
                "prompt_count": prompt_count,
                "estimated_seconds_from_measured_throughput": prompt_count / requests_per_second,
                "estimated_seconds_from_mean_latency": prompt_count * mean_latency_ms / 1000.0,
                "estimated_seconds_from_p50_latency": prompt_count * p50_latency_ms / 1000.0,
                "estimated_seconds_from_p95_latency": prompt_count * p95_latency_ms / 1000.0,
            }
        )

    full_matrix_prompt_count = 10000 * full_matrix_config_count
    full_matrix_seconds = full_matrix_prompt_count / requests_per_second
    runpod: dict[str, dict[str, object]] = {}
    for gpu_name, target in runpod_targets.items():
        raw_price = target.get("hourly_price_usd")
        raw_multiplier = target.get("throughput_multiplier_vs_rtx3070")
        price = _float_value(raw_price) if raw_price is not None else None
        multiplier = _float_value(raw_multiplier) if raw_multiplier is not None else None
        if price is not None and price < 0:
            msg = f"{gpu_name} hourly_price_usd must be >= 0"
            raise ValueError(msg)
        if multiplier is not None and multiplier <= 0:
            msg = f"{gpu_name} throughput_multiplier_vs_rtx3070 must be > 0"
            raise ValueError(msg)
        estimated_seconds = full_matrix_seconds / multiplier if multiplier is not None else None
        estimated_cost = (
            estimated_seconds / 3600.0 * price
            if estimated_seconds is not None and price is not None
            else None
        )
        runpod[gpu_name] = {
            "hourly_price_usd": price,
            "throughput_multiplier_vs_rtx3070": multiplier,
            "estimated_full_matrix_seconds": estimated_seconds,
            "estimated_full_matrix_hours": (
                estimated_seconds / 3600.0 if estimated_seconds is not None else None
            ),
            "estimated_full_matrix_cost_usd": estimated_cost,
            "status": (
                "estimated"
                if estimated_cost is not None
                else "placeholder_requires_price_and_measured_multiplier"
            ),
        }

    return {
        "projection_type": "b1_measured_concurrency_one_linear_estimate",
        "is_guarantee": False,
        "measured_prompt_count": measured_prompt_count,
        "measured_wall_seconds": measured_wall_seconds,
        "measured_requests_per_second": requests_per_second,
        "latency_reference_ms": {
            "mean": mean_latency_ms,
            "p50": p50_latency_ms,
            "p95": p95_latency_ms,
        },
        "rtx3070_prompt_projections": projections,
        "rtx3070_full_matrix_projection": {
            "config_count": full_matrix_config_count,
            "prompts_per_config": 10000,
            "total_prompt_executions": full_matrix_prompt_count,
            "estimated_seconds": full_matrix_seconds,
            "estimated_hours": full_matrix_seconds / 3600.0,
        },
        "runpod_full_matrix_placeholders": runpod,
        "assumptions": [
            "concurrency remains 1",
            "prompt and output length distributions remain comparable",
            "the server remains warm and free of competing GPU workloads",
            "network and queue conditions remain comparable",
            "RunPod estimates require measured throughput multipliers and current hourly prices",
        ],
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    """Return a mean for report construction without inventing missing samples."""

    return statistics.fmean(values) if values else None
