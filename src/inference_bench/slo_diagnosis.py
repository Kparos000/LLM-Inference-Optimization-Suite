"""Failed-SLO-only deterministic diagnosis engine."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from inference_bench.bottleneck_catalog import load_bottleneck_catalog
from inference_bench.slo_profiles import SelectedSlo, SloProfile, select_slos

METRIC_OBSERVATION_ALIASES = {
    "groundedness_min": ("grounded_rate", "groundedness"),
    "citation_accuracy_min": ("citation_accuracy_rate", "citation_accuracy"),
    "evidence_match_min": ("evidence_match_rate", "evidence_match"),
    "task_success_min": ("task_success_rate", "success_rate"),
    "format_validity_min": (
        "generation_contract_valid_rate",
        "format_valid_rate",
        "format_validity",
    ),
    "safety_violations_max": ("safety_violation_count",),
    "ttft_p50_ms_max": ("p50_ttft_ms", "ttft_p50_ms"),
    "ttft_p95_ms_max": ("p95_ttft_ms", "ttft_p95_ms"),
    "ttft_p99_ms_max": ("p99_ttft_ms", "ttft_p99_ms"),
    "itl_p50_ms_max": ("p50_itl_ms", "itl_p50_ms"),
    "itl_p95_ms_max": ("p95_itl_ms", "itl_p95_ms"),
    "itl_p99_ms_max": ("p99_itl_ms", "itl_p99_ms"),
    "tpot_p50_ms_max": ("p50_tpot_ms", "tpot_p50_ms"),
    "tpot_p95_ms_max": ("p95_tpot_ms", "tpot_p95_ms"),
    "tpot_p99_ms_max": ("p99_tpot_ms", "tpot_p99_ms"),
    "e2e_p50_ms_max": ("p50_e2e_latency_ms", "e2e_p50_ms"),
    "e2e_p95_ms_max": ("p95_e2e_latency_ms", "e2e_p95_ms"),
    "e2e_p99_ms_max": ("p99_e2e_latency_ms", "e2e_p99_ms"),
    "tokens_per_second_min": (
        "aggregate_tokens_per_second",
        "tokens_per_second",
    ),
    "gpu_utilization_min_pct": (
        "mean_gpu_utilization_percent",
        "gpu_utilization_percent",
    ),
    "gpu_memory_peak_gb_max": ("gpu_memory_peak_gb",),
    "cpu_utilization_max_pct": ("cpu_utilization_percent",),
    "ram_usage_gb_max": ("ram_usage_gb",),
    "retrieval_rounds_max": ("mean_retrieval_rounds", "retrieval_rounds"),
    "generation_attempts_max": ("mean_generation_attempts", "generation_attempts"),
    "repair_attempts_max": ("mean_repair_attempts", "repair_attempts"),
    "tool_calls_max": ("mean_tool_call_count", "tool_call_count"),
    "trace_completeness_min": ("trace_completeness",),
}
METRIC_TO_BOTTLENECK = {
    "groundedness_min": "low_groundedness",
    "evidence_match_min": "low_evidence_match",
    "citation_accuracy_min": "low_citation_accuracy",
    "format_validity_min": "low_contract_validity",
    "json_validity_min": "low_json_validity",
    "safety_violations_max": "safety_violations",
    "truncation_rate_max": "high_truncation_rate",
    "insufficient_evidence_accuracy_min": "insufficient_evidence_misuse",
    "candidate_recall_at_20_min": "low_candidate_recall",
    "candidate_recall_at_50_min": "low_candidate_recall",
    "final_recall_at_5_min": "low_final_recall",
    "mrr_min": "low_mrr",
    "retrieval_latency_ms_max": "high_retrieval_latency",
    "input_tokens_per_request_max": "high_input_tokens",
    "context_tokens_max": "excessive_context_tokens",
    "compression_recall_loss_max": "compression_recall_loss",
    "context_ordering_quality_min": "poor_context_ordering",
    "ttft_p50_ms_max": "high_ttft_p50",
    "ttft_p95_ms_max": "high_ttft_p95",
    "ttft_p99_ms_max": "high_ttft_p99",
    "tpot_p50_ms_max": "high_tpot",
    "tpot_p95_ms_max": "high_tpot",
    "tpot_p99_ms_max": "high_tpot",
    "itl_p95_ms_max": "high_itl_p95",
    "itl_p99_ms_max": "high_itl_p99",
    "e2e_p95_ms_max": "high_e2e_p95",
    "e2e_p99_ms_max": "high_e2e_p99",
    "queue_delay_ms_max": "high_queue_delay",
    "prefill_time_ms_max": "high_prefill_time",
    "decode_time_ms_max": "high_decode_time",
    "requests_per_second_min": "low_requests_per_second",
    "tokens_per_second_min": "low_tokens_per_second",
    "successful_requests_per_second_min": "low_successful_requests_per_second",
    "gpu_utilization_min_pct": "low_gpu_utilization",
    "gpu_memory_utilization_max_pct": "high_gpu_memory_utilization",
    "cpu_utilization_max_pct": "high_cpu_utilization",
    "ram_usage_gb_max": "high_ram_usage",
    "api_cost_per_request_usd_max": "high_api_cost_per_request",
    "gpu_cost_per_request_usd_max": "high_gpu_cost_per_request",
    "api_cost_per_successful_answer_usd_max": "high_cost_per_successful_answer",
    "gpu_cost_per_successful_answer_usd_max": "high_cost_per_successful_answer",
    "api_cost_per_grounded_successful_answer_usd_max": "high_cost_per_grounded_answer",
    "gpu_cost_per_grounded_successful_answer_usd_max": "high_cost_per_grounded_answer",
    "tokens_per_gpu_dollar_min": "low_tokens_per_gpu_dollar",
}


def _observation(run_metrics: dict[str, Any], metric_name: str) -> float | bool | None:
    for candidate in (metric_name, *METRIC_OBSERVATION_ALIASES.get(metric_name, ())):
        value = run_metrics.get(candidate)
        if value not in (None, ""):
            if isinstance(value, bool):
                return value
            return float(str(value))
    if metric_name == "gpu_memory_utilization_max_pct":
        used = run_metrics.get("max_gpu_memory_used_mb")
        total = run_metrics.get("gpu_memory_total_mb")
        if used not in (None, "") and total not in (None, "", 0):
            return float(str(used)) / float(str(total)) * 100.0
    if metric_name == "gpu_memory_peak_gb_max":
        used = run_metrics.get("max_gpu_memory_used_mb")
        if used not in (None, ""):
            return float(str(used)) / 1024.0
    return None


def _evaluate(selected: SelectedSlo, observed: float | bool) -> tuple[bool, float]:
    if selected.direction == "required":
        passed = bool(observed) is bool(selected.target)
        return passed, 0.0 if passed else -1.0
    target = float(selected.target)
    value = float(observed)
    if selected.direction == "min":
        return value >= target, value - target
    return value <= target, target - value


def _row(
    selected: SelectedSlo,
    *,
    status: str,
    observed: float | bool | None,
    gap: float | None,
) -> dict[str, Any]:
    payload = asdict(selected)
    payload.update({"status": status, "observed": observed, "gap": gap})
    return payload


def _normalized_severity(row: dict[str, Any]) -> float:
    target = float(row["target"])
    observed = float(row["observed"])
    if target == 0:
        return min(1.0, abs(observed))
    return min(1.0, abs(float(row["gap"])) / abs(target))


def _build_bottlenecks(
    failed_slos: list[dict[str, Any]],
    *,
    catalog_path: str,
) -> list[dict[str, Any]]:
    catalog = load_bottleneck_catalog(catalog_path)
    evidence_by_id: dict[str, list[dict[str, Any]]] = {}
    for failure in failed_slos:
        bottleneck_id = METRIC_TO_BOTTLENECK.get(str(failure["metric_name"]))
        if bottleneck_id is None:
            continue
        evidence_by_id.setdefault(bottleneck_id, []).append(
            {
                "slo_id": failure["id"],
                "group": failure["group"],
                "metric_name": failure["metric_name"],
                "target": failure["target"],
                "observed": failure["observed"],
                "gap": failure["gap"],
            }
        )
    bottlenecks: list[dict[str, Any]] = []
    for bottleneck_id, evidence in sorted(evidence_by_id.items()):
        definition = catalog[bottleneck_id]
        available_metric_names = {str(item["metric_name"]) for item in evidence}
        required = set(definition.required_metrics)
        coverage = len(available_metric_names & required) / max(len(required), 1)
        bottlenecks.append(
            {
                "id": bottleneck_id,
                "category": definition.category,
                "description": definition.description,
                "severity": round(
                    max(
                        _normalized_severity(item)
                        for item in failed_slos
                        if item["metric_name"] in available_metric_names
                    ),
                    6,
                ),
                "confidence": round(max(0.5, min(1.0, coverage)), 6),
                "evidence": evidence,
                "possible_causes": list(definition.possible_causes),
                "compatible_optimizations": list(definition.compatible_optimizations),
                "severity_logic": definition.severity_logic,
                "confidence_logic": definition.confidence_logic,
            }
        )
    return bottlenecks


def diagnose_slos(
    *,
    run_metrics: dict[str, Any],
    profile: SloProfile,
    experiment_config: dict[str, Any],
    model_metadata: dict[str, Any],
    hardware_profile: dict[str, Any],
    engine: str,
    memory_mode: str,
    vertical: str,
    telemetry_available: bool,
    gpu_hourly_price: float | None = None,
    backend_type: str | None = None,
    bottleneck_catalog_path: str = "configs/bottleneck_catalog.yaml",
    optimization_catalog_path: str = "configs/optimization_catalog.yaml",
) -> dict[str, Any]:
    """Evaluate selected SLOs, then diagnose only observed failures."""

    selected_slos = select_slos(
        profile,
        vertical=vertical,
        memory_mode=memory_mode,
        engine=engine,
        hardware_name=str(
            hardware_profile.get("hardware_alias") or hardware_profile.get("name") or ""
        ),
        telemetry_available=telemetry_available,
        gpu_hourly_price=gpu_hourly_price,
        backend_type=backend_type,
    )
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for selected in selected_slos:
        if not selected.applicable:
            not_applicable.append(_row(selected, status="NOT_APPLICABLE", observed=None, gap=None))
            continue
        observed = _observation(run_metrics, selected.metric_name)
        if observed is None:
            unavailable.append(_row(selected, status="UNAVAILABLE", observed=None, gap=None))
            continue
        metric_passed, gap = _evaluate(selected, observed)
        target_rows = passed if metric_passed else failed
        target_rows.append(
            _row(
                selected,
                status="PASS" if metric_passed else "FAILED",
                observed=observed,
                gap=round(gap, 6),
            )
        )

    bottlenecks = _build_bottlenecks(failed, catalog_path=bottleneck_catalog_path)
    diagnosis: dict[str, Any] = {
        "profile": {
            "name": profile.name,
            "priority_mode": profile.priority_mode,
            "enabled_groups": list(profile.enabled_groups),
            "disabled_groups": list(profile.disabled_groups),
        },
        "context": {
            "vertical": vertical,
            "engine": engine,
            "memory_mode": memory_mode,
            "telemetry_available": telemetry_available,
            "gpu_hourly_price": gpu_hourly_price,
            "experiment_config": experiment_config,
            "model_metadata": model_metadata,
            "hardware_profile": hardware_profile,
            "run_metrics": run_metrics,
        },
        "selected_slos": [asdict(item) for item in selected_slos],
        "passed_slos": passed,
        "failed_slos": failed,
        "not_applicable_slos": not_applicable,
        "unavailable_metrics": unavailable,
        "bottlenecks": bottlenecks,
        "confidence_scores": {str(item["id"]): item["confidence"] for item in bottlenecks},
        "evidence_used": {str(item["id"]): item["evidence"] for item in bottlenecks},
        "diagnosis_scope": "failed_slos_only",
        "diagnosed_failed_slo_count": len(failed),
        "llm_used": False,
    }
    from inference_bench.optimization_recommender import recommend_optimizations

    recommendation = recommend_optimizations(diagnosis, catalog_path=optimization_catalog_path)
    diagnosis["recommended_optimizations"] = recommendation["recommendations"]
    diagnosis["incompatible_optimizations"] = recommendation["incompatible_optimizations"]
    diagnosis["already_active_capabilities"] = recommendation["already_active_capabilities"]
    diagnosis["next_experiment_suggestions"] = recommendation["next_experiment_suggestions"]
    diagnosis["primary_recommendation"] = recommendation["primary_recommendation"]
    return diagnosis
