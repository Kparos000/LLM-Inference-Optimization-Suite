"""UI-facing optimization intelligence for the official Main_Inference_V1 run.

This module is intentionally an adapter over the existing deterministic SLO
diagnosis, bottleneck catalog, optimization catalog, negative-rule registry,
and recommender. It does not run inference and does not create optimized
experiment results.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from inference_bench.bottleneck_catalog import (
    BottleneckDefinition,
    load_bottleneck_catalog,
)
from inference_bench.config import ModelConfig, load_project_config
from inference_bench.optimization_catalog import (
    OptimizationDefinition,
    load_optimization_catalog,
)
from inference_bench.optimization_negative_rules import (
    OptimizationNegativeRule,
    load_optimization_negative_rules,
)
from inference_bench.slo_diagnosis import diagnose_slos
from inference_bench.slo_profiles import resolve_slo_profile

DEFAULT_EXPERIMENT_ROOT = Path("experiments/main/main_inference_v1")
DEFAULT_OUTPUT_ROOT = DEFAULT_EXPERIMENT_ROOT / "processed"
RUN_ID = "main_inference_v1"


@dataclass(frozen=True)
class ScorecardMetricMapping:
    """Map official Main_Inference scorecard labels into existing SLO semantics."""

    scorecard_label: str
    ui_metric_id: str
    metric_name: str
    status_field: str | None
    bottleneck_id: str | None
    group: str


METRIC_MAPPINGS: dict[str, ScorecardMetricMapping] = {
    "JSON validity": ScorecardMetricMapping(
        scorecard_label="JSON validity",
        ui_metric_id="json_validity",
        metric_name="json_validity_min",
        status_field="slo_json_validity",
        bottleneck_id="low_json_validity",
        group="quality",
    ),
    "Contract validity": ScorecardMetricMapping(
        scorecard_label="Contract validity",
        ui_metric_id="contract_validity",
        metric_name="format_validity_min",
        status_field="slo_contract_validity",
        bottleneck_id="low_contract_validity",
        group="quality",
    ),
    "Format validity": ScorecardMetricMapping(
        scorecard_label="Format validity",
        ui_metric_id="format_validity",
        metric_name="format_validity_min",
        status_field="slo_contract_validity",
        bottleneck_id="low_contract_validity",
        group="quality",
    ),
    "Evidence match": ScorecardMetricMapping(
        scorecard_label="Evidence match",
        ui_metric_id="evidence_match",
        metric_name="evidence_match_min",
        status_field="slo_evidence_match",
        bottleneck_id="low_evidence_match",
        group="quality",
    ),
    "Groundedness": ScorecardMetricMapping(
        scorecard_label="Groundedness",
        ui_metric_id="groundedness",
        metric_name="groundedness_min",
        status_field="slo_groundedness",
        bottleneck_id="low_groundedness",
        group="quality",
    ),
    "Safety findings": ScorecardMetricMapping(
        scorecard_label="Safety findings",
        ui_metric_id="safety_findings",
        metric_name="safety_violations_max",
        status_field="slo_safety",
        bottleneck_id="safety_violations",
        group="safety",
    ),
    "Runtime": ScorecardMetricMapping(
        scorecard_label="Runtime",
        ui_metric_id="runtime",
        metric_name="runtime_slo",
        status_field=None,
        bottleneck_id=None,
        group="runtime",
    ),
    "Cost": ScorecardMetricMapping(
        scorecard_label="Cost",
        ui_metric_id="cost",
        metric_name="cost_slo",
        status_field=None,
        bottleneck_id=None,
        group="cost",
    ),
}

QUALITY_STATUS_FIELDS = (
    "slo_contract_validity",
    "slo_evidence_match",
    "slo_groundedness",
    "slo_safety",
)


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        value = value.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        pass
    return value.as_posix()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "n/a"):
        return default
    return float(str(value))


def _int(value: object, default: int = 0) -> int:
    if value in (None, "", "n/a"):
        return default
    return int(float(str(value)))


def _parse_target(target: str) -> tuple[str, float | None]:
    normalized = target.strip()
    if normalized.startswith(">="):
        return "min", float(normalized.removeprefix(">=").strip())
    if normalized.startswith("<="):
        return "max", float(normalized.removeprefix("<=").strip())
    if normalized.startswith("="):
        return "max", float(normalized.removeprefix("=").strip())
    return "informational", None


def _normalized_severity(
    *,
    direction: str,
    target: float | None,
    observed: float,
    difference: float,
) -> float:
    if target is None:
        return 0.0
    if direction == "max" and target == 0:
        return min(1.0, abs(observed))
    if target == 0:
        return min(1.0, abs(difference))
    return round(min(1.0, abs(difference) / abs(target)), 6)


def _display_name(optimization_id: str) -> str:
    overrides = {
        "use_mm4_agentic_repair": "Use MM4 Bounded Agentic Repair",
        "switch_engine_to_vllm": "Switch Engine To vLLM",
        "switch_engine_to_sglang": "Switch Engine To SGLang",
        "switch_engine_to_tensorrt_llm": "Switch Engine To TensorRT-LLM",
        "enable_awq_int4": "Enable AWQ INT4",
        "enable_gptq_int4": "Enable GPTQ INT4",
        "enable_fp8_where_supported": "Enable FP8 Where Supported",
    }
    return overrides.get(
        optimization_id,
        optimization_id.replace("_", " ").title().replace("Json", "JSON"),
    )


def _expected_improvement(definition: OptimizationDefinition) -> dict[str, Any]:
    gain = dict(definition.expected_gain_range)
    if gain.get("basis") == "must_measure":
        summary = "Expected improvement must be measured in a controlled optimized rerun."
    else:
        summary = (
            f"Expected gain range: {gain.get('min_pct')}% to {gain.get('max_pct')}% "
            f"on basis {gain.get('basis')}."
        )
    return {
        "summary": summary,
        "improves": list(definition.improves),
        "expected_gain_range": gain,
    }


def _model_metadata(model: ModelConfig) -> dict[str, Any]:
    return asdict(model)


def _hardware_profile_for_row(row: dict[str, str]) -> dict[str, Any]:
    if row.get("backend_type") == "api_provider":
        return {
            "hardware_alias": "provider_managed",
            "name": "provider_managed",
            "capabilities": [],
        }
    return {
        "hardware_alias": "a100_sxm_80gb",
        "name": "NVIDIA A100-SXM4-80GB",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "vram_gb": 80,
        "gpu_count": 1,
        "capabilities": ["gpu"],
    }


def _hardware_type_for_row(row: dict[str, str]) -> str:
    return "provider_managed" if row.get("backend_type") == "api_provider" else "a100_sxm_80gb"


def _run_metrics_for_config(row: dict[str, str]) -> dict[str, Any]:
    return {
        "generation_contract_valid_rate": _float(row.get("generation_contract_valid_rate")),
        "format_valid_rate": _float(row.get("generation_contract_valid_rate")),
        "evidence_match_rate": _float(row.get("evidence_match_rate")),
        "grounded_rate": _float(row.get("grounded_rate")),
        "safety_violation_count": _int(row.get("safety_violation_count")),
        "tokens_per_second": _float(row.get("mean_total_tokens_per_second")),
        "mean_ttft_ms": _float(row.get("mean_ttft_ms")),
        "mean_tpot_ms": _float(row.get("mean_tpot_ms")),
        "mean_e2e_latency_ms": _float(row.get("mean_e2e_latency_ms")),
    }


def _build_config_diagnoses(
    slo_summary_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    project = load_project_config()
    profile = resolve_slo_profile(enabled_groups=["quality"])
    diagnoses: dict[str, dict[str, Any]] = {}
    for row in slo_summary_rows:
        alias = str(row["model_alias"])
        model = project.resolve_model_config(alias)
        diagnosis = diagnose_slos(
            run_metrics=_run_metrics_for_config(row),
            profile=profile,
            experiment_config={
                "experiment_id": row["config_id"],
                "concurrency": _int(row.get("concurrency"), default=1),
            },
            model_metadata=_model_metadata(model),
            hardware_profile=_hardware_profile_for_row(row),
            engine=str(row["engine"]),
            memory_mode=str(row["memory_mode"]),
            vertical="finance",
            telemetry_available=row.get("backend_type") == "self_hosted_gpu",
            gpu_hourly_price=1.49 if row.get("backend_type") == "self_hosted_gpu" else None,
            backend_type=str(row.get("backend_type") or ""),
        )
        diagnosis["main_inference_config_row"] = dict(row)
        diagnoses[str(row["config_id"])] = diagnosis
    return diagnoses


def _run_facts(
    *,
    slo_report: dict[str, Any],
    eval_report: dict[str, Any],
    context_preflight: dict[str, Any],
) -> dict[str, Any]:
    summary = cast(dict[str, Any], eval_report.get("summary", {}))
    latency = cast(dict[str, Any], eval_report.get("latency_summary", {}))
    gpu = cast(dict[str, Any], eval_report.get("gpu_summary", {}))
    max_memory = _stat_number(gpu.get("max_memory_used_mb"), preferred_key="max")
    total_memory = _stat_number(gpu.get("memory_total_mb"), preferred_key="max", default=1.0)
    return {
        "quality_failed": slo_report.get("quality_slo_verdict") == "FAIL",
        "safety_failed": slo_report.get("safety_slo_verdict") == "FAIL",
        "runtime_failed": slo_report.get("runtime_slo_verdict") == "FAIL",
        "cost_failed": slo_report.get("cost_slo_verdict") == "FAIL",
        "contract_failed": float(summary.get("generation_contract_valid_rate") or 1.0) < 0.95,
        "evidence_failed": float(summary.get("evidence_match_rate") or 1.0) < 0.95,
        "groundedness_failed": float(summary.get("grounded_rate") or 1.0) < 0.98,
        "json_failed": float(summary.get("json_valid_rate") or 1.0) < 0.95,
        "truncation_rate": float(summary.get("truncation_rate") or 0.0),
        "ttft_failed": False,
        "tpot_failed": False,
        "latency_passed": slo_report.get("runtime_slo_verdict") == "PASS",
        "gpu_cost_priced": (
            cast(dict[str, Any], slo_report.get("cost_report", {})).get(
                "self_hosted_gpu_hourly_price_usd"
            )
            is not None
        ),
        "context_absent_or_partial_count": int(
            context_preflight.get("unrecoverable_row_count") or 0
        ),
        "context_absent_count": int(context_preflight.get("absent_count") or 0),
        "checkpoint_resume_available": True,
        "gpu_telemetry_available": int(gpu.get("sample_count") or 0) > 0,
        "prefix_reuse_metrics_available": "prefix_reuse_potential" in summary,
        "cache_hit_telemetry_available": "prefix_cache_hit_rate" in summary,
        "queue_prefill_decode_telemetry_available": all(
            key in latency for key in ("queue_delay_ms", "prefill_time_ms", "decode_time_ms")
        ),
        "single_gpu": True,
        "draft_model_registered": False,
        "acceptance_rate_telemetry_available": False,
        "model_fits_one_gpu": max_memory < total_memory,
    }


def _stat_number(value: object, *, preferred_key: str, default: float = 0.0) -> float:
    if isinstance(value, dict):
        raw = value.get(preferred_key)
        if raw is None:
            raw = next(iter(value.values()), default)
        return _float(raw, default=default)
    return _float(value, default=default)


def _negative_condition_triggered(
    condition: str,
    *,
    run_facts: dict[str, Any],
    failed_slo: dict[str, Any],
    definition: OptimizationDefinition,
) -> tuple[bool, str]:
    metric_id = str(failed_slo["metric_id"])
    condition_lower = condition.lower()
    if "quality gate is already failing" in condition_lower:
        return bool(run_facts["quality_failed"]), "Main_Inference quality SLO verdict is FAIL."
    if "calibrated kernels or quantized weights are unavailable" in condition_lower:
        triggered = definition.implementation_status == "planned"
        return triggered, "The catalog marks this optimization as planned for this project."
    if "target model already fits" in condition_lower:
        return bool(run_facts["model_fits_one_gpu"]), "A100 telemetry shows the model fits in VRAM."
    if "comparison would mix model precision" in condition_lower:
        return True, "This UI phase allows one diagnosed factor at a time."
    if "prefix reuse potential is low" in condition_lower:
        triggered = not bool(run_facts["prefix_reuse_metrics_available"])
        return triggered, "No prefix-reuse metric is present in the Main_Inference artifacts."
    if "cache-hit telemetry cannot be collected" in condition_lower:
        triggered = not bool(run_facts["cache_hit_telemetry_available"])
        return triggered, "No prefix-cache hit telemetry exists for Main_Inference_V1."
    if "no compatible draft model is registered" in condition_lower:
        return not bool(run_facts["draft_model_registered"]), "No draft model registry is present."
    if "quality or json/contract validity is below target" in condition_lower:
        triggered = bool(run_facts["quality_failed"] or run_facts["contract_failed"])
        return triggered, "Quality/contract SLOs are below target."
    if "tpot is not the diagnosed bottleneck" in condition_lower:
        return metric_id != "tpot", f"The selected failed SLO is {metric_id}, not TPOT."
    if "acceptance-rate telemetry cannot be measured" in condition_lower:
        triggered = not bool(run_facts["acceptance_rate_telemetry_available"])
        return triggered, "No speculative-decoding acceptance-rate telemetry exists."
    if "model fits and meets latency on one gpu" in condition_lower:
        triggered = bool(run_facts["model_fits_one_gpu"] and run_facts["latency_passed"])
        return triggered, "The A100 run passed runtime SLOs and fit the active model."
    if "multi-gpu interconnect" in condition_lower:
        return bool(run_facts["single_gpu"]), "Main_Inference_V1 was measured on one A100 GPU."
    if "prefill time or ttft is not the diagnosed bottleneck" in condition_lower:
        return metric_id != "ttft", f"The selected failed SLO is {metric_id}, not TTFT."
    if "queue, prefill, and decode telemetry are unavailable" in condition_lower:
        triggered = not bool(run_facts["queue_prefill_decode_telemetry_available"])
        return triggered, "Main_Inference telemetry does not split queue/prefill/decode time."
    if "runtime deployment does not support separate prefill/decode workers" in condition_lower:
        return True, "No deployed disaggregated prefill/decode runtime is registered."
    if "retrieval final recall or evidence match is below target" in condition_lower:
        return bool(run_facts["evidence_failed"]), "Evidence match is below target."
    if "token savings would be measured without a groundedness guard" in condition_lower:
        return False, "The project has groundedness evaluation, so this guard is available."
    if "quality gate has not passed at concurrency one" in condition_lower:
        return bool(run_facts["quality_failed"]), "Quality SLOs failed in Main_Inference_V1."
    if "p95 or p99 latency is already above target" in condition_lower:
        return bool(run_facts["runtime_failed"]), "Runtime SLO verdict did not fail."
    if "checkpoint/resume is missing" in condition_lower:
        triggered = not bool(run_facts["checkpoint_resume_available"])
        return triggered, "A checkpoint artifact is present for Main_Inference_V1."
    if "gpu memory headroom or oom telemetry is unavailable" in condition_lower:
        triggered = not bool(run_facts["gpu_telemetry_available"])
        return triggered, "A100 GPU telemetry is present."
    if "retrieval/gold evidence is absent from rendered context" in condition_lower:
        triggered = int(run_facts["context_absent_or_partial_count"]) > 0
        return triggered, "Context preflight reports partial or absent required evidence rows."
    if "provider or gpu cost is unpriced" in condition_lower:
        return not bool(run_facts["gpu_cost_priced"]), "GPU hourly price is recorded."
    if "smaller-model failures are caused by truncation or prompt-contract bugs" in condition_lower:
        triggered = bool(run_facts["contract_failed"] or run_facts["truncation_rate"] > 0)
        return triggered, "Contract validity is a failed SLO."
    return False, "No Main_Inference fact maps this advisory condition to true."


def _negative_rule_checks(
    optimization_id: str,
    *,
    definition: OptimizationDefinition,
    rules: dict[str, OptimizationNegativeRule],
    run_facts: dict[str, Any],
    failed_slo: dict[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rule in rules.values():
        if optimization_id not in set(rule.optimization_ids):
            continue
        for condition in rule.when_not_to_use:
            triggered, evidence = _negative_condition_triggered(
                condition,
                run_facts=run_facts,
                failed_slo=failed_slo,
                definition=definition,
            )
            checks.append(
                {
                    "rule_id": rule.id,
                    "condition": condition,
                    "triggered": triggered,
                    "evidence": evidence,
                }
            )
    return checks


def _triggered_negative_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [check for check in checks if bool(check["triggered"])]


def _context_for_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "config_id": row["config_id"],
        "engine": row["engine"],
        "runtime": row["runtime"],
        "backend_type": row["backend_type"],
        "memory_mode": row["memory_mode"],
        "model_alias": row["model_alias"],
        "model_id": row["model_id"],
        "hardware": _hardware_type_for_row(row),
        "concurrency": _int(row.get("concurrency")),
    }


def _supports_metric(row: dict[str, str], mapping: ScorecardMetricMapping) -> bool:
    if mapping.status_field is None:
        return False
    return row.get(mapping.status_field) == "FAIL"


def _candidate_ids_for_bottleneck(
    bottleneck: BottleneckDefinition,
    catalog: dict[str, OptimizationDefinition],
) -> list[str]:
    ids = list(bottleneck.compatible_optimizations)
    for optimization_id, definition in catalog.items():
        if bottleneck.id in set(definition.compatible_bottlenecks) and optimization_id not in ids:
            ids.append(optimization_id)
    return ids


def _aggregate_recommendation_support(
    *,
    failed_slo: dict[str, Any],
    mapping: ScorecardMetricMapping,
    source_rows: list[dict[str, str]],
    diagnoses_by_config: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    support: dict[str, list[dict[str, Any]]] = {}
    incompatible: dict[str, list[str]] = {}
    bottleneck_id = str(failed_slo["bottleneck"]["id"])
    for row in source_rows:
        if not _supports_metric(row, mapping):
            continue
        diagnosis = diagnoses_by_config[row["config_id"]]
        for recommendation in cast(list[dict[str, Any]], diagnosis["recommended_optimizations"]):
            if bottleneck_id not in set(cast(list[str], recommendation["matched_bottlenecks"])):
                continue
            support.setdefault(str(recommendation["optimization_id"]), []).append(
                {
                    "context": _context_for_row(row),
                    "rank_score": recommendation["rank_score"],
                    "reason": recommendation["reason"],
                }
            )
        for item in cast(list[dict[str, str]], diagnosis["incompatible_optimizations"]):
            incompatible.setdefault(str(item["optimization_id"]), []).append(str(item["reason"]))
    return support, incompatible


def _requires_gpu_rerun(contexts: list[dict[str, Any]]) -> bool:
    return any(str(item["context"]["backend_type"]) == "self_hosted_gpu" for item in contexts)


def _requires_api_rerun(contexts: list[dict[str, Any]]) -> bool:
    return any(str(item["context"]["backend_type"]) == "api_provider" for item in contexts)


def _unique_context_field(contexts: list[dict[str, Any]], field: str) -> list[str]:
    return sorted({str(item["context"][field]) for item in contexts})


def _build_option(
    *,
    failed_slo: dict[str, Any],
    optimization_id: str,
    definition: OptimizationDefinition,
    contexts: list[dict[str, Any]],
    negative_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "optimization_id": optimization_id,
        "display_name": _display_name(optimization_id),
        "explanation": definition.description,
        "why_it_applies": (
            f"Targets bottleneck {failed_slo['bottleneck']['id']} caused by failed "
            f"SLO {failed_slo['metric_label']}."
        ),
        "expected_improvement": _expected_improvement(definition),
        "expected_tradeoffs": list(definition.may_hurt),
        "risks": {
            "quality_risk": definition.quality_risk,
            "cost_risk": definition.cost_risk,
            "safety_notes": list(definition.experiment_safety_notes),
        },
        "implementation_status": definition.implementation_status,
        "application_method": definition.application_method,
        "requires_gpu_rerun": _requires_gpu_rerun(contexts),
        "requires_api_rerun": _requires_api_rerun(contexts),
        "compatible_engines": _unique_context_field(contexts, "engine"),
        "compatible_memory_modes": _unique_context_field(contexts, "memory_mode"),
        "compatible_hardware": _unique_context_field(contexts, "hardware"),
        "compatible_models": _unique_context_field(contexts, "model_alias"),
        "compatible_config_count": len({str(item["context"]["config_id"]) for item in contexts}),
        "source_contexts": contexts,
        "negative_rule_checks": negative_checks,
        "source_catalog": "configs/optimization_catalog.yaml",
    }


def _build_rejection(
    *,
    optimization_id: str,
    definition: OptimizationDefinition | None,
    reason: str,
    negative_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    triggered = _triggered_negative_checks(negative_checks)
    return {
        "optimization_id": optimization_id,
        "display_name": _display_name(optimization_id),
        "reason_rejected": reason,
        "negative_rule_triggered": triggered[0]["rule_id"] if triggered else None,
        "negative_rule_checks": negative_checks,
        "implementation_status": (
            definition.implementation_status if definition is not None else "unknown"
        ),
    }


def _failure_explanation(
    *,
    row: dict[str, str],
    mapping: ScorecardMetricMapping,
    bottleneck: BottleneckDefinition,
) -> str:
    return (
        f"{mapping.scorecard_label} missed the official Main_Inference_V1 target "
        f"({row['target']}) with observed value {row['observed_main_inference_v1_value']}. "
        f"The existing bottleneck catalog maps this to {bottleneck.id}: "
        f"{bottleneck.description}"
    )


def build_ui_diagnosis(
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
) -> dict[str, Any]:
    """Build UI diagnosis and option payloads without running inference."""

    root = _repo_path(experiment_root)
    processed = root / "processed"
    raw = root / "raw"
    slo_report = _read_json(processed / "main_inference_v1_slo_report.json")
    eval_report = _read_json(processed / "main_inference_v1_eval_report.json")
    context_preflight = _read_json(processed / "main_inference_v1_context_preflight_report.json")
    manifest = _read_json(raw / "main_inference_v1_manifest.json")
    scorecard_rows = _read_csv(processed / "main_inference_v1_slo_scorecard.csv")
    slo_summary_rows = _read_csv(processed / "main_inference_v1_slo_summary.csv")
    catalog = load_optimization_catalog()
    bottlenecks = load_bottleneck_catalog()
    negative_rules = load_optimization_negative_rules()
    diagnoses_by_config = _build_config_diagnoses(slo_summary_rows)
    run_facts = _run_facts(
        slo_report=slo_report,
        eval_report=eval_report,
        context_preflight=context_preflight,
    )

    failed_slos: list[dict[str, Any]] = []
    options_by_slo: dict[str, list[dict[str, Any]]] = {}
    rejected_by_slo: dict[str, list[dict[str, Any]]] = {}
    for row in scorecard_rows:
        if row.get("status") != "FAIL":
            continue
        mapping = METRIC_MAPPINGS.get(str(row["slo_metric"]))
        if mapping is None or mapping.bottleneck_id is None:
            continue
        bottleneck = bottlenecks[mapping.bottleneck_id]
        direction, target_value = _parse_target(row["target"])
        observed = _float(row["observed_main_inference_v1_value"])
        difference = _float(row["difference"])
        severity = _normalized_severity(
            direction=direction,
            target=target_value,
            observed=observed,
            difference=difference,
        )
        failed_slo = {
            "slo_id": f"{RUN_ID}.{mapping.ui_metric_id}",
            "metric_id": mapping.ui_metric_id,
            "metric_label": mapping.scorecard_label,
            "metric_name": mapping.metric_name,
            "group": mapping.group,
            "target": row["target"],
            "observed": observed,
            "difference": difference,
            "status": "FAIL",
            "severity": severity,
            "confidence": 1.0,
            "bottleneck": {
                "id": bottleneck.id,
                "category": bottleneck.category,
                "description": bottleneck.description,
                "possible_causes": list(bottleneck.possible_causes),
                "confidence": 1.0,
            },
            "explanation": _failure_explanation(
                row=row,
                mapping=mapping,
                bottleneck=bottleneck,
            ),
            "source_artifact": (
                "experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv"
            ),
        }
        failed_slos.append(failed_slo)

        source_rows = [
            source_row for source_row in slo_summary_rows if _supports_metric(source_row, mapping)
        ]
        support, incompatible = _aggregate_recommendation_support(
            failed_slo=failed_slo,
            mapping=mapping,
            source_rows=source_rows,
            diagnoses_by_config=diagnoses_by_config,
        )
        allowed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for optimization_id in _candidate_ids_for_bottleneck(bottleneck, catalog):
            definition = catalog.get(optimization_id)
            if definition is None:
                rejected.append(
                    _build_rejection(
                        optimization_id=optimization_id,
                        definition=None,
                        reason=(
                            "Optimization is referenced by the bottleneck catalog but "
                            "missing from the optimization catalog."
                        ),
                        negative_checks=[],
                    )
                )
                continue
            negative_checks = _negative_rule_checks(
                optimization_id,
                definition=definition,
                rules=negative_rules,
                run_facts=run_facts,
                failed_slo=failed_slo,
            )
            triggered = _triggered_negative_checks(negative_checks)
            if triggered:
                rejected.append(
                    _build_rejection(
                        optimization_id=optimization_id,
                        definition=definition,
                        reason=(
                            "Rejected by optimization negative-rule filtering: "
                            + "; ".join(str(item["condition"]) for item in triggered)
                        ),
                        negative_checks=negative_checks,
                    )
                )
                continue
            contexts = support.get(optimization_id, [])
            if not contexts:
                reasons = incompatible.get(optimization_id, [])
                reason = (
                    "; ".join(sorted(set(reasons)))
                    if reasons
                    else (
                        "No failed Main_Inference config supports this optimization under "
                        "the current engine, memory mode, model, and hardware constraints."
                    )
                )
                rejected.append(
                    _build_rejection(
                        optimization_id=optimization_id,
                        definition=definition,
                        reason=reason,
                        negative_checks=negative_checks,
                    )
                )
                continue
            allowed.append(
                _build_option(
                    failed_slo=failed_slo,
                    optimization_id=optimization_id,
                    definition=definition,
                    contexts=contexts,
                    negative_checks=negative_checks,
                )
            )
        allowed.sort(
            key=lambda item: (
                -int(item["compatible_config_count"]),
                str(item["implementation_status"]) == "planned",
                str(item["optimization_id"]),
            )
        )
        rejected.sort(key=lambda item: str(item["optimization_id"]))
        options_by_slo[str(failed_slo["slo_id"])] = allowed
        rejected_by_slo[str(failed_slo["slo_id"])] = rejected

    generated_at = datetime.now(timezone.utc).isoformat()
    diagnosis_payload = {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at,
        "status": "UI_DIAGNOSIS_READY",
        "decision_source": "deterministic_slo_diagnosis_and_yaml_catalogs",
        "llm_used": False,
        "inference_executed": False,
        "source_artifacts": {
            "manifest": _display_path(root / "raw/main_inference_v1_manifest.json"),
            "slo_report": _display_path(processed / "main_inference_v1_slo_report.json"),
            "slo_scorecard": _display_path(processed / "main_inference_v1_slo_scorecard.csv"),
            "slo_summary": _display_path(processed / "main_inference_v1_slo_summary.csv"),
            "eval_report": _display_path(processed / "main_inference_v1_eval_report.json"),
            "context_preflight": _display_path(
                processed / "main_inference_v1_context_preflight_report.json"
            ),
        },
        "run_context": {
            "run_id": manifest.get("run_id"),
            "baseline_or_optimized": manifest.get("baseline_or_optimized"),
            "model_alias": manifest.get("model_alias"),
            "engine": manifest.get("engine"),
            "memory_mode": manifest.get("memory_mode"),
            "hardware": manifest.get("hardware"),
            "traffic_profile": manifest.get("traffic_profile"),
            "completed_count": manifest.get("completed_count"),
            "failed_count": manifest.get("failed_count"),
            "deployability_verdict": slo_report.get("overall_deployability_verdict"),
        },
        "failed_slos": failed_slos,
        "passed_slo_count": sum(1 for row in scorecard_rows if row.get("status") == "PASS"),
        "failed_slo_count": len(failed_slos),
        "negative_rule_filtering_applied": True,
    }

    options_payload = {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at,
        "status": "UI_OPTIMIZATION_OPTIONS_READY",
        "decision_source": "existing_recommender_plus_catalog_negative_rules",
        "llm_used": False,
        "inference_executed": False,
        "options_by_failed_slo": options_by_slo,
        "rejected_optimizations_by_failed_slo": rejected_by_slo,
        "ui_guardrails": [
            "Only show options under the failed SLO that produced them.",
            "Hide rejected options from the selectable dropdown.",
            "Show rejected options only in an audit/details panel.",
            "Do not execute inference from these UI artifacts.",
        ],
    }
    apply_plan = _build_apply_plan(
        diagnosis_payload=diagnosis_payload,
        options_payload=options_payload,
        generated_at=generated_at,
    )
    story = _build_story(
        diagnosis_payload=diagnosis_payload,
        options_payload=options_payload,
        apply_plan=apply_plan,
        generated_at=generated_at,
    )
    return {
        "diagnosis": diagnosis_payload,
        "optimization_options": options_payload,
        "apply_plan": apply_plan,
        "story": story,
    }


def _exact_changes_for_optimization(optimization_id: str) -> list[str]:
    changes = {
        "prompt_contract_repair": [
            "Tighten generation contract instructions and repair prompts.",
            "Preserve gold data, evaluator semantics, prompt IDs, and source retrieval.",
        ],
        "improve_evidence_formatting": [
            "Reformat model-facing evidence blocks and citation instructions.",
            "Keep the same selected context IDs and short evidence labels.",
        ],
        "use_mm4_agentic_repair": [
            "Route eligible contextual rows through bounded mm4 agentic repair.",
            "Allow only the existing capped retrieval/generation/repair workflow.",
        ],
        "enable_escalation_path": [
            (
                "Enable explicit escalation/insufficient-evidence handling for unsafe or "
                "unsupported answers."
            ),
            "Keep safety evaluator semantics unchanged.",
        ],
        "reduce_max_new_tokens": [
            "Lower or tune max output token budget on the same prompt set.",
            "Measure truncation, contract validity, and groundedness after the change.",
        ],
    }
    return changes.get(
        optimization_id,
        [
            f"Apply catalog optimization {optimization_id} as a single controlled factor.",
            "Measure the same SLOs against the frozen Main_Inference prompt set.",
        ],
    )


def _build_apply_plan(
    *,
    diagnosis_payload: dict[str, Any],
    options_payload: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    plans: dict[str, dict[str, Any]] = {}
    failed_slos = {
        str(item["slo_id"]): item
        for item in cast(list[dict[str, Any]], diagnosis_payload["failed_slos"])
    }
    for slo_id, options in cast(
        dict[str, list[dict[str, Any]]],
        options_payload["options_by_failed_slo"],
    ).items():
        for option in options:
            optimization_id = str(option["optimization_id"])
            plan = plans.setdefault(
                optimization_id,
                {
                    "plan_id": f"{RUN_ID}.{optimization_id}",
                    "optimization_id": optimization_id,
                    "display_name": option["display_name"],
                    "affected_failed_slos": [],
                    "exact_changes": _exact_changes_for_optimization(optimization_id),
                    "hold_constant": [
                        "gold data",
                        "evaluator semantics",
                        "source Main_Inference_V1 artifacts",
                        "reported baseline metrics",
                        "dataset split",
                    ],
                    "requires_gpu_rerun": False,
                    "requires_api_rerun": False,
                    "creates_optimized_inference_v1": False,
                    "execution_mode": "plan_only_no_inference",
                },
            )
            plan["requires_gpu_rerun"] = bool(plan["requires_gpu_rerun"]) or bool(
                option["requires_gpu_rerun"]
            )
            plan["requires_api_rerun"] = bool(plan["requires_api_rerun"]) or bool(
                option["requires_api_rerun"]
            )
            cast(list[dict[str, Any]], plan["affected_failed_slos"]).append(
                {
                    "slo_id": slo_id,
                    "metric_label": failed_slos[slo_id]["metric_label"],
                    "bottleneck_id": failed_slos[slo_id]["bottleneck"]["id"],
                }
            )
    ordered_plans = sorted(plans.values(), key=lambda item: str(item["optimization_id"]))
    return {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at,
        "status": "UI_APPLY_PLAN_READY",
        "inference_executed": False,
        "optimized_result_created": False,
        "plan_semantics": (
            "The UI may show what would change. A real optimized result requires a separate "
            "approved optimized rerun and artifact import."
        ),
        "plans": ordered_plans,
        "apply_all_plan": {
            "status": "PLAN_ONLY",
            "optimization_ids": [str(item["optimization_id"]) for item in ordered_plans],
            "conflict_policy": (
                "Apply prompt/contract and evidence-presentation fixes before model, "
                "agentic, concurrency, or hardware changes. Measure one factor at a time "
                "unless an approved optimized experiment explicitly bundles them."
            ),
            "does_not_execute": True,
        },
    }


def _build_story(
    *,
    diagnosis_payload: dict[str, Any],
    options_payload: dict[str, Any],
    apply_plan: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    failed_slos = cast(list[dict[str, Any]], diagnosis_payload["failed_slos"])
    option_counts = {
        slo_id: len(options)
        for slo_id, options in cast(
            dict[str, list[dict[str, Any]]],
            options_payload["options_by_failed_slo"],
        ).items()
    }
    return {
        "run_id": RUN_ID,
        "generated_at_utc": generated_at,
        "status": "UI_STORY_READY",
        "title": "Optimization Intelligence For Main_Inference_V1",
        "summary": (
            "Main_Inference_V1 completed operationally but failed quality and safety SLOs. "
            "The UI reasoning layer maps each failed SLO to a bottleneck and only exposes "
            "catalog-backed optimizations that survived compatibility and negative-rule filters."
        ),
        "steps": [
            {
                "step": "select_failed_slo",
                "ui_action": "User clicks a failed SLO row.",
                "backend_artifact": "main_inference_v1_ui_diagnosis.json",
            },
            {
                "step": "show_bottleneck",
                "ui_action": "Show target, observed value, severity, confidence, and bottleneck.",
                "backend_artifact": "main_inference_v1_ui_diagnosis.json",
            },
            {
                "step": "show_filtered_options",
                "ui_action": "Dropdown contains only compatible, non-rejected optimizations.",
                "backend_artifact": "main_inference_v1_ui_optimization_options.json",
            },
            {
                "step": "inspect_rejections",
                "ui_action": "Optional details panel explains rejected optimizations and rules.",
                "backend_artifact": "main_inference_v1_ui_optimization_options.json",
            },
            {
                "step": "apply_plan",
                "ui_action": "Apply shows the exact plan; it does not run inference.",
                "backend_artifact": "main_inference_v1_ui_apply_plan.json",
            },
        ],
        "failed_slo_count": len(failed_slos),
        "failed_slo_option_counts": option_counts,
        "apply_plan_count": len(cast(list[dict[str, Any]], apply_plan["plans"])),
        "no_gpu_required_for_ui_replay": True,
    }


def write_ui_artifacts(
    *,
    experiment_root: str | Path = DEFAULT_EXPERIMENT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    """Write the four Main_Inference UI optimization artifacts."""

    payloads = build_ui_diagnosis(experiment_root=experiment_root)
    output = _repo_path(output_root)
    paths = {
        "diagnosis": output / "main_inference_v1_ui_diagnosis.json",
        "optimization_options": output / "main_inference_v1_ui_optimization_options.json",
        "apply_plan": output / "main_inference_v1_ui_apply_plan.json",
        "story": output / "main_inference_v1_ui_story.json",
    }
    for key, path in paths.items():
        _write_json(path, cast(dict[str, Any], payloads[key]))
    return paths
