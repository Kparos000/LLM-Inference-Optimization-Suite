"""Deterministic optimization recommendation and compatibility filtering."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from inference_bench.optimization_catalog import (
    OptimizationDefinition,
    load_optimization_catalog,
)

QUALITY_BOTTLENECKS = {
    "low_groundedness",
    "low_evidence_match",
    "low_citation_accuracy",
    "low_contract_validity",
    "low_json_validity",
    "safety_violations",
    "high_truncation_rate",
    "insufficient_evidence_misuse",
}
RETRIEVAL_BOTTLENECKS = {
    "low_candidate_recall",
    "low_final_recall",
    "low_mrr",
    "high_retrieval_latency",
    "compression_recall_loss",
}
HARDWARE_UPGRADES = {
    "increase_gpu_size",
    "move_to_runpod_rtx4090",
    "move_to_runpod_l40s",
    "move_to_runpod_a100",
    "move_to_runpod_h100",
}
STATUS_SCORE = {
    "implemented": 0.20,
    "engine_builtin": 0.15,
    "config_only": 0.10,
    "planned": 0.0,
}


def _hardware_capabilities(hardware_profile: dict[str, Any]) -> set[str]:
    capabilities = {str(item) for item in hardware_profile.get("capabilities", [])}
    if hardware_profile.get("gpu_name") or hardware_profile.get("vram_gb"):
        capabilities.add("gpu")
    alias = str(
        hardware_profile.get("hardware_alias") or hardware_profile.get("name") or ""
    ).lower()
    if "runpod" in alias:
        capabilities.add("runpod")
    if int(hardware_profile.get("gpu_count") or 1) > 1:
        capabilities.add("multi_gpu")
    if bool(hardware_profile.get("supports_fp8")):
        capabilities.add("fp8_gpu")
    return capabilities


def _compatibility_reason(
    optimization: OptimizationDefinition,
    *,
    engine: str,
    memory_mode: str,
    hardware_profile: dict[str, Any],
    model_metadata: dict[str, Any],
) -> str | None:
    if (
        optimization.compatible_memory_modes
        and memory_mode not in optimization.compatible_memory_modes
    ):
        return f"memory mode {memory_mode} is not in compatible_memory_modes"
    if memory_mode in optimization.incompatible_memory_modes:
        return f"memory mode {memory_mode} is explicitly incompatible"
    if optimization.required_engines and engine not in optimization.required_engines:
        return (
            f"current engine {engine} does not satisfy required_engines "
            f"{list(optimization.required_engines)}"
        )
    missing_hardware = set(optimization.required_hardware) - _hardware_capabilities(
        hardware_profile
    )
    if missing_hardware:
        return f"missing hardware capabilities: {sorted(missing_hardware)}"
    allowed_backends = {
        str(item).removesuffix("_optional") for item in model_metadata.get("allowed_backends", [])
    }
    target_engine = {
        "switch_engine_to_vllm": "vllm",
        "switch_engine_to_sglang": "sglang",
        "switch_engine_to_tensorrt_llm": "tensorrt_llm",
    }.get(optimization.id)
    if target_engine and allowed_backends and target_engine not in allowed_backends:
        return (
            f"model metadata does not list {target_engine} as an allowed backend; "
            f"allowed={sorted(allowed_backends)}"
        )
    execution_target = str(model_metadata.get("execution_target") or "")
    if (
        optimization.id
        in {
            "use_quantized_model",
            "enable_int8_quantization",
            "enable_awq_int4",
            "enable_gptq_int4",
            "enable_fp8_where_supported",
        }
        and "api" in execution_target
    ):
        return "provider API model precision is not controlled by this project"
    return None


def _rule_candidates(
    bottleneck_ids: set[str],
    *,
    engine: str,
    memory_mode: str,
    concurrency: int,
    failed_groups: set[str],
    passed_groups: set[str],
    run_metrics: dict[str, Any],
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    ordered: list[str] = []
    bonuses: dict[str, float] = {}
    reasons: dict[str, str] = {}

    def add(optimization_id: str, bonus: float, reason: str) -> None:
        if optimization_id not in ordered:
            ordered.append(optimization_id)
        bonuses[optimization_id] = max(bonuses.get(optimization_id, 0.0), bonus)
        reasons.setdefault(optimization_id, reason)

    if "low_gpu_utilization" in bottleneck_ids and concurrency == 1:
        add(
            "concurrency_sweep",
            3.0,
            "GPU utilization missed its SLO at concurrency 1; measure a bounded sweep first.",
        )
        add(
            "increase_concurrency",
            2.2,
            "Higher concurrency may improve device occupancy, subject to tail-latency guards.",
        )
        if engine in {"vllm", "sglang"}:
            add(
                "enable_continuous_batching",
                1.5,
                "The serving engine supports continuous batching; verify its active configuration.",
            )

    high_ttft = bool(bottleneck_ids & {"high_ttft_p50", "high_ttft_p95", "high_ttft_p99"})
    high_input = bool(bottleneck_ids & {"high_input_tokens", "excessive_context_tokens"}) or bool(
        run_metrics.get("high_input_tokens")
    )
    if high_ttft and high_input:
        add(
            "reduce_context_tokens",
            3.2,
            "High TTFT coincides with high input/context tokens.",
        )
        if memory_mode == "mm2_hybrid_top5":
            add(
                "enable_context_compression",
                2.8,
                "mm2 can be compared with the existing deterministic mm3 compression path.",
            )
        add("reduce_top_k", 2.2, "Reducing retrieved context can lower prefill work.")
        if bool(run_metrics.get("prefix_reuse_available")):
            add("enable_prefix_cache", 2.0, "Measured prefix reuse makes cache testing relevant.")
        if engine == "huggingface":
            add(
                "switch_engine_to_vllm",
                1.8,
                "HF Transformers lacks serving-oriented batching and paged KV management.",
            )

    quality_failed = bool(bottleneck_ids & QUALITY_BOTTLENECKS)
    retrieval_failed = bool(bottleneck_ids & RETRIEVAL_BOTTLENECKS) or (
        "retrieval" in failed_groups
    )
    if quality_failed:
        if retrieval_failed:
            add(
                "repair_retrieval",
                4.0,
                "Quality failed while retrieval SLOs also failed; repair evidence supply first.",
            )
        else:
            retrieval_passed = "retrieval" in passed_groups
            qualifier = "passed" if retrieval_passed else "did not fail"
            add(
                "use_stronger_model",
                3.6,
                f"Quality failed while retrieval {qualifier}; isolate model capability next.",
            )
            add(
                "improve_evidence_formatting",
                2.9,
                "Present the same evidence more clearly without changing retrieval or gold data.",
            )
            add(
                "prompt_contract_repair",
                2.7,
                "Contract and citation instructions can be tested under the unchanged evaluator.",
            )
            if memory_mode in {"mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}:
                add(
                    "use_mm4_agentic_repair",
                    2.2,
                    (
                        "The bounded mm4 path can test one repair/escalation step "
                        "on the same evidence."
                    ),
                )

    if engine == "huggingface" and bottleneck_ids & {
        "high_gpu_memory_utilization",
        "low_vram_headroom",
        "engine_not_serving_optimized",
        "high_ttft_p95",
        "low_requests_per_second",
    }:
        add(
            "switch_engine_to_vllm",
            3.0,
            "Switch to vLLM to test PagedAttention and continuous batching as engine capabilities.",
        )

    return ordered, bonuses, reasons


def _already_active_capabilities(engine: str) -> list[dict[str, str]]:
    active: list[dict[str, str]] = []
    if engine == "vllm":
        active.append(
            {
                "optimization_id": "use_pagedattention_capable_engine",
                "status": "already_active",
                "reason": "PagedAttention is an engine_builtin capability of the active vLLM path.",
            }
        )
        active.append(
            {
                "optimization_id": "enable_continuous_batching",
                "status": "already_active",
                "reason": "Continuous batching is provided by the active vLLM serving engine.",
            }
        )
    elif engine == "sglang":
        active.append(
            {
                "optimization_id": "enable_continuous_batching",
                "status": "already_active",
                "reason": "Continuous batching is provided by the active SGLang serving engine.",
            }
        )
    return active


def _next_experiment(
    primary: dict[str, Any] | None,
    *,
    experiment_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if primary is None:
        return []
    optimization_id = str(primary["optimization_id"])
    factor = {
        "concurrency_sweep": "concurrency",
        "increase_concurrency": "concurrency",
        "use_stronger_model": "model_alias",
        "repair_retrieval": "retrieval_configuration",
        "reduce_context_tokens": "context_token_budget",
        "enable_context_compression": "memory_mode",
        "prompt_contract_repair": "prompt_renderer",
        "improve_evidence_formatting": "evidence_format",
        "switch_engine_to_vllm": "engine",
        "use_mm4_agentic_repair": "memory_mode",
    }.get(optimization_id, optimization_id)
    return [
        {
            "optimization_id": optimization_id,
            "change_exactly_one_factor": factor,
            "hold_constant": [
                key
                for key in (
                    "prompt_ids",
                    "model_alias",
                    "engine",
                    "hardware",
                    "memory_mode",
                    "temperature",
                    "max_new_tokens",
                    "evaluator",
                )
                if key != factor
            ],
            "safety_gate": (
                "Use the smallest frozen workload that can measure the failed SLOs; "
                "stop on OOM, request failure, quality regression, or safety regression."
            ),
            "source_experiment": experiment_config.get("experiment_id"),
        }
    ]


def recommend_optimizations(
    diagnosis: dict[str, Any],
    *,
    catalog_path: str = "configs/optimization_catalog.yaml",
) -> dict[str, Any]:
    """Rank deterministic recommendations for diagnosed bottlenecks."""

    catalog = load_optimization_catalog(catalog_path)
    context = cast(dict[str, Any], diagnosis.get("context", {}))
    engine = str(context.get("engine") or "")
    memory_mode = str(context.get("memory_mode") or "")
    experiment_config = cast(dict[str, Any], context.get("experiment_config", {}))
    hardware_profile = cast(dict[str, Any], context.get("hardware_profile", {}))
    model_metadata = cast(dict[str, Any], context.get("model_metadata", {}))
    run_metrics = cast(dict[str, Any], context.get("run_metrics", {}))
    concurrency = int(experiment_config.get("concurrency") or 1)
    bottlenecks = cast(list[dict[str, Any]], diagnosis.get("bottlenecks", []))
    bottleneck_ids = {str(item["id"]) for item in bottlenecks}
    severity_by_id = {str(item["id"]): float(item.get("severity") or 0.0) for item in bottlenecks}
    confidence_by_id = {
        str(item["id"]): float(item.get("confidence") or 0.0) for item in bottlenecks
    }
    failed_groups = {
        str(item["group"]) for item in cast(list[dict[str, Any]], diagnosis["failed_slos"])
    }
    passed_groups = {
        str(item["group"]) for item in cast(list[dict[str, Any]], diagnosis["passed_slos"])
    }
    candidate_ids: list[str] = []
    for bottleneck in bottlenecks:
        for optimization_id in cast(list[str], bottleneck["compatible_optimizations"]):
            if optimization_id not in candidate_ids:
                candidate_ids.append(optimization_id)
    rule_ids, bonuses, rule_reasons = _rule_candidates(
        bottleneck_ids,
        engine=engine,
        memory_mode=memory_mode,
        concurrency=concurrency,
        failed_groups=failed_groups,
        passed_groups=passed_groups,
        run_metrics=run_metrics,
    )
    for optimization_id in reversed(rule_ids):
        if optimization_id in candidate_ids:
            candidate_ids.remove(optimization_id)
        candidate_ids.insert(0, optimization_id)

    active = _already_active_capabilities(engine)
    active_ids = {item["optimization_id"] for item in active}
    compatible: list[dict[str, Any]] = []
    incompatible: list[dict[str, str]] = []
    for optimization_id in candidate_ids:
        definition = catalog.get(optimization_id)
        if definition is None:
            incompatible.append(
                {
                    "optimization_id": optimization_id,
                    "reason": "optimization is not present in the catalog",
                }
            )
            continue
        if optimization_id in active_ids:
            continue
        incompatibility = _compatibility_reason(
            definition,
            engine=engine,
            memory_mode=memory_mode,
            hardware_profile=hardware_profile,
            model_metadata=model_metadata,
        )
        if incompatibility:
            incompatible.append({"optimization_id": optimization_id, "reason": incompatibility})
            continue
        matched_bottlenecks = sorted(bottleneck_ids & set(definition.compatible_bottlenecks))
        severity = max(
            (severity_by_id.get(item, 0.0) for item in matched_bottlenecks),
            default=0.0,
        )
        confidence = max(
            (confidence_by_id.get(item, 0.0) for item in matched_bottlenecks),
            default=0.5,
        )
        score = (
            bonuses.get(optimization_id, 0.0)
            + severity
            + confidence * 0.5
            + STATUS_SCORE[definition.implementation_status]
        )
        if optimization_id in HARDWARE_UPGRADES:
            score -= 1.5
        compatible.append(
            {
                "optimization_id": optimization_id,
                "rank_score": round(score, 6),
                "reason": rule_reasons.get(
                    optimization_id,
                    ("Catalog mapping from failed bottlenecks: " + ", ".join(matched_bottlenecks)),
                ),
                "evidence": [
                    evidence
                    for bottleneck in bottlenecks
                    if str(bottleneck["id"]) in matched_bottlenecks
                    for evidence in cast(list[dict[str, Any]], bottleneck["evidence"])
                ],
                "matched_bottlenecks": matched_bottlenecks,
                "implementation_status": definition.implementation_status,
                "application_method": definition.application_method,
                "quality_risk": definition.quality_risk,
                "cost_risk": definition.cost_risk,
                "expected_gain_range": definition.expected_gain_range,
                "experiment_safety_notes": list(definition.experiment_safety_notes),
                "catalog_definition": asdict(definition),
            }
        )
    compatible.sort(key=lambda item: (-float(item["rank_score"]), str(item["optimization_id"])))
    primary = compatible[0] if compatible else None
    for index, item in enumerate(compatible):
        item["recommendation_tier"] = "primary" if index == 0 else "secondary"
    return {
        "primary_recommendation": primary,
        "recommendations": compatible,
        "already_active_capabilities": active,
        "incompatible_optimizations": incompatible,
        "next_experiment_suggestions": _next_experiment(
            primary, experiment_config=experiment_config
        ),
        "decision_source": "deterministic_rules_and_yaml_catalogs",
        "llm_used": False,
    }
