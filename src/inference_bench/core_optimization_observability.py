"""Observability contracts for planned core inference optimizations.

This module builds instrumentation plans and UI-facing observability artifacts.
It never runs inference, starts a serving engine, mutates measured
Main_Inference_V1 results, or creates Optimized_Inference_V1.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from inference_bench.core_optimization_planning import (
    REPAIR_IDS,
    build_core_optimization_taxonomy,
)

JsonDict = dict[str, Any]

RUN_ID = "main_inference_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path("experiments/main/main_inference_v1")
MAIN_PROCESSED = MAIN_ROOT / "processed"
MAIN_RAW = MAIN_ROOT / "raw"
MAIN_LOGS = MAIN_ROOT / "logs"
WORKLOAD_ROOT = Path("data/workloads/final_10000")
AUTHORITATIVE_WORKLOAD_PATH = WORKLOAD_ROOT / "prompt_plus_metadata" / "mm2_hybrid_top5.jsonl"

OBSERVABILITY_CONFIG_PATH = Path("configs/core_optimization_observability.yaml")
SCENARIO_REGISTRY_PATH = Path("configs/core_optimization_scenario_registry.yaml")

OBSERVABILITY_REGISTRY_JSON_PATH = MAIN_PROCESSED / "core_optimization_observability_registry.json"
OBSERVABILITY_READINESS_JSON_PATH = (
    MAIN_PROCESSED / "core_optimization_observability_readiness.json"
)
OBSERVABILITY_READINESS_CSV_PATH = MAIN_PROCESSED / "core_optimization_observability_readiness.csv"
OBSERVABILITY_INVENTORY_JSON_PATH = (
    MAIN_PROCESSED / "main_inference_v1_observability_inventory.json"
)
OBSERVABILITY_INVENTORY_CSV_PATH = MAIN_PROCESSED / "main_inference_v1_observability_inventory.csv"
PREFIX_ANALYSIS_JSON_PATH = (
    MAIN_PROCESSED / "coreopt_prefix_layout_static_v1_prefix_opportunity_analysis.json"
)
PREFIX_ANALYSIS_CSV_PATH = (
    MAIN_PROCESSED / "coreopt_prefix_layout_static_v1_prefix_opportunity_analysis.csv"
)
EVENT_SCHEMA_JSON_PATH = MAIN_PROCESSED / "core_optimization_event_schema.json"
UI_OBSERVABILITY_CONTRACT_PATH = MAIN_PROCESSED / "core_optimization_ui_observability_contract.json"
ADAPTER_COVERAGE_PATH = MAIN_PROCESSED / "core_optimization_adapter_coverage.json"

SCENARIO_PLAN_PATHS = {
    "coreopt_prefix_layout_static_v1": (
        MAIN_PROCESSED / "coreopt_prefix_layout_static_v1_instrumentation_plan.json"
    ),
    "coreopt_scheduler_batch_vllm_v1": (
        MAIN_PROCESSED / "coreopt_scheduler_batch_vllm_v1_instrumentation_plan.json"
    ),
    "coreopt_prefix_cache_vllm_v1": (
        MAIN_PROCESSED / "coreopt_prefix_cache_vllm_v1_instrumentation_plan.json"
    ),
    "coreopt_chunked_prefill_sglang_v1": (
        MAIN_PROCESSED / "coreopt_chunked_prefill_sglang_v1_instrumentation_plan.json"
    ),
}

READINESS_STATES = {
    "ready_existing",
    "ready_derivable",
    "requires_runner_instrumentation",
    "requires_engine_metrics",
    "requires_external_profiler",
    "unsupported_current_runtime",
    "future_architecture",
}

MEASUREMENT_TYPES = {
    "measured",
    "engine_reported",
    "derived",
    "estimated",
    "unavailable",
}

OPTIMIZATION_DOMAINS = {
    "prompt_workload",
    "prefix_cache",
    "scheduler_batching",
    "kv_cache",
    "prefill_decode_balance",
    "routing",
    "quantization",
    "speculative_decoding",
    "model_selection",
    "model_compression",
    "parallelism",
    "cache_offloading",
    "kernel_compiler",
    "disaggregation",
    "distributed_serving",
}

EVENT_TYPES = {
    "run_started",
    "config_started",
    "request_arrived",
    "request_queued",
    "request_scheduled",
    "prefill_started",
    "prefill_chunk_completed",
    "prefix_cache_lookup",
    "prefix_cache_hit",
    "prefix_cache_miss",
    "kv_cache_allocated",
    "kv_cache_evicted",
    "batch_iteration",
    "decode_token",
    "request_completed",
    "request_failed",
    "telemetry_sample",
    "quality_evaluation",
    "optimization_decision",
    "prompt_layout_rendered",
    "prefix_family_assigned",
    "static_metric_computed",
    "run_completed",
}


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _display_path(path: str | Path) -> str:
    value = _repo_path(path)
    try:
        return value.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object at {path}"
        raise ValueError(msg)
    return cast(JsonDict, payload)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _iter_jsonl(path: str | Path) -> Iterable[JsonDict]:
    with _repo_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield cast(JsonDict, value)


def _write_json(path: str | Path, payload: JsonDict) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: str | Path, payload: JsonDict) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        payload,
        Dumper=_NoAliasSafeDumper,
        sort_keys=False,
        allow_unicode=False,
    )
    target.write_text(text, encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fields: list[str]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    target = _repo_path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(text: str) -> list[str]:
    # Deterministic local token proxy. It is exact for this analysis, but it is
    # not claimed as historical engine tokenizer telemetry.
    return re.findall(r"\S+|[^\w\s]", text, flags=re.UNICODE)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(ordered[int(index)], 6)
    weight = index - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _common_prefix_length(token_lists: Sequence[Sequence[str]]) -> int:
    if not token_lists:
        return 0
    shortest = min(len(items) for items in token_lists)
    count = 0
    for index in range(shortest):
        value = token_lists[0][index]
        if all(items[index] == value for items in token_lists[1:]):
            count += 1
        else:
            break
    return count


def _render_messages(row: JsonDict) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    rendered: list[str] = []
    for item in messages:
        if isinstance(item, dict):
            rendered.append(f"{item.get('role', '')}: {item.get('content', '')}")
    return "\n".join(rendered)


def _static_prefix_text(row: JsonDict) -> str:
    """Return the leading text before request-specific context starts."""

    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    rendered: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role == "user" and "\n\nContext:\n\n" in content:
            content = content.split("\n\nContext:\n\n", 1)[0] + "\n\nContext:"
            rendered.append(f"{role}: {content}")
            break
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def _bucket(value: int) -> str:
    if value < 128:
        return "000_127"
    if value < 256:
        return "128_255"
    if value < 512:
        return "256_511"
    if value < 1024:
        return "512_1023"
    if value < 2048:
        return "1024_2047"
    return "2048_plus"


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _core_optimizations_by_id() -> dict[str, JsonDict]:
    taxonomy = build_core_optimization_taxonomy()
    layers = taxonomy["layers"]
    return {
        str(item["optimization_id"]): item
        for item in cast(list[JsonDict], layers["B_engineer_applied_core_optimizations"])
    }


METRIC_ENVELOPES: dict[str, list[str]] = {
    "run_identity": [
        "run_id",
        "parent_run_id",
        "scenario_id",
        "optimization_id",
        "changed_factor",
        "held_constants",
        "model",
        "engine",
        "backend",
        "memory_mode",
        "concurrency",
        "hardware",
        "precision",
        "git_commit",
        "config_hash",
        "workload_hash",
        "start_time",
        "end_time",
        "status",
    ],
    "request_lifecycle": [
        "request_id",
        "prompt_id",
        "config_id",
        "arrival_time",
        "queue_enter_time",
        "scheduling_time",
        "prefill_start",
        "prefill_end",
        "first_token_time",
        "decode_start",
        "decode_end",
        "completion_time",
        "failure_retry_status",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ],
    "performance": [
        "ttft_ms",
        "queue_wait_ms",
        "prefill_latency_ms",
        "tpot_ms",
        "itl_ms",
        "e2e_latency_ms",
        "requests_per_second",
        "tokens_per_second",
        "successful_requests_per_second",
        "batch_level_throughput",
    ],
    "resources": [
        "gpu_utilization_percent",
        "vram_used_mb",
        "vram_available_mb",
        "power_draw_w",
        "temperature_c",
        "cpu_utilization_percent",
        "ram_usage_gb",
        "model_memory_mb",
        "kv_cache_memory_mb",
        "telemetry_timestamp",
        "active_engine_process",
    ],
    "quality_safety_protection": [
        "json_validity",
        "contract_validity",
        "format_validity",
        "evidence_id_presence",
        "evidence_match",
        "groundedness",
        "safety_findings",
        "truncation",
        "completion_failure_rate",
        "escalation_correctness",
        "insufficient_evidence_correctness",
        "mm4_bound_compliance",
    ],
    "cost": [
        "gpu_cost_usd",
        "api_cost_usd",
        "total_cost_usd",
        "cost_per_request_usd",
        "cost_per_1000_requests_usd",
        "cost_per_successful_request_usd",
        "tokens_per_gpu_dollar",
        "tokens_per_api_dollar",
    ],
}


DOMAIN_SPECS: dict[str, JsonDict] = {
    "prompt_prefix_layout_optimization": {
        "domain": "prompt_workload",
        "readiness": "ready_derivable",
        "primary_metrics": [
            "exact_shared_prefix_token_count",
            "reusable_token_ratio",
            "prefix_family_count",
            "ttft_ms",
        ],
        "secondary_metrics": ["input_tokens", "prefill_latency_ms", "tokens_per_second"],
        "required": [
            "rendered_prompt_hash",
            "tokenized_prefix_hash",
            "exact_shared_prefix_token_count",
            "total_input_token_count",
            "reusable_token_ratio",
            "prefix_family_id",
            "requests_per_prefix_family",
            "longest_exact_common_prefix",
            "common_system_prompt_tokens",
            "common_generation_contract_tokens",
            "common_evidence_schema_tokens",
            "common_tool_definition_tokens",
            "variable_suffix_token_count",
            "prefix_entropy",
        ],
        "optional": ["engine_cache_hit_count", "prefix_cache_lookup_latency_ms"],
        "existing": ["rendered messages", "input token aggregates", "workload hashes"],
        "missing": ["engine cache-hit telemetry"],
        "derivation": "Deterministic static token analysis over rendered workload messages.",
        "sampling": "once per workload artifact",
        "temporal_resolution": "static per prompt and aggregate per family",
        "experiment_type": "cpu_static_audit_then_one_factor_latency_rerun",
        "ui": ["token_ribbon", "stable_prefix_vs_variable_suffix", "prefix_family_histogram"],
        "acceptance": ["larger exact leading prefix", "quality/safety unchanged after rerun"],
        "rejection": ["prefix families fragment", "contract or evidence changes"],
    },
    "prefix_cache_verification_tuning": {
        "domain": "prefix_cache",
        "readiness": "requires_engine_metrics",
        "primary_metrics": ["cache_hit_rate", "hit_token_count", "ttft_ms_for_hits"],
        "secondary_metrics": ["cache_lookup_latency_ms", "cache_occupancy", "prefill_latency_ms"],
        "required": [
            "cache_lookup_count",
            "cache_hit_count",
            "cache_miss_count",
            "cache_hit_rate",
            "hit_token_count",
            "miss_token_count",
            "reused_token_ratio",
            "cache_lookup_latency_ms",
            "ttft_ms_for_hits",
            "ttft_ms_for_misses",
            "prefill_latency_ms_for_hits",
            "prefill_latency_ms_for_misses",
            "cache_occupancy",
            "cache_blocks_used",
            "cache_blocks_free",
            "cache_blocks_total",
            "eviction_count",
            "insertion_count",
            "cache_invalidation_count",
            "cache_salt_or_tenant_partition",
            "engine_configuration_proving_cache_state",
        ],
        "optional": ["startup log cache flag", "metrics endpoint scrape"],
        "existing": ["rendered prompts", "TTFT aggregates"],
        "missing": ["cache hit/miss counters", "cache occupancy", "hit/miss TTFT split"],
        "derivation": "No derivation for hit/miss; exact prefix opportunity is only a precursor.",
        "sampling": "per request plus engine metrics scrape",
        "temporal_resolution": "per request and per engine interval",
        "experiment_type": "gpu_one_factor_engine_flag",
        "ui": ["hit_miss_flow", "reused_tokens", "hit_miss_ttft_comparison"],
        "acceptance": ["non-zero measured hits", "TTFT improves for hits", "quality gates pass"],
        "rejection": ["no hits", "eviction overhead", "quality/safety regression"],
    },
    "scheduler_batch_tuning": {
        "domain": "scheduler_batching",
        "readiness": "requires_runner_instrumentation",
        "primary_metrics": ["queue_wait_ms", "active_batch_size", "token_budget_utilization"],
        "secondary_metrics": ["ttft_ms", "e2e_latency_ms", "tokens_per_second"],
        "required": [
            "request_arrival_time",
            "queue_wait_ms",
            "waiting_request_count",
            "running_request_count",
            "active_batch_size",
            "batch_composition",
            "scheduled_tokens",
            "maximum_token_budget",
            "prefill_tokens_per_iteration",
            "decode_tokens_per_iteration",
            "maximum_running_sequences",
            "maximum_batched_tokens",
            "admission_decision",
            "preemption_count",
            "preemption_reason",
            "recomputation_count",
            "scheduling_policy",
            "priority_class",
            "batch_waiting_delay_ms",
            "iteration_duration_ms",
            "gpu_busy_idle_interval",
        ],
        "optional": ["engine metrics endpoint", "scheduler debug logs"],
        "existing": ["configured concurrency", "aggregate latency", "aggregate throughput"],
        "missing": ["actual active batch size", "queue wait", "engine iteration counters"],
        "derivation": "Configured concurrency is available but is not active batch size.",
        "sampling": "per request and per engine iteration",
        "temporal_resolution": "request lifecycle and scheduler iteration",
        "experiment_type": "gpu_one_factor_runtime_config",
        "ui": ["queue_animation", "active_batch", "token_budget_utilization"],
        "acceptance": ["higher throughput without tail latency or quality regression"],
        "rejection": ["p95/p99 latency regression", "OOM", "preemption spikes"],
    },
    "kv_cache_capacity_allocation_tuning": {
        "domain": "kv_cache",
        "readiness": "requires_engine_metrics",
        "primary_metrics": ["kv_blocks_used", "kv_blocks_free", "cache_occupancy_percent"],
        "secondary_metrics": ["vram_used_mb", "oom_count", "preemption_count"],
        "required": [
            "total_kv_blocks",
            "used_kv_blocks",
            "free_kv_blocks",
            "cache_occupancy_percent",
            "kv_bytes_per_token",
            "estimated_kv_bytes_per_request",
            "cache_allocation_failures",
            "eviction_count",
            "eviction_reason",
            "preemption_count",
            "recomputation_count",
            "oom_count",
            "maximum_concurrent_sequences",
            "maximum_sequence_length",
            "gpu_memory_utilization_setting",
            "block_or_page_size",
            "watermark",
            "cache_retention_eviction_policy",
            "session_or_tenant_partition",
        ],
        "optional": ["engine block-manager counters"],
        "existing": ["VRAM telemetry", "input/output token aggregates"],
        "missing": ["KV block counts", "allocation failures", "eviction reasons"],
        "derivation": "Formula estimates may be produced from tokens and model config only.",
        "sampling": "engine interval plus run summary",
        "temporal_resolution": "per interval, per config, and per length bucket",
        "experiment_type": "gpu_one_factor_runtime_config",
        "ui": ["kv_block_map", "occupancy", "eviction_preemption_events"],
        "acceptance": ["higher useful KV capacity", "no OOM/preemption regression"],
        "rejection": ["evictions or OOMs increase", "quality gates regress"],
    },
    "chunked_prefill_tuning": {
        "domain": "prefill_decode_balance",
        "readiness": "requires_engine_metrics",
        "primary_metrics": ["ttft_by_input_length_bucket", "chunk_count", "decode_interference"],
        "secondary_metrics": ["tpot_by_input_length_bucket", "e2e_latency_by_input_length_bucket"],
        "required": [
            "chunked_prefill_enabled",
            "configured_chunk_size",
            "actual_chunk_count",
            "tokens_per_chunk",
            "maximum_partial_prefills",
            "long_prompt_threshold",
            "prefill_token_budget",
            "prefill_queue_delay_ms",
            "short_request_bypass_behavior",
            "decode_interference",
            "ttft_by_input_length_bucket",
            "tpot_by_input_length_bucket",
            "e2e_by_input_length_bucket",
            "head_of_line_blocking_indicators",
            "short_long_request_co_residency",
            "engine_iteration_composition",
        ],
        "optional": ["engine debug traces"],
        "existing": ["input token aggregate by config", "TTFT aggregate by config"],
        "missing": ["chunk counters", "iteration composition", "short/long co-residency"],
        "derivation": "Length-bucket opportunity can be estimated; chunk events cannot.",
        "sampling": "per request and per engine iteration",
        "temporal_resolution": "request length bucket and scheduler iteration",
        "experiment_type": "gpu_one_factor_engine_config",
        "ui": ["long_prompt_chunks", "concurrent_decode_work", "ttft_by_length_bucket"],
        "acceptance": ["long-prompt TTFT improves without short-request regression"],
        "rejection": ["decode interference increases", "tail latency worsens"],
    },
    "cache_workload_aware_routing": {
        "domain": "routing",
        "readiness": "requires_runner_instrumentation",
        "primary_metrics": ["routing_decision", "cache_affinity_key", "route_level_latency"],
        "secondary_metrics": ["route_level_cost", "route_level_quality"],
        "required": [
            "routing_decision",
            "selected_route_model_engine",
            "decision_reason",
            "prompt_length_bucket",
            "vertical",
            "complexity_class",
            "expected_status",
            "cache_affinity_key",
            "session_affinity",
            "queue_depth",
            "estimated_cost",
            "predicted_latency",
            "fallback_escalation_route",
            "routing_success_failure",
            "route_level_quality",
            "route_level_latency",
            "route_level_cost",
        ],
        "optional": ["policy simulation trace"],
        "existing": ["vertical", "prompt length aggregates", "route comparison CSVs"],
        "missing": ["routing decision trace", "cache-affinity key", "queue-depth input"],
        "derivation": "Historical route comparisons can inform planning but not route decisions.",
        "sampling": "per request routing decision",
        "temporal_resolution": "per request and route aggregate",
        "experiment_type": "planned_policy_simulation_then_measured_rerun",
        "ui": ["routing_map", "cache_affinity_clusters", "route_cost_latency_quality"],
        "acceptance": ["route improves latency/cost while protected metrics pass"],
        "rejection": ["routing harms quality or creates unfair route drift"],
    },
    "model_selection_routing": {
        "domain": "model_selection",
        "readiness": "requires_runner_instrumentation",
        "primary_metrics": ["selected_model", "decision_reason", "route_level_quality"],
        "secondary_metrics": ["route_level_latency", "route_level_cost"],
        "required": [
            "routing_decision",
            "selected_route_model_engine",
            "decision_reason",
            "prompt_length_bucket",
            "vertical",
            "complexity_class",
            "expected_status",
            "estimated_cost",
            "predicted_latency",
            "fallback_escalation_route",
            "routing_success_failure",
            "route_level_quality",
            "route_level_latency",
            "route_level_cost",
        ],
        "optional": ["confidence threshold", "fallback trace"],
        "existing": ["model comparison CSV", "API versus self-hosted comparison CSV"],
        "missing": ["per-request router decision", "policy thresholds"],
        "derivation": "Aggregate model comparison can plan a policy but cannot prove routing.",
        "sampling": "per request routing decision",
        "temporal_resolution": "per request and route aggregate",
        "experiment_type": "planned_policy_simulation_then_measured_rerun",
        "ui": ["model_route_tree", "quality_cost_latency_tradeoff"],
        "acceptance": ["better deployability or lower cost at protected quality"],
        "rejection": ["route quality regression", "cost increase without quality gain"],
    },
    "quantization": {
        "domain": "quantization",
        "readiness": "requires_runner_instrumentation",
        "primary_metrics": ["precision_format", "model_weight_vram_mb", "tpot_ms"],
        "secondary_metrics": ["e2e_latency_ms", "throughput", "power", "quality_regression"],
        "required": [
            "precision_format",
            "quantization_method",
            "checkpoint_model_id",
            "weight_format",
            "activation_format",
            "kv_cache_format",
            "group_size",
            "calibration_method_dataset",
            "excluded_layers",
            "model_file_size",
            "load_time_ms",
            "model_weight_vram_mb",
            "total_vram_mb",
            "kv_cache_capacity",
            "ttft_ms",
            "tpot_ms",
            "e2e_latency_ms",
            "throughput",
            "power",
            "cost",
            "json_validity",
            "contract_validity",
            "evidence_match",
            "groundedness",
            "safety",
            "long_context_quality",
            "numerical_quality_regression",
        ],
        "optional": ["quantization calibration report"],
        "existing": ["A100 VRAM telemetry", "quality/cost aggregates"],
        "missing": ["quantized checkpoint", "precision metadata", "quality regression run"],
        "derivation": "No quantization result can be derived from unquantized baseline.",
        "sampling": "per run and per request aggregate",
        "temporal_resolution": "per config and quality evaluation",
        "experiment_type": "future_gpu_model_variant",
        "ui": ["precision_ladder", "model_memory_footprint", "speed_quality_tradeoff"],
        "acceptance": ["cost/latency improves while quality and safety pass"],
        "rejection": ["quality/safety regress", "kernel unsupported"],
    },
    "speculative_decoding": {
        "domain": "speculative_decoding",
        "readiness": "unsupported_current_runtime",
        "primary_metrics": ["acceptance_rate", "accepted_tokens_per_target_pass", "tpot_ms"],
        "secondary_metrics": ["additional_vram", "draft_latency", "throughput"],
        "required": [
            "target_model",
            "draft_model",
            "speculation_method",
            "drafted_tokens",
            "accepted_tokens",
            "rejected_tokens",
            "acceptance_rate",
            "accepted_tokens_per_target_pass",
            "draft_latency_ms",
            "target_verification_latency_ms",
            "speculation_depth",
            "draft_token_count",
            "additional_vram_mb",
            "draft_model_memory_mb",
            "tpot_itl_change",
            "throughput_change",
            "concurrency_effect",
            "quality_safety_protections",
            "automatic_disable_reason",
        ],
        "optional": ["engine speculation counters"],
        "existing": ["none"],
        "missing": ["draft model", "accept/reject telemetry", "verification latency"],
        "derivation": "Speculation acceptance cannot exist without a speculative run.",
        "sampling": "per decode verification step",
        "temporal_resolution": "per request and token verification step",
        "experiment_type": "future_gpu_engine_config",
        "ui": ["draft_tokens", "accepted_rejected_tokens", "acceptance_rate"],
        "acceptance": ["high acceptance rate and TPOT improvement with protected quality"],
        "rejection": ["low acceptance rate", "extra VRAM/cost outweighs gains"],
    },
    "model_compression": {
        "domain": "model_compression",
        "readiness": "future_architecture",
        "primary_metrics": ["model_size", "quality_regression", "latency_change"],
        "secondary_metrics": ["training_or_distillation_cost", "vram"],
        "required": ["compressed_model_id", "compression_method", "quality_regression"],
        "optional": ["distillation report"],
        "existing": ["model registry"],
        "missing": ["compressed model artifact"],
        "derivation": "No model compression is present in current artifacts.",
        "sampling": "per model artifact",
        "temporal_resolution": "future training/evaluation run",
        "experiment_type": "future_model_artifact",
        "ui": ["model_size_delta", "quality_tradeoff"],
        "acceptance": ["smaller/faster model with acceptable quality"],
        "rejection": ["quality loss exceeds budget"],
    },
    "multi_gpu_parallelism": {
        "domain": "parallelism",
        "readiness": "unsupported_current_runtime",
        "primary_metrics": ["per_gpu_utilization", "communication_time", "imbalance"],
        "secondary_metrics": ["all_reduce_time", "nvlink_nccl_metrics", "throughput"],
        "required": [
            "communication_time",
            "all_reduce_all_to_all_time",
            "per_gpu_utilization",
            "imbalance",
            "nvlink_nccl_metrics",
        ],
        "optional": ["NCCL trace"],
        "existing": ["single A100 telemetry only"],
        "missing": ["multi-GPU run", "NCCL metrics"],
        "derivation": "Single-GPU data cannot prove multi-GPU parallelism.",
        "sampling": "per GPU interval and collective operation",
        "temporal_resolution": "future multi-GPU run",
        "experiment_type": "future_hardware_architecture",
        "ui": ["per_gpu_heatmap", "communication_bar"],
        "acceptance": ["larger model or throughput improvement justifies overhead"],
        "rejection": ["communication overhead dominates"],
    },
    "kv_cache_offloading_hierarchical_cache": {
        "domain": "cache_offloading",
        "readiness": "future_architecture",
        "primary_metrics": ["offload_bytes", "transfer_latency", "cache_hit_rate"],
        "secondary_metrics": ["vram_saved", "ttft_ms", "e2e_latency_ms"],
        "required": ["offload_bytes", "transfer_latency_ms", "tier_hit_rate", "vram_saved_mb"],
        "optional": ["host memory bandwidth", "remote cache latency"],
        "existing": ["VRAM telemetry"],
        "missing": ["offload tier metrics"],
        "derivation": "Current run has no hierarchical cache layer.",
        "sampling": "per cache movement event",
        "temporal_resolution": "future architecture",
        "experiment_type": "future_cache_architecture",
        "ui": ["cache_tiers", "transfer_path", "vram_saved"],
        "acceptance": ["serves larger contexts without unacceptable latency"],
        "rejection": ["transfer overhead defeats benefit"],
    },
    "manual_kernel_compiler_optimization": {
        "domain": "kernel_compiler",
        "readiness": "requires_external_profiler",
        "primary_metrics": ["kernel_duration", "launch_count", "occupancy"],
        "secondary_metrics": ["cpu_gaps", "memory_bandwidth", "profiler_trace_paths"],
        "required": [
            "kernel_duration",
            "launch_count",
            "cpu_gaps",
            "occupancy",
            "memory_bandwidth",
            "profiler_trace_paths",
        ],
        "optional": ["Nsight Systems", "Nsight Compute", "PyTorch profiler"],
        "existing": ["none"],
        "missing": ["profiler traces"],
        "derivation": "Kernel behavior requires external profiling.",
        "sampling": "profiler trace",
        "temporal_resolution": "kernel/event timeline",
        "experiment_type": "future_profiler_guided_optimization",
        "ui": ["kernel_timeline", "occupancy", "cpu_gap_view"],
        "acceptance": ["kernel bottleneck reduced with end-to-end benefit"],
        "rejection": ["microbenchmark gain does not improve serving SLOs"],
    },
    "prefill_decode_disaggregation": {
        "domain": "disaggregation",
        "readiness": "future_architecture",
        "primary_metrics": ["prefill_queue", "decode_queue", "kv_transfer_time"],
        "secondary_metrics": ["transfer_bytes", "worker_utilization", "transfer_failures"],
        "required": [
            "prefill_queue",
            "decode_queue",
            "kv_transfer_time",
            "transfer_bytes",
            "prefill_decode_worker_utilization",
            "transfer_failures",
        ],
        "optional": ["network topology", "worker logs"],
        "existing": ["none"],
        "missing": ["separate prefill/decode workers"],
        "derivation": "Current single-node serving cannot derive disaggregation metrics.",
        "sampling": "per request and per transfer",
        "temporal_resolution": "future distributed run",
        "experiment_type": "future_disaggregated_serving",
        "ui": ["prefill_decode_queues", "kv_transfer_flow"],
        "acceptance": ["tail latency/throughput improves under mixed prompt lengths"],
        "rejection": ["transfer overhead or failures dominate"],
    },
    "distributed_capacity_serving_architecture": {
        "domain": "distributed_serving",
        "readiness": "future_architecture",
        "primary_metrics": ["replica_count", "queue_depth", "scale_events"],
        "secondary_metrics": ["cache_locality", "failover", "cross_region_latency", "cost"],
        "required": [
            "replica_count",
            "queue_depth",
            "routing_decisions",
            "scale_events",
            "cache_locality",
            "failover",
            "cross_region_latency",
            "cost",
        ],
        "optional": ["load balancer logs", "autoscaler events"],
        "existing": ["single run matrix route labels"],
        "missing": ["replicas", "autoscaler", "traffic router"],
        "derivation": "No distributed serving layer exists in current artifacts.",
        "sampling": "per route and per scale event",
        "temporal_resolution": "future distributed serving run",
        "experiment_type": "future_distributed_architecture",
        "ui": ["replica_map", "queue_depth", "scale_event_timeline"],
        "acceptance": ["capacity improves without quality/cost regression"],
        "rejection": ["complexity or cost outweighs SLO benefit"],
    },
}


ENGINE_FIELD_SUPPORT: dict[str, dict[str, str]] = {
    "vllm": {
        "ttft_ms": "available_from_existing_eval_aggregates",
        "gpu_utilization_percent": "available_from_saved_gpu_telemetry",
        "prefix_cache_hit_count": "requires_engine_metrics_or_wrapper",
        "kv_blocks_used": "requires_engine_metrics",
        "active_batch_size": "requires_wrapper_or_engine_iteration_logs",
        "chunked_prefill_state": "startup_flag_missing",
        "speculative_acceptance_rate": "unavailable_without_speculative_run",
    },
    "sglang": {
        "ttft_ms": "available_from_existing_eval_aggregates",
        "gpu_utilization_percent": "available_from_saved_gpu_telemetry",
        "radixattention_cache_metrics": "requires_runtime_logs_or_metrics",
        "kv_blocks_used": "requires_engine_metrics",
        "active_batch_size": "requires_wrapper_or_scheduler_logs",
        "chunked_prefill_state": "startup_flag_missing",
        "speculative_acceptance_rate": "unavailable_without_speculative_run",
    },
    "api_provider_route": {
        "ttft_ms": "available_from_existing_eval_aggregates",
        "api_cost_usd": "available_from_cost_report",
        "gpu_utilization_percent": "not_applicable_provider_managed",
        "prefix_cache_hit_count": "unavailable_provider_hidden",
        "kv_blocks_used": "unavailable_provider_hidden",
    },
}


class _BasePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunStartedPayload(_BasePayload):
    payload_type: Literal["run_started"] = "run_started"
    status: str
    expected_count: int | None = None


class ConfigStartedPayload(_BasePayload):
    payload_type: Literal["config_started"] = "config_started"
    memory_mode: str | None = None
    concurrency: int | None = None


class RequestArrivedPayload(_BasePayload):
    payload_type: Literal["request_arrived"] = "request_arrived"
    request_id: str
    prompt_id: str
    arrival_time: str | None = None


class RequestQueuedPayload(_BasePayload):
    payload_type: Literal["request_queued"] = "request_queued"
    request_id: str
    queue_enter_time: str | None = None


class RequestScheduledPayload(_BasePayload):
    payload_type: Literal["request_scheduled"] = "request_scheduled"
    request_id: str
    active_batch_size: int | None = None
    scheduled_tokens: int | None = None


class PrefillStartedPayload(_BasePayload):
    payload_type: Literal["prefill_started"] = "prefill_started"
    request_id: str
    input_tokens: int | None = None


class PrefillChunkCompletedPayload(_BasePayload):
    payload_type: Literal["prefill_chunk_completed"] = "prefill_chunk_completed"
    request_id: str
    chunk_index: int | None = None
    tokens_in_chunk: int | None = None


class PrefixCacheLookupPayload(_BasePayload):
    payload_type: Literal["prefix_cache_lookup"] = "prefix_cache_lookup"
    request_id: str
    prefix_hash: str | None = None
    lookup_latency_ms: float | None = None


class PrefixCacheHitPayload(_BasePayload):
    payload_type: Literal["prefix_cache_hit"] = "prefix_cache_hit"
    request_id: str
    hit_tokens: int | None = None


class PrefixCacheMissPayload(_BasePayload):
    payload_type: Literal["prefix_cache_miss"] = "prefix_cache_miss"
    request_id: str
    miss_tokens: int | None = None


class KvCacheAllocatedPayload(_BasePayload):
    payload_type: Literal["kv_cache_allocated"] = "kv_cache_allocated"
    request_id: str | None = None
    kv_blocks_used: int | None = None
    kv_blocks_free: int | None = None


class KvCacheEvictedPayload(_BasePayload):
    payload_type: Literal["kv_cache_evicted"] = "kv_cache_evicted"
    request_id: str | None = None
    evicted_blocks: int | None = None
    reason: str | None = None


class BatchIterationPayload(_BasePayload):
    payload_type: Literal["batch_iteration"] = "batch_iteration"
    iteration_index: int
    running_request_count: int | None = None
    waiting_request_count: int | None = None
    scheduled_tokens: int | None = None


class DecodeTokenPayload(_BasePayload):
    payload_type: Literal["decode_token"] = "decode_token"
    request_id: str
    token_index: int | None = None
    inter_token_latency_ms: float | None = None


class RequestCompletedPayload(_BasePayload):
    payload_type: Literal["request_completed"] = "request_completed"
    request_id: str
    output_tokens: int | None = None
    e2e_latency_ms: float | None = None


class RequestFailedPayload(_BasePayload):
    payload_type: Literal["request_failed"] = "request_failed"
    request_id: str
    error_type: str | None = None
    retry_count: int | None = None


class TelemetrySamplePayload(_BasePayload):
    payload_type: Literal["telemetry_sample"] = "telemetry_sample"
    gpu_utilization_percent: float | None = None
    vram_used_mb: float | None = None
    power_draw_w: float | None = None
    temperature_c: float | None = None


class QualityEvaluationPayload(_BasePayload):
    payload_type: Literal["quality_evaluation"] = "quality_evaluation"
    prompt_id: str | None = None
    json_valid: bool | None = None
    contract_valid: bool | None = None
    evidence_match: bool | None = None
    grounded: bool | None = None
    safety_findings: int | None = None


class OptimizationDecisionPayload(_BasePayload):
    payload_type: Literal["optimization_decision"] = "optimization_decision"
    optimization_id: str
    decision: str
    reason: str


class PromptLayoutRenderedPayload(_BasePayload):
    payload_type: Literal["prompt_layout_rendered"] = "prompt_layout_rendered"
    prompt_id: str
    layout_id: str
    memory_mode: str | None = None
    input_tokens: int | None = None


class PrefixFamilyAssignedPayload(_BasePayload):
    payload_type: Literal["prefix_family_assigned"] = "prefix_family_assigned"
    prompt_id: str
    layout_id: str
    prefix_family_id: str
    prefix_hash: str
    reusable_prefix_tokens: int | None = None


class StaticMetricComputedPayload(_BasePayload):
    payload_type: Literal["static_metric_computed"] = "static_metric_computed"
    metric_name: str
    layout_id: str
    value: float
    unit: str | None = None


class RunCompletedPayload(_BasePayload):
    payload_type: Literal["run_completed"] = "run_completed"
    status: str
    completed_count: int | None = None
    failed_count: int | None = None


ObservabilityPayload = Annotated[
    RunStartedPayload
    | ConfigStartedPayload
    | RequestArrivedPayload
    | RequestQueuedPayload
    | RequestScheduledPayload
    | PrefillStartedPayload
    | PrefillChunkCompletedPayload
    | PrefixCacheLookupPayload
    | PrefixCacheHitPayload
    | PrefixCacheMissPayload
    | KvCacheAllocatedPayload
    | KvCacheEvictedPayload
    | BatchIterationPayload
    | DecodeTokenPayload
    | RequestCompletedPayload
    | RequestFailedPayload
    | TelemetrySamplePayload
    | QualityEvaluationPayload
    | OptimizationDecisionPayload
    | PromptLayoutRenderedPayload
    | PrefixFamilyAssignedPayload
    | StaticMetricComputedPayload
    | RunCompletedPayload,
    Field(discriminator="payload_type"),
]


class ObservabilityEvent(BaseModel):
    """Discriminated optimization observability event."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["core_optimization_observability.v1"] = (
        "core_optimization_observability.v1"
    )
    timestamp: str
    run_id: str
    scenario_id: str
    config_id: str | None = None
    engine: str | None = None
    model: str | None = None
    optimization_id: str | None = None
    source: str
    measurement_type: Literal[
        "measured",
        "engine_reported",
        "derived",
        "estimated",
        "unavailable",
    ]
    event_type: Literal[
        "run_started",
        "config_started",
        "request_arrived",
        "request_queued",
        "request_scheduled",
        "prefill_started",
        "prefill_chunk_completed",
        "prefix_cache_lookup",
        "prefix_cache_hit",
        "prefix_cache_miss",
        "kv_cache_allocated",
        "kv_cache_evicted",
        "batch_iteration",
        "decode_token",
        "request_completed",
        "request_failed",
        "telemetry_sample",
        "quality_evaluation",
        "optimization_decision",
        "prompt_layout_rendered",
        "prefix_family_assigned",
        "static_metric_computed",
        "run_completed",
    ]
    payload: ObservabilityPayload

    @model_validator(mode="after")
    def payload_matches_event_type(self) -> ObservabilityEvent:
        if self.event_type != self.payload.payload_type:
            msg = "event_type must match payload.payload_type"
            raise ValueError(msg)
        return self


OBSERVABILITY_EVENT_ADAPTER = TypeAdapter(ObservabilityEvent)


@dataclass(frozen=True)
class AdapterFieldStatus:
    field_name: str
    support_state: Literal["captured", "derivable", "missing", "not_applicable"]
    measurement_type: Literal[
        "measured",
        "engine_reported",
        "derived",
        "estimated",
        "unavailable",
    ]
    source_artifact: str
    note: str


@dataclass(frozen=True)
class AdapterReport:
    adapter_id: str
    adapter_name: str
    live_engine_required: bool
    mutates_artifacts: bool
    parsed_source_artifacts: list[str]
    field_statuses: list[AdapterFieldStatus] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "adapter_id": self.adapter_id,
            "adapter_name": self.adapter_name,
            "live_engine_required": self.live_engine_required,
            "mutates_artifacts": self.mutates_artifacts,
            "parsed_source_artifacts": self.parsed_source_artifacts,
            "field_statuses": [
                {
                    "field_name": item.field_name,
                    "support_state": item.support_state,
                    "measurement_type": item.measurement_type,
                    "source_artifact": item.source_artifact,
                    "note": item.note,
                }
                for item in self.field_statuses
            ],
        }


class ObservabilityAdapter(Protocol):
    adapter_id: str

    def inspect(self) -> AdapterReport:
        """Inspect saved artifacts and report field availability."""


class ManifestAdapter:
    adapter_id = "main_inference_manifest"

    def inspect(self) -> AdapterReport:
        path = MAIN_RAW / "main_inference_v1_manifest.json"
        fields = [
            "run_id",
            "config_id",
            "git_commit",
            "model_alias",
            "model_id",
            "memory_mode",
            "runtime",
            "engine",
            "backend_type",
            "hardware",
            "provider",
            "concurrency",
            "traffic_profile",
            "prompt_count",
            "dataset_workload_hash",
            "config_hash",
            "status",
        ]
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="Main_Inference_V1 manifest adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[_display_path(path)],
            field_statuses=[
                AdapterFieldStatus(
                    field_name=item,
                    support_state="captured",
                    measurement_type="measured",
                    source_artifact=_display_path(path),
                    note="Captured in the saved run manifest.",
                )
                for item in fields
            ],
        )


class ProgressLogAdapter:
    adapter_id = "main_inference_progress_log"

    def inspect(self) -> AdapterReport:
        path = MAIN_LOGS / "main_inference_v1_progress.jsonl"
        captured = [
            "completed_requests",
            "failure_count",
            "elapsed_seconds",
            "estimated_remaining_seconds",
            "current_config_id",
            "runtime",
            "engine",
            "memory_mode",
            "concurrency",
            "vertical",
            "approximate_cost_so_far_usd",
        ]
        missing = ["request_arrival_time", "queue_enter_time", "prefill_start", "decode_end"]
        statuses = [
            AdapterFieldStatus(
                field_name=item,
                support_state="captured",
                measurement_type="measured",
                source_artifact=_display_path(path),
                note="Captured at progress-checkpoint cadence, not per request.",
            )
            for item in captured
        ]
        statuses.extend(
            AdapterFieldStatus(
                field_name=item,
                support_state="missing",
                measurement_type="unavailable",
                source_artifact=_display_path(path),
                note="Progress log does not expose per-request lifecycle events.",
            )
            for item in missing
        )
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="Progress log adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[_display_path(path)],
            field_statuses=statuses,
        )


class GpuTelemetryAdapter:
    adapter_id = "main_inference_gpu_telemetry"

    def inspect(self) -> AdapterReport:
        path = MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl"
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="GPU telemetry adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[_display_path(path)],
            field_statuses=[
                AdapterFieldStatus(
                    field_name=item,
                    support_state="captured",
                    measurement_type="measured",
                    source_artifact=_display_path(path),
                    note="Captured for self-hosted GPU phases; API route is provider-managed.",
                )
                for item in [
                    "gpu_utilization_percent",
                    "vram_used_mb",
                    "vram_total_mb",
                    "power_draw_w",
                    "temperature_c",
                    "active_engine_process",
                    "telemetry_timestamp",
                ]
            ],
        )


class EvalReportAdapter:
    adapter_id = "main_inference_eval_report"

    def inspect(self) -> AdapterReport:
        path = MAIN_PROCESSED / "main_inference_v1_eval_report.json"
        captured = [
            "ttft_ms",
            "tpot_ms",
            "e2e_latency_ms",
            "tokens_per_second",
            "json_validity",
            "contract_validity",
            "format_validity",
            "evidence_match",
            "groundedness",
            "safety_findings",
            "truncation",
            "input_tokens",
            "output_tokens",
        ]
        missing = ["queue_wait_ms", "prefill_latency_ms", "active_batch_size"]
        statuses = [
            AdapterFieldStatus(
                field_name=item,
                support_state="captured",
                measurement_type="measured",
                source_artifact=_display_path(path),
                note="Available as aggregate or per-config aggregate evaluation output.",
            )
            for item in captured
        ]
        statuses.extend(
            AdapterFieldStatus(
                field_name=item,
                support_state="missing",
                measurement_type="unavailable",
                source_artifact=_display_path(path),
                note="Not present in saved evaluation report.",
            )
            for item in missing
        )
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="Evaluation report adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[_display_path(path)],
            field_statuses=statuses,
        )


class CostReportAdapter:
    adapter_id = "main_inference_cost_report"

    def inspect(self) -> AdapterReport:
        path = MAIN_PROCESSED / "main_inference_v1_cost_report.json"
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="Cost report adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[_display_path(path)],
            field_statuses=[
                AdapterFieldStatus(
                    field_name=item,
                    support_state="captured",
                    measurement_type="measured",
                    source_artifact=_display_path(path),
                    note="Captured in the saved post-run cost report.",
                )
                for item in [
                    "gpu_cost_usd",
                    "api_cost_usd",
                    "total_cost_usd",
                    "gpu_hourly_price_usd",
                    "self_hosted_request_count",
                    "api_request_count",
                ]
            ],
        )


class EngineMetricsAdapter:
    adapter_id = "engine_metrics_unavailable"

    def inspect(self) -> AdapterReport:
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="vLLM/SGLang engine metrics adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[
                "missing vLLM metrics endpoint scrape",
                "missing SGLang metrics/log scrape",
            ],
            field_statuses=[
                AdapterFieldStatus(
                    field_name=item,
                    support_state="missing",
                    measurement_type="unavailable",
                    source_artifact="not captured in Main_Inference_V1",
                    note="Adapter interface is defined, but saved engine counters are absent.",
                )
                for item in [
                    "cache_hit_count",
                    "cache_miss_count",
                    "kv_blocks_used",
                    "active_batch_size",
                    "chunk_count",
                    "speculative_acceptance_rate",
                ]
            ],
        )


class PrefixLayoutStaticAdapter:
    adapter_id = "prefix_layout_static_analysis"

    def inspect(self) -> AdapterReport:
        return AdapterReport(
            adapter_id=self.adapter_id,
            adapter_name="Static prefix-layout adapter",
            live_engine_required=False,
            mutates_artifacts=False,
            parsed_source_artifacts=[_display_path(path) for path in _workload_paths()],
            field_statuses=[
                AdapterFieldStatus(
                    field_name=item,
                    support_state="derivable",
                    measurement_type="estimated",
                    source_artifact=_display_path(WORKLOAD_ROOT),
                    note=(
                        "Derived from deterministic prompt text tokenization; not historical "
                        "engine cache telemetry."
                    ),
                )
                for item in [
                    "rendered_prompt_hash",
                    "tokenized_prefix_hash",
                    "exact_shared_prefix_token_count",
                    "reusable_token_ratio",
                    "prefix_family_id",
                    "variable_suffix_token_count",
                ]
            ],
        )


def all_adapters() -> list[ObservabilityAdapter]:
    return [
        ManifestAdapter(),
        ProgressLogAdapter(),
        GpuTelemetryAdapter(),
        EvalReportAdapter(),
        CostReportAdapter(),
        EngineMetricsAdapter(),
        PrefixLayoutStaticAdapter(),
    ]


def build_metric_envelopes() -> JsonDict:
    return {
        "run_identity": {
            "description": "Stable run and scenario identity fields.",
            "fields": METRIC_ENVELOPES["run_identity"],
        },
        "request_lifecycle": {
            "description": (
                "Nullable per-request timing events; runtimes emit only what they expose."
            ),
            "fields": METRIC_ENVELOPES["request_lifecycle"],
        },
        "performance": {
            "description": "Latency, queueing, throughput, and batch efficiency metrics.",
            "fields": METRIC_ENVELOPES["performance"],
        },
        "resources": {
            "description": "GPU/CPU/RAM/power/process resource telemetry.",
            "fields": METRIC_ENVELOPES["resources"],
        },
        "quality_safety_protection": {
            "description": "Protected answer-quality and safety gates.",
            "fields": METRIC_ENVELOPES["quality_safety_protection"],
        },
        "cost": {
            "description": "GPU/API/total cost and normalized economics.",
            "fields": METRIC_ENVELOPES["cost"],
        },
    }


def _observability_entry(optimization_id: str, definition: JsonDict) -> JsonDict:
    spec = DOMAIN_SPECS[optimization_id]
    return {
        "optimization_id": optimization_id,
        "display_name": definition["display_name"],
        "optimization_domain": spec["domain"],
        "difficulty_tier": definition["difficulty_tier"],
        "problem_statement": _problem_statement(optimization_id, definition),
        "mechanism": spec.get("mechanism") or definition["mechanism"],
        "hypothesis": _hypothesis(optimization_id, definition),
        "primary_metrics": spec["primary_metrics"],
        "secondary_metrics": spec["secondary_metrics"],
        "protected_quality_safety_metrics": definition["protected_metrics"],
        "required_instrumentation": spec["required"],
        "optional_instrumentation": spec["optional"],
        "existing_instrumentation": spec["existing"],
        "missing_instrumentation": spec["missing"],
        "derivation_method": spec["derivation"],
        "sampling_frequency": spec["sampling"],
        "temporal_resolution": spec["temporal_resolution"],
        "aggregation_levels": _aggregation_levels(optimization_id),
        "engine_specific_fields": _engine_specific_fields(optimization_id),
        "model_specific_fields": _model_specific_fields(optimization_id),
        "device_requirements": _device_requirements(optimization_id, definition),
        "external_profiler_requirements": _external_profiler_requirements(optimization_id),
        "experiment_type": spec["experiment_type"],
        "ui_visualization_contract": {
            "visualizations": spec["ui"],
            "state_labels": [
                "instrumented",
                "partially_instrumented",
                "instrumentation_missing",
                "planned",
                "measured",
                "unavailable",
                "future",
            ],
            "measurement_labels_required": True,
        },
        "acceptance_evidence": spec["acceptance"],
        "rejection_evidence": spec["rejection"],
        "instrumentation_readiness_state": spec["readiness"],
        "instrumentation_readiness_state_does_not_unblock_optimization": True,
    }


def _problem_statement(optimization_id: str, definition: JsonDict) -> str:
    problem_by_id = {
        "prompt_prefix_layout_optimization": (
            "Rendered prompts may not put stable instructions and schemas into the longest "
            "possible exact leading prefix."
        ),
        "prefix_cache_verification_tuning": (
            "The run cannot prove whether cross-request prefix caching was active or useful."
        ),
        "scheduler_batch_tuning": (
            "Configured concurrency does not prove actual batch composition or scheduler "
            "efficiency."
        ),
        "kv_cache_capacity_allocation_tuning": (
            "VRAM was sampled, but KV block occupancy and eviction pressure were not measured."
        ),
        "chunked_prefill_tuning": (
            "Long context can increase TTFT and interfere with decode work if prefill is "
            "not balanced."
        ),
        "cache_workload_aware_routing": (
            "Requests are not yet routed by cache affinity or workload shape."
        ),
        "model_selection_routing": (
            "The platform has measured model/route differences but no per-request selection policy."
        ),
        "quantization": (
            "Lower precision could reduce cost or memory, but quality failed and "
            "quantization is blocked."
        ),
        "speculative_decoding": (
            "Decode speed may improve with a draft model, but no speculation support or "
            "telemetry exists."
        ),
    }
    return problem_by_id.get(optimization_id, str(definition["definition"]))


def _hypothesis(optimization_id: str, definition: JsonDict) -> str:
    hypothesis_by_id = {
        "prompt_prefix_layout_optimization": (
            "If stable prompt sections move earlier and remain token-identical, prefix reuse "
            "opportunity increases without changing answer semantics."
        ),
        "prefix_cache_verification_tuning": (
            "If exact prefix families exist and engine cache hits are recorded, TTFT and prefill "
            "work should improve for hit requests."
        ),
        "scheduler_batch_tuning": (
            "If queueing and batch budgets are tuned from observed scheduler metrics, throughput "
            "can improve without tail-latency regression."
        ),
        "kv_cache_capacity_allocation_tuning": (
            "If KV allocation pressure is visible, memory settings can reduce eviction "
            "and OOM risk."
        ),
        "chunked_prefill_tuning": (
            "If long prefills are chunked appropriately, long-request TTFT improves while decode "
            "continues for shorter requests."
        ),
    }
    return hypothesis_by_id.get(
        optimization_id,
        f"{definition['display_name']} can improve {', '.join(definition['target_metrics'])} "
        "only if its mechanism is instrumented and protected quality gates pass.",
    )


def _aggregation_levels(optimization_id: str) -> list[str]:
    common = ["per_run", "per_config", "per_engine", "per_memory_mode", "per_vertical"]
    if optimization_id in {
        "prompt_prefix_layout_optimization",
        "prefix_cache_verification_tuning",
        "chunked_prefill_tuning",
    }:
        return [*common, "per_prompt_length_bucket", "per_prefix_family"]
    if optimization_id in {"scheduler_batch_tuning", "kv_cache_capacity_allocation_tuning"}:
        return [*common, "per_request", "per_engine_iteration", "per_concurrency_level"]
    if optimization_id in {"cache_workload_aware_routing", "model_selection_routing"}:
        return [*common, "per_route", "per_complexity_class"]
    return common


def _engine_specific_fields(optimization_id: str) -> JsonDict:
    if optimization_id == "prefix_cache_verification_tuning":
        return {
            "vllm": {
                "cache_lookup_count": "requires engine metrics endpoint or wrapper",
                "cache_hit_count": "requires engine metrics endpoint or wrapper",
                "cache_blocks": "requires engine metrics",
            },
            "sglang": {
                "radixattention_cache_metrics": "requires runtime logs or metrics",
                "cache_hit_count": "requires wrapper instrumentation if not logged",
            },
        }
    if optimization_id == "scheduler_batch_tuning":
        return {
            "vllm": {
                "active_batch_size": "requires scheduler/iteration telemetry",
                "max_num_seqs": "startup/runtime config field",
                "max_num_batched_tokens": "startup/runtime config field",
            },
            "sglang": {
                "running_waiting_requests": "requires scheduler logs",
                "batch_waiting_delay": "runtime config and observed queue timing",
            },
        }
    if optimization_id == "chunked_prefill_tuning":
        return {
            "vllm": {"chunked_prefill_enabled": "startup flag plus engine iteration metrics"},
            "sglang": {"chunked_prefill_enabled": "startup flag plus chunk counters"},
        }
    return ENGINE_FIELD_SUPPORT


def _model_specific_fields(optimization_id: str) -> JsonDict:
    if optimization_id == "quantization":
        return {
            "model3_7b": ["precision_format", "weight_format", "calibration_dataset"],
            "model6_gated": ["provider_hidden_precision", "provider_route_cost"],
        }
    if optimization_id == "speculative_decoding":
        return {
            "target_model": "model3_7b",
            "draft_model": "must be explicitly selected before experiment",
        }
    return {"model3_7b": ["tokenizer_name", "model_id"], "model6_gated": ["api_provider_route"]}


def _device_requirements(optimization_id: str, definition: JsonDict) -> list[str]:
    if optimization_id == "prompt_prefix_layout_optimization":
        return ["CPU for static prefix audit", "GPU/API only for later measured latency rerun"]
    if definition.get("hardware_compatibility"):
        return list(cast(list[str], definition["hardware_compatibility"]))
    if optimization_id in {"multi_gpu_parallelism", "distributed_capacity_serving_architecture"}:
        return ["future multi-GPU or distributed serving hardware"]
    return ["future architecture or provider-managed route"]


def _external_profiler_requirements(optimization_id: str) -> list[str]:
    if optimization_id == "manual_kernel_compiler_optimization":
        return ["PyTorch profiler optional", "Nsight Systems", "Nsight Compute"]
    if optimization_id in {"multi_gpu_parallelism", "prefill_decode_disaggregation"}:
        return ["distributed trace optional", "NCCL/network telemetry optional"]
    return []


def build_observability_registry() -> JsonDict:
    core_by_id = _core_optimizations_by_id()
    missing = sorted(set(core_by_id) - set(DOMAIN_SPECS))
    if missing:
        msg = "Missing observability specs: " + ", ".join(missing)
        raise ValueError(msg)
    entries = [_observability_entry(item, core_by_id[item]) for item in sorted(core_by_id)]
    return {
        "version": 1,
        "status": "OBSERVABILITY_REGISTRY_READY_PLANNING_ONLY",
        "does_not_execute_inference": True,
        "does_not_create_optimized_inference_v1": True,
        "does_not_mutate_main_inference_v1": True,
        "readiness_states": sorted(READINESS_STATES),
        "measurement_types": sorted(MEASUREMENT_TYPES),
        "optimization_domains": sorted(OPTIMIZATION_DOMAINS),
        "measurement_semantics": {
            "measured": "Captured from saved run, engine, evaluator, or telemetry artifacts.",
            "engine_reported": "Reported directly by a serving engine metrics endpoint or log.",
            "derived": "Calculated from measured fields without inventing missing counters.",
            "estimated": (
                "Planning estimate from static analysis or formulas; not measured telemetry."
            ),
            "unavailable": "Not captured and not derivable from current artifacts.",
        },
        "metric_envelopes": build_metric_envelopes(),
        "optimizations": entries,
        "deployability_repairs_excluded": sorted(REPAIR_IDS),
    }


def load_observability_registry(
    path: str | Path = OBSERVABILITY_CONFIG_PATH,
) -> JsonDict:
    payload = yaml.safe_load(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Core optimization observability registry must be a mapping: {path}"
        raise ValueError(msg)
    return cast(JsonDict, payload)


def build_observability_readiness(registry: JsonDict | None = None) -> JsonDict:
    payload = registry or build_observability_registry()
    entries = cast(list[JsonDict], payload["optimizations"])
    rows: list[JsonDict] = []
    for entry in entries:
        required = cast(list[str], entry["required_instrumentation"])
        missing = cast(list[str], entry["missing_instrumentation"])
        available = cast(list[str], entry["existing_instrumentation"])
        rows.append(
            {
                "optimization_id": entry["optimization_id"],
                "display_name": entry["display_name"],
                "optimization_domain": entry["optimization_domain"],
                "difficulty_tier": entry["difficulty_tier"],
                "instrumentation_readiness_state": entry["instrumentation_readiness_state"],
                "required_field_count": len(required),
                "currently_available_field_count": len(available),
                "missing_field_count": len(missing),
                "external_profiler_required": bool(entry["external_profiler_requirements"]),
                "requires_gpu_to_measure_effect": entry["experiment_type"]
                not in {"cpu_static_audit_then_one_factor_latency_rerun"},
                "can_be_shown_in_public_ui": True,
            }
        )
    return {
        "version": 1,
        "run_id": RUN_ID,
        "status": "OBSERVABILITY_READINESS_PLANNING_ONLY",
        "result_type": "planned",
        "summary": {
            "optimization_count": len(rows),
            "ready_existing": sum(
                item["instrumentation_readiness_state"] == "ready_existing" for item in rows
            ),
            "ready_derivable": sum(
                item["instrumentation_readiness_state"] == "ready_derivable" for item in rows
            ),
            "requires_instrumentation_or_engine_metrics": sum(
                item["instrumentation_readiness_state"]
                in {"requires_runner_instrumentation", "requires_engine_metrics"}
                for item in rows
            ),
            "future_or_unsupported": sum(
                item["instrumentation_readiness_state"]
                in {"future_architecture", "unsupported_current_runtime"}
                for item in rows
            ),
        },
        "rows": rows,
        "no_optimization_marked_measured": True,
    }


def build_adapter_coverage() -> JsonDict:
    reports = [adapter.inspect().to_dict() for adapter in all_adapters()]
    return {
        "version": 1,
        "run_id": RUN_ID,
        "status": "ADAPTER_COVERAGE_REPORTED_NO_LIVE_ENGINE_REQUIRED",
        "result_type": "planned",
        "adapters": reports,
        "summary": {
            "adapter_count": len(reports),
            "live_engine_required": any(report["live_engine_required"] for report in reports),
            "mutates_artifacts": any(report["mutates_artifacts"] for report in reports),
            "missing_field_count": sum(
                1
                for report in reports
                for field_status in cast(list[JsonDict], report["field_statuses"])
                if field_status["support_state"] == "missing"
            ),
        },
    }


def build_observability_inventory() -> JsonDict:
    rows = [
        _inventory_row(
            "run_id",
            captured=True,
            derivable=False,
            source=MAIN_RAW / "main_inference_v1_manifest.json",
            resolution="run",
            engine_support="all routes",
            model_support="all measured models",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "config_id",
            captured=True,
            derivable=False,
            source=MAIN_PROCESSED / "main_inference_v1_api_vs_self_hosted_comparison.csv",
            resolution="per config aggregate",
            engine_support="vLLM, SGLang, API provider route",
            model_support="model3_7b, model6_gated",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "ttft_ms",
            captured=True,
            derivable=False,
            source=MAIN_PROCESSED / "main_inference_v1_eval_report.json",
            resolution="aggregate and per config aggregate",
            engine_support="vLLM, SGLang, API provider route",
            model_support="model3_7b, model6_gated",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "tpot_ms",
            captured=True,
            derivable=False,
            source=MAIN_PROCESSED / "main_inference_v1_eval_report.json",
            resolution="aggregate and per config aggregate",
            engine_support="vLLM, SGLang, API provider route",
            model_support="model3_7b, model6_gated",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "e2e_latency_ms",
            captured=True,
            derivable=False,
            source=MAIN_PROCESSED / "main_inference_v1_eval_report.json",
            resolution="aggregate and per config aggregate",
            engine_support="vLLM, SGLang, API provider route",
            model_support="model3_7b, model6_gated",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "tokens_per_second",
            captured=True,
            derivable=False,
            source=MAIN_PROCESSED / "main_inference_v1_eval_report.json",
            resolution="aggregate and per config aggregate",
            engine_support="vLLM, SGLang, API provider route",
            model_support="model3_7b, model6_gated",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "input_output_tokens",
            captured=True,
            derivable=True,
            source=MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv",
            resolution="per config aggregate",
            engine_support="all measured routes",
            model_support="all measured models",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "gpu_utilization_percent",
            captured=True,
            derivable=False,
            source=MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl",
            resolution="telemetry sample interval",
            engine_support="self-hosted GPU only",
            model_support="model3_7b",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "vram_power_temperature",
            captured=True,
            derivable=False,
            source=MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl",
            resolution="telemetry sample interval",
            engine_support="self-hosted GPU only",
            model_support="model3_7b",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "quality_safety_metrics",
            captured=True,
            derivable=False,
            source=MAIN_PROCESSED / "main_inference_v1_eval_report.json",
            resolution="aggregate and per config aggregate",
            engine_support="all measured routes",
            model_support="all measured models",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "cost_metrics",
            captured=True,
            derivable=True,
            source=MAIN_PROCESSED / "main_inference_v1_cost_report.json",
            resolution="run aggregate",
            engine_support="all measured routes",
            model_support="all measured models",
            code_changes=False,
            external_profiler=False,
        ),
        _inventory_row(
            "exact_prefix_reuse_opportunity",
            captured=False,
            derivable=True,
            source=WORKLOAD_ROOT,
            resolution="static per prompt and per prefix family",
            engine_support="all routes as planning estimate",
            model_support="tokenizer-dependent planning estimate",
            code_changes=False,
            external_profiler=False,
            measurement_type="estimated",
        ),
        _inventory_row(
            "queue_wait_ms",
            captured=False,
            derivable=False,
            source="not captured",
            resolution="unavailable",
            engine_support="requires vLLM/SGLang/API wrapper instrumentation",
            model_support="all measured models after instrumentation",
            code_changes=True,
            external_profiler=False,
            public_safe=True,
        ),
        _inventory_row(
            "active_batch_size",
            captured=False,
            derivable=False,
            source="not captured",
            resolution="unavailable",
            engine_support="requires engine iteration or wrapper instrumentation",
            model_support="self-hosted GPU models",
            code_changes=True,
            external_profiler=False,
            public_safe=True,
        ),
        _inventory_row(
            "prefix_cache_hit_miss",
            captured=False,
            derivable=False,
            source="not captured",
            resolution="unavailable",
            engine_support="requires vLLM/SGLang engine metrics",
            model_support="self-hosted GPU models",
            code_changes=True,
            external_profiler=False,
            public_safe=True,
        ),
        _inventory_row(
            "kv_cache_blocks",
            captured=False,
            derivable=False,
            source="not captured",
            resolution="unavailable",
            engine_support="requires vLLM/SGLang engine metrics",
            model_support="self-hosted GPU models",
            code_changes=True,
            external_profiler=False,
            public_safe=True,
        ),
        _inventory_row(
            "speculative_acceptance_rate",
            captured=False,
            derivable=False,
            source="not captured",
            resolution="unavailable",
            engine_support="requires speculative decoding run",
            model_support="future target/draft model pair",
            code_changes=True,
            external_profiler=False,
            public_safe=True,
        ),
        _inventory_row(
            "kernel_profiler_trace",
            captured=False,
            derivable=False,
            source="not captured",
            resolution="unavailable",
            engine_support="requires external profiler",
            model_support="self-hosted GPU models",
            code_changes=True,
            external_profiler=True,
            public_safe=False,
        ),
    ]
    return {
        "version": 1,
        "run_id": RUN_ID,
        "status": "OBSERVABILITY_AUDIT_COMPLETE_PLANNING_ONLY",
        "result_type": "planned",
        "rows": rows,
        "summary": {
            "captured_count": sum(1 for row in rows if row["currently_captured"]),
            "derivable_count": sum(1 for row in rows if row["derivable"]),
            "missing_count": sum(1 for row in rows if row["missing"]),
            "public_ui_safe_count": sum(1 for row in rows if row["safe_for_public_ui"]),
        },
        "audit_note": (
            "Captured values come from saved Main_Inference_V1 artifacts. Missing telemetry "
            "remains unavailable, not zero."
        ),
    }


def _inventory_row(
    metric: str,
    *,
    captured: bool,
    derivable: bool,
    source: str | Path,
    resolution: str,
    engine_support: str,
    model_support: str,
    code_changes: bool,
    external_profiler: bool,
    public_safe: bool = True,
    measurement_type: str | None = None,
) -> JsonDict:
    return {
        "metric_or_instrumentation_field": metric,
        "currently_captured": captured,
        "derivable": derivable,
        "missing": not captured and not derivable,
        "source_artifact": _display_path(source) if source != "not captured" else "not captured",
        "temporal_resolution": resolution,
        "engine_support": engine_support,
        "model_support": model_support,
        "code_changes_required": code_changes,
        "external_profiling_required": external_profiler,
        "safe_for_public_ui": public_safe,
        "measurement_type": measurement_type
        or ("measured" if captured else "derived" if derivable else "unavailable"),
    }


def _workload_paths() -> list[Path]:
    if _repo_path(AUTHORITATIVE_WORKLOAD_PATH).exists():
        return [AUTHORITATIVE_WORKLOAD_PATH]
    return []


def _related_workload_artifacts() -> list[str]:
    roots = [WORKLOAD_ROOT / "prompt_plus_metadata", WORKLOAD_ROOT]
    related: dict[str, str] = {}
    for root in roots:
        target = _repo_path(root)
        if not target.exists():
            continue
        for path in sorted(target.glob("mm*_*.jsonl")):
            if "prompt_plus_source_hints" in path.as_posix():
                continue
            relative = Path(path).relative_to(REPO_ROOT)
            related[relative.as_posix()] = _sha256_file(relative)
    return sorted(related)


def analyze_prefix_opportunity(
    workload_paths: Sequence[str | Path] | None = None,
) -> JsonDict:
    paths = (
        [Path(path) for path in workload_paths] if workload_paths is not None else _workload_paths()
    )
    family_counter: Counter[str] = Counter()
    family_prefix_tokens: dict[str, int] = {}
    family_examples: dict[str, JsonDict] = {}
    vertical_totals: dict[str, list[float]] = defaultdict(list)
    memory_totals: dict[str, list[float]] = defaultdict(list)
    length_buckets: Counter[str] = Counter()
    token_lists_by_path: dict[str, list[list[str]]] = defaultdict(list)
    rows_seen = 0
    source_hashes: dict[str, str] = {}
    detail_rows: list[JsonDict] = []

    for path in paths:
        if not _repo_path(path).exists():
            continue
        source_hashes[_display_path(path)] = _sha256_file(path)
        for row in _iter_jsonl(path):
            rows_seen += 1
            rendered = _render_messages(row)
            prefix_text = _static_prefix_text(row)
            full_tokens = _tokens(rendered)
            prefix_tokens = _tokens(prefix_text)
            suffix_count = max(0, len(full_tokens) - len(prefix_tokens))
            ratio = len(prefix_tokens) / max(len(full_tokens), 1)
            prefix_hash = _sha256_text("\n".join(prefix_tokens))
            family_id = f"prefix_family_{prefix_hash[:12]}"
            family_counter[family_id] += 1
            family_prefix_tokens[family_id] = len(prefix_tokens)
            vertical = str(row.get("vertical") or "unknown")
            memory_mode = str(row.get("memory_mode") or Path(path).stem)
            vertical_totals[vertical].append(ratio)
            memory_totals[memory_mode].append(ratio)
            length_buckets[_bucket(len(full_tokens))] += 1
            token_lists_by_path[_display_path(path)].append(full_tokens)
            family_examples.setdefault(
                family_id,
                {
                    "prefix_family_id": family_id,
                    "example_prompt_id": row.get("prompt_id"),
                    "memory_mode": memory_mode,
                    "vertical": vertical,
                    "tokenized_prefix_hash": prefix_hash,
                    "exact_shared_prefix_token_count": len(prefix_tokens),
                    "rendered_prompt_hash": _sha256_text(rendered),
                },
            )
            detail_rows.append(
                {
                    "prefix_family_id": family_id,
                    "prompt_id": row.get("prompt_id"),
                    "vertical": vertical,
                    "memory_mode": memory_mode,
                    "total_input_tokens_estimated": len(full_tokens),
                    "exact_shared_prefix_tokens_estimated": len(prefix_tokens),
                    "variable_suffix_tokens_estimated": suffix_count,
                    "reusable_token_ratio_estimated": round(ratio, 6),
                    "tokenizer": "repo_regex_tokenizer_v1",
                    "measurement_type": "estimated",
                }
            )

    family_rows = [
        {
            "prefix_family_id": family_id,
            "request_count": count,
            "exact_shared_prefix_token_count": family_prefix_tokens[family_id],
            "example_prompt_id": family_examples[family_id]["example_prompt_id"],
            "memory_mode": family_examples[family_id]["memory_mode"],
            "vertical_example": family_examples[family_id]["vertical"],
            "tokenized_prefix_hash": family_examples[family_id]["tokenized_prefix_hash"],
        }
        for family_id, count in family_counter.most_common()
    ]
    ratios = [
        float(row["reusable_token_ratio_estimated"])
        for row in detail_rows
        if row["reusable_token_ratio_estimated"] is not None
    ]
    longest_common_by_source = {
        source: _common_prefix_length(token_lists)
        for source, token_lists in token_lists_by_path.items()
    }
    source_artifacts = [_display_path(path) for path in paths if _repo_path(path).exists()]
    return {
        "version": 1,
        "run_id": RUN_ID,
        "scenario_id": "coreopt_prefix_layout_static_v1",
        "status": "PREFIX_OPPORTUNITY_ANALYZED_PLANNING_ONLY",
        "result_type": "planned",
        "inference_executed": False,
        "optimization_applied": False,
        "historical_cache_reuse_claimed": False,
        "semantic_similarity_counted": False,
        "tokenizer": {
            "name": "repo_regex_tokenizer_v1",
            "version": "1",
            "hash": _sha256_text("repo_regex_tokenizer_v1:\\S+|[^\\w\\s]")[:16],
            "label": "planning estimate tokenizer, not engine historical tokenizer",
        },
        "source_artifacts": source_artifacts,
        "source_hashes": source_hashes,
        "related_rendered_workload_artifacts": _related_workload_artifacts(),
        "missing_memory_mode_artifacts": [
            item
            for item in ["mm4_bounded_agentic"]
            if not any(item in source for source in source_artifacts)
        ],
        "summary": {
            "rows_scanned": rows_seen,
            "prefix_family_count": len(family_rows),
            "largest_prefix_family_request_count": family_rows[0]["request_count"]
            if family_rows
            else 0,
            "mean_reusable_token_ratio_estimated": _mean(ratios),
            "p50_reusable_token_ratio_estimated": _percentile(ratios, 0.50),
            "p95_reusable_token_ratio_estimated": _percentile(ratios, 0.95),
            "longest_exact_common_prefix_by_source": longest_common_by_source,
        },
        "distribution_by_vertical": [
            {
                "vertical": vertical,
                "request_count": len(values),
                "mean_reusable_token_ratio_estimated": _mean(values),
            }
            for vertical, values in sorted(vertical_totals.items())
        ],
        "distribution_by_memory_mode": [
            {
                "memory_mode": memory_mode,
                "request_count": len(values),
                "mean_reusable_token_ratio_estimated": _mean(values),
            }
            for memory_mode, values in sorted(memory_totals.items())
        ],
        "input_length_buckets": dict(sorted(length_buckets.items())),
        "prefix_families": family_rows[:50],
        "stable_sections": {
            "common_system_prompt_tokens": "derived from leading system message",
            "common_generation_contract_tokens": "derived from shared instruction text",
            "common_evidence_schema_tokens": "derived from context-header scaffolding",
            "common_tool_definition_tokens": "none in current rendered workload",
        },
        "candidate_layout_structure": [
            "system instruction",
            "generation contract",
            "evidence schema/citation rules",
            "memory mode and route labels",
            "context records",
            "request-specific question",
        ],
        "measurement_plan": [
            "Freeze baseline prompt layout and candidate layout.",
            "Hash rendered prompts and tokenized prefixes.",
            "Confirm gold IDs are not leaked into prompts beyond allowed context IDs.",
            "Run one-factor latency experiment only after deployability repairs stay protected.",
            "Measure actual engine prefix cache hits before claiming cache reuse.",
        ],
        "csv_detail_rows": detail_rows,
    }


def compare_prefix_layouts(
    baseline_prompts: Sequence[str],
    candidate_prompts: Sequence[str],
) -> JsonDict:
    if len(baseline_prompts) != len(candidate_prompts):
        msg = "baseline_prompts and candidate_prompts must have the same length"
        raise ValueError(msg)
    baseline_tokens = [_tokens(item) for item in baseline_prompts]
    candidate_tokens = [_tokens(item) for item in candidate_prompts]
    baseline_common = _common_prefix_length(baseline_tokens)
    candidate_common = _common_prefix_length(candidate_tokens)
    baseline_total = sum(len(item) for item in baseline_tokens)
    candidate_total = sum(len(item) for item in candidate_tokens)
    return {
        "baseline_longest_exact_common_prefix": baseline_common,
        "candidate_longest_exact_common_prefix": candidate_common,
        "common_prefix_token_delta": candidate_common - baseline_common,
        "baseline_total_tokens": baseline_total,
        "candidate_total_tokens": candidate_total,
        "token_savings": baseline_total - candidate_total,
        "semantic_similarity_used": False,
        "measurement_type": "derived",
    }


def build_event_schema() -> JsonDict:
    schema = OBSERVABILITY_EVENT_ADAPTER.json_schema()
    return {
        "version": 1,
        "schema_name": "core_optimization_observability_event",
        "status": "EVENT_SCHEMA_READY_PLANNING_ONLY",
        "event_types": sorted(EVENT_TYPES),
        "measurement_types": sorted(MEASUREMENT_TYPES),
        "schema": schema,
        "example_event": {
            "schema_version": "core_optimization_observability.v1",
            "timestamp": "2026-07-09T00:00:00+00:00",
            "run_id": RUN_ID,
            "scenario_id": "coreopt_prefix_layout_static_v1",
            "config_id": "static_prefix_audit",
            "engine": None,
            "model": "model3_7b",
            "optimization_id": "prompt_prefix_layout_optimization",
            "source": "data/workloads/final_10000/prompt_plus_metadata/mm2_hybrid_top5.jsonl",
            "measurement_type": "estimated",
            "event_type": "optimization_decision",
            "payload": {
                "payload_type": "optimization_decision",
                "optimization_id": "prompt_prefix_layout_optimization",
                "decision": "instrumentation_ready_for_static_audit",
                "reason": "Rendered prompts are available for deterministic prefix analysis.",
            },
        },
    }


def build_instrumentation_plan(
    scenario_id: str,
    registry: JsonDict | None = None,
) -> JsonDict:
    scenario_to_optimization = {
        "coreopt_prefix_layout_static_v1": "prompt_prefix_layout_optimization",
        "coreopt_scheduler_batch_vllm_v1": "scheduler_batch_tuning",
        "coreopt_prefix_cache_vllm_v1": "prefix_cache_verification_tuning",
        "coreopt_chunked_prefill_sglang_v1": "chunked_prefill_tuning",
    }
    optimization_id = scenario_to_optimization[scenario_id]
    payload = registry or build_observability_registry()
    entry = next(
        item
        for item in cast(list[JsonDict], payload["optimizations"])
        if item["optimization_id"] == optimization_id
    )
    required = cast(list[str], entry["required_instrumentation"])
    missing = set(cast(list[str], entry["missing_instrumentation"]))
    available = cast(list[str], entry["existing_instrumentation"])
    return {
        "version": 1,
        "run_id": RUN_ID,
        "scenario_id": scenario_id,
        "optimization_id": optimization_id,
        "status": "INSTRUMENTATION_PLAN_READY_NO_EXPERIMENT_RUN",
        "result_type": "planned",
        "instrumentation_readiness": entry["instrumentation_readiness_state"],
        "required_fields": required,
        "currently_available_fields": available,
        "missing_fields": sorted(missing),
        "implementation_blockers": _implementation_blockers(entry),
        "promotion_criteria": [
            "instrumentation fields are captured or explicitly marked unavailable",
            "one-factor changed variable is frozen",
            "protected quality and safety gates are evaluated",
            "no missing telemetry is rendered as zero",
            "human approval is recorded before champion selection",
        ],
        "ui_replay_readiness": entry["instrumentation_readiness_state"]
        in {"ready_existing", "ready_derivable"},
        "does_not_change_scenario_result_type": True,
    }


def _implementation_blockers(entry: JsonDict) -> list[str]:
    state = str(entry["instrumentation_readiness_state"])
    if state == "ready_derivable":
        return ["No live blocker for static audit; GPU needed only for measured latency rerun."]
    if state == "requires_runner_instrumentation":
        return ["Add request lifecycle and decision logging to the runner/wrapper."]
    if state == "requires_engine_metrics":
        return ["Capture serving-engine metrics endpoint or logs during the one-factor run."]
    if state == "requires_external_profiler":
        return ["Collect external profiler traces before claiming kernel/compiler effects."]
    if state == "unsupported_current_runtime":
        return ["Execution path is not implemented in the current project runtime."]
    return ["Future architecture is outside the current one-A100 saved-artifact demo."]


def build_updated_scenario_registry(existing_path: str | Path = SCENARIO_REGISTRY_PATH) -> JsonDict:
    existing = yaml.safe_load(_repo_path(existing_path).read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        msg = "Scenario registry must be a YAML mapping"
        raise ValueError(msg)
    registry = cast(JsonDict, existing)
    readiness_by_scenario = {
        scenario_id: build_instrumentation_plan(scenario_id) for scenario_id in SCENARIO_PLAN_PATHS
    }
    scenarios = cast(list[JsonDict], registry.get("scenarios", []))
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id") or "")
        if scenario_id in readiness_by_scenario:
            plan = readiness_by_scenario[scenario_id]
            scenario["instrumentation_readiness"] = plan["instrumentation_readiness"]
            scenario["required_fields"] = plan["required_fields"]
            scenario["currently_available_fields"] = plan["currently_available_fields"]
            scenario["missing_fields"] = plan["missing_fields"]
            scenario["implementation_blockers"] = plan["implementation_blockers"]
            scenario["promotion_criteria"] = plan["promotion_criteria"]
            scenario["ui_replay_readiness"] = plan["ui_replay_readiness"]
    registry["status"] = "SCENARIO_REGISTRY_PLANNED_WITH_OBSERVABILITY"
    registry["observability_framework"] = {
        "configured": True,
        "optimized_inference_v1_created": False,
        "champion_selected": False,
    }
    return registry


def build_ui_observability_contract(
    registry: JsonDict | None = None,
    readiness: JsonDict | None = None,
    prefix: JsonDict | None = None,
) -> JsonDict:
    registry_payload = registry or build_observability_registry()
    readiness_payload = readiness or build_observability_readiness(registry_payload)
    prefix_payload = prefix or analyze_prefix_opportunity()
    cards: list[JsonDict] = []
    readiness_rows = {
        str(row["optimization_id"]): row for row in cast(list[JsonDict], readiness_payload["rows"])
    }
    for entry in cast(list[JsonDict], registry_payload["optimizations"]):
        visualization = _visualization_spec(entry)
        state = readiness_rows[str(entry["optimization_id"])]["instrumentation_readiness_state"]
        cards.append(
            {
                "optimization_id": entry["optimization_id"],
                "display_name": entry["display_name"],
                "domain": entry["optimization_domain"],
                "optimization_domain": entry["optimization_domain"],
                "difficulty_tier": entry["difficulty_tier"],
                "state": state,
                "instrumentation_state": state,
                "problem": entry["problem_statement"],
                "mechanism": entry["mechanism"],
                "experiment": entry["experiment_type"],
                "required_instrumentation": entry["required_instrumentation"],
                "missing_instrumentation": entry["missing_instrumentation"],
                "primary_metrics": entry["primary_metrics"],
                "visualization": {
                    "hero": visualization["hero_visualization"],
                    "workload_grounded": visualization["workload_grounded_visualization"],
                    "live_experiment": visualization["live_experiment_visualization"],
                    "final_result": visualization["final_result_visualization"],
                    "empty_state": visualization["empty_unavailable_state"],
                },
                "source_label": visualization["source_label"],
                "story_flow": {
                    "problem": entry["problem_statement"],
                    "mechanism": entry["mechanism"],
                    "instrumentation": entry["required_instrumentation"][:6],
                    "one_factor_experiment": entry["experiment_type"],
                    "measured_result": "not available until scenario is run",
                    "engineering_decision": "pending",
                },
                "visualizations": visualization,
                "empty_state": (
                    "Instrumentation contract exists, but no measured optimization result is "
                    "available yet."
                ),
            }
        )
    return {
        "version": 1,
        "run_id": RUN_ID,
        "status": "UI_OBSERVABILITY_CONTRACT_READY_PLANNING_ONLY",
        "result_type": "planned",
        "endpoints": [
            "/api/optimizations/observability/registry",
            "/api/optimizations/observability/readiness",
            "/api/optimizations/observability/inventory",
            "/api/optimizations/observability/prefix-opportunity",
            "/api/optimizations/observability/event-schema",
            "/api/optimizations/observability/missing-instrumentation",
            "/api/optimizations/observability/cards",
        ],
        "state_labels": [
            "instrumented",
            "partially_instrumented",
            "instrumentation_missing",
            "planned",
            "measured",
            "unavailable",
            "future",
        ],
        "measurement_label_rules": {
            "show_measured_only_from_saved_artifacts": True,
            "show_derived_when_formula_based": True,
            "show_estimated_for_static_prefix_planning": True,
            "never_render_missing_as_zero": True,
        },
        "optimization_cards": cards,
        "prefix_opportunity_snapshot": {
            "rows_scanned": prefix_payload["summary"]["rows_scanned"],
            "prefix_family_count": prefix_payload["summary"]["prefix_family_count"],
            "mean_reusable_token_ratio_estimated": prefix_payload["summary"][
                "mean_reusable_token_ratio_estimated"
            ],
            "measurement_type": "estimated",
        },
        "no_inference_executed": True,
    }


def _visualization_spec(entry: JsonDict) -> JsonDict:
    domain = str(entry["optimization_domain"])
    specs = {
        "prompt_workload": {
            "hero_visualization": "token ribbon",
            "workload_grounded_visualization": "stable prefix versus variable suffix",
            "live_experiment_visualization": "prefix family size and reusable-token percent",
            "final_result_visualization": "TTFT/quality deltas after rerun",
        },
        "prefix_cache": {
            "hero_visualization": "hit/miss flow",
            "workload_grounded_visualization": "reused-token histogram",
            "live_experiment_visualization": "hit/miss TTFT comparison",
            "final_result_visualization": "cache occupancy and latency delta",
        },
        "scheduler_batching": {
            "hero_visualization": "queue animation",
            "workload_grounded_visualization": "waiting/running requests",
            "live_experiment_visualization": "active batch and token-budget utilization",
            "final_result_visualization": "throughput and tail-latency decision",
        },
        "kv_cache": {
            "hero_visualization": "KV block allocation map",
            "workload_grounded_visualization": "occupancy and length buckets",
            "live_experiment_visualization": "eviction/preemption events",
            "final_result_visualization": "memory-pressure and latency decision",
        },
        "prefill_decode_balance": {
            "hero_visualization": "long prompt chunks",
            "workload_grounded_visualization": "TTFT by input-length bucket",
            "live_experiment_visualization": "concurrent decode work",
            "final_result_visualization": "long/short request tradeoff",
        },
        "quantization": {
            "hero_visualization": "precision ladder",
            "workload_grounded_visualization": "model memory footprint",
            "live_experiment_visualization": "speed/quality tradeoff",
            "final_result_visualization": "regression gate",
        },
        "speculative_decoding": {
            "hero_visualization": "draft token stream",
            "workload_grounded_visualization": "draft/target model pair",
            "live_experiment_visualization": "accepted versus rejected tokens",
            "final_result_visualization": "acceptance-rate and TPOT decision",
        },
    }
    default = {
        "hero_visualization": f"{domain} architecture diagram",
        "workload_grounded_visualization": "workload applicability map",
        "live_experiment_visualization": "planned event timeline",
        "final_result_visualization": "protected SLO decision",
    }
    spec = specs.get(domain, default)
    return {
        **spec,
        "explanation_copy": entry["hypothesis"],
        "source_label": "repo artifact or planned instrumentation contract",
        "interaction_behavior": (
            "click card -> inspect problem/mechanism/instrumentation/experiment"
        ),
        "empty_unavailable_state": "Do not show values until captured or explicitly estimated.",
        "measured_derived_estimated_distinction": True,
    }


def build_missing_instrumentation_report(registry: JsonDict | None = None) -> JsonDict:
    registry_payload = registry or build_observability_registry()
    rows: list[JsonDict] = []
    for entry in cast(list[JsonDict], registry_payload["optimizations"]):
        for field_name in cast(list[str], entry["missing_instrumentation"]):
            rows.append(
                {
                    "optimization_id": entry["optimization_id"],
                    "display_name": entry["display_name"],
                    "optimization_domain": entry["optimization_domain"],
                    "field_name": field_name,
                    "measurement_type": "unavailable",
                    "reason": "Field is not captured in saved Main_Inference_V1 artifacts.",
                    "required_before_measured_claim": True,
                }
            )
    return {
        "version": 1,
        "run_id": RUN_ID,
        "status": "MISSING_INSTRUMENTATION_REPORTED",
        "result_type": "planned",
        "rows": rows,
        "missing_field_count": len(rows),
        "missing_is_not_zero": True,
    }


def write_core_optimization_observability_artifacts() -> dict[str, str]:
    registry = build_observability_registry()
    readiness = build_observability_readiness(registry)
    inventory = build_observability_inventory()
    adapter_coverage = build_adapter_coverage()
    prefix = analyze_prefix_opportunity()
    event_schema = build_event_schema()
    ui_contract = build_ui_observability_contract(registry, readiness, prefix)
    missing = build_missing_instrumentation_report(registry)
    scenario_registry = build_updated_scenario_registry()

    _write_yaml(OBSERVABILITY_CONFIG_PATH, registry)
    _write_json(OBSERVABILITY_REGISTRY_JSON_PATH, registry)
    _write_json(OBSERVABILITY_READINESS_JSON_PATH, readiness)
    _write_csv(
        OBSERVABILITY_READINESS_CSV_PATH,
        cast(list[JsonDict], readiness["rows"]),
        [
            "optimization_id",
            "display_name",
            "optimization_domain",
            "difficulty_tier",
            "instrumentation_readiness_state",
            "required_field_count",
            "currently_available_field_count",
            "missing_field_count",
            "external_profiler_required",
            "requires_gpu_to_measure_effect",
            "can_be_shown_in_public_ui",
        ],
    )
    _write_json(OBSERVABILITY_INVENTORY_JSON_PATH, inventory)
    _write_csv(
        OBSERVABILITY_INVENTORY_CSV_PATH,
        cast(list[JsonDict], inventory["rows"]),
        [
            "metric_or_instrumentation_field",
            "currently_captured",
            "derivable",
            "missing",
            "source_artifact",
            "temporal_resolution",
            "engine_support",
            "model_support",
            "code_changes_required",
            "external_profiling_required",
            "safe_for_public_ui",
            "measurement_type",
        ],
    )
    _write_json(
        PREFIX_ANALYSIS_JSON_PATH, {k: v for k, v in prefix.items() if k != "csv_detail_rows"}
    )
    _write_csv(
        PREFIX_ANALYSIS_CSV_PATH,
        cast(list[JsonDict], prefix["csv_detail_rows"]),
        [
            "prefix_family_id",
            "prompt_id",
            "vertical",
            "memory_mode",
            "total_input_tokens_estimated",
            "exact_shared_prefix_tokens_estimated",
            "variable_suffix_tokens_estimated",
            "reusable_token_ratio_estimated",
            "tokenizer",
            "measurement_type",
        ],
    )
    _write_json(EVENT_SCHEMA_JSON_PATH, event_schema)
    _write_json(UI_OBSERVABILITY_CONTRACT_PATH, ui_contract)
    _write_json(ADAPTER_COVERAGE_PATH, adapter_coverage)
    _write_json(MAIN_PROCESSED / "core_optimization_missing_instrumentation.json", missing)
    for scenario_id, path in SCENARIO_PLAN_PATHS.items():
        _write_json(path, build_instrumentation_plan(scenario_id, registry))
    _write_yaml(SCENARIO_REGISTRY_PATH, scenario_registry)

    return {
        "observability_config": _display_path(OBSERVABILITY_CONFIG_PATH),
        "observability_registry_json": _display_path(OBSERVABILITY_REGISTRY_JSON_PATH),
        "observability_readiness_json": _display_path(OBSERVABILITY_READINESS_JSON_PATH),
        "observability_readiness_csv": _display_path(OBSERVABILITY_READINESS_CSV_PATH),
        "observability_inventory_json": _display_path(OBSERVABILITY_INVENTORY_JSON_PATH),
        "observability_inventory_csv": _display_path(OBSERVABILITY_INVENTORY_CSV_PATH),
        "prefix_analysis_json": _display_path(PREFIX_ANALYSIS_JSON_PATH),
        "prefix_analysis_csv": _display_path(PREFIX_ANALYSIS_CSV_PATH),
        "event_schema_json": _display_path(EVENT_SCHEMA_JSON_PATH),
        "ui_observability_contract": _display_path(UI_OBSERVABILITY_CONTRACT_PATH),
        "adapter_coverage": _display_path(ADAPTER_COVERAGE_PATH),
        "scenario_registry": _display_path(SCENARIO_REGISTRY_PATH),
    }
