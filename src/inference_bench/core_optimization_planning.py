"""Core optimization planning artifacts for the product platform.

This module builds audit and design artifacts only. It never runs inference,
mutates measured Main_Inference_V1 artifacts, or creates optimized results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import yaml

from inference_bench.optimization_catalog import (
    OptimizationDefinition,
    load_optimization_catalog,
)

JsonDict = dict[str, Any]


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: object) -> bool:
        return True


RUN_ID = "main_inference_v1"
REPAIR_RUN_ID = "deployability_repair_validation_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path("experiments/main/main_inference_v1")
MAIN_PROCESSED = MAIN_ROOT / "processed"
MAIN_RAW = MAIN_ROOT / "raw"
REPAIR_ROOT = Path("experiments/repairs/deployability_repair_validation_v1")
WORKLOAD_PATH = Path("data/workloads/final_10000/prompt_plus_metadata/mm2_hybrid_top5.jsonl")

TAXONOMY_PATH = Path("configs/core_optimization_taxonomy.yaml")
SCENARIO_REGISTRY_PATH = Path("configs/core_optimization_scenario_registry.yaml")
DOC_PATH = Path("docs/130_core_optimization_planning_baseline_capability_audit.md")
SUMMARY_DOC_PATH = Path("docs/summaries/blockCoreOptimizationPlanningBaselineAudit_summary.md")

ENGINE_BASELINE_REPORT_PATH = (
    MAIN_PROCESSED / "main_inference_v1_engine_baseline_capability_report.json"
)
ENGINE_BASELINE_SUMMARY_PATH = (
    MAIN_PROCESSED / "main_inference_v1_engine_baseline_capability_summary.csv"
)
APPLICABILITY_JSON_PATH = MAIN_PROCESSED / "core_optimization_applicability_matrix.json"
APPLICABILITY_CSV_PATH = MAIN_PROCESSED / "core_optimization_applicability_matrix.csv"
WORKLOAD_OPPORTUNITY_PATH = MAIN_PROCESSED / "core_optimization_workload_opportunity_report.json"
INSTRUMENTATION_GAP_PATH = MAIN_PROCESSED / "core_optimization_instrumentation_gap_report.json"
ONE_FACTOR_PLAN_PATH = MAIN_PROCESSED / "core_optimization_one_factor_experiment_plan.json"
SCALE_PLAN_PATH = MAIN_PROCESSED / "core_optimization_experiment_scale_plan.json"
ACCEPTANCE_GATE_PATH = MAIN_PROCESSED / "core_optimization_acceptance_gate_schema.json"
UI_CONTRACT_PATH = MAIN_PROCESSED / "core_optimization_ui_contract.json"
CHAMPION_FRAMEWORK_PATH = MAIN_PROCESSED / "core_optimization_champion_selection_framework.json"

REPAIR_IDS = {
    "prompt_contract_repair",
    "improve_evidence_formatting",
    "enable_escalation_path",
    "use_mm4_agentic_repair",
    "enable_bounded_citation_repair",
    "repair_retrieval",
}

ALLOWED_ACTIVATION_STATES = {
    "confirmed_active",
    "engine_inherent",
    "likely_default",
    "not_enabled",
    "unknown",
}


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
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(JsonDict, payload)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _iter_jsonl(path: str | Path) -> Iterable[JsonDict]:
    with _repo_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield cast(JsonDict, payload)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _common_prefix_length(token_lists: list[list[str]]) -> int:
    if not token_lists:
        return 0
    shortest = min(len(items) for items in token_lists)
    length = 0
    for index in range(shortest):
        value = token_lists[0][index]
        if all(items[index] == value for items in token_lists[1:]):
            length += 1
        else:
            break
    return length


def _message_text(row: JsonDict) -> str:
    messages = cast(list[JsonDict], row.get("messages", []))
    return "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in messages)


def _pre_context_text(row: JsonDict) -> str:
    messages = cast(list[JsonDict], row.get("messages", []))
    parts: list[str] = []
    for item in messages:
        content = str(item.get("content", ""))
        if item.get("role") == "user" and "\n\nContext:\n\n" in content:
            content = content.split("\n\nContext:\n\n", 1)[0] + "\n\nContext:"
        parts.append(f"{item.get('role', '')}: {content}")
    return "\n".join(parts)


def _bucket(value: int, buckets: list[tuple[str, int, int | None]]) -> str:
    for label, low, high in buckets:
        if value >= low and (high is None or value <= high):
            return label
    return "unknown"


def _float(value: object) -> float:
    if value in (None, "", "n/a"):
        return 0.0
    return float(str(value))


def _average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _engine_rows() -> list[dict[str, str]]:
    return _read_csv(MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv")


def _manifest() -> JsonDict:
    return _read_json(MAIN_RAW / "main_inference_v1_manifest.json")


def _repair_gate() -> JsonDict:
    return _read_json(
        REPAIR_ROOT / "processed/deployability_repair_validation_v1_validation_gate_report.json"
    )


def _source(path: str | Path, note: str) -> JsonDict:
    return {"path": _display_path(path), "note": note}


def _engine_baseline_capabilities() -> list[JsonDict]:
    source_manifest = _source(
        MAIN_RAW / "main_inference_v1_manifest.json",
        "Main_Inference_V1 selected vLLM, SGLang, and API provider route.",
    )
    source_catalog = _source(
        "configs/optimization_catalog.yaml",
        "Catalog marks some serving features as engine_builtin.",
    )
    source_runtime = _source(
        "configs/runtime_engines.yaml",
        "Runtime registry marks vLLM and SGLang as ready self-hosted GPU engines.",
    )
    no_telemetry = "No engine-specific cache/kernel telemetry was saved for this field."
    return [
        {
            "capability_id": "vllm_pagedattention_block_kv",
            "display_name": "PagedAttention / block KV management",
            "engine": "vllm",
            "category": "automatic_engine_native",
            "mechanism": "Paged KV blocks reduce fragmentation and improve serving memory layout.",
            "target_metrics": ["vram", "throughput", "ttft", "e2e_latency"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [source_manifest, source_catalog, source_runtime],
            "exact_version_config_dependence": "unknown; vLLM package version is not recorded.",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": (
                "The baseline used vLLM, so the platform should describe PagedAttention as "
                "a vLLM baseline capability, not as a newly applied optimization."
            ),
            "source_artifact_config": _display_path("configs/optimization_catalog.yaml"),
        },
        {
            "capability_id": "vllm_continuous_batching",
            "display_name": "Continuous batching",
            "engine": "vllm",
            "category": "automatic_engine_native",
            "mechanism": "The serving engine schedules active requests together during decode.",
            "target_metrics": ["throughput", "gpu_utilization", "e2e_latency"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [source_manifest, source_catalog],
            "exact_version_config_dependence": "unknown; batch internals were not logged.",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": (
                "vLLM was not raw single-request generation. It used a serving runtime that "
                "can batch concurrent requests, but scheduler tuning was not changed."
            ),
            "source_artifact_config": _display_path("configs/optimization_catalog.yaml"),
        },
        {
            "capability_id": "vllm_per_request_kv_cache",
            "display_name": "Per-request KV cache",
            "engine": "vllm",
            "category": "automatic_engine_native",
            "mechanism": "Autoregressive decoding reuses each request's own past keys and values.",
            "target_metrics": ["tpot", "e2e_latency"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [source_manifest],
            "exact_version_config_dependence": "not recorded",
            "separately_measured": False,
            "tunable_later": False,
            "ui_explanation": (
                "This is ordinary decode-time KV reuse inside a request, not cross-request "
                "prefix caching."
            ),
            "source_artifact_config": _display_path(MAIN_RAW / "main_inference_v1_manifest.json"),
        },
        {
            "capability_id": "vllm_scheduler_admission",
            "display_name": "Scheduler and admission defaults",
            "engine": "vllm",
            "category": "default_enabled_configurable",
            "mechanism": (
                "The server schedules queued requests and token work using engine defaults."
            ),
            "target_metrics": ["queue_delay", "ttft", "throughput"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [source_manifest],
            "exact_version_config_dependence": "unknown; scheduler mode was not recorded.",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Scheduler tuning remains a future one-factor experiment.",
            "source_artifact_config": _display_path(
                MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv"
            ),
        },
        {
            "capability_id": "vllm_attention_kernel_dispatch",
            "display_name": "Optimized attention/kernel dispatch",
            "engine": "vllm",
            "category": "supported_not_automatic",
            "mechanism": "Runtime may select optimized attention kernels depending on install.",
            "target_metrics": ["tpot", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": (
                "unknown; package and attention backend logs missing."
            ),
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Do not claim a specific kernel backend for the baseline.",
            "source_artifact_config": "missing engine startup logs",
        },
        {
            "capability_id": "vllm_cuda_graph_state",
            "display_name": "CUDA Graph state",
            "engine": "vllm",
            "category": "supported_not_automatic",
            "mechanism": "Graph capture can reduce CPU overhead for steady decode paths.",
            "target_metrics": ["tpot", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": (
                "CUDA Graph status must be instrumented before being taught as active."
            ),
            "source_artifact_config": "missing engine startup logs",
        },
        {
            "capability_id": "vllm_chunked_prefill_state",
            "display_name": "Chunked prefill state",
            "engine": "vllm",
            "category": "supported_not_automatic",
            "mechanism": "Long prefills can be split to reduce decode interference.",
            "target_metrics": ["ttft", "e2e_latency", "tail_latency"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "This is a candidate, not a proven baseline feature.",
            "source_artifact_config": "missing engine startup logs",
        },
        {
            "capability_id": "vllm_prefix_cache_state",
            "display_name": "Cross-request prefix cache state",
            "engine": "vllm",
            "category": "supported_not_automatic",
            "mechanism": "Identical request prefixes can reuse prefetched KV across requests.",
            "target_metrics": ["ttft", "prefill_time", "gpu_utilization"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Prefix cache needs an explicit hit/miss experiment.",
            "source_artifact_config": "missing engine cache telemetry",
        },
        {
            "capability_id": "vllm_gpu_memory_kv_block_allocation",
            "display_name": "GPU memory / KV-block allocation",
            "engine": "vllm",
            "category": "automatic_engine_native",
            "mechanism": "The runtime allocates GPU memory for model weights and KV blocks.",
            "target_metrics": ["vram", "oom_rate", "throughput"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [
                source_manifest,
                _source(
                    MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl",
                    "GPU memory was sampled, but KV block counts were not captured.",
                ),
            ],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "VRAM was measured; KV block occupancy still needs instrumentation.",
            "source_artifact_config": _display_path(
                MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl"
            ),
        },
        {
            "capability_id": "sglang_radixattention_prefix_reuse",
            "display_name": "RadixAttention / prefix reuse",
            "engine": "sglang",
            "category": "supported_not_automatic",
            "mechanism": "SGLang can organize prefixes for reuse when cache conditions are met.",
            "target_metrics": ["ttft", "prefill_time", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown; SGLang package version not recorded.",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Do not claim RadixAttention cache hits without engine telemetry.",
            "source_artifact_config": "missing SGLang cache telemetry",
        },
        {
            "capability_id": "sglang_paged_kv_storage",
            "display_name": "Paged KV storage",
            "engine": "sglang",
            "category": "supported_not_automatic",
            "mechanism": "Engine may use paged KV storage depending on runtime configuration.",
            "target_metrics": ["vram", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Supported engine capability is not proof of baseline activation.",
            "source_artifact_config": "missing SGLang startup logs",
        },
        {
            "capability_id": "sglang_continuous_batching",
            "display_name": "Continuous batching",
            "engine": "sglang",
            "category": "automatic_engine_native",
            "mechanism": "The server schedules concurrent request token work together.",
            "target_metrics": ["throughput", "gpu_utilization", "e2e_latency"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [source_manifest, source_catalog, source_runtime],
            "exact_version_config_dependence": "unknown; scheduler internals were not logged.",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": (
                "SGLang baseline is an optimized serving baseline, not raw generation."
            ),
            "source_artifact_config": _display_path("configs/optimization_catalog.yaml"),
        },
        {
            "capability_id": "sglang_cache_aware_scheduling",
            "display_name": "Cache-aware scheduling",
            "engine": "sglang",
            "category": "supported_not_automatic",
            "mechanism": "Scheduler can exploit cache locality when measurable cache hits exist.",
            "target_metrics": ["ttft", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Needs cache hit/miss and queue telemetry before claims.",
            "source_artifact_config": "missing SGLang scheduler telemetry",
        },
        {
            "capability_id": "sglang_per_request_kv_cache",
            "display_name": "Per-request KV cache",
            "engine": "sglang",
            "category": "automatic_engine_native",
            "mechanism": "Autoregressive decoding reuses past keys and values within a request.",
            "target_metrics": ["tpot", "e2e_latency"],
            "baseline_activation_state": "engine_inherent",
            "activation_evidence": [source_manifest],
            "exact_version_config_dependence": "not recorded",
            "separately_measured": False,
            "tunable_later": False,
            "ui_explanation": "Per-request KV reuse is distinct from cross-request prefix reuse.",
            "source_artifact_config": _display_path(MAIN_RAW / "main_inference_v1_manifest.json"),
        },
        {
            "capability_id": "sglang_attention_backend_selection",
            "display_name": "Attention backend selection",
            "engine": "sglang",
            "category": "supported_not_automatic",
            "mechanism": "Runtime can choose attention backends based on install and flags.",
            "target_metrics": ["tpot", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Kernel/backend state must be logged in later experiments.",
            "source_artifact_config": "missing SGLang startup logs",
        },
        {
            "capability_id": "sglang_cuda_graph_state",
            "display_name": "CUDA Graph state",
            "engine": "sglang",
            "category": "supported_not_automatic",
            "mechanism": "Graph capture can reduce CPU overhead for repeated decode shapes.",
            "target_metrics": ["tpot", "throughput"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [{"path": "not available", "note": no_telemetry}],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "No baseline claim without startup/runtime evidence.",
            "source_artifact_config": "missing SGLang startup logs",
        },
        {
            "capability_id": "sglang_chunked_prefill_state",
            "display_name": "Chunked prefill state",
            "engine": "sglang",
            "category": "supported_not_automatic",
            "mechanism": "Long prompts can be prefilled in chunks to reduce blocking.",
            "target_metrics": ["ttft", "tail_latency"],
            "baseline_activation_state": "unknown",
            "activation_evidence": [
                _source(
                    "configs/serving_profiles.yaml",
                    "A profile contains a chunked-prefill command, but the exact live baseline "
                    "startup log is unavailable.",
                )
            ],
            "exact_version_config_dependence": "unknown",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "Treat chunked prefill as a candidate until measured.",
            "source_artifact_config": _display_path("configs/serving_profiles.yaml"),
        },
        {
            "capability_id": "tensorrt_llm_future_engine",
            "display_name": "TensorRT-LLM planned engine",
            "engine": "tensorrt_llm",
            "category": "supported_not_automatic",
            "mechanism": "Compiled TensorRT-LLM serving path planned for future work.",
            "target_metrics": ["tpot", "throughput", "latency", "cost"],
            "baseline_activation_state": "not_enabled",
            "activation_evidence": [
                _source(
                    "configs/runtime_engines.yaml",
                    "Runtime registry marks TensorRT-LLM planned and smoke_tested false.",
                ),
                source_manifest,
            ],
            "exact_version_config_dependence": "not installed or smoke-tested in repo evidence",
            "separately_measured": False,
            "tunable_later": True,
            "ui_explanation": "TensorRT-LLM belongs in future architecture, not baseline claims.",
            "source_artifact_config": _display_path("configs/runtime_engines.yaml"),
        },
    ]


def _core_optimizations() -> list[JsonDict]:
    common_quality_gates = [
        "json_validity",
        "contract_validity",
        "format_validity",
        "evidence_match",
        "groundedness",
        "safety_findings",
        "truncation",
        "completion_failure_rate",
    ]
    return [
        {
            "optimization_id": "prompt_prefix_layout_optimization",
            "display_name": "Prompt and prefix-layout optimization",
            "difficulty_tier": 1,
            "category": "workload_and_configuration",
            "definition": (
                "Move stable instructions, schema, and reusable scaffolding into identical "
                "leading token spans without changing the answer contract."
            ),
            "mechanism": "Increase token-identical leading prefix length for cacheability.",
            "implementation_mechanism": ["code_change", "workload_change"],
            "target_metrics": ["ttft", "prefill_time", "tokens_per_second"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["rendered prompt diff", "no gold leakage check"],
            "engine_compatibility": ["vllm", "sglang", "api_provider_route"],
            "model_compatibility": ["model3_7b", "model6_gated"],
            "hardware_compatibility": ["a100_sxm_80gb", "provider_managed"],
            "baseline_state": "not_applied_as_project_optimization",
            "already_active": False,
            "negative_rules": ["context_compression"],
            "risks": ["accidental contract change", "lower evidence clarity"],
            "expected_telemetry": ["exact_reusable_prefix_tokens", "prompt_token_distribution"],
            "one_factor_experiment_design": (
                "Change prompt layout only; keep context and model fixed."
            ),
            "gpu_api_cpu_requirements": "CPU for static diff; GPU/API only for measured latency.",
            "result_availability": "planned",
            "ui_visualization_concept": (
                "Prefix waterfall showing fixed vs per-request token spans."
            ),
            "future_paper_evidence_required": "Prefix-cache and prompt-layout literature review.",
            "source_catalog_ids": ["reduce_context_tokens"],
        },
        {
            "optimization_id": "prefix_cache_verification_tuning",
            "display_name": "Prefix-cache verification and tuning",
            "difficulty_tier": 1,
            "category": "workload_and_configuration",
            "definition": "Enable and verify cross-request prefix cache behavior where supported.",
            "mechanism": "Reuse KV for exact matching prefixes across requests.",
            "implementation_mechanism": ["engine_flag", "runtime_config"],
            "target_metrics": ["ttft", "prefill_time", "gpu_utilization"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["exact prefix audit", "cache hit/miss telemetry"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "unknown",
            "already_active": False,
            "negative_rules": ["prefix_caching"],
            "risks": ["no hits if prefixes diverge", "cache eviction overhead"],
            "expected_telemetry": [
                "cache_lookup_count",
                "hit_count",
                "miss_count",
                "hit_token_count",
                "cache_occupancy",
            ],
            "one_factor_experiment_design": (
                "Enable prefix cache only; compare against same engine."
            ),
            "gpu_api_cpu_requirements": "GPU required for measured cache behavior.",
            "result_availability": "planned",
            "ui_visualization_concept": "Cache hit/miss timeline and reusable-token histogram.",
            "future_paper_evidence_required": "Engine-specific prefix caching docs and papers.",
            "source_catalog_ids": ["enable_prefix_cache"],
        },
        {
            "optimization_id": "scheduler_batch_tuning",
            "display_name": "Scheduler and batch tuning",
            "difficulty_tier": 1,
            "category": "workload_and_configuration",
            "definition": "Tune request scheduling, max running requests, and batching limits.",
            "mechanism": "Balance queueing, prefill, decode, and batch composition.",
            "implementation_mechanism": ["engine_flag", "runtime_config"],
            "target_metrics": ["throughput", "ttft", "e2e_latency", "gpu_utilization"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["queue wait telemetry", "active batch telemetry"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "default_engine_behavior_untuned",
            "already_active": False,
            "negative_rules": ["concurrency_increase"],
            "risks": ["tail latency regression", "OOM", "lower reliability"],
            "expected_telemetry": ["queue_wait_ms", "active_batch_size", "running_requests"],
            "one_factor_experiment_design": "Change one scheduler/batch setting group only.",
            "gpu_api_cpu_requirements": "GPU required.",
            "result_availability": "planned",
            "ui_visualization_concept": "Queue depth, active batch size, and latency overlay.",
            "future_paper_evidence_required": "Continuous batching and scheduling references.",
            "source_catalog_ids": [
                "tune_scheduler",
                "tune_max_num_seqs",
                "concurrency_sweep",
            ],
        },
        {
            "optimization_id": "kv_cache_capacity_allocation_tuning",
            "display_name": "KV-cache capacity/allocation tuning",
            "difficulty_tier": 1,
            "category": "workload_and_configuration",
            "definition": "Tune allocation headroom for model weights, KV blocks, and request mix.",
            "mechanism": "Improve sequence capacity while avoiding OOM and eviction pressure.",
            "implementation_mechanism": ["engine_flag", "runtime_config"],
            "target_metrics": ["vram", "throughput", "ttft", "oom_rate"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["KV block metrics", "VRAM headroom telemetry"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "engine_allocated_but_not_instrumented",
            "already_active": False,
            "negative_rules": [],
            "risks": ["OOM", "lower concurrency", "cache thrash"],
            "expected_telemetry": ["blocks_total", "blocks_used", "blocks_free", "evictions"],
            "one_factor_experiment_design": "Change one KV allocation setting only.",
            "gpu_api_cpu_requirements": "GPU required.",
            "result_availability": "planned",
            "ui_visualization_concept": "KV block occupancy gauge and per-length pressure bands.",
            "future_paper_evidence_required": "Paged KV and memory management references.",
            "source_catalog_ids": ["tune_kv_cache", "tune_gpu_memory_utilization"],
        },
        {
            "optimization_id": "chunked_prefill_tuning",
            "display_name": "Chunked-prefill tuning",
            "difficulty_tier": 1,
            "category": "workload_and_configuration",
            "definition": "Split long prompt prefill work to reduce head-of-line blocking.",
            "mechanism": "Interleave long-prefill requests with decode work.",
            "implementation_mechanism": ["engine_flag", "runtime_config"],
            "target_metrics": ["ttft", "tail_latency", "e2e_latency"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["length-bucket telemetry", "prefill/decode timing"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "unknown",
            "already_active": False,
            "negative_rules": ["disaggregated_prefill"],
            "risks": ["throughput loss", "scheduler instability"],
            "expected_telemetry": ["chunk_count", "chunk_size", "long_prefill_queue_time"],
            "one_factor_experiment_design": "Change chunked-prefill setting only.",
            "gpu_api_cpu_requirements": "GPU required.",
            "result_availability": "planned",
            "ui_visualization_concept": "Long-prefill lane showing chunk boundaries and TTFT.",
            "future_paper_evidence_required": "Chunked prefill engine docs and studies.",
            "source_catalog_ids": [],
        },
        {
            "optimization_id": "cache_workload_aware_routing",
            "display_name": "Cache-aware/workload-aware routing",
            "difficulty_tier": 1,
            "category": "workload_and_configuration",
            "definition": "Route similar-prefix or similar-length requests to improve locality.",
            "mechanism": "Improve cache locality and reduce mixed-length interference.",
            "implementation_mechanism": ["routing_layer", "deployment_architecture"],
            "target_metrics": ["ttft", "throughput", "cache_hit_rate"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["prefix family IDs", "length buckets", "routing telemetry"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "not_applied",
            "already_active": False,
            "negative_rules": ["prefix_caching"],
            "risks": ["load imbalance", "operational complexity"],
            "expected_telemetry": ["route_id", "prefix_family_id", "worker_cache_hit_rate"],
            "one_factor_experiment_design": (
                "Change routing policy only after cache telemetry exists."
            ),
            "gpu_api_cpu_requirements": "GPU required for live routing; CPU for simulation.",
            "result_availability": "planned",
            "ui_visualization_concept": "Router map from prefix families to workers.",
            "future_paper_evidence_required": "Cache-aware serving/routing evidence.",
            "source_catalog_ids": ["route_long_and_short_requests_separately"],
        },
        {
            "optimization_id": "quantization",
            "display_name": "Quantization",
            "difficulty_tier": 2,
            "category": "model_execution",
            "definition": "Use lower-precision weights or kernels to reduce memory/cost.",
            "mechanism": "Lower weight memory and sometimes increase throughput.",
            "implementation_mechanism": ["model_checkpoint", "engine_flag"],
            "target_metrics": ["vram", "throughput", "cost"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["registered quantized checkpoint", "quality guard"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "not_enabled_by_project",
            "already_active": False,
            "negative_rules": ["quantization"],
            "risks": ["quality loss", "kernel incompatibility"],
            "expected_telemetry": ["model_memory_mb", "precision_format", "quality_metrics"],
            "one_factor_experiment_design": "Change model precision/checkpoint only.",
            "gpu_api_cpu_requirements": "GPU required.",
            "result_availability": "planned",
            "ui_visualization_concept": "Memory footprint before/after with quality guard overlay.",
            "future_paper_evidence_required": "Quantization method evidence for Qwen2.5.",
            "source_catalog_ids": [
                "enable_int8_quantization",
                "enable_awq_int4",
                "enable_gptq_int4",
                "enable_fp8_where_supported",
            ],
        },
        {
            "optimization_id": "speculative_decoding",
            "display_name": "Speculative decoding",
            "difficulty_tier": 2,
            "category": "model_execution",
            "definition": "Use a draft model to propose tokens and verify with the target model.",
            "mechanism": "Reduce target-model decode work when draft tokens are accepted.",
            "implementation_mechanism": ["engine_flag", "model_checkpoint"],
            "target_metrics": ["tpot", "throughput", "e2e_latency"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["compatible draft model", "acceptance-rate telemetry"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "not_enabled_by_project",
            "already_active": False,
            "negative_rules": ["speculative_decoding"],
            "risks": ["extra VRAM", "low acceptance rate", "complexity"],
            "expected_telemetry": ["drafted_tokens", "accepted_tokens", "acceptance_rate"],
            "one_factor_experiment_design": "Enable speculation only with one draft model.",
            "gpu_api_cpu_requirements": "GPU required.",
            "result_availability": "planned",
            "ui_visualization_concept": "Draft vs accepted token stream.",
            "future_paper_evidence_required": "Speculative decoding acceptance references.",
            "source_catalog_ids": ["enable_speculative_decoding"],
        },
        {
            "optimization_id": "model_selection_routing",
            "display_name": "Model selection and routing",
            "difficulty_tier": 2,
            "category": "model_execution",
            "definition": "Route tasks to smaller, stronger, or provider models by workload class.",
            "mechanism": "Match model capacity/cost to task difficulty.",
            "implementation_mechanism": ["routing_layer", "model_checkpoint", "api_route"],
            "target_metrics": ["cost", "quality", "latency"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["task classifier", "per-class quality reports"],
            "engine_compatibility": ["vllm", "sglang", "api_provider_route"],
            "model_compatibility": ["model3_7b", "model6_gated"],
            "hardware_compatibility": ["a100_sxm_80gb", "provider_managed"],
            "baseline_state": "matrix_compared_models_but_no_router",
            "already_active": False,
            "negative_rules": ["stronger_model_escalation"],
            "risks": ["routing errors", "cost regression"],
            "expected_telemetry": ["route_decision", "model_alias", "per_route_quality"],
            "one_factor_experiment_design": "Change routing policy only.",
            "gpu_api_cpu_requirements": "GPU and API credentials for measured routing.",
            "result_availability": "planned",
            "ui_visualization_concept": "Decision tree from prompt class to model route.",
            "future_paper_evidence_required": "LLM routing/model cascade references.",
            "source_catalog_ids": ["use_smaller_model", "use_stronger_model"],
        },
        {
            "optimization_id": "model_compression",
            "display_name": "Model compression",
            "difficulty_tier": 3,
            "category": "model_hardware_architecture",
            "definition": "Use distillation, pruning, or compressed checkpoints.",
            "mechanism": "Reduce model size while preserving useful behavior.",
            "implementation_mechanism": ["model_checkpoint"],
            "target_metrics": ["vram", "cost", "throughput"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["compressed checkpoint", "quality validation"],
            "engine_compatibility": ["vllm", "sglang"],
            "model_compatibility": ["future_compressed_model"],
            "hardware_compatibility": ["a100_sxm_80gb", "smaller_gpu_future"],
            "baseline_state": "not_implemented",
            "already_active": False,
            "negative_rules": ["quantization"],
            "risks": ["quality loss", "training cost"],
            "expected_telemetry": ["model_memory_mb", "quality_metrics"],
            "one_factor_experiment_design": "Change checkpoint only.",
            "gpu_api_cpu_requirements": "GPU required after checkpoint exists.",
            "result_availability": "future",
            "ui_visualization_concept": "Model size vs quality frontier.",
            "future_paper_evidence_required": "Compression/distillation evidence.",
            "source_catalog_ids": ["use_distilled_model"],
        },
        {
            "optimization_id": "multi_gpu_parallelism",
            "display_name": "Multi-GPU parallelism",
            "difficulty_tier": 3,
            "category": "model_hardware_architecture",
            "definition": "Shard model or work across multiple GPUs.",
            "mechanism": "Increase capacity or throughput with tensor/pipeline/data parallelism.",
            "implementation_mechanism": ["deployment_architecture", "engine_flag"],
            "target_metrics": ["throughput", "vram_headroom", "latency"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["multi-GPU hardware", "interconnect control", "cost registry"],
            "engine_compatibility": ["vllm", "sglang", "tensorrt_llm"],
            "model_compatibility": ["model3_7b", "model4_32b_future"],
            "hardware_compatibility": ["multi_gpu"],
            "baseline_state": "not_applicable_single_a100",
            "already_active": False,
            "negative_rules": ["tensor_parallelism"],
            "risks": ["communication overhead", "cost increase"],
            "expected_telemetry": ["gpu_count", "per_gpu_utilization", "interconnect_metrics"],
            "one_factor_experiment_design": "Change GPU topology only.",
            "gpu_api_cpu_requirements": "Multi-GPU rental required.",
            "result_availability": "future",
            "ui_visualization_concept": "GPU shard topology and cost/latency tradeoff.",
            "future_paper_evidence_required": "Parallel serving references.",
            "source_catalog_ids": [
                "use_tensor_parallelism",
                "use_pipeline_parallelism",
                "use_data_parallelism",
            ],
        },
        {
            "optimization_id": "kv_cache_offloading_hierarchical_cache",
            "display_name": "KV-cache offloading / hierarchical caching",
            "difficulty_tier": 3,
            "category": "model_hardware_architecture",
            "definition": "Move KV state across GPU/CPU/storage tiers when memory is constrained.",
            "mechanism": "Trade memory capacity against transfer overhead.",
            "implementation_mechanism": ["deployment_architecture", "runtime_config"],
            "target_metrics": ["vram", "sequence_capacity", "cost"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["engine support", "eviction/offload telemetry"],
            "engine_compatibility": ["future"],
            "model_compatibility": ["future"],
            "hardware_compatibility": ["future"],
            "baseline_state": "not_implemented",
            "already_active": False,
            "negative_rules": [],
            "risks": ["latency regression", "complexity"],
            "expected_telemetry": ["offload_bytes", "evictions", "recomputations"],
            "one_factor_experiment_design": "Change offload policy only after support exists.",
            "gpu_api_cpu_requirements": "Future GPU/host-memory setup.",
            "result_availability": "future",
            "ui_visualization_concept": "Cache hierarchy diagram with movement counters.",
            "future_paper_evidence_required": "KV offload/hierarchical cache references.",
            "source_catalog_ids": [],
        },
        {
            "optimization_id": "manual_kernel_compiler_optimization",
            "display_name": "Manual kernel/compiler optimization",
            "difficulty_tier": 3,
            "category": "model_hardware_architecture",
            "definition": (
                "Change attention kernels, compiler settings, or runtime compilation path."
            ),
            "mechanism": "Reduce kernel latency or CPU overhead.",
            "implementation_mechanism": ["runtime_config", "code_change"],
            "target_metrics": ["tpot", "throughput", "latency"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["engine version logs", "kernel/backend telemetry"],
            "engine_compatibility": ["vllm", "sglang", "tensorrt_llm"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["a100_sxm_80gb"],
            "baseline_state": "unknown_kernel_state",
            "already_active": False,
            "negative_rules": [],
            "risks": ["build instability", "non-reproducibility"],
            "expected_telemetry": ["attention_backend", "kernel_name", "compile_state"],
            "one_factor_experiment_design": "Change one kernel/backend flag only.",
            "gpu_api_cpu_requirements": "GPU required.",
            "result_availability": "future",
            "ui_visualization_concept": "Kernel path card with measured TPOT before/after.",
            "future_paper_evidence_required": "Kernel/compiler docs and references.",
            "source_catalog_ids": [
                "enable_cuda_graphs",
                "use_flashattention_where_available",
                "use_flashinfer_where_available",
            ],
        },
        {
            "optimization_id": "prefill_decode_disaggregation",
            "display_name": "Prefill/decode disaggregation",
            "difficulty_tier": 4,
            "category": "distributed_architecture",
            "definition": "Separate prefill and decode workers or stages.",
            "mechanism": "Specialize infrastructure for long-input and decode-heavy work.",
            "implementation_mechanism": ["deployment_architecture", "routing_layer"],
            "target_metrics": ["ttft", "throughput", "tail_latency"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["prefill/decode telemetry", "multi-worker deployment"],
            "engine_compatibility": ["future"],
            "model_compatibility": ["model3_7b"],
            "hardware_compatibility": ["multi_gpu_or_multi_worker"],
            "baseline_state": "not_implemented",
            "already_active": False,
            "negative_rules": ["disaggregated_prefill"],
            "risks": ["operational complexity", "higher infrastructure cost"],
            "expected_telemetry": ["prefill_time_ms", "decode_time_ms", "worker_role"],
            "one_factor_experiment_design": "Change serving architecture only.",
            "gpu_api_cpu_requirements": "Future multi-worker GPU environment.",
            "result_availability": "future",
            "ui_visualization_concept": "Prefill lane and decode lane flow diagram.",
            "future_paper_evidence_required": "Disaggregated serving references.",
            "source_catalog_ids": ["prefill_decode_disaggregation"],
        },
        {
            "optimization_id": "distributed_capacity_serving_architecture",
            "display_name": "Distributed capacity and serving architecture",
            "difficulty_tier": 4,
            "category": "distributed_architecture",
            "definition": "Scale serving across workers, queues, and admission control.",
            "mechanism": "Increase capacity through routing, admission, and worker placement.",
            "implementation_mechanism": ["deployment_architecture", "routing_layer"],
            "target_metrics": ["throughput", "availability", "cost_per_request"],
            "protected_metrics": common_quality_gates,
            "prerequisites": ["multi-worker setup", "load probe", "admission telemetry"],
            "engine_compatibility": ["vllm", "sglang", "api_provider_route"],
            "model_compatibility": ["model3_7b", "model6_gated"],
            "hardware_compatibility": ["multi_worker"],
            "baseline_state": "not_implemented",
            "already_active": False,
            "negative_rules": ["concurrency_increase", "stronger_model_escalation"],
            "risks": ["cost regression", "operational complexity"],
            "expected_telemetry": ["worker_id", "admission_decision", "queue_wait_ms"],
            "one_factor_experiment_design": "Change deployment topology only.",
            "gpu_api_cpu_requirements": "Future multi-worker GPU/API setup.",
            "result_availability": "future",
            "ui_visualization_concept": "Capacity map and request flow diagram.",
            "future_paper_evidence_required": "Serving architecture and load balancing references.",
            "source_catalog_ids": ["admission_control", "request_queue_tuning"],
        },
    ]


def build_core_optimization_taxonomy() -> JsonDict:
    return {
        "version": 1,
        "status": "planning_audit_complete",
        "principles": [
            "Deployability repairs are separate from core inference optimization.",
            "Engine support does not prove baseline activation.",
            "Engine-native baseline capabilities are not new optimizations.",
            "Every measured experiment changes one factor.",
            "Quality and safety gates remain protected.",
        ],
        "layers": {
            "A_engine_baseline_capabilities": _engine_baseline_capabilities(),
            "B_engineer_applied_core_optimizations": _core_optimizations(),
            "C_applicable_experiment_candidates": [],
            "D_measured_optimization_scenarios": [
                {
                    "scenario_id": RUN_ID,
                    "scenario_type": "engine_baseline",
                    "result_type": "measured",
                    "artifact_root": _display_path(MAIN_ROOT),
                },
                {
                    "scenario_id": REPAIR_RUN_ID,
                    "scenario_type": "repair_validation",
                    "result_type": "measured",
                    "artifact_root": _display_path(REPAIR_ROOT),
                },
            ],
        },
        "removed_from_core_list": sorted(REPAIR_IDS),
    }


def _catalog_entry_payload(definition: OptimizationDefinition) -> JsonDict:
    return {
        "id": definition.id,
        "category": definition.category,
        "implementation_status": definition.implementation_status,
        "application_method": definition.application_method,
        "improves": list(definition.improves),
        "required_engines": list(definition.required_engines),
        "current_project_support": definition.current_project_support,
    }


def build_current_catalog_audit() -> JsonDict:
    catalog = load_optimization_catalog()
    engine_builtin = {
        key: definition
        for key, definition in catalog.items()
        if definition.implementation_status == "engine_builtin"
    }
    repairs = {
        key: definition
        for key, definition in catalog.items()
        if key in REPAIR_IDS or definition.category == "agentic"
    }
    engineer_configurable = {
        key: definition
        for key, definition in catalog.items()
        if key not in repairs
        and definition.implementation_status in {"config_only", "implemented", "planned"}
        and definition.category
        in {"serving_engine", "concurrency_capacity", "model", "hardware", "workload_context"}
    }
    future_architecture = {
        key: definition
        for key, definition in catalog.items()
        if definition.category == "hardware" or "future" in definition.current_project_support
    }
    ambiguous_groups = [
        {
            "group": "quantization_family",
            "ids": [
                "use_quantized_model",
                "enable_int8_quantization",
                "enable_awq_int4",
                "enable_gptq_int4",
                "enable_fp8_where_supported",
            ],
            "issue": "Generic and method-specific quantization entries overlap.",
        },
        {
            "group": "concurrency_family",
            "ids": ["increase_concurrency", "decrease_concurrency", "concurrency_sweep"],
            "issue": "Sweep and directional changes should be modeled as separate experiments.",
        },
        {
            "group": "engine_switch_vs_engine_builtin",
            "ids": [
                "switch_engine_to_vllm",
                "switch_engine_to_sglang",
                "use_pagedattention_capable_engine",
                "enable_continuous_batching",
            ],
            "issue": "Engine selection and engine-native capabilities are currently adjacent.",
        },
        {
            "group": "context_quality_vs_serving",
            "ids": [
                "prompt_contract_repair",
                "improve_evidence_formatting",
                "repair_retrieval",
                "reduce_context_tokens",
                "enable_context_compression",
            ],
            "issue": "Deployability repairs and token/workload optimizations need separation.",
        },
    ]
    return {
        "run_id": RUN_ID,
        "status": "CATALOG_AUDITED",
        "true_core_inference_optimizations": [
            item["optimization_id"] for item in _core_optimizations()
        ],
        "engine_native_baseline_capability_ids": sorted(engine_builtin),
        "engineer_configurable_catalog_entries": [
            _catalog_entry_payload(item) for item in engineer_configurable.values()
        ],
        "advanced_future_architecture_entries": [
            _catalog_entry_payload(item) for item in future_architecture.values()
        ],
        "deployability_repairs_to_keep_out_of_core": [
            _catalog_entry_payload(item) for item in repairs.values()
        ],
        "duplicated_or_ambiguous_groups": ambiguous_groups,
        "unsupported_by_evidence_states": [
            {
                "issue": (
                    "Current UI catalogs can show engine built-ins near planned optimizations."
                ),
                "required_fix": (
                    "Render baseline capabilities separately from engineer-applied candidates."
                ),
            },
            {
                "issue": (
                    "No engine cache hit, CUDA Graph, attention backend, or KV block telemetry."
                ),
                "required_fix": "Mark exact baseline state unknown unless direct evidence exists.",
            },
        ],
        "missing_instrumentation_fields": _instrumentation_fields(),
    }


def build_engine_baseline_capability_report() -> JsonDict:
    manifest = _manifest()
    capabilities = _engine_baseline_capabilities()
    invalid = [
        item["capability_id"]
        for item in capabilities
        if item["baseline_activation_state"] not in ALLOWED_ACTIVATION_STATES
    ]
    if invalid:
        msg = f"Invalid activation states for {invalid}"
        raise ValueError(msg)
    return {
        "run_id": RUN_ID,
        "status": "ENGINE_BASELINE_CAPABILITY_AUDITED",
        "source_manifest": _display_path(MAIN_RAW / "main_inference_v1_manifest.json"),
        "main_inference_v1_manifest_optimization_flags": manifest.get("optimization_flags", []),
        "baseline_summary": {
            "engine": manifest.get("engine"),
            "runtime": manifest.get("runtime"),
            "hardware": manifest.get("hardware"),
            "model_alias": manifest.get("model_alias"),
            "prompt_count": manifest.get("prompt_count"),
            "optimization_flags": manifest.get("optimization_flags", []),
            "optimization_flags_empty": manifest.get("optimization_flags", []) == [],
        },
        "exact_engine_state": {
            "vllm": {
                "package_version": "unknown",
                "model_precision": "unknown",
                "quantization_state": "not_enabled_by_project",
                "prefix_caching_state": "unknown",
                "chunked_prefill_state": "unknown",
                "cuda_graph_state": "unknown",
                "attention_backend": "unknown",
                "max_running_sequences": "unknown",
                "max_batched_tokens": "unknown",
                "context_model_length": "unknown",
                "gpu_memory_utilization_setting": "unknown",
                "cache_block_page_settings": "unknown",
                "scheduler_mode": "unknown",
                "speculative_decoding_state": "not_enabled_by_project",
                "parallelism_settings": "not_enabled_by_project",
                "offloading_state": "not_enabled_by_project",
            },
            "sglang": {
                "package_version": "unknown",
                "model_precision": "unknown",
                "quantization_state": "not_enabled_by_project",
                "prefix_caching_state": "unknown",
                "chunked_prefill_state": "unknown",
                "cuda_graph_state": "unknown",
                "attention_backend": "unknown",
                "max_running_requests": "unknown",
                "max_batched_tokens": "unknown",
                "context_model_length": "unknown",
                "gpu_memory_utilization_setting": "unknown",
                "cache_block_page_settings": "unknown",
                "scheduler_mode": "unknown",
                "speculative_decoding_state": "not_enabled_by_project",
                "parallelism_settings": "not_enabled_by_project",
                "offloading_state": "not_enabled_by_project",
            },
            "tensorrt_llm": {
                "package_version": "not_applicable",
                "baseline_state": "not_enabled",
                "reason": "Runtime registry marks TensorRT-LLM planned and smoke_tested false.",
            },
        },
        "baseline_capabilities_are_new_optimizations": False,
        "capabilities": capabilities,
        "interpretation": (
            "The baseline used optimized serving engines, but no project-selected core "
            "optimization flags were recorded. Engine-native capabilities must be taught as "
            "baseline capabilities unless a later experiment changes or tunes them."
        ),
    }


def build_engine_baseline_capability_summary_rows() -> list[dict[str, Any]]:
    rows = []
    for item in _engine_baseline_capabilities():
        rows.append(
            {
                "capability_id": item["capability_id"],
                "display_name": item["display_name"],
                "engine": item["engine"],
                "category": item["category"],
                "baseline_activation_state": item["baseline_activation_state"],
                "separately_measured": item["separately_measured"],
                "tunable_later": item["tunable_later"],
                "source_artifact_config": item["source_artifact_config"],
            }
        )
    return rows


def _engine_metric_summary() -> JsonDict:
    rows = [row for row in _engine_rows() if row["backend_type"] == "self_hosted_gpu"]
    payload: JsonDict = {}
    for engine in ["vllm", "sglang"]:
        engine_rows = [row for row in rows if row["engine"] == engine]
        payload[engine] = {
            "row_count": len(engine_rows),
            "completed_requests": sum(int(row["requests_completed"]) for row in engine_rows),
            "mean_e2e_latency_ms": _average(
                _float(row["mean_e2e_latency_ms"]) for row in engine_rows
            ),
            "mean_ttft_ms": _average(_float(row["mean_ttft_ms"]) for row in engine_rows),
            "mean_tpot_ms": _average(_float(row["mean_tpot_ms"]) for row in engine_rows),
            "mean_total_tokens_per_second": _average(
                _float(row["mean_total_tokens_per_second"]) for row in engine_rows
            ),
            "by_concurrency": {},
        }
        by_concurrency = cast(JsonDict, payload[engine]["by_concurrency"])
        for concurrency in ["16", "32"]:
            subset = [row for row in engine_rows if row["concurrency"] == concurrency]
            by_concurrency[concurrency] = {
                "row_count": len(subset),
                "mean_e2e_latency_ms": _average(
                    _float(row["mean_e2e_latency_ms"]) for row in subset
                ),
                "mean_ttft_ms": _average(_float(row["mean_ttft_ms"]) for row in subset),
                "mean_tpot_ms": _average(_float(row["mean_tpot_ms"]) for row in subset),
                "mean_total_tokens_per_second": _average(
                    _float(row["mean_total_tokens_per_second"]) for row in subset
                ),
            }
    return payload


def _workload_scan() -> JsonDict:
    prompt_buckets = [
        ("short_0_512", 0, 512),
        ("medium_513_1024", 513, 1024),
        ("long_1025_2048", 1025, 2048),
        ("very_long_2049_plus", 2049, None),
    ]
    context_buckets = [
        ("no_context_0", 0, 0),
        ("small_1_256", 1, 256),
        ("medium_257_512", 257, 512),
        ("large_513_1024", 513, 1024),
        ("very_large_1025_plus", 1025, None),
    ]
    prefix_hash_counts: Counter[str] = Counter()
    prefix_lengths: list[int] = []
    all_prefix_tokens: list[list[str]] = []
    full_prompt_lengths: list[int] = []
    context_lengths: list[int] = []
    vertical_counts: Counter[str] = Counter()
    prompt_bucket_counts: Counter[str] = Counter()
    context_bucket_counts: Counter[str] = Counter()
    vertical_prompt_lengths: dict[str, list[int]] = defaultdict(list)
    vertical_context_lengths: dict[str, list[int]] = defaultdict(list)
    rows_scanned = 0

    for row in _iter_jsonl(WORKLOAD_PATH):
        rows_scanned += 1
        vertical = str(row.get("vertical") or "unknown")
        vertical_counts[vertical] += 1
        prefix_text = _pre_context_text(row)
        prefix_token_list = _tokens(prefix_text)
        all_prefix_tokens.append(prefix_token_list)
        prefix_lengths.append(len(prefix_token_list))
        prefix_hash = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()[:16]
        prefix_hash_counts[prefix_hash] += 1
        prompt_length = len(_tokens(_message_text(row)))
        context_length = int(row.get("context_token_estimate") or 0)
        full_prompt_lengths.append(prompt_length)
        context_lengths.append(context_length)
        vertical_prompt_lengths[vertical].append(prompt_length)
        vertical_context_lengths[vertical].append(context_length)
        prompt_bucket_counts[_bucket(prompt_length, prompt_buckets)] += 1
        context_bucket_counts[_bucket(context_length, context_buckets)] += 1

    common_all = _common_prefix_length(all_prefix_tokens[:2000])
    top_prefixes = prefix_hash_counts.most_common(10)
    return {
        "source_workload_path": _display_path(WORKLOAD_PATH),
        "historical_tokenizer_reproducibility": (
            "Exact historical model tokenizer settings were not saved locally; this audit uses "
            "deterministic whitespace-token estimates for planning only."
        ),
        "semantic_similarity_counted": False,
        "rows_scanned": rows_scanned,
        "prefix_reuse": {
            "method": "token-identical leading prefix using deterministic whitespace tokens",
            "common_prefix_tokens_first_2000_rows": common_all,
            "prefix_family_count": len(prefix_hash_counts),
            "top_prefix_families": [
                {"prefix_hash": key, "request_count": count} for key, count in top_prefixes
            ],
            "max_prefix_family_share": (
                max(prefix_hash_counts.values()) / rows_scanned if rows_scanned else 0.0
            ),
            "median_prefix_length_estimate": sorted(prefix_lengths)[len(prefix_lengths) // 2]
            if prefix_lengths
            else 0,
            "measured_cache_hits": None,
            "cache_hit_telemetry_available": False,
        },
        "prompt_context_length_distribution": {
            "prompt_token_estimate": {
                "mean": _average(float(value) for value in full_prompt_lengths),
                "min": min(full_prompt_lengths) if full_prompt_lengths else 0,
                "max": max(full_prompt_lengths) if full_prompt_lengths else 0,
                "buckets": dict(sorted(prompt_bucket_counts.items())),
            },
            "context_token_estimate": {
                "mean": _average(float(value) for value in context_lengths),
                "min": min(context_lengths) if context_lengths else 0,
                "max": max(context_lengths) if context_lengths else 0,
                "buckets": dict(sorted(context_bucket_counts.items())),
            },
            "per_vertical": {
                vertical: {
                    "request_count": vertical_counts[vertical],
                    "mean_prompt_tokens": _average(
                        float(value) for value in vertical_prompt_lengths[vertical]
                    ),
                    "mean_context_tokens": _average(
                        float(value) for value in vertical_context_lengths[vertical]
                    ),
                }
                for vertical in sorted(vertical_counts)
            },
        },
    }


def build_workload_opportunity_report() -> JsonDict:
    scan = _workload_scan()
    engine_metrics = _engine_metric_summary()
    vllm_c16 = cast(JsonDict, cast(JsonDict, engine_metrics["vllm"])["by_concurrency"])["16"]
    vllm_c32 = cast(JsonDict, cast(JsonDict, engine_metrics["vllm"])["by_concurrency"])["32"]
    sglang_c16 = cast(JsonDict, cast(JsonDict, engine_metrics["sglang"])["by_concurrency"])["16"]
    sglang_c32 = cast(JsonDict, cast(JsonDict, engine_metrics["sglang"])["by_concurrency"])["32"]
    return {
        "run_id": RUN_ID,
        "status": "WORKLOAD_OPPORTUNITY_AUDITED",
        "semantic_similarity_counted": False,
        "measured_cache_hits_available": False,
        "cache_hits_measured": False,
        "prefix_cache_hit_rate": None,
        "workload_scan": scan,
        "source_artifacts": [
            _display_path(WORKLOAD_PATH),
            _display_path(MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv"),
            _display_path(MAIN_RAW / "main_inference_v1_gpu_telemetry.jsonl"),
        ],
        "prefix_reuse_potential": scan["prefix_reuse"],
        "prompt_context_length_distribution": scan["prompt_context_length_distribution"],
        "concurrency_scheduler_opportunity": {
            "engine_metric_summary": engine_metrics,
            "vllm_c32_minus_c16": {
                "e2e_latency_ms": _float(vllm_c32["mean_e2e_latency_ms"])
                - _float(vllm_c16["mean_e2e_latency_ms"]),
                "ttft_ms": _float(vllm_c32["mean_ttft_ms"]) - _float(vllm_c16["mean_ttft_ms"]),
                "tokens_per_second": _float(vllm_c32["mean_total_tokens_per_second"])
                - _float(vllm_c16["mean_total_tokens_per_second"]),
            },
            "sglang_c32_minus_c16": {
                "e2e_latency_ms": _float(sglang_c32["mean_e2e_latency_ms"])
                - _float(sglang_c16["mean_e2e_latency_ms"]),
                "ttft_ms": _float(sglang_c32["mean_ttft_ms"]) - _float(sglang_c16["mean_ttft_ms"]),
                "tokens_per_second": _float(sglang_c32["mean_total_tokens_per_second"])
                - _float(sglang_c16["mean_total_tokens_per_second"]),
            },
            "interpretation": (
                "The saved comparison suggests c32 increased latency and reduced average "
                "tokens/sec versus c16 for both self-hosted engines. That points toward "
                "controlled scheduler/batch tuning, not a simple concurrency increase."
            ),
        },
        "kv_cache_opportunity": {
            "exact_kv_bytes_per_token": None,
            "reason": (
                "Model architecture, KV block counts, and engine cache occupancy were not "
                "recorded. Later experiments need KV telemetry before making cache claims."
            ),
            "vram_telemetry_available": True,
            "kv_block_telemetry_available": False,
        },
        "chunked_prefill_opportunity": {
            "state": "applicable_after_instrumentation",
            "evidence": (
                "The workload contains long prompt/context buckets and mixed memory modes, but "
                "prefill chunk timing was not captured."
            ),
            "required_measurement": [
                "prefill_time_ms",
                "chunk_count",
                "decode_interference",
                "length_bucket_ttft",
            ],
        },
        "quantization_opportunity": {
            "state": "blocked_by_negative_rule",
            "baseline_precision": "unknown",
            "quantized_checkpoint_registered": False,
            "reason": (
                "Main_Inference quality/safety failed and no quantized Qwen2.5-7B checkpoint "
                "is registered in configs/models.yaml."
            ),
        },
        "speculative_decoding_opportunity": {
            "state": "not_implemented",
            "draft_model_registered": False,
            "acceptance_telemetry_available": False,
            "reason": "No compatible draft model or acceptance-rate instrumentation is registered.",
        },
        "limitations": [
            "No measured cache hits, cache evictions, or speculation acceptance are claimed.",
            "Exact historical tokenizer state is unavailable; token counts are planning estimates.",
            "Main raw 250k response rows are not required for this planning audit.",
        ],
    }


def _candidate_state(optimization: JsonDict) -> str:
    oid = str(optimization["optimization_id"])
    if oid in {"prompt_prefix_layout_optimization", "scheduler_batch_tuning"}:
        return "selected_for_one_factor_test"
    if oid in {
        "prefix_cache_verification_tuning",
        "kv_cache_capacity_allocation_tuning",
        "chunked_prefill_tuning",
    }:
        return "applicable_after_instrumentation"
    if oid == "quantization":
        return "blocked_by_negative_rule"
    if oid == "speculative_decoding":
        return "not_implemented"
    if oid == "multi_gpu_parallelism":
        return "not_applicable_current_project"
    if oid in {
        "model_compression",
        "kv_cache_offloading_hierarchical_cache",
        "manual_kernel_compiler_optimization",
        "prefill_decode_disaggregation",
        "distributed_capacity_serving_architecture",
    }:
        return "future_architecture"
    return "applicable_candidate"


def build_applicability_matrix() -> JsonDict:
    repair_gate = _repair_gate()
    states = []
    for optimization in _core_optimizations():
        state = _candidate_state(optimization)
        oid = str(optimization["optimization_id"])
        states.append(
            {
                "optimization_id": oid,
                "display_name": optimization["display_name"],
                "difficulty_tier": optimization["difficulty_tier"],
                "category": optimization["category"],
                "applicability_state": state,
                "why": _applicability_reason(oid, state),
                "observed_opportunity": _observed_opportunity(oid),
                "causally_isolatable": state
                in {
                    "selected_for_one_factor_test",
                    "applicable_candidate",
                    "applicable_after_instrumentation",
                },
                "instrument_first": state == "applicable_after_instrumentation",
                "feasible_on_one_a100": "a100_sxm_80gb"
                in cast(list[str], optimization["hardware_compatibility"]),
                "quality_safety_protected": True,
                "implementation_cost": _implementation_cost(oid),
                "paper_demo_value": _paper_demo_value(oid),
                "result_type": "planned",
            }
        )
    return {
        "run_id": RUN_ID,
        "status": "CORE_OPTIMIZATION_APPLICABILITY_READY",
        "context": {
            "target_self_hosted_model": "Qwen/Qwen2.5-7B-Instruct",
            "api_model": "meta-llama/Llama-3.1-8B-Instruct",
            "hardware": "one A100-SXM4-80GB",
            "traffic_profile": "online_low_latency",
            "mandatory_repairs_included_future_run": bool(
                repair_gate.get("deployability_repair_sample_validated")
            ),
        },
        "states": states,
        "state_counts": {
            state: sum(1 for item in states if item["applicability_state"] == state)
            for state in sorted({str(item["applicability_state"]) for item in states})
        },
    }


def _applicability_reason(optimization_id: str, state: str) -> str:
    reasons = {
        "selected_for_one_factor_test": "Evidence is available and the factor can be isolated.",
        "applicable_after_instrumentation": (
            "Likely relevant, but missing telemetry would make claims weak."
        ),
        "blocked_by_negative_rule": (
            "Negative rules block this until quality/safety are protected and support exists."
        ),
        "not_implemented": "Required model/runtime support or instrumentation is absent.",
        "not_applicable_current_project": "Current final run target is one A100, not multi-GPU.",
        "future_architecture": "Important concept, but outside the current one-A100 program.",
        "applicable_candidate": "Applicable after higher-priority one-factor tests.",
    }
    return reasons[state]


def _observed_opportunity(optimization_id: str) -> str:
    mapping = {
        "prompt_prefix_layout_optimization": "Exact shared prompt scaffolding exists.",
        "prefix_cache_verification_tuning": (
            "Reusable prefix families exist, but cache hits are unmeasured."
        ),
        "scheduler_batch_tuning": "c32 underperformed c16 in saved comparison artifacts.",
        "kv_cache_capacity_allocation_tuning": (
            "A100 VRAM was heavily allocated; KV details missing."
        ),
        "chunked_prefill_tuning": "Long-context and mixed-length requests create prefill risk.",
        "quantization": "Potential VRAM/cost lever, currently gated by quality risk.",
        "speculative_decoding": "Potential TPOT lever, but no draft model is registered.",
    }
    return mapping.get(optimization_id, "No direct measured opportunity yet.")


def _implementation_cost(optimization_id: str) -> str:
    if optimization_id in {"prompt_prefix_layout_optimization"}:
        return "low"
    if optimization_id in {
        "prefix_cache_verification_tuning",
        "scheduler_batch_tuning",
        "kv_cache_capacity_allocation_tuning",
        "chunked_prefill_tuning",
    }:
        return "medium"
    return "high"


def _paper_demo_value(optimization_id: str) -> str:
    if optimization_id in {
        "prompt_prefix_layout_optimization",
        "prefix_cache_verification_tuning",
        "scheduler_batch_tuning",
    }:
        return "high"
    if optimization_id in {"quantization", "speculative_decoding"}:
        return "medium_when_supported"
    return "future"


def build_applicability_matrix_rows() -> list[dict[str, Any]]:
    payload = build_applicability_matrix()
    return [
        {
            "optimization_id": row["optimization_id"],
            "display_name": row["display_name"],
            "difficulty_tier": row["difficulty_tier"],
            "category": row["category"],
            "applicability_state": row["applicability_state"],
            "feasible_on_one_a100": row["feasible_on_one_a100"],
            "instrument_first": row["instrument_first"],
            "result_type": row["result_type"],
            "why": row["why"],
        }
        for row in cast(list[JsonDict], payload["states"])
    ]


def _instrumentation_fields() -> list[JsonDict]:
    return [
        {"field": "exact_reusable_prefix_tokens", "area": "prefix_cache", "state": "derivable"},
        {"field": "cache_lookup_count", "area": "prefix_cache", "state": "missing"},
        {"field": "cache_hit_count", "area": "prefix_cache", "state": "missing"},
        {"field": "cache_miss_count", "area": "prefix_cache", "state": "missing"},
        {"field": "hit_token_count", "area": "prefix_cache", "state": "missing"},
        {"field": "hit_miss_ttft_ms", "area": "prefix_cache", "state": "missing"},
        {"field": "cache_occupancy", "area": "prefix_cache", "state": "missing"},
        {"field": "eviction_count", "area": "prefix_cache", "state": "missing"},
        {"field": "queue_wait_ms", "area": "scheduler_batching", "state": "missing"},
        {"field": "active_batch_size", "area": "scheduler_batching", "state": "missing"},
        {"field": "scheduled_tokens", "area": "scheduler_batching", "state": "missing"},
        {"field": "prefill_tokens_per_iteration", "area": "scheduler_batching", "state": "missing"},
        {"field": "decode_tokens_per_iteration", "area": "scheduler_batching", "state": "missing"},
        {"field": "running_waiting_requests", "area": "scheduler_batching", "state": "missing"},
        {"field": "kv_blocks_total", "area": "kv_cache", "state": "missing"},
        {"field": "kv_blocks_free", "area": "kv_cache", "state": "missing"},
        {"field": "kv_blocks_used", "area": "kv_cache", "state": "missing"},
        {"field": "kv_bytes_per_token", "area": "kv_cache", "state": "missing"},
        {"field": "oom_count", "area": "kv_cache", "state": "already_captured_indirectly"},
        {"field": "chunk_count", "area": "chunked_prefill", "state": "missing"},
        {"field": "chunk_size", "area": "chunked_prefill", "state": "missing"},
        {"field": "length_bucket_ttft_ms", "area": "chunked_prefill", "state": "derivable"},
        {"field": "model_memory_mb", "area": "quantization", "state": "missing"},
        {"field": "precision_format", "area": "quantization", "state": "missing"},
        {"field": "drafted_tokens", "area": "speculation", "state": "missing"},
        {"field": "accepted_tokens", "area": "speculation", "state": "missing"},
        {"field": "acceptance_rate", "area": "speculation", "state": "missing"},
    ]


def build_instrumentation_gap_report() -> JsonDict:
    fields = _instrumentation_fields()
    return {
        "run_id": RUN_ID,
        "status": "INSTRUMENTATION_GAPS_IDENTIFIED",
        "fields": fields,
        "state_counts": {
            state: sum(1 for item in fields if item["state"] == state)
            for state in sorted({str(item["state"]) for item in fields})
        },
        "candidate_requirements": {
            "prefix_cache_verification_tuning": [
                "cache_lookup_count",
                "cache_hit_count",
                "cache_miss_count",
                "hit_token_count",
                "cache_occupancy",
            ],
            "scheduler_batch_tuning": [
                "queue_wait_ms",
                "active_batch_size",
                "scheduled_tokens",
                "running_waiting_requests",
            ],
            "kv_cache_capacity_allocation_tuning": [
                "kv_blocks_total",
                "kv_blocks_free",
                "kv_blocks_used",
                "kv_bytes_per_token",
            ],
            "chunked_prefill_tuning": [
                "chunk_count",
                "chunk_size",
                "length_bucket_ttft_ms",
            ],
            "quantization": ["model_memory_mb", "precision_format", "quality_metrics"],
            "speculative_decoding": ["drafted_tokens", "accepted_tokens", "acceptance_rate"],
        },
    }


def build_one_factor_experiment_plan() -> JsonDict:
    protected = [
        "json_validity",
        "contract_validity",
        "format_validity",
        "evidence_match",
        "groundedness",
        "safety_findings",
        "truncation",
        "completion_failure_rate",
        "escalation_behavior",
        "mm4_trace_bounds",
    ]
    common_constants = [
        "parent_run_id",
        "prompt_ids",
        "dataset_version",
        "gold_records",
        "evaluator",
        "mandatory_deployability_repairs",
        "model_alias_except_model_routing_tests",
        "temperature",
        "traffic_profile",
        "memory_mode_except_memory_specific_tests",
    ]
    experiments = [
        {
            "rank": 1,
            "experiment_id": "coreopt_prefix_layout_static_v1",
            "parent_run": RUN_ID,
            "optimization_id": "prompt_prefix_layout_optimization",
            "hypothesis": (
                "A more stable leading prefix increases cacheability without changing quality."
            ),
            "changed_variable": "rendered_prompt_prefix_layout",
            "held_constant_variables": common_constants,
            "workload_sample": "static CPU diff over final_10000 source workload",
            "sample_stratification": ["vertical", "memory_mode", "expected_status"],
            "engine_model": "planning-only; later vLLM/SGLang measured run",
            "baseline_config": "Main_Inference_V1 rendered prompt layout",
            "candidate_configs": ["prefix_grouped_schema_first_layout"],
            "telemetry_required": ["exact_reusable_prefix_tokens", "prompt_token_distribution"],
            "target_metrics": ["reusable_prefix_tokens", "future_ttft"],
            "protected_quality_safety_metrics": protected,
            "acceptance_criteria": [
                "no gold leakage",
                "same evaluator contract",
                "larger exact prefix",
            ],
            "rejection_criteria": [
                "contract text changes semantics",
                "evidence visibility worsens",
            ],
            "estimated_runtime_cost": "CPU only for static diff",
            "device_requirements": "cpu",
            "artifact_outputs": [
                "coreopt_prefix_layout_static_v1_prompt_diff_report.json",
                "coreopt_prefix_layout_static_v1_summary.csv",
            ],
            "ui_replay_requirements": ["prefix waterfall", "changed-token diff"],
            "paper_recruiter_learning_value": (
                "Shows that prefix caching starts with prompt layout."
            ),
            "does_not_execute_automatically": True,
        },
        {
            "rank": 2,
            "experiment_id": "coreopt_scheduler_batch_vllm_v1",
            "parent_run": RUN_ID,
            "optimization_id": "scheduler_batch_tuning",
            "hypothesis": (
                "A controlled scheduler/batch setting improves latency/throughput balance."
            ),
            "changed_variable": "scheduler_batch_config",
            "held_constant_variables": common_constants,
            "workload_sample": "deterministic targeted self-hosted sample, approval required",
            "sample_stratification": ["vertical", "memory_mode", "length_bucket"],
            "engine_model": "vLLM + model3_7b",
            "baseline_config": "Main_Inference_V1 vLLM baseline scenario",
            "candidate_configs": ["one_scheduler_batch_variant_at_a_time"],
            "telemetry_required": ["queue_wait_ms", "active_batch_size", "running_requests"],
            "target_metrics": ["ttft", "tpot", "e2e_latency", "tokens_per_second"],
            "protected_quality_safety_metrics": protected,
            "acceptance_criteria": [
                "quality/safety gates pass",
                "request completion remains complete",
                "measured latency or throughput improves",
            ],
            "rejection_criteria": ["tail latency worsens", "OOM", "quality/safety gate fails"],
            "estimated_runtime_cost": "small A100 run, exact count pending approval",
            "device_requirements": "a100_sxm_80gb",
            "artifact_outputs": [
                "coreopt_scheduler_batch_vllm_v1_manifest.json",
                "coreopt_scheduler_batch_vllm_v1_eval_report.json",
            ],
            "ui_replay_requirements": ["queue chart", "batch-size timeline", "latency overlay"],
            "paper_recruiter_learning_value": "Explains why more concurrency is not always faster.",
            "does_not_execute_automatically": True,
        },
        {
            "rank": 3,
            "experiment_id": "coreopt_prefix_cache_vllm_v1",
            "parent_run": RUN_ID,
            "optimization_id": "prefix_cache_verification_tuning",
            "hypothesis": "Exact shared prefixes can reduce TTFT when cache hits are recorded.",
            "changed_variable": "prefix_cache_flag",
            "held_constant_variables": common_constants,
            "workload_sample": "prefix-family stratified sample, approval required",
            "sample_stratification": ["prefix_family", "vertical", "memory_mode"],
            "engine_model": "vLLM + model3_7b",
            "baseline_config": "same vLLM scenario with prefix cache off/unknown",
            "candidate_configs": ["prefix_cache_enabled"],
            "telemetry_required": [
                "hit_count",
                "miss_count",
                "hit_token_count",
                "hit_miss_ttft_ms",
            ],
            "target_metrics": ["ttft", "prefill_time", "tokens_per_second"],
            "protected_quality_safety_metrics": protected,
            "acceptance_criteria": ["cache hits measured", "TTFT improves for hit rows"],
            "rejection_criteria": ["no cache hits", "quality/safety regression"],
            "estimated_runtime_cost": "small A100 run after instrumentation",
            "device_requirements": "a100_sxm_80gb",
            "artifact_outputs": [
                "coreopt_prefix_cache_vllm_v1_cache_report.json",
                "coreopt_prefix_cache_vllm_v1_eval_report.json",
            ],
            "ui_replay_requirements": ["hit/miss chart", "reused-token distribution"],
            "paper_recruiter_learning_value": "Connects workload construction to serving speed.",
            "does_not_execute_automatically": True,
        },
        {
            "rank": 4,
            "experiment_id": "coreopt_chunked_prefill_sglang_v1",
            "parent_run": RUN_ID,
            "optimization_id": "chunked_prefill_tuning",
            "hypothesis": "Chunked prefill reduces long-context TTFT and queue interference.",
            "changed_variable": "chunked_prefill_config",
            "held_constant_variables": common_constants,
            "workload_sample": "long-context stratified sample, approval required",
            "sample_stratification": ["length_bucket", "vertical", "memory_mode"],
            "engine_model": "SGLang + model3_7b",
            "baseline_config": "Main_Inference_V1 SGLang baseline scenario",
            "candidate_configs": ["one_chunked_prefill_variant_at_a_time"],
            "telemetry_required": ["chunk_count", "chunk_size", "length_bucket_ttft_ms"],
            "target_metrics": ["ttft", "p95_e2e_latency", "p99_e2e_latency"],
            "protected_quality_safety_metrics": protected,
            "acceptance_criteria": ["long-bucket TTFT improves", "no completion regression"],
            "rejection_criteria": ["decode throughput worsens materially", "quality/safety fails"],
            "estimated_runtime_cost": "small A100 run after instrumentation",
            "device_requirements": "a100_sxm_80gb",
            "artifact_outputs": ["coreopt_chunked_prefill_sglang_v1_eval_report.json"],
            "ui_replay_requirements": ["long-prefill lane", "length-bucket TTFT chart"],
            "paper_recruiter_learning_value": "Shows prefill/decode tradeoffs visually.",
            "does_not_execute_automatically": True,
        },
    ]
    return {
        "run_id": RUN_ID,
        "status": "ONE_FACTOR_EXPERIMENT_PROGRAM_PLANNED",
        "does_not_execute_inference": True,
        "does_not_create_optimized_inference_v1": True,
        "selection_rule": (
            "Ranked by prerequisite order, causal clarity, instrumentation readiness, cost, "
            "and recruiter learning value."
        ),
        "experiments": experiments,
        "not_selected_yet": [
            {
                "optimization_id": "quantization",
                "reason": (
                    "Blocked until repair path is full-run validated and checkpoint support exists."
                ),
            },
            {
                "optimization_id": "speculative_decoding",
                "reason": "No compatible draft model or acceptance telemetry exists.",
            },
            {
                "optimization_id": "multi_gpu_parallelism",
                "reason": "Current final target is one A100.",
            },
        ],
    }


def build_experiment_scale_plan() -> JsonDict:
    return {
        "version": 1,
        "status": "SCALE_STRATEGY_PLANNED",
        "scales": [
            {
                "scale_id": "A_static_cpu_audit",
                "purpose": (
                    "Validate taxonomy, prefix reuse, prompt diff, and instrumentation plan."
                ),
                "request_count": 0,
                "sampling_method": (
                    "deterministic full-workload metadata scan where local files exist"
                ),
                "vertical_coverage": "all five verticals",
                "memory_mode_coverage": "source workload plus processed matrix summaries",
                "model_engine_coverage": "no live model",
                "device": "cpu",
                "telemetry": ["static token estimates", "prefix family counts"],
                "artifact_requirements": ["planning reports", "no measured inference outputs"],
                "promotion_gate": "no leakage, deterministic outputs, reviewed plan",
            },
            {
                "scale_id": "B_targeted_one_factor_validation",
                "purpose": "Measure one changed serving factor on a small representative sample.",
                "request_count": "approval_required",
                "sampling_method": (
                    "deterministic stratified by vertical, memory mode, and length bucket"
                ),
                "vertical_coverage": "all five where feasible",
                "memory_mode_coverage": "minimum mm2 plus targeted modes relevant to the factor",
                "model_engine_coverage": "model3_7b on one selected engine per experiment",
                "device": "A100 only when execution is approved",
                "telemetry": ["factor-specific telemetry", "quality/safety metrics"],
                "artifact_requirements": ["manifest", "raw results", "telemetry", "eval report"],
                "promotion_gate": "quality/safety gates pass and factor improves target metrics",
            },
            {
                "scale_id": "C_medium_candidate_bundle_validation",
                "purpose": "Combine validated compatible factors before full run.",
                "request_count": "approval_required",
                "sampling_method": "deterministic larger sample preserving matrix shape",
                "vertical_coverage": "all five",
                "memory_mode_coverage": "candidate champion modes",
                "model_engine_coverage": "selected champion route only",
                "device": "A100",
                "telemetry": ["full runtime and factor-specific telemetry"],
                "artifact_requirements": ["bundle manifest", "comparison report", "repair traces"],
                "promotion_gate": "no gate failure and clear Pareto improvement",
            },
            {
                "scale_id": "D_full_optimized_inference_v1",
                "purpose": "Run final champion against official workload and evaluation framework.",
                "request_count": 250000,
                "sampling_method": "official full matrix workload",
                "vertical_coverage": "all five",
                "memory_mode_coverage": "selected champion configuration",
                "model_engine_coverage": "approved champion",
                "device": "A100 or approved provider route",
                "telemetry": ["full manifest", "checkpoint", "GPU/API cost", "SLO report"],
                "artifact_requirements": ["optimized_inference_v1 complete artifact set"],
                "promotion_gate": "SLO pass and human approval",
            },
        ],
    }


def build_acceptance_gate_schema() -> JsonDict:
    return {
        "version": 1,
        "status": "SCHEMA_ONLY_NO_THRESHOLDS_INVENTED",
        "protected_metrics": [
            "json_validity",
            "contract_validity",
            "format_validity",
            "evidence_match",
            "groundedness",
            "safety_findings",
            "truncation",
            "completion_failure_rate",
            "escalation_behavior",
            "mm4_trace_bounds",
        ],
        "performance_metrics": [
            "ttft",
            "tpot",
            "itl",
            "e2e_latency",
            "throughput",
            "gpu_utilization",
            "vram",
            "cost",
            "tokens_per_gpu_dollar",
        ],
        "regression_budget_policy": {
            "repo_explicit_one_factor_tolerances_present": False,
            "schema_fields": [
                "metric_name",
                "baseline_value",
                "candidate_value",
                "target",
                "allowed_regression",
                "approval_required",
            ],
            "note": (
                "The repo has SLO targets, but no explicit one-factor regression budgets. "
                "This schema preserves a place for them without inventing values."
            ),
        },
    }


def build_scenario_registry() -> JsonDict:
    return {
        "version": 1,
        "status": "SCENARIO_REGISTRY_PLANNED",
        "scenario_types": [
            "engine_baseline",
            "repair_validation",
            "one_factor",
            "candidate_bundle",
            "champion_full_run",
            "educational_future",
        ],
        "scenarios": [
            {
                "scenario_id": RUN_ID,
                "scenario_type": "engine_baseline",
                "result_type": "measured",
                "lineage": {"parent_run_id": None},
                "changed_factor": None,
                "held_constants": [],
                "artifact_paths": [
                    _display_path(MAIN_RAW / "main_inference_v1_manifest.json"),
                    _display_path(MAIN_PROCESSED / "main_inference_v1_eval_report.json"),
                    _display_path(MAIN_PROCESSED / "main_inference_v1_engine_comparison.csv"),
                ],
                "ui_replay_available": True,
                "conclusion_available": True,
            },
            {
                "scenario_id": REPAIR_RUN_ID,
                "scenario_type": "repair_validation",
                "result_type": "measured",
                "lineage": {"parent_run_id": RUN_ID},
                "changed_factor": "deployability_repair_logic",
                "held_constants": [
                    "no_core_optimization",
                    "no_a100",
                    "no_inference_results_mutation",
                ],
                "artifact_paths": [
                    _display_path(
                        REPAIR_ROOT
                        / "processed/deployability_repair_validation_v1_validation_gate_report.json"
                    )
                ],
                "ui_replay_available": True,
                "conclusion_available": True,
            },
            {
                "scenario_id": "coreopt_prefix_layout_static_v1",
                "scenario_type": "one_factor",
                "result_type": "planned",
                "lineage": {"parent_run_id": RUN_ID},
                "changed_factor": "rendered_prompt_prefix_layout",
                "held_constants": ["dataset", "evaluator", "gold_records", "repairs"],
                "artifact_paths": [],
                "ui_replay_available": False,
                "conclusion_available": False,
            },
            {
                "scenario_id": "coreopt_scheduler_batch_vllm_v1",
                "scenario_type": "one_factor",
                "result_type": "planned",
                "lineage": {"parent_run_id": RUN_ID},
                "changed_factor": "scheduler_batch_config",
                "held_constants": ["dataset", "evaluator", "model3_7b", "repairs"],
                "artifact_paths": [],
                "ui_replay_available": False,
                "conclusion_available": False,
            },
            {
                "scenario_id": "coreopt_prefix_cache_vllm_v1",
                "scenario_type": "one_factor",
                "result_type": "planned",
                "lineage": {"parent_run_id": RUN_ID},
                "changed_factor": "prefix_cache_flag",
                "held_constants": ["dataset", "evaluator", "model3_7b", "repairs"],
                "artifact_paths": [],
                "ui_replay_available": False,
                "conclusion_available": False,
            },
            {
                "scenario_id": "coreopt_chunked_prefill_sglang_v1",
                "scenario_type": "one_factor",
                "result_type": "planned",
                "lineage": {"parent_run_id": RUN_ID},
                "changed_factor": "chunked_prefill_config",
                "held_constants": ["dataset", "evaluator", "model3_7b", "repairs"],
                "artifact_paths": [],
                "ui_replay_available": False,
                "conclusion_available": False,
            },
            {
                "scenario_id": "optimized_inference_v1",
                "scenario_type": "champion_full_run",
                "result_type": "missing_not_created",
                "lineage": {"parent_run_id": RUN_ID},
                "changed_factor": "human_approved_champion_recipe",
                "held_constants": ["official_workload", "slo_targets", "evaluator"],
                "artifact_paths": [],
                "ui_replay_available": False,
                "conclusion_available": False,
            },
            {
                "scenario_id": "tensorrt_llm_future",
                "scenario_type": "educational_future",
                "result_type": "future",
                "lineage": {"parent_run_id": None},
                "changed_factor": "future_engine_architecture",
                "held_constants": [],
                "artifact_paths": [],
                "ui_replay_available": False,
                "conclusion_available": False,
            },
        ],
        "champion_selected": False,
    }


def build_ui_contract() -> JsonDict:
    return {
        "version": 1,
        "status": "UI_CONTRACT_DESIGNED_NO_MEASURED_OPTIMIZATION_RESULTS",
        "does_not_create_optimized_inference_v1": True,
        "ui_state_labels": {
            "measured_baseline_capability": "Measured baseline capability",
            "engine_inherent": "Engine inherent",
            "unknown": "Unknown, not instrumented",
            "planned_candidate": "Planned candidate",
            "requires_instrumentation": "Requires instrumentation",
            "blocked_by_negative_rule": "Blocked by negative rule",
            "future_architecture": "Future architecture",
            "missing_result": "Missing result",
        },
        "frontend_must_not_display_as_measured": [
            "core optimization candidates",
            "modeled expected improvements",
            "future TensorRT-LLM or multi-GPU concepts",
        ],
        "stages": {
            "main_inference_simulation": {
                "section": "Serving Engine Optimization Baseline",
                "cards": [
                    "baseline capability cards",
                    "activation state",
                    "mechanism",
                    "evidence source",
                    "untuned settings",
                    "future experiment link",
                ],
                "callouts": [
                    "vLLM/SGLang are optimized serving baselines.",
                    "Engine-native capability is not the same as an applied experiment.",
                ],
            },
            "core_optimization_lab": {
                "organization": "difficulty_tier",
                "labels": [
                    "selected_for_one_factor_test",
                    "applicable_candidate",
                    "applicable_after_instrumentation",
                    "already_active_baseline",
                    "not_applicable_current_project",
                    "blocked_by_negative_rule",
                    "not_implemented",
                    "future_architecture",
                ],
                "visuals": [
                    "exact prefix reuse",
                    "cache hits/misses",
                    "scheduler queues",
                    "chunked prefill",
                    "KV blocks",
                    "quantization memory",
                    "speculative token acceptance",
                ],
            },
            "optimized_inference_simulation": {
                "show_only": "selected champion recipe after human approval",
                "requires": [
                    "mandatory repairs",
                    "selected core optimizations",
                    "config deltas",
                    "progress events",
                    "telemetry",
                    "quality protection",
                    "final measured result",
                ],
                "current_state": "not_available",
            },
        },
        "example_records": {
            "baseline_capability": _engine_baseline_capabilities()[0],
            "planned_core_optimization": _core_optimizations()[0],
            "planned_scenario": build_scenario_registry()["scenarios"][2],
        },
        "no_fabricated_values": True,
    }


def build_champion_selection_framework() -> JsonDict:
    return {
        "version": 1,
        "status": "FRAMEWORK_READY_NO_CHAMPION_SELECTED",
        "mandatory_conditions": [
            "quality/safety protection gates pass",
            "no critical reliability regression",
            "exact experiment artifact is complete",
            "human approval recorded",
        ],
        "ranking_dimensions": [
            "ttft_improvement",
            "tpot_improvement",
            "e2e_improvement",
            "throughput_improvement",
            "gpu_utilization",
            "vram_efficiency",
            "cost_efficiency",
            "operational_complexity",
            "implementation_risk",
            "reproducibility",
        ],
        "scorecard_schema": {
            "candidate_id": "string",
            "measured_artifact_root": "path",
            "protected_gate_status": "PASS|FAIL",
            "performance_scores": "object",
            "pareto_frontier_member": "bool",
            "recommended_by_system": "bool",
            "approved_by_engineer": "bool",
            "approval_record_path": "path|null",
        },
        "pareto_frontier_design": (
            "Compare only candidates with passing protection gates; maximize speed/cost gains "
            "while minimizing operational risk."
        ),
        "human_approval_record": {
            "required": True,
            "fields": ["approver", "timestamp_utc", "candidate_id", "reason", "accepted_risks"],
        },
        "champion_selected": False,
    }


def _markdown_report(
    *,
    audit: JsonDict,
    capability_report: JsonDict,
    workload: JsonDict,
    applicability: JsonDict,
    plan: JsonDict,
) -> str:
    selected = [
        row["optimization_id"]
        for row in cast(list[JsonDict], applicability["states"])
        if row["applicability_state"] == "selected_for_one_factor_test"
    ]
    unknown_capabilities = [
        item["capability_id"]
        for item in cast(list[JsonDict], capability_report["capabilities"])
        if item["baseline_activation_state"] == "unknown"
    ]
    removed_core = ", ".join(
        cast(list[str], audit["removed_from_core_list"])
        if "removed_from_core_list" in audit
        else sorted(REPAIR_IDS)
    )
    ranked_program = "\n".join(
        (f"{item['rank']}. `{item['experiment_id']}` changes `{item['changed_variable']}` only.")
        for item in cast(list[JsonDict], plan["experiments"])
    )
    return f"""# Core Optimization Planning And Baseline Capability Audit

Status: planning audit completed; no inference executed.

## Scope

This document converts the broad optimization catalog into a measurable core
inference optimization program. It does not modify Main_Inference_V1, create
Optimized_Inference_V1, or claim a core optimization result.

## Audit Findings

- Catalog entries audited: {len(load_optimization_catalog())}
- Engine-native baseline capability entries:
  {len(cast(list[str], audit["engine_native_baseline_capability_ids"]))}
- Deployability repair IDs removed from the core list:
  {removed_core}
- Engine support is not treated as confirmed activation.
- Main_Inference_V1 manifest records `optimization_flags: []`.

## Baseline Engine Capability State

The baseline used vLLM, SGLang, and an API provider route. vLLM and SGLang
therefore supplied engine-native serving behavior, but the repo does not contain
enough startup or engine telemetry to prove every cache, kernel, CUDA Graph, or
chunked-prefill state.

Unknown or unproven capability states:

{chr(10).join(f"- `{item}`" for item in unknown_capabilities)}

## Workload Opportunity

- Workload source scanned:
  `{workload["source_artifacts"][0]}`
- Prefix audit counts token-identical leading tokens only.
- Measured cache hits: not available.
- KV block telemetry: not available.
- Speculation acceptance telemetry: not available.

The saved comparison suggests concurrency 32 increased latency and reduced
average tokens/sec versus concurrency 16 for both self-hosted engines, so the
first serving experiment should test scheduler/batch behavior carefully rather
than simply increasing concurrency.

## Selected First Candidates

{chr(10).join(f"- `{item}`" for item in selected)}

## Ranked One-Factor Program

{ranked_program}

## Four-Layer Taxonomy

1. Engine baseline capabilities: serving behavior supplied by vLLM, SGLang,
   API provider routes, or planned TensorRT-LLM support.
2. Engineer-applied core optimizations: deliberate workload, runtime, model,
   hardware, or distributed-serving changes.
3. Applicable experiment candidates: core optimizations filtered for the
   current one-A100, model3/model6, vLLM/SGLang/API-provider project state.
4. Measured optimization scenarios: saved scenarios only. Main_Inference_V1 and
   Deployability_Repair_Validation_V1 are measured; one-factor core candidates
   are planned; Optimized_Inference_V1 is missing/not created.

## Output Artifacts

Configuration artifacts:

- `configs/core_optimization_taxonomy.yaml`
- `configs/core_optimization_scenario_registry.yaml`

Processed planning artifacts under
`experiments/main/main_inference_v1/processed/`:

- `main_inference_v1_engine_baseline_capability_report.json`
- `core_optimization_applicability_matrix.json`
- `core_optimization_workload_opportunity_report.json`
- `core_optimization_instrumentation_gap_report.json`
- `core_optimization_one_factor_experiment_plan.json`
- `core_optimization_ui_contract.json`
- `core_optimization_champion_selection_framework.json`

## Instrumentation Gaps

Later core optimization experiments need cache hits/misses, hit token count,
cache occupancy, queue wait, active batch size, scheduled tokens, KV block
counts, prefill/decode split timing, chunk counts, quantization metadata,
speculative acceptance, and engine version/startup flags. Until those fields
exist, the UI must show these states as planned or unknown, not measured.

## Recommended First Core Task

Start with `coreopt_prefix_layout_static_v1`. It is CPU-only, isolates one
changed factor, protects the evaluator and gold data, and teaches why prefix
caching begins with stable prompt layout. The first live GPU follow-up is
`coreopt_scheduler_batch_vllm_v1`, but it should run only after the static
layout audit and after the required queue/batch telemetry fields are available.

## Current State

`Deployability_Repair_Validation_V1` is sample-validated. Core optimization
planning is complete. One-factor experiments have not run. `Optimized_Inference_V1`
does not exist yet.
"""


def build_all_core_optimization_planning_artifacts() -> dict[str, JsonDict]:
    taxonomy = build_core_optimization_taxonomy()
    audit = build_current_catalog_audit()
    capability_report = build_engine_baseline_capability_report()
    applicability = build_applicability_matrix()
    workload = build_workload_opportunity_report()
    instrumentation = build_instrumentation_gap_report()
    plan = build_one_factor_experiment_plan()
    scale = build_experiment_scale_plan()
    acceptance = build_acceptance_gate_schema()
    scenario_registry = build_scenario_registry()
    ui_contract = build_ui_contract()
    champion = build_champion_selection_framework()
    taxonomy["catalog_audit"] = audit
    taxonomy["layers"]["C_applicable_experiment_candidates"] = applicability["states"]
    return {
        "taxonomy": taxonomy,
        "catalog_audit": audit,
        "capability_report": capability_report,
        "applicability": applicability,
        "workload": workload,
        "instrumentation": instrumentation,
        "plan": plan,
        "scale": scale,
        "acceptance": acceptance,
        "scenario_registry": scenario_registry,
        "ui_contract": ui_contract,
        "champion": champion,
    }


def write_core_optimization_planning_artifacts() -> dict[str, str]:
    artifacts = build_all_core_optimization_planning_artifacts()
    _write_yaml(TAXONOMY_PATH, artifacts["taxonomy"])
    _write_json(ENGINE_BASELINE_REPORT_PATH, artifacts["capability_report"])
    _write_csv(
        ENGINE_BASELINE_SUMMARY_PATH,
        build_engine_baseline_capability_summary_rows(),
        [
            "capability_id",
            "display_name",
            "engine",
            "category",
            "baseline_activation_state",
            "separately_measured",
            "tunable_later",
            "source_artifact_config",
        ],
    )
    _write_json(APPLICABILITY_JSON_PATH, artifacts["applicability"])
    _write_csv(
        APPLICABILITY_CSV_PATH,
        build_applicability_matrix_rows(),
        [
            "optimization_id",
            "display_name",
            "difficulty_tier",
            "category",
            "applicability_state",
            "feasible_on_one_a100",
            "instrument_first",
            "result_type",
            "why",
        ],
    )
    _write_json(WORKLOAD_OPPORTUNITY_PATH, artifacts["workload"])
    _write_json(INSTRUMENTATION_GAP_PATH, artifacts["instrumentation"])
    _write_json(ONE_FACTOR_PLAN_PATH, artifacts["plan"])
    _write_json(SCALE_PLAN_PATH, artifacts["scale"])
    _write_json(ACCEPTANCE_GATE_PATH, artifacts["acceptance"])
    _write_yaml(SCENARIO_REGISTRY_PATH, artifacts["scenario_registry"])
    _write_json(UI_CONTRACT_PATH, artifacts["ui_contract"])
    _write_json(CHAMPION_FRAMEWORK_PATH, artifacts["champion"])
    _repo_path(DOC_PATH).write_text(
        _markdown_report(
            audit=artifacts["catalog_audit"],
            capability_report=artifacts["capability_report"],
            workload=artifacts["workload"],
            applicability=artifacts["applicability"],
            plan=artifacts["plan"],
        ),
        encoding="utf-8",
    )
    _repo_path(SUMMARY_DOC_PATH).write_text(
        """# Core Optimization Planning Baseline Audit Summary

Status: planning audit completed; no inference executed.

The repo now separates engine-native baseline capabilities from
engineer-applied core inference optimizations. Main_Inference_V1 remains
immutable, deployability repairs are sample-validated, one-factor core
experiments are planned but not run, and Optimized_Inference_V1 is still
pending.
""",
        encoding="utf-8",
    )
    return {
        key: _display_path(path)
        for key, path in {
            "taxonomy": TAXONOMY_PATH,
            "engine_baseline_report": ENGINE_BASELINE_REPORT_PATH,
            "engine_baseline_summary": ENGINE_BASELINE_SUMMARY_PATH,
            "applicability_json": APPLICABILITY_JSON_PATH,
            "applicability_csv": APPLICABILITY_CSV_PATH,
            "workload_opportunity": WORKLOAD_OPPORTUNITY_PATH,
            "instrumentation_gap": INSTRUMENTATION_GAP_PATH,
            "one_factor_plan": ONE_FACTOR_PLAN_PATH,
            "scale_plan": SCALE_PLAN_PATH,
            "acceptance_gate": ACCEPTANCE_GATE_PATH,
            "scenario_registry": SCENARIO_REGISTRY_PATH,
            "ui_contract": UI_CONTRACT_PATH,
            "champion_framework": CHAMPION_FRAMEWORK_PATH,
            "doc": DOC_PATH,
            "summary_doc": SUMMARY_DOC_PATH,
        }.items()
    }
