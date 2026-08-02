# Core Optimization Planning And Baseline Capability Audit

Status: planning audit completed; no inference executed.

## Scope

This document converts the broad optimization catalog into a measurable core
inference optimization program. It does not modify Main_Inference_V1, create
Optimized_Inference_V1, or claim a core optimization result.

## Audit Findings

- Catalog entries audited: 57
- Engine-native baseline capability entries:
  4
- Deployability repair IDs removed from the core list:
  enable_bounded_citation_repair, enable_escalation_path, improve_evidence_formatting, prompt_contract_repair, repair_retrieval, use_mm4_agentic_repair
- Engine support is not treated as confirmed activation.
- Main_Inference_V1 manifest records `optimization_flags: []`.

## Baseline Engine Capability State

The baseline used vLLM, SGLang, and an API provider route. vLLM and SGLang
therefore supplied engine-native serving behavior, but the repo does not contain
enough startup or engine telemetry to prove every cache, kernel, CUDA Graph, or
chunked-prefill state.

Unknown or unproven capability states:

- `vllm_attention_kernel_dispatch`
- `vllm_cuda_graph_state`
- `vllm_chunked_prefill_state`
- `vllm_prefix_cache_state`
- `sglang_radixattention_prefix_reuse`
- `sglang_paged_kv_storage`
- `sglang_cache_aware_scheduling`
- `sglang_attention_backend_selection`
- `sglang_cuda_graph_state`
- `sglang_chunked_prefill_state`

## Workload Opportunity

- Workload source scanned:
  `data/workloads/final_10000/prompt_plus_metadata/mm2_hybrid_top5.jsonl`
- Prefix audit counts token-identical leading tokens only.
- Measured cache hits: not available.
- KV block telemetry: not available.
- Speculation acceptance telemetry: not available.

The saved comparison suggests concurrency 32 increased latency and reduced
average tokens/sec versus concurrency 16 for both self-hosted engines, so the
first serving experiment should test scheduler/batch behavior carefully rather
than simply increasing concurrency.

## Selected First Candidates

- `prompt_prefix_layout_optimization`
- `scheduler_batch_tuning`

## Ranked One-Factor Program

1. `coreopt_prefix_layout_static_v1` changes `rendered_prompt_prefix_layout` only.
2. `coreopt_scheduler_batch_vllm_v1` changes `scheduler_batch_config` only.
3. `coreopt_prefix_cache_vllm_v1` changes `prefix_cache_flag` only.
4. `coreopt_chunked_prefill_sglang_v1` changes `chunked_prefill_config` only.

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

## Observability Follow-On

The instrumentation layer for those gaps is now implemented in
`docs/131_core_optimization_observability_framework.md`. It adds
`configs/core_optimization_observability.yaml`, an event schema, adapter
coverage, missing-instrumentation reports, UI observability cards, and
scenario-level readiness fields. The one-factor scenarios remain planned and
no optimization result is claimed.

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
