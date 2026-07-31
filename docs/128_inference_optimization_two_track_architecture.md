# Inference Optimization Two-Track Architecture

Status: product architecture corrected on July 31, 2026

This document defines the product-facing optimization architecture for the
interactive AI Inference Engineering Platform. It is grounded in the saved
`Main_Inference_V1` artifacts and existing repository logic.

The platform is not a live GPU dashboard. It is a replay and reasoning layer
over measured artifacts. It must not imply that a UI click runs inference or
creates `Optimized_Inference_V1`.

## Audit Findings

The repository already had the core deterministic pieces:

- SLO diagnosis in `src/inference_bench/slo_diagnosis.py`
- bottleneck definitions in `configs/bottleneck_catalog.yaml`
- optimization definitions in `configs/optimization_catalog.yaml`
- negative rules in `configs/optimization_negative_rules.yaml`
- recommender logic in `src/inference_bench/optimization_recommender.py`
- Main_Inference UI adapter in
  `src/inference_bench/main_inference_optimization_ui.py`

The issue was product semantics. Mandatory deployability repairs and core
inference optimizations were presented too closely together as one optimization
recipe. That could make the product look like prompt repair, quality repair,
concurrency tuning, quantization, and engine tuning are all the same kind of
action.

They are not.

## Product Principle

Passing an SLO means the measured system cleared the configured minimum target.
It does not mean the system is optimally served.

`Main_Inference_V1` is the example:

- runtime SLO: `PASS`
- cost SLO: `PASS`
- quality SLO: `FAIL`
- safety SLO: `FAIL`
- deployability verdict: `NOT_DEPLOYABLE_SLO_FAILURES`

Therefore the next step is not to tune throughput first. The next step is to
repair deployability failures, validate those repairs in a measured run, and
only then plan core inference optimization.

## Two Tracks

### 1. Deployability Repairs

Deployability repairs are mandatory system-quality or workflow changes needed
because the measured result failed quality or safety SLOs.

Current repair-track IDs:

- `improve_evidence_formatting`
- `prompt_contract_repair`
- `use_mm4_agentic_repair`
- `enable_escalation_path`
- `enable_bounded_citation_repair`

These are shown as repair plans, not core inference optimizations. They can
change prompt contracts, evidence presentation, bounded repair behavior, or
escalation behavior. They must hold gold data, evaluator semantics, baseline
artifacts, and dataset split constant.

### 2. Core Inference Optimizations

Core optimizations improve serving, latency, throughput, memory, hardware
efficiency, or cost after the system is deployable enough to optimize.

Examples:

- prefix caching
- KV-cache tuning
- scheduler tuning
- continuous batching checks
- concurrency sweeps
- request queue tuning
- route long and short requests separately
- engine selection between vLLM and SGLang
- CUDA graph / attention kernel configuration
- speculative decoding
- quantization
- tensor parallelism
- hardware scaling
- prefill/decode disaggregation
- TensorRT-LLM as a planned future engine

These remain visible in the UI for education, but most are not selectable while
the deployability repair gate is not validated.

## Stage Gate

The product exposes these stages:

1. `MAIN_INFERENCE_MEASURED`
2. `DEPLOYABILITY_REPAIR_REQUIRED`
3. `DEPLOYABILITY_REPAIR_PLANNED`
4. `DEPLOYABILITY_REPAIR_VALIDATED`
5. `CORE_OPTIMIZATION_ELIGIBLE`
6. `CORE_OPTIMIZATION_PLANNED`
7. `OPTIMIZED_INFERENCE_READY`

The current repo state is:

```text
DEPLOYABILITY_REPAIR_PLANNED
```

The repair gate is:

```text
NOT_MEASURED
```

That means the product can explain the repair plan and core optimization
catalog, but it cannot claim `Optimized_Inference_V1` exists.

## UI Contracts

The Main_Inference product layer now writes these UI-ready files:

- `main_inference_v1_ui_deployability_repairs.json`
- `main_inference_v1_ui_repair_gate.json`
- `main_inference_v1_ui_core_optimization_catalog.json`
- `main_inference_v1_ui_core_optimization_applicability.json`
- `main_inference_v1_ui_experiment_stage.json`
- `main_inference_v1_ui_optimization_story.json`

The older compatibility files remain present:

- `main_inference_v1_ui_diagnosis.json`
- `main_inference_v1_ui_optimization_options.json`
- `main_inference_v1_ui_apply_plan.json`
- `main_inference_v1_ui_story.json`

## UI Interaction Contract

The Optimization Lab should tell this story:

1. User inspects failed SLOs.
2. UI shows target, observed value, severity, bottleneck, and evidence.
3. User opens the deployability repair track.
4. UI shows only repair actions that apply to the failed deployability SLOs.
5. User creates a repair plan.
6. UI explains this is plan-only and no inference is executed.
7. Repair gate remains `NOT_MEASURED` until measured repaired artifacts exist.
8. User opens the core optimization track.
9. UI shows every core technique as educational content.
10. UI disables core techniques that are locked by stage gates or negative
    rules.
11. After repair validation passes, the product can allow a controlled core
    optimization experiment plan.

There is no single combined "Apply All Optimizations" action at this stage.

## Negative Rule Semantics

Negative rules are mandatory filters. They prevent invalid recommendations
from appearing as selectable options.

Examples:

- Quantization is blocked while quality is already failing.
- Prefix caching is blocked without prefix-reuse and cache-hit telemetry.
- Speculative decoding is blocked without a draft model and acceptance-rate
  telemetry.
- Concurrency increase is blocked while quality has not passed.
- Tensor parallelism is blocked when the model fits and meets runtime on one
  GPU.
- Disaggregated prefill is blocked without queue/prefill/decode telemetry and
  runtime support.

The UI can still show blocked strategies as educational cards, but it must
explain why they are blocked.

## Backend API

New read-only endpoints:

- `/api/optimizations/deployability-repairs`
- `/api/optimizations/repair-gate`
- `/api/optimizations/core-catalog-v2`
- `/api/optimizations/core-applicability`
- `/api/optimizations/experiment-stage`
- `/api/optimizations/story`

The existing endpoints remain available for compatibility.

## Implementation Boundary

This architecture does not:

- execute inference;
- mutate Main_Inference results;
- fabricate optimized metrics;
- create `Optimized_Inference_V1`;
- authorize core optimization before repair validation.

The next measured work is a repair-validation run. Core inference optimization
comes after that gate passes.
