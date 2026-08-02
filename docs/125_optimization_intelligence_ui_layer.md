# Optimization Intelligence UI Layer

Status: implemented for `Main_Inference_V1` as a saved-artifact reasoning
layer.

This document describes the UI-facing optimization intelligence layer that
turns the completed Main_Inference run into an explainable product workflow.
It does not run inference, mutate benchmark results, or create
`Optimized_Inference_V1`.

Update: the product-facing layer now uses the two-track architecture described
in `docs/128_inference_optimization_two_track_architecture.md`.
Deployability repairs and core inference optimizations are separate stages.
The targeted repair gate is currently `SAMPLE_VALIDATED`, so core strategies
are now eligible for planning. Full deployability still requires measured
`Optimized_Inference_V1` artifacts.

Core optimization planning is now grounded by
`docs/130_core_optimization_planning_baseline_capability_audit.md` and the
generated planning artifacts:

- `configs/core_optimization_taxonomy.yaml`
- `configs/core_optimization_scenario_registry.yaml`
- `experiments/main/main_inference_v1/processed/main_inference_v1_engine_baseline_capability_report.json`
- `experiments/main/main_inference_v1/processed/core_optimization_applicability_matrix.json`
- `experiments/main/main_inference_v1/processed/core_optimization_one_factor_experiment_plan.json`
- `experiments/main/main_inference_v1/processed/core_optimization_ui_contract.json`

These files keep engine-native baseline behavior, planned one-factor
experiments, and missing optimized results in separate UI states.

Core optimization observability is now grounded by
`docs/131_core_optimization_observability_framework.md` and:

- `configs/core_optimization_observability.yaml`
- `experiments/main/main_inference_v1/processed/core_optimization_observability_registry.json`
- `experiments/main/main_inference_v1/processed/core_optimization_observability_readiness.json`
- `experiments/main/main_inference_v1/processed/core_optimization_event_schema.json`
- `experiments/main/main_inference_v1/processed/core_optimization_ui_observability_contract.json`
- `experiments/main/main_inference_v1/processed/coreopt_prefix_layout_static_v1_prefix_opportunity_analysis.json`

This layer answers what must be instrumented before the platform can claim an
optimization worked. It does not apply a strategy or create optimized results.

## Purpose

The final platform needs more than a static benchmark dashboard. When a user
clicks a failed SLO, the platform should explain:

- which metric failed;
- what target was missed;
- what bottleneck the repository logic diagnosed;
- which optimizations are compatible;
- why each optimization applies;
- which optimizations were rejected and why.

The UI must never show a selectable optimization that is irrelevant to the
selected failed SLO or blocked by a negative rule.

## Implementation

The implementation is:

```text
src/inference_bench/main_inference_optimization_ui.py
scripts/phase4/build_main_inference_optimization_ui.py
tests/test_main_inference_optimization_ui.py
```

It wraps existing repository logic:

- `src/inference_bench/slo_diagnosis.py`
- `src/inference_bench/optimization_recommender.py`
- `src/inference_bench/bottleneck_catalog.py`
- `src/inference_bench/optimization_catalog.py`
- `src/inference_bench/optimization_negative_rules.py`

It consumes only saved Main_Inference artifacts:

- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_context_preflight_report.json`
- `experiments/main/main_inference_v1/raw/main_inference_v1_manifest.json`

It also reads the targeted repair-validation gate when present:

- `experiments/repairs/deployability_repair_validation_v1/processed/deployability_repair_validation_v1_validation_gate_report.json`

## Generated UI Artifacts

The script writes:

```text
experiments/main/main_inference_v1/processed/main_inference_v1_ui_diagnosis.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_optimization_options.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_apply_plan.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_story.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_deployability_repairs.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_repair_gate.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_core_optimization_catalog.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_core_optimization_applicability.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_experiment_stage.json
experiments/main/main_inference_v1/processed/main_inference_v1_ui_optimization_story.json
```

Each artifact includes:

- `inference_executed: false`;
- `llm_used: false` where applicable;
- source artifact references;
- deterministic catalog-backed explanations.

## User Flow

```text
Click failed SLO
-> show target, observed value, severity, and bottleneck
-> show deployability repair options for failed quality/safety SLOs
-> show explanation, tradeoffs, risk, and implementation status
-> optional rejected-option audit
-> repair action produces a plan only
-> targeted repair gate is SAMPLE_VALIDATED after deterministic sample validation
-> core optimization catalog becomes eligible for planning, while negative
   rules still block invalid strategies
-> observability cards explain problem, mechanism, instrumentation, one-factor
   experiment, result, and decision requirements
-> saved optimized artifacts can be replayed later when they exist
```

## Current Main_Inference Diagnosis

The UI diagnosis exposes five failed SLO rows:

| Failed SLO | Bottleneck |
| --- | --- |
| Contract validity | `low_contract_validity` |
| Format validity | `low_contract_validity` |
| Evidence match | `low_evidence_match` |
| Groundedness | `low_groundedness` |
| Safety findings | `safety_violations` |

Runtime and cost passed, so latency, throughput, GPU-utilization, and cost
optimizations are not required deployability repairs for this run. They remain
visible in the core catalog because passing an SLO means minimum acceptance,
not optimal serving.

## Negative-Rule Filtering

The layer applies `configs/optimization_negative_rules.yaml` before an option
can appear in a failed-SLO dropdown.

Examples:

- Stronger-model escalation is rejected for this run because prompt-contract
  failures and context preflight gaps still exist.
- Quantization is rejected because quality is already failing and quantized
  runtime support is not a measured fix for the failed quality SLOs.
- Concurrency increase is not shown because quality failed and runtime did not
  fail.
- Prefix caching is not shown because prefix reuse and cache-hit telemetry were
  not diagnosed as failed SLO evidence.

Rejected optimizations are retained in the JSON for audit panels, but the UI
should not present them as selectable dropdown options.

## Apply Plan Semantics

`main_inference_v1_ui_apply_plan.json` is retained for compatibility and is
plan-only. The corrected product flow should use the repair plan and repair
gate contracts first, then create a separate core optimization experiment plan
only after targeted repair validation passes.

It explicitly does not:

- execute inference;
- modify Main_Inference artifacts;
- create `Optimized_Inference_V1`;
- weaken evaluator semantics;
- modify gold data.

The plan is suitable for a product demo because users can inspect the
reasoning and later replay saved optimized artifacts without needing GPU
access.

## Command

```powershell
python scripts/phase4/build_main_inference_optimization_ui.py
```

## Product Contract

Frontend behavior:

- Load `main_inference_v1_ui_diagnosis.json` for failed SLO rows.
- Load `main_inference_v1_ui_optimization_options.json` for dropdown options.
- Use `options_by_failed_slo[slo_id]` as the only selectable option source.
- Show `rejected_optimizations_by_failed_slo[slo_id]` only in an audit drawer.
- Load `main_inference_v1_ui_apply_plan.json` when the user clicks apply.
- Load `main_inference_v1_ui_story.json` for the guided explanation page.
- Load `main_inference_v1_ui_repair_gate.json` to distinguish
  `SAMPLE_VALIDATED` from full-scale optimized `PASS`.
- Load `main_inference_v1_ui_core_optimization_applicability.json` for
  post-repair core strategy eligibility.
- Load `core_optimization_ui_contract.json` and
  `core_optimization_scenario_registry.yaml` before rendering planned core
  experiments or optimized-result placeholders.

Backend behavior:

- Serve these JSON files as read-only artifacts.
- Do not expose endpoints that start inference from this product layer.
- Keep optimized-run replay separate from plan generation.
