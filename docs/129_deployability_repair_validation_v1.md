# Deployability Repair Validation V1

Status: targeted sample validated on July 31, 2026

Artifact root:

```text
experiments/repairs/deployability_repair_validation_v1/
```

`Deployability_Repair_Validation_V1` validates the already implemented
deployability repair paths that sit between `Main_Inference_V1` and core
inference optimization. It is intentionally small, deterministic, and
artifact-backed.

It does not run live model inference, rent an A100, mutate
`Main_Inference_V1`, change gold data, change evaluator semantics, apply core
inference optimizations, or create `Optimized_Inference_V1`.

## Result

```text
SAMPLE_VALIDATED
```

The targeted validation covered 10 deterministic rows across all five
verticals and all repair families. The validation ran with CPU fallback because
`nvidia-smi` was unavailable locally. No GPU inference was executed.

## Scope

| Field | Value |
| --- | --- |
| Run ID | `deployability_repair_validation_v1` |
| Parent run | `main_inference_v1` |
| Sample count | 10 rows |
| Verticals | airline, healthcare_admin, retail, finance, research_ai |
| Live inference | false |
| A100 selected | false |
| Core optimization flags | none |
| Full-scale optimized run | not created |
| Backup verification | passed |
| Backup completeness score | 1.0 |

## Repair Families Validated

| Repair family | Representative behavior |
| --- | --- |
| `prompt_contract` | Converts malformed or incomplete output into the five-field generation contract. |
| `evidence_formatting` | Rewrites invalid evidence labels to the visible E1-E5 label contract. |
| `bounded_citation` | Adds missing visible citations without exposing canonical gold IDs. |
| `safety_wording` | Removes prohibited unsafe wording while preserving the structured response. |
| `escalation` | Routes insufficient, out-of-scope, and escalation cases instead of forcing answers. |
| `mm4_bounded` | Exercises the bounded LangGraph repair path and confirms tool/repair limits. |

## Validation Metrics

Source:

```text
experiments/repairs/deployability_repair_validation_v1/processed/deployability_repair_validation_v1_eval_summary.csv
```

| Metric | Value |
| --- | ---: |
| Row count | 10 |
| JSON validity | 1.0 |
| Generation contract validity | 1.0 |
| Format validity | 1.0 |
| Status behavior correctness | 1.0 |
| Safety violation count | 0 |
| Safety violation rate | 0.0 |
| Truncation count | 0 |
| Truncation rate | 0.0 |

Evidence match and groundedness are below 1.0 in the aggregate summary because
the sample deliberately includes non-answer rows such as insufficient evidence,
out-of-scope, escalation, and safety-boundary cases. The gate uses
`status_behavior_correct_rate`, not answer-only evidence metrics, as the
cross-status repair criterion.

## Held Constant

The validation held these controls constant:

- Main_Inference_V1 artifacts.
- Gold datasets.
- Evaluator semantics.
- SLO targets and profiles.
- Model aliases and runtime routes.
- Engine choices.
- Concurrency.
- Scheduler settings.
- Quantization.
- Prefix caching.
- Speculative decoding.
- Tensor parallelism.
- TensorRT-LLM.

Only the existing repair logic was exercised.

## UI State Change

The Main_Inference UI optimization payloads now distinguish three states:

- `NOT_MEASURED`: no repair validation artifacts exist.
- `SAMPLE_VALIDATED`: targeted deployability repair validation passed.
- `PASS`: full-scale optimized validation artifacts exist.

Current state:

```text
SAMPLE_VALIDATED
```

This means core inference optimization planning can begin. It does not mean the
system is deployable, and it does not mean `Optimized_Inference_V1` exists.

The subsequent planning audit is now saved in
`docs/130_core_optimization_planning_baseline_capability_audit.md` with
machine-readable outputs under `experiments/main/main_inference_v1/processed/`.
That audit keeps the validated repair layer separate from core serving
experiments such as prefix layout, scheduler/batch tuning, prefix-cache
verification, KV-cache tuning, chunked prefill, quantization, and future
TensorRT-LLM planning.

## Authoritative Artifacts

| Artifact | Purpose |
| --- | --- |
| `raw/deployability_repair_validation_v1_manifest.json` | Run identity, parent run, selected sample, config/workload hashes, and status. |
| `raw/deployability_repair_validation_v1_selected_sample.jsonl` | Deterministic 10-row validation sample. |
| `raw/deployability_repair_validation_v1_results.jsonl` | Repaired row outputs. |
| `raw/deployability_repair_validation_v1_repair_traces.jsonl` | Per-row repair decisions, prompts, and MM4 trace details. |
| `processed/deployability_repair_validation_v1_validation_gate_report.json` | Gate status and pass/fail checks. |
| `processed/deployability_repair_validation_v1_eval_report.json` | Deterministic evaluator output for the repaired sample. |
| `processed/deployability_repair_validation_v1_repair_effectiveness_report.json` | Repair-family effectiveness summary. |
| `processed/deployability_repair_validation_v1_core_optimization_handoff.json` | Handoff from deployability repair validation into core optimization planning. |
| `processed/deployability_repair_validation_v1_artifact_sync_report.json` | Local sync and backup verification result. |
| `checksums/SHA256SUMS.txt` | Checksums for generated validation artifacts. |

## Product Interpretation

The Optimization Lab can now tell the correct sequence:

```text
Main_Inference_V1
-> failed quality/safety SLOs
-> targeted deployability repair validation
-> core inference optimization planning
-> future Optimized_Inference_V1
```

Core optimization can be planned because the deployability repair sample passed.
The future optimized run must still be measured separately before the platform
can show a before/after improvement claim.
