# Project State

Status as of June 15, 2026.

## Current Decision

```text
READY_FOR_SMALL_MODEL_SERVING_EXPERIMENTS
QUALITY_BLOCKED_FOR_SCALE
```

Blocks A1 through A6 validated the RTX 3070 vLLM/SGLang serving paths, GPU
telemetry, and bounded mm4 workflow.

Phase B1 loaded Qwen2.5-1.5B on vLLM and completed a balanced 100-prompt smoke
without OOM. The model fit in 8 GB VRAM with 6,534 MB peak sampled memory.

Phase B2 implemented modular SLO profiles, structured bottleneck and
optimization catalogs, failed-SLO-only diagnosis, deterministic compatibility
filtering, and one-factor next-experiment recommendations. It ran no new
inference. Existing A1, A2/A3, and A5/A6 metrics were diagnosed from local
artifacts.

Phase B3 audited all 65 failed B1 rows from existing artifacts. It found that
52 failures lacked at least one required gold evidence ID in the rendered E1-E5
context, while 18 failures had available evidence that the model did not cite.
No inference ran and no evaluator, gold, or promoted retrieval data changed.

## B1 Quality Gate

- JSON validity: 93%, required 95%
- Contract validity: 92%, required 85%
- Evidence match: 35%, required 60%
- Groundedness: 35%, required 60%
- Safety violations: 2, required 0

Result: `QUALITY_BLOCKED`.

The exact 50-prompt A1 overlap improved contract validity from 72% to 94%,
evidence match from 30% to 44%, and groundedness from 28% to 44%. The gain is
real but insufficient for scale.

## B2 Deterministic Diagnosis

- SLO profile: `default_enterprise`, backed by `configs/slo_targets.yaml`
- Bottleneck catalog: 51 IDs
- Optimization catalog: 57 IDs
- Existing diagnoses: 15 run/vertical slices across A1, A2/A3, and A5/A6
- Primary recommendation for all slices: `use_stronger_model`
- Secondary A1/A2 capacity action: bounded concurrency sweep
- Decision source: deterministic rules and YAML catalogs
- LLM calls: none

PagedAttention is represented as an active vLLM engine capability, not an
optimization toggle.

## B3 Quality Root Cause

- Failed B1 rows audited: 65
- Required gold absent from E1-E5: 52
- Evidence present but not cited: 18
- Partial multi-evidence citation: 27
- Invalid JSON / contract: 7 / 8
- Truncation: 6
- Finance failures with required evidence absent: 18 of 19
- Finance safety violations: 0

Finance is primarily a frozen workload/rendered-context alignment problem, with
a secondary citation-selection and truncation problem. This does not revise the
promoted retrieval source of truth.

## Next Step

Run `B3R1_FROZEN_WORKLOAD_CONTEXT_ALIGNMENT_REPAIR`: trace and re-export the
same 100 prompt contexts, require every expected evidence ID to map to E1-E5,
and rerun the offline audit. Only after that gate passes should a maximum
five-prompt Finance replay isolate model citation selection. Do not run a larger
benchmark, concurrency sweep, SGLang, mm4, or RunPod from the B1 decision.

See `docs/summaries/blockB1_vllm_1_5b_quality_smoke_summary.md` for the measured
result and comparison. See
`docs/99_modular_slo_diagnosis_and_optimization_catalog.md` for the B2 decision
architecture and `docs/100_generation_quality_root_cause_audit.md` for B3.
