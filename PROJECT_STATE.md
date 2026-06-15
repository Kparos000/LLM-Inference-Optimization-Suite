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

## Next Step

Repair and isolate grounded-output quality before increasing prompt count or
concurrency. Finance evidence selection, 128-token truncation, and prohibited
phrase emission are the current diagnostic priorities. Do not run a larger
benchmark, SGLang, mm4, or RunPod from the B1 decision.

See `docs/summaries/blockB1_vllm_1_5b_quality_smoke_summary.md` for the measured
result and comparison. See
`docs/99_modular_slo_diagnosis_and_optimization_catalog.md` for the B2 decision
architecture.
