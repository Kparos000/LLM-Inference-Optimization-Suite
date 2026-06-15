# Project State

Status as of June 15, 2026.

## Current Decision

```text
READY_FOR_SMALL_MODEL_SERVING_EXPERIMENTS
QUALITY_READY_FOR_FROZEN_100
CONTROLLED_SCALE_GATE_PENDING
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

Phase B4 executed the frozen context-alignment repair and reran the exact
100-prompt Qwen2.5-1.5B vLLM matrix. All required gold evidence now maps to
E1-E5, including Finance 20/20, but the quality gate remains blocked because
the two Airline safety violations persisted.

Phase B5 repaired the safety wording and multi-evidence citation-selection
path on the same frozen matrix. The targeted 25 failed-row replay passed the
B5 gate and triggered a full frozen 100 rerun. The full rerun reached 99% JSON
and contract validity, 96% evidence match and groundedness, and zero safety
violations.

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

## B4 Context Alignment And Quality Repair

- Context-aligned runner rows: 100
- All required gold evidence present in E1-E5: 100/100
- Finance all-required-gold-present rate: 20/20, up from 2/20 in B1
- JSON validity: 97%, up from 93%
- Contract validity: 97%, up from 92%
- Evidence match: 76%, up from 35%
- Groundedness: 76%, up from 35%
- Safety violations: 2, unchanged
- Truncation: 3%, down from 6%
- Mean TTFT: 137.966 ms
- Mean TPOT: 11.280 ms
- Mean E2E latency: 1,701.776 ms
- Mean/peak GPU utilization: 83.02% / 100%
- Peak sampled GPU memory: 6,602 MB

Result: `QUALITY_BLOCKED`.

The post-B4 audit found 25 failed rows. Zero failed rows lacked required gold
evidence in E1-E5. Evidence was present but not cited in 24 failed rows, and
all 25 failed rows were classified as model instruction-following failures.
Finance is now primarily a model citation-selection problem, not a
retrieval/context availability problem or safety problem.

## B5 Final Generation Quality Hardening

- Targeted B4 failed rows replayed: 25
- Targeted JSON validity: 100%
- Targeted contract validity: 100%
- Targeted evidence match: 92%
- Targeted groundedness: 92%
- Targeted safety violations: 0
- Targeted truncation: 0%
- Lexical-guard repairs: 2
- Missing-label retry triggers: 4
- Full frozen 100 rerun triggered: yes
- Full JSON validity: 99%
- Full contract validity: 99%
- Full evidence match: 96%
- Full groundedness: 96%
- Full safety violations: 0
- Full truncation: 1%
- Mean full-run TTFT: 142.102 ms
- Mean full-run TPOT: 10.718 ms
- Mean full-run E2E latency: 1,473.156 ms

Result: `QUALITY_READY_FOR_FROZEN_100`.

Residual full-run failures remain: one Airline citation miss, two Finance
citation misses, and one Research AI truncated JSON output. The B5 result is a
frozen 100-prompt gate, not a final scale benchmark or concurrency claim.

## Next Step

Run `B6_CONTROLLED_SCALE_AND_CONCURRENCY_GATE`: first run a controlled
500-prompt quality gate at concurrency one. Only run concurrency 2/4 if the
500-prompt gate maintains evidence and groundedness above target and safety
violations remain zero. Do not run a larger 2,000 or 10,000 record benchmark
until that gate passes.

See `docs/summaries/blockB1_vllm_1_5b_quality_smoke_summary.md` for the measured
result and comparison. See
`docs/99_modular_slo_diagnosis_and_optimization_catalog.md` for the B2 decision
architecture, `docs/100_generation_quality_root_cause_audit.md` for B3,
`docs/101_context_alignment_and_generation_quality_repair.md` for B4, and
`docs/102_final_generation_quality_hardening.md` for B5.
