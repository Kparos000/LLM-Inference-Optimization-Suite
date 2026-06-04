# Phase 4 Handoff and Retrieval Promotion

Date: 2026-06-04

This document freezes the Phase 3 retrieval baseline for future Phase 4
execution work. It does not describe a new retrieval experiment, model
inference run, GPU run, or API call.

## Promotion Decision

The repaired 2,000-record retrieval validation is now the canonical retrieval
source of truth.

Canonical files:

- `data/generated/context_engineering/retrieval_source_of_truth_manifest.json`
- `data/generated/context_engineering/retrieval_promotion_registry.json`
- `data/generated/context_engineering/repaired_retrieval_validation_report.json`
- `data/generated/context_engineering/repaired_retrieval_validation_summary.csv`
- `data/generated/context_engineering/repaired_retrieval_promotion_plan.json`

The legacy `retrieval_evaluation_report.json` remains useful historical output,
but it is no longer the source for Phase 4 retrieval readiness.

## Final Retrieval Metrics

The active baseline is:

- Dataset variant: `repaired_generated`
- Stage size: `2000`
- Ablation mode: `prompt_plus_metadata`
- Dense backend: `qdrant_vector`
- Vector store: `qdrant_local`
- Measurement: `retrieval_dataset_alignment`

| Vertical | Candidate@20 | Candidate@50 | Recall@5 | MRR | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Airline | 1.000000 | 1.000000 | 1.000000 | 1.000000 | PASS |
| Healthcare Admin | 1.000000 | 1.000000 | 1.000000 | 0.994250 | PASS |
| Retail | 0.974333 | 0.982083 | 0.959917 | 0.922592 | PASS |
| Finance | 0.948875 | 0.955750 | 0.939000 | 0.941833 | PASS |
| Research AI | 0.975172 | 0.979826 | 0.917460 | 0.953233 | PASS |

All five verticals pass the retrieval SLO targets:

- Candidate Recall@20 >= 0.90
- Candidate Recall@50 >= 0.95
- Final Recall@5 >= 0.90
- MRR >= 0.85, with Finance and Research AI using the stricter 0.90 target in
  `configs/slo_targets.yaml`

## Validation Status

Qdrant validation: pass.

All active rows use `qdrant_vector` with `qdrant_local`.

Compression validation: pass.

The final 10,000-record prompt-plus-metadata compression diagnostics retain
retrieval recall while reducing context tokens by more than 20 percent for every
vertical.

Leakage validation: pass.

The promoted baseline uses `prompt_plus_metadata`, not direct gold IDs or
source IDs. Source-hint-assisted retrieval remains documented as an assisted
upper bound, not as the final claim baseline.

## SLO Reporting Change

`scripts/phase3/evaluate_slo_readiness.py` now reads the promoted retrieval
source-of-truth manifest by default:

```powershell
python scripts/phase3/evaluate_slo_readiness.py
```

Expected retrieval state:

- Retrieval readiness: PASS
- Retrieval blocked count: `0`
- Overall status: `READY_WITH_GAPS`

The remaining gaps are expected because latency, quality, throughput, hardware,
and cost metrics require Phase 4 and Phase 5 inference runs.

## Phase 4 Starting Point

Phase 4 should start from the promoted retrieval baseline and existing runner
adapter path:

1. Export a small `smoke_500` workload from the promoted workload files.
2. Run mock runner plumbing validation.
3. Run local Hugging Face smoke validation with `model1_0_5b`.
4. Run OpenAI-compatible dry-run validation, then a tiny vLLM server smoke when
   the server is available.
5. Evaluate generation outputs by joining on `prompt_id`.

Phase 4 should not use the legacy retrieval evaluation report for final
retrieval claims.

## Remaining Gaps Before Inference Scaling

Retrieval blockers: none.

Still unmeasured:

- Generated-answer quality
- Groundedness and citation quality after model generation
- TTFT, TPOT, ITL, and end-to-end latency
- Throughput at concurrency
- GPU utilization, memory, power, and temperature telemetry
- Self-hosted GPU cost and API-priced token cost

These gaps are expected and should be addressed by Phase 4 and Phase 5
execution, not by more retrieval-only repair.
