# Airline + Healthcare Retrieval Enrichment and Research AI Scale Validation

Block 16B is a retrieval-only repair and validation pass. It does not run model
inference, GPU work, external APIs, or source/gold-ID assisted retrieval.

## What Changed

Airline retrieval now has domain-specific query expansion and scoring for:

- booking, refund, cancellation, ticket-change, and travel-credit language
- delay, disruption, rebooking, weather, schedule-change, and compensation language
- baggage delay/damage terminology
- partner-airline, codeshare, operating-carrier, and escalation terminology
- visa, passport, documentation, and international-entry terminology
- a constrained top-5 selector that places the likely primary policy first and
  then covers related support policies when visible query terms justify it

Healthcare Admin retrieval now has query expansion and scoring for:

- appointment booking, rescheduling, and cancellation terminology
- billing, payment, insurance, authorization, referral, and records terminology
- identity-verification, privacy, proxy-access, and safety-boundary signals
- admin procedure and boundary metadata extracted from context records

Research AI was not changed in this block. It was validated at 500 and 2,000
records using Qdrant-backed hybrid retrieval and the current reranker.

## Leakage Guard

The generated Block 16B reports show:

- gold IDs used as query terms: false
- source IDs used as query terms: false
- direct hint leakage detected: 0

The enrichment uses prompt-visible fields such as `support_type`, `route`,
`travel_type`, `department`, `expected_queue`, and `safety_boundary`, plus
corpus-side public metadata such as tags, document type, title, and boundary
flags.

## Airline Results

Baseline from the previous all-vertical repair report at 500 records:

| Metric | Before |
| --- | ---: |
| Candidate Recall@20 | 0.744000 |
| Candidate Recall@50 | 0.786333 |
| Recall@5 | 0.759000 |
| MRR | 0.901467 |

Block 16B after enrichment at 500 records:

| Metric | After |
| --- | ---: |
| Candidate Recall@20 | 0.954000 |
| Candidate Recall@50 | 0.985000 |
| Recall@5 | 0.915000 |
| MRR | 0.820533 |

Airline now passes the Recall@5 target on the staged 500-record validation, but
MRR remains below the SLO target. The remaining blocker is rank ordering: the
right evidence is usually in the candidate set and top five, but primary vs.
supporting policy order still needs tuning.

## Healthcare Results

Baseline from the previous all-vertical repair report at 500 records:

| Metric | Before |
| --- | ---: |
| Candidate Recall@20 | 0.840000 |
| Candidate Recall@50 | 0.868333 |
| Recall@5 | 0.840000 |
| MRR | 0.949667 |

Block 16B current query path at 500 records:

| Metric | Current Query |
| --- | ---: |
| Candidate Recall@20 | 0.964667 |
| Candidate Recall@50 | 0.994000 |
| Recall@5 | 0.979667 |
| MRR | 0.870000 |

Block 16B aggressive enrichment at 500 records:

| Metric | After Enrichment |
| --- | ---: |
| Candidate Recall@20 | 1.000000 |
| Candidate Recall@50 | 1.000000 |
| Recall@5 | 0.946000 |
| MRR | 0.790733 |

Healthcare now passes the Recall@5 target in both current-query and enriched
paths. The current-query path is stronger overall because aggressive privacy and
identity enrichment can over-rank supporting boundary documents. The recommended
path for Phase 4 is to keep Healthcare on the current enriched retrieval scorer,
not the extra aggressive query rewrite, until rank calibration is improved.

## Research AI Scale Validation

| Stage | Candidate Recall@20 | Candidate Recall@50 | Recall@5 | MRR |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 1.000000 | 1.000000 | 0.989333 | 1.000000 |
| 2,000 | 0.913807 | 0.939707 | 0.776176 | 0.747267 |

Research AI does not remain stable at 2,000 records. The report marks candidate
degradation as true and reranking degradation as false because the main drop is
already visible at candidate Recall@20/50. The next repair should focus on
candidate recall for the larger Research AI prompt distribution before using it
for inference scaling claims.

## SLO Status

The regenerated canonical SLO readiness report remains `BLOCKED`.

Reasons:

- the official SLO evaluator still reads the canonical full retrieval report
- Block 16B staged reports are diagnostic/repair outputs, not the canonical final
  retrieval report
- Airline staged Recall@5 now passes, but MRR is still below target
- Healthcare staged Recall@5 passes, but aggressive enrichment hurts MRR
- Research AI 2,000-record validation shows scale drift
- Retail and Finance remain part of the broader canonical retrieval gate

## Regeneration Commands

```powershell
python scripts/phase3/enrich_airline_healthcare_validate_research.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/generated/context_engineering `
  --stage-sizes 250 500 `
  --research-stage-sizes 500 2000

python scripts/phase3/evaluate_slo_readiness.py `
  --slo-config configs/slo_targets.yaml `
  --retrieval-report data/generated/context_engineering/retrieval_evaluation_report.json `
  --quality-gate-report data/generated/context_engineering/retrieval_quality_gate_report.json `
  --output-root data/generated/context_engineering
```

## Generated Reports

- `data/generated/context_engineering/airline_healthcare_enrichment_report.json`
- `data/generated/context_engineering/airline_healthcare_enrichment_summary.csv`
- `data/generated/context_engineering/research_ai_scale_validation_report.json`
- `data/generated/context_engineering/research_ai_scale_validation_summary.csv`
- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`

## Recommended Next Step

Do not start inference scaling from the full 10,000-record workload yet. The next
retrieval block should repair Research AI candidate recall at 2,000 records and
calibrate Airline/Healthcare rank ordering so Recall@5 and MRR pass together.
