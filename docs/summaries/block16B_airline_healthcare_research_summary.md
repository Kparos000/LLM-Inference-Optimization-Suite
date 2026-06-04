# Block 16B Airline Healthcare Research Summary

Summary file path:
`docs/summaries/block16B_airline_healthcare_research_summary.md`

## Files Changed

- `src/inference_bench/retrieval.py`
- `src/inference_bench/vertical_retrieval_repair.py`
- `src/inference_bench/airline_healthcare_research_validation.py`
- `scripts/phase3/enrich_airline_healthcare_validate_research.py`
- `tests/test_phase3_airline_healthcare_research_validation.py`
- `docs/76_airline_healthcare_research_validation.md`
- `data/generated/context_engineering/airline_healthcare_enrichment_report.json`
- `data/generated/context_engineering/airline_healthcare_enrichment_summary.csv`
- `data/generated/context_engineering/research_ai_scale_validation_report.json`
- `data/generated/context_engineering/research_ai_scale_validation_summary.csv`
- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`

## Retrieval Logic Changed

- Added Airline policy synonym expansion for booking, refund, cancellation,
  disruption, baggage, escalation, partner/codeshare, and travel-documentation
  language.
- Added Healthcare Admin synonym expansion for appointments, billing, identity
  verification, referrals, privacy, safety boundaries, and admin procedures.
- Added Airline/Healthcare metadata feature extraction from context title, tags,
  document type, and boundary metadata.
- Added Airline primary-family top-5 selection so primary policies stay visible
  while related support policies are covered.
- Kept direct source/gold identifier leakage guards enabled.

## Airline Before vs After

500-record previous baseline:

| Metric | Before |
| --- | ---: |
| Candidate Recall@20 | 0.744000 |
| Candidate Recall@50 | 0.786333 |
| Recall@5 | 0.759000 |
| MRR | 0.901467 |

500-record Block 16B after enrichment:

| Metric | After |
| --- | ---: |
| Candidate Recall@20 | 0.954000 |
| Candidate Recall@50 | 0.985000 |
| Recall@5 | 0.915000 |
| MRR | 0.820533 |

Airline Recall@5 now passes the staged 0.90 target. MRR remains below target.

## Healthcare Before vs After

500-record previous baseline:

| Metric | Before |
| --- | ---: |
| Candidate Recall@20 | 0.840000 |
| Candidate Recall@50 | 0.868333 |
| Recall@5 | 0.840000 |
| MRR | 0.949667 |

500-record current query path:

| Metric | Current Query |
| --- | ---: |
| Candidate Recall@20 | 0.964667 |
| Candidate Recall@50 | 0.994000 |
| Recall@5 | 0.979667 |
| MRR | 0.870000 |

500-record aggressive enrichment:

| Metric | After Enrichment |
| --- | ---: |
| Candidate Recall@20 | 1.000000 |
| Candidate Recall@50 | 1.000000 |
| Recall@5 | 0.946000 |
| MRR | 0.790733 |

Healthcare Recall@5 passes. The current query path is preferred over aggressive
query rewriting because it preserves better rank ordering.

## Research AI 500 vs 2000

| Stage | Candidate Recall@20 | Candidate Recall@50 | Recall@5 | MRR |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 1.000000 | 1.000000 | 0.989333 | 1.000000 |
| 2,000 | 0.913807 | 0.939707 | 0.776176 | 0.747267 |

Research AI shows scale drift at 2,000 records. Candidate degradation is true,
so the next repair should target candidate recall before reranking.

## SLO Status

The regenerated canonical SLO readiness report remains `BLOCKED`.

Passing in staged Block 16B:

- Airline Recall@5
- Healthcare Recall@5

Still failing or blocking:

- Airline MRR/rank ordering
- Healthcare aggressive-enrichment MRR/rank ordering
- Research AI 2,000-record candidate recall and Recall@5
- canonical full-report SLO gate remains blocked

## Commands Run

```powershell
pytest tests/test_phase3_airline_healthcare_research_validation.py
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

Full verification commands were run after implementation; see the final Codex
report for exact pass/fail status.

Commit hash after push: pending until commit is created.
