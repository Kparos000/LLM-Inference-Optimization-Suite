# All-Vertical Retrieval SLO Repair

This block makes retrieval repair vertical-aware across all five benchmark
verticals. It does not run inference, GPU work, paid API calls, or change SLO
targets.

## Why Finance-Only Repair Was Not Enough

The production SLO framework showed 15 blocked retrieval metrics across Airline,
Retail, Healthcare Admin, Finance, and Research AI. Finance needed special
attention because prompt/gold ambiguity was severe, but inference scaling cannot
start from a Finance-only repair. Every vertical needs retrieval SLO coverage
before model-serving experiments can produce credible results.

## Repair Approach

The repair module is `src/inference_bench/vertical_retrieval_repair.py`.

It adds vertical-specific audit and query-enrichment profiles:

- Airline: policy type, travel issue, route region, escalation type.
- Healthcare Admin: admin task type, safety boundary, privacy sensitivity,
  policy type.
- Retail: product category, product title terms, review issue type, sentiment
  signal, policy type.
- Finance: company, ticker, metric family, period, fiscal year, fiscal quarter,
  filing type, section type, XBRL concept.
- Research AI: public paper label, topic, section type, method/result/limitation
  signals.

Gold IDs are used only for offline recall measurement. The query rewrite layer
scrubs direct evidence/source identifiers and reported zero direct-hint leakage
in staged validation.

## Staged Qdrant Validation

The full Finance Qdrant repair run had previously timed out, so this block uses
staged validation:

- 250 records per vertical
- 500 records per vertical
- 2,000 records per vertical is deferred until smaller stages are healthy

Run:

```powershell
python scripts/phase3/repair_all_vertical_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --slo-config configs/slo_targets.yaml `
  --output-root data/generated/context_engineering `
  --stage-sizes 250 500
```

Outputs:

- `data/generated/context_engineering/all_vertical_retrieval_repair_report.json`
- `data/generated/context_engineering/all_vertical_retrieval_repair_summary.csv`
- `data/generated/context_engineering/all_vertical_retrieval_repair_examples.jsonl`

## Baseline vs 500-Record Staged Repair

Baseline values are from the current canonical `final_10000`,
`prompt_plus_metadata`, `mm2_hybrid_top5` retrieval report. Staged repair values
are Qdrant-backed 500-record validation rows.

| Vertical | Baseline Cand@20 | Repair Cand@20 | Baseline Cand@50 | Repair Cand@50 | Baseline Recall@5 | Repair Recall@5 | Baseline MRR | Repair MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| airline | 0.701208 | 0.744000 | 0.761042 | 0.786333 | 0.706500 | 0.759000 | 0.936125 | 0.901467 |
| retail | 0.907083 | 0.955000 | 0.924667 | 0.986000 | 0.221583 | 0.249000 | 0.248417 | 0.245000 |
| healthcare_admin | 0.794083 | 0.840000 | 0.816333 | 0.868333 | 0.794333 | 0.840000 | 0.936125 | 0.949667 |
| finance | 0.489625 | 0.470000 | 0.820750 | 0.805000 | 0.280750 | 0.216000 | 0.174333 | 0.119467 |
| research_ai | 0.796851 | 1.000000 | 0.866486 | 1.000000 | 0.641715 | 0.989333 | 0.618225 | 1.000000 |

## SLO Result by Vertical

At the 500-record staged validation:

- Research AI passed retrieval SLOs.
- Retail passed candidate-recall targets but still failed final Recall@5 and
  MRR, so the primary blocker is final top-5 selection.
- Airline and Healthcare Admin improved but still failed candidate recall.
- Finance remained blocked, confirming that prompt-visible non-ID metadata does
  not yet reproduce the earlier gold-linked local fallback repair result.

## Remaining Blockers

Current official SLO readiness remains `BLOCKED` because the canonical
retrieval report still fails 15 retrieval metrics. Staged repair is diagnostic
and should not replace the canonical report until a full regenerated retrieval
evaluation passes.

Recommended next actions:

- Airline: improve policy issue extraction and policy-section metadata in
  candidate retrieval.
- Healthcare Admin: improve admin-issue normalization and candidate recall.
- Retail: keep candidate repair, then tune final top-5 selection and parent-child
  reranking.
- Finance: materialize non-ID period/metric/filing metadata into workload inputs
  before another Qdrant run.
- Research AI: validate the 2,000-record stage because the 500-record stage
  passed.

The project is still not inference-ready. Retrieval SLOs remain the blocking
gate before Phase 4/5 scaling.

