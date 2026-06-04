# Retail + Finance Retrieval Recovery

Block 16A repairs the two retrieval paths that were blocking inference scaling:
Retail final top-5 ranking and Finance metadata materialization. This work is
retrieval-only. It does not run model inference, GPU experiments, paid API
calls, or SGLang.

## Scope

Inputs:

- `data/scaleup_2000_full/`
- `data/generated/context_engineering/corpora/`
- Qdrant-backed retrieval when the local index is available

Outputs:

- `data/generated/context_engineering/retail_failure_report.json`
- `data/generated/context_engineering/retail_failure_summary.csv`
- `data/generated/context_engineering/finance_metadata_flow_report.json`
- `data/generated/context_engineering/finance_metadata_flow_summary.csv`
- `data/generated/context_engineering/retail_finance_recovery_report.json`
- `data/generated/context_engineering/retail_finance_recovery_summary.csv`

## What Changed

Retail retrieval now includes:

- explicit Retail evidence-kind classification for review, summary, policy,
  metadata, and multicategory support rows
- product-title and category-aware scoring
- review issue and policy-aware scoring
- parent-child product/review balancing in the final top-5 selector
- top-50 candidate expansion followed by deterministic final selection

Finance recovery now includes:

- metadata-flow reporting for all 2,000 Finance prompts
- prompt-visible filing, metric, period, and section materialization checks
- non-leaking query materialization that blocks gold IDs and direct source IDs
- guarded expansion so broad Finance terms are not applied when no metric,
  period, or section is recoverable from the prompt

## Results

Incoming blocker baseline from the previous staged report:

| Vertical | Recall@5 | MRR |
| --- | ---: | ---: |
| Retail | 0.249000 | 0.245000 |
| Finance | 0.216000 | 0.119467 |

Block 16A staged 500-record result:

| Vertical | Candidate@20 | Candidate@50 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Retail | 0.991000 | 0.991000 | 0.293000 | 0.227933 |
| Finance | 0.454000 | 0.812000 | 0.266000 | 0.171500 |

Retail Recall@5 improved, but Retail MRR remains below the incoming MRR
baseline. Finance improved on both Recall@5 and MRR, but candidate Recall@20 is
still too low for inference scaling.

## Retail Root Cause

Retail failures are not mostly candidate-generation failures. On the 500-record
stage:

- failed Retail queries: 361
- candidate retrieval failures: 10, or 2.77%
- reranker failures: 351, or 97.23%
- product title mismatches: 0
- category mismatches: 0
- review issue mismatches: 10
- policy mismatches: 62

The dominant remaining blocker is exact final ranking among many near-duplicate
same-product support rows. In many failed cases, the correct generated evidence
row is in the top-50 candidate pool, but the prompt does not expose a safe
non-leaking discriminator that identifies that exact row.

## Finance Metadata Flow

The Finance metadata-flow report covers all 2,000 Finance prompts:

- filing materialized: 1,497
- metric materialized: 214
- period materialized: 20
- section materialized: 0
- direct hint leakage detected: 0

This confirms the earlier diagnosis: Finance prompts usually expose ticker,
company, and filing form, but rarely expose period, metric, or section. That
limits both candidate retrieval and final ranking.

## Leakage Policy

Strict modes still do not use:

- gold evidence IDs
- source IDs
- parent IDs
- document IDs
- filing IDs
- answer-side direct evidence hints

`prompt_plus_source_hints` remains an assisted upper bound and is not used for
the Block 16A recovery claims.

## Regeneration

```powershell
python scripts/phase3/recover_retail_finance_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/generated/context_engineering `
  --stage-sizes 250 500
```

## Remaining Blockers

Retail needs a safe final-rank discriminator for same-product generated support
rows. Without source IDs or gold IDs, many candidates are intentionally similar.

Finance needs better prompt-side or workload-side non-ID metadata before further
reranking. The current benchmark prompts often do not expose metric, period, or
section.

The next retrieval block should focus on safe non-ID workload metadata
materialization before running more inference-scale experiments.
