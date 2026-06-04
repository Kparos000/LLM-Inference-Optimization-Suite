# Block 16A Retail + Finance Retrieval Recovery Summary

## Files Changed

- `src/inference_bench/retrieval.py`
- `src/inference_bench/vertical_retrieval_repair.py`
- `src/inference_bench/retail_finance_recovery.py`
- `scripts/phase3/recover_retail_finance_retrieval.py`
- `tests/test_phase3_retail_finance_recovery.py`
- `docs/75_retail_finance_retrieval_recovery.md`
- `docs/summaries/block16A_retail_finance_recovery_summary.md`
- `data/generated/context_engineering/retail_failure_report.json`
- `data/generated/context_engineering/retail_failure_summary.csv`
- `data/generated/context_engineering/finance_metadata_flow_report.json`
- `data/generated/context_engineering/finance_metadata_flow_summary.csv`
- `data/generated/context_engineering/retail_finance_recovery_report.json`
- `data/generated/context_engineering/retail_finance_recovery_summary.csv`

## Root Causes

Retail was primarily a final top-5 ranking problem, not a candidate retrieval
problem. In the 500-record staged report, only 2.77% of failed Retail queries
were candidate failures; 97.23% were reranker/final-selector failures.

Finance was primarily a metadata materialization problem. Across 2,000 Finance
prompts, filing context was often recoverable, but metric, period, and section
signals were sparse.

## Retrieval Changes

Retail:

- added Retail evidence-kind classification
- added product-title, category, review-issue, and policy-aware scoring
- added parent-child Retail final selection
- preserved strict no-gold/no-source-ID leakage guards

Finance:

- added metadata-flow reporting
- added non-ID filing/metric/period materialization checks
- guarded broad query expansion when prompts lack metric, period, or section

## Before vs After

Incoming Block 15 blocker baseline:

| Vertical | Recall@5 | MRR |
| --- | ---: | ---: |
| Retail | 0.249000 | 0.245000 |
| Finance | 0.216000 | 0.119467 |

Block 16A 500-record staged result:

| Vertical | Candidate@20 | Candidate@50 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Retail | 0.991000 | 0.991000 | 0.293000 | 0.227933 |
| Finance | 0.454000 | 0.812000 | 0.266000 | 0.171500 |

Retail Recall@5 improved, but MRR remains below the incoming baseline. Finance
improved on both Recall@5 and MRR, but candidate Recall@20 remains too low.

## Retail Failure Decomposition

500-record stage:

- failed queries: 361
- candidate failures: 10
- reranker failures: 351
- metadata failures: 72
- product title mismatches: 0
- category mismatches: 0
- review issue mismatches: 10
- policy mismatches: 62

## Finance Metadata Flow

2,000 Finance prompts:

- filing materialized: 1,497
- metric materialized: 214
- period materialized: 20
- section materialized: 0
- direct hint leakage detected: 0

## Remaining Gaps

Retail still needs a safe non-ID discriminator for near-duplicate same-product
support rows. Finance still needs non-ID metric, period, and section metadata
materialized before additional reranking can be expected to help.

## Commands Run

```powershell
pytest tests/test_phase3_retail_finance_recovery.py
pytest tests/test_phase3_retail_finance_recovery.py tests/test_phase3_strict_retrieval_upgrade.py
python scripts/phase3/recover_retail_finance_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/generated/context_engineering `
  --stage-sizes 250 500
```

Full verification commands are recorded in the final response for this block.

Commit hash after push: see final response.
