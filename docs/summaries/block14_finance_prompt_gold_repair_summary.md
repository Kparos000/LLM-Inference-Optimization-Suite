# Block 14 Finance Prompt/Gold Repair Summary

## Files Changed

- `src/inference_bench/finance_retrieval_repair.py`
- `scripts/phase3/repair_finance_retrieval.py`
- `tests/test_phase3_finance_retrieval_repair.py`
- `docs/73_finance_prompt_gold_repair.md`
- `docs/summaries/block14_finance_prompt_gold_repair_summary.md`
- `data/generated/context_engineering/finance_prompt_quality_report.json`
- `data/generated/context_engineering/finance_prompt_quality_summary.csv`
- `data/generated/context_engineering/finance_gold_quality_report.json`
- `data/generated/context_engineering/finance_gold_quality_summary.csv`
- `data/generated/context_engineering/finance_metadata_enrichment_report.json`
- `data/generated/context_engineering/finance_retrieval_repair_impact_report.json`
- `data/generated/context_engineering/finance_retrieval_repair_impact_summary.csv`
- `data/generated/context_engineering/finance_retrieval_repair_report.json`

## Root Causes Found

Finance prompts expose company/ticker reliably, but not the retrieval cues needed
for strict matching:

- prompts missing period: 2,000
- prompts missing metric: 1,835
- prompts missing filing type: 484
- prompts missing section: 1,988

Gold records also lack period text:

- gold records missing explicit period: 2,000
- gold records missing explicit metric: 1,174
- gold records missing explicit filing type: 503
- gold records missing explicit section: 1,683

## Metadata Enriched

The enrichment layer derived:

- ticker: 2,000 prompts
- company: 2,000 prompts
- filing type: 1,516 prompts
- filing section: 1,127 prompts
- period: 1,147 prompts
- fiscal year: 1,127 prompts
- XBRL concept: 873 prompts
- metric family: 2,000 prompts

The rewrite pipeline scrubs direct evidence IDs and adds only human-readable
period, metric, filing, section, company, ticker, and XBRL concept terms.

## Retrieval Impact

The full 2,000-prompt Finance repair measurement was run with
`dense_backend=local_fallback` and is labeled that way in the reports.

| Ablation | Before Final Recall@5 | After Final Recall@5 | Before MRR | After MRR |
| --- | ---: | ---: | ---: | ---: |
| prompt_text_only | 0.162500 | 0.647125 | 0.091292 | 0.454942 |
| prompt_plus_metadata | 0.288625 | 0.629500 | 0.190917 | 0.442133 |

Candidate recall improved as well:

| Ablation | Before Recall@20 | After Recall@20 | Before Recall@50 | After Recall@50 |
| --- | ---: | ---: | ---: | ---: |
| prompt_text_only | 0.304000 | 0.824625 | 0.566500 | 0.854500 |
| prompt_plus_metadata | 0.495625 | 0.800000 | 0.794375 | 0.840750 |

## Remaining Gaps

- The repaired score is a feasibility measurement, not a final paper claim.
- The Qdrant-backed full repair measurement needs batch-warmed query embeddings
  before rerunning all Finance prompts.
- Period metadata is only recoverable for 1,147 prompts, so the Finance workload
  needs explicit non-ID period fields before final strict retrieval claims.
- Once non-ID metadata is materialized, reranking/final selection should be tuned
  against Qdrant strict ablation results.

## Commands Run

```powershell
pytest tests/test_phase3_finance_retrieval_repair.py
mypy src/inference_bench/finance_retrieval_repair.py tests/test_phase3_finance_retrieval_repair.py
ruff check src/inference_bench/finance_retrieval_repair.py scripts/phase3/repair_finance_retrieval.py tests/test_phase3_finance_retrieval_repair.py
python scripts/phase3/repair_finance_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/generated/context_engineering `
  --dense-backend local_fallback
```

