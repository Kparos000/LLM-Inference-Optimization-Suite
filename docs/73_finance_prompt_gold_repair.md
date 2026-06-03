# Finance Prompt and Gold Repair Audit

This report closes the Finance retrieval ambiguity audit before more reranking work.
It does not run inference, GPU experiments, paid APIs, or modify the promoted
benchmark dataset.

## Why This Block Exists

The Finance retrieval root-cause report showed that strict retrieval failures
were dominated by missing prompt-side metadata:

- `prompt_missing_period`
- `prompt_missing_metric`
- candidate retrieval was not the primary blocker
- final top-5 selection was a secondary blocker

That means another reranking pass would be premature. The first question is
whether Finance prompts and gold records expose enough human-readable retrieval
cues for strict retrieval to succeed without direct source IDs.

## Generated Reports

The repair pipeline writes these reports under
`data/generated/context_engineering/`:

- `finance_prompt_quality_report.json`
- `finance_prompt_quality_summary.csv`
- `finance_gold_quality_report.json`
- `finance_gold_quality_summary.csv`
- `finance_metadata_enrichment_report.json`
- `finance_retrieval_repair_impact_report.json`
- `finance_retrieval_repair_impact_summary.csv`
- `finance_retrieval_repair_report.json`

Regenerate them with:

```powershell
python scripts/phase3/repair_finance_retrieval.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --output-root data/generated/context_engineering `
  --dense-backend local_fallback
```

The script uses retrieval only. It does not call a model.

## Prompt Quality Results

All 2,000 Finance prompts were audited for company/ticker, metric, period,
filing type, and filing section cues.

| Field | Present | Missing |
| --- | ---: | ---: |
| Company/ticker | 2,000 | 0 |
| Metric | 165 | 1,835 |
| Period | 0 | 2,000 |
| Filing type | 1,516 | 484 |
| Filing section | 12 | 1,988 |

The exact blocker is now measurable: Finance prompts usually identify the
company/ticker, but they generally do not expose period, metric, or section
language that a strict retriever can use.

## Gold Quality Results

Gold/eval records were audited for explicit recoverability and linked-context
recoverability.

| Field | Explicit In Gold | Missing From Gold | Recoverable From Linked Context |
| --- | ---: | ---: | ---: |
| Metric | 826 | 1,174 | 2,000 |
| Period | 0 | 2,000 | 1,127 |
| Filing type | 1,497 | 503 | 1,127 |
| Filing section | 317 | 1,683 | 1,127 |

The gold records are better than prompts for filing type and metric language, but
period remains absent from gold text. Linked corpus metadata can recover much of
the missing structure, especially filing section, filing type, and fiscal period.

## Metadata Enrichment

The enrichment layer derives human-readable fields for retrieval repair:

- ticker
- company
- filing type
- filing section
- period
- fiscal quarter
- fiscal year
- XBRL concept
- metric family

Coverage from the generated report:

| Enriched Field | Count |
| --- | ---: |
| Ticker | 2,000 |
| Company | 2,000 |
| Filing type | 1,516 |
| Filing section | 1,127 |
| Period | 1,147 |
| Fiscal year | 1,127 |
| XBRL concept | 873 |
| Metric family | 2,000 |

The rewrite pipeline adds these human-readable cues to retrieval queries after
scrubbing direct evidence identifiers such as `finance_kb_*`, `finance_sec_*`,
`sec://`, `source_id`, `parent_id`, and gold evidence fields.

## Retrieval Impact

The repair measurement compares current strict Finance queries against
metadata-repaired queries. It was run with `dense_backend=local_fallback`, and
the reports label that backend explicitly. A full Qdrant repair run was attempted
but exceeded the local timeout; the next Qdrant run should batch-warm query
embeddings/searches before measuring all 2,000 prompts.

| Ablation | Measurement | Candidate Recall@20 | Candidate Recall@50 | Final Recall@5 | MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| prompt_text_only | before | 0.304000 | 0.566500 | 0.162500 | 0.091292 |
| prompt_text_only | after metadata repair | 0.824625 | 0.854500 | 0.647125 | 0.454942 |
| prompt_plus_metadata | before | 0.495625 | 0.794375 | 0.288625 | 0.190917 |
| prompt_plus_metadata | after metadata repair | 0.800000 | 0.840750 | 0.629500 | 0.442133 |

The repaired queries substantially improve candidate recall and final recall.
This confirms that Finance retrieval is primarily blocked by missing
human-readable metric/period/filing context, not by corpus absence.

## Interpretation

This is not a final strict retrieval claim for the paper. The repaired score uses
gold-linked corpus metadata to test whether adding human-readable metadata would
fix retrieval after direct evidence IDs are removed. It should be treated as a
repair feasibility result.

The next implementation step should materialize non-ID Finance metadata into the
workload/query-building layer in a way that is available at retrieval time, then
rerun the strict Qdrant ablation. Only after that should we tune final top-5
reranking.

