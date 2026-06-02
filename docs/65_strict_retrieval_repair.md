# Strict Retrieval Repair

This block hardens the retrieval evaluation before Phase 4 execution. It does
not run model inference, GPU work, external APIs, or paid calls.

## Why Source-Hint Retrieval Is Not Enough

The project reports three retrieval ablations:

- `prompt_text_only`: uses only user-visible prompt text.
- `prompt_plus_metadata`: uses prompt text plus realistic prompt metadata such as
  vertical, task type, company, ticker, form, category, or topic.
- `prompt_plus_source_hints`: uses direct source/evidence hints when present and
  is explicitly an assisted upper bound.

The source-hint-assisted score remains useful for validating corpus alignment,
but it is not the honest score for final retrieval claims. The final claim should
use `prompt_text_only` or `prompt_plus_metadata`, with the source-hint score
reported only as an upper bound.

## What Changed

- Added strict leakage guards for `prompt_text_only` and `prompt_plus_metadata`.
- Scrubbed generated evidence/document IDs from strict query text.
- Added deterministic query enrichment for visible finance terms, periods,
  company/ticker aliases, and Research AI terms.
- Added finance-aware reranking using ticker, company, filing form, fiscal period,
  section, and metric/concept overlap.
- Improved Qdrant indexed text with title, body text, and selected metadata.
- Added vector-query compaction so source-hint strings do not dominate dense
  embeddings.
- Added a Qdrant vector snapshot path for high-volume local workload generation.
  The index is still built in Qdrant; the snapshot avoids tens of thousands of
  slow local Qdrant point queries during report generation.
- Added report fields for ablation mode, source-hint use, query enrichment,
  reranking, dense backend, and vector store.

## Final 10,000-Record Results

`mm2_hybrid_top5` final split:

| Ablation | Overall recall@5 | Overall MRR | Finance recall@5 | Finance MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.442638 | 0.488373 | 0.160625 | 0.091908 |
| `prompt_plus_metadata` | 0.528491 | 0.569042 | 0.218000 | 0.131867 |
| `prompt_plus_source_hints` | 0.978887 | 0.990592 | 0.991625 | 0.990083 |

The target strict scores were not reached. This is not treated as success. The
main reason is that many Finance and Retail prompts are under-specified without
direct source hints: ticker/form or product title often identifies a broad group
of plausible records, but not the exact filing accession, section, review, or
XBRL concept required by the gold record.

## Compression

`mm3_compressed_hybrid_top5` remains safe:

- `prompt_text_only`: 26.3149% token reduction, 0.0 recall loss.
- `prompt_plus_metadata`: 26.0301% token reduction, 0.0 recall loss.
- `prompt_plus_source_hints`: 26.3504% token reduction, 0.0 recall loss.

## Remaining Gaps

- Finance strict retrieval needs better query-side specificity or a realistic
  filing/section selection signal before it can reach the requested targets.
- Retail strict retrieval has same-product decoys; boosting one evidence type
  globally hurt the full dataset because some gold rows legitimately point to
  seed-expanded evidence.
- Research AI strict retrieval is strong when the paper title is visible, but
  weak for generic scenario prompts without a title or section clue.

The project should keep reporting strict and assisted scores separately in Phase
4 and Phase 5.
