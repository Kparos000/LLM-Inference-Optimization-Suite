# Strict Retrieval Upgrade And Finance Reranking

Block 11 upgrades the strict retrieval baseline before larger Phase 4/5
inference runs. It does not run model inference, GPU work, paid API calls, or
gated model calls.

## Why Strict Retrieval Was Weak

The source-hint ablation remains high because it can use direct source/evidence
hints from the workload. That is useful as an assisted upper bound, but it is
not the right score for final retrieval-quality claims.

The honest strict modes are harder:

- `prompt_text_only` uses only visible prompt text.
- `prompt_plus_metadata` adds realistic prompt metadata such as vertical, task
  type, company/ticker, category, output format, and filing form.
- Both strict modes still block gold evidence IDs, source IDs, parent IDs,
  filing IDs, accession numbers, and answer-side hints.

Finance is the hardest vertical because many prompts say only “selected
financial metric” or name a company/form without a fiscal period, accession,
section, or exact XBRL concept. The correct evidence often reaches the candidate
pool, but the prompt does not contain enough honest signal to rank it into the
final top 5.

## What Changed

Retrieval now separates:

1. candidate generation
2. candidate reranking
3. final top-k selection

Default settings:

- dense candidate top-k: `50`
- lexical candidate top-k: `50`
- final top-k: `5`

For hybrid retrieval, BM25 uses expanded query variants while Qdrant uses the
primary enriched query to avoid multiplying local embedding cost. The reranker
scores merged candidates using:

- original BM25 score
- Qdrant dense score
- company/ticker match
- metric synonym match
- period match
- section match
- title and metadata overlap
- source-hint match only for `prompt_plus_source_hints`

## Query Expansion

Strict retrieval now builds multiple deterministic query variants:

- normalized original query
- synonym-expanded query
- finance metric-expanded query
- entity-normalized query when company/ticker appears in visible text
- period-normalized query when years, quarters, annual, or latest-period terms
  appear
- metadata-expanded query for `prompt_plus_metadata`
- XBRL concept-expanded query when concepts exist in the corpus

## Finance-Specific Retrieval

Finance retrieval adds:

- corpus-derived company/ticker resolver
- finance metric synonym mapper
- fiscal period and quarter extraction
- XBRL concept mapping from corpus concepts only
- section-aware reranking for risk, financial metric, business performance, and
  guidance/outlook questions

No fabricated XBRL concepts are introduced. The concept map is built only from
normalized context-corpus metadata.

## Results

Final `final_10000`, `mm2_hybrid_top5`:

| Ablation | Before recall@5 | After recall@5 | Before MRR | After MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.442638 | 0.430112 | 0.488373 | 0.477390 |
| `prompt_plus_metadata` | 0.528491 | 0.535084 | 0.569042 | 0.576252 |
| `prompt_plus_source_hints` | 0.978887 | 0.957858 | 0.990592 | 0.977587 |

Finance `final_10000`, `mm2_hybrid_top5`:

| Ablation | Before recall@5 | After recall@5 | Before MRR | After MRR |
|---|---:|---:|---:|---:|
| `prompt_text_only` | 0.160625 | 0.156000 | 0.091908 | 0.086383 |
| `prompt_plus_metadata` | 0.218000 | 0.277250 | 0.131867 | 0.186617 |
| `prompt_plus_source_hints` | 0.991625 | 0.997375 | 0.990083 | 0.992500 |

The strict target was not reached. This is not treated as success. Candidate
expansion did reveal substantial recoverable evidence:

- finance `prompt_text_only` candidate recall@50: `0.596250`
- finance `prompt_plus_metadata` candidate recall@50: `0.873875`
- finance `prompt_plus_metadata` reranker rescues: `397`

The remaining gap is final ranking among many same-company/same-form candidates
when the prompt lacks exact period, section, accession, or metric information.

## Compression

`mm3_compressed_hybrid_top5` remains within safety targets:

- `prompt_text_only`: 24.3680% token reduction, 0.0 recall loss
- `prompt_plus_metadata`: 23.9112% token reduction, 0.0 recall loss
- `prompt_plus_source_hints`: 24.4265% token reduction, 0.0 recall loss

## Leakage Prevention

Strict modes still block:

- gold evidence IDs
- source IDs
- parent IDs
- document IDs
- filing IDs
- accession numbers
- answer-side hints
- direct evidence identifiers

`prompt_plus_source_hints` remains clearly labeled as an assisted upper bound
and should not be used as the only retrieval score in final claims.

## Remaining Gaps

- Finance strict prompts often lack exact period or metric specificity.
- Retail still has product/review decoys for same-product evidence.
- Some correct evidence reaches top 50 but cannot be honestly selected into top
  5 without direct source hints.
- A future dataset revision could add realistic retrieval-time metadata such as
  fiscal year, filing date, section intent, or metric label to improve strict
  final ranking without leakage.
