# RAG Quality Gate And Final Top-5 Retrieval

Block 12 adds a hard retrieval quality gate before larger inference runs. It
does not run model inference, GPU work, paid API calls, gated model calls, or
local HF/vLLM/SGLang generation.

## Why Inference Is Blocked

The project now has three retrieval ablations:

- `prompt_text_only`: visible prompt text only.
- `prompt_plus_metadata`: prompt text plus realistic prompt-side metadata.
- `prompt_plus_source_hints`: direct source hints, explicitly treated as an
  assisted upper bound.

The source-hint score is useful for proving that the corpus and gold labels can
align, but it is not the honest score for final retrieval claims. The main
quality gate therefore focuses on `prompt_plus_metadata` and `prompt_text_only`.

The target for metadata-assisted retrieval is 80%+ recall@5 because the future
generation benchmark will only receive five evidence records. If the correct
evidence is not in that final top 5, the generation system will often be forced
to answer with incomplete or incorrect context.

## Candidate Vs Final Selection

Block 11 showed that candidate retrieval was often better than final top-5
selection. Finance was the clearest case: gold evidence frequently appeared in
the candidate pool, but did not survive final ranking.

Block 12 adds report fields for top 100 and top 200 diagnostics. The practical
full-build candidate depth remains 50 because two full top-200 attempts timed
out before report generation on the local CPU workflow. The reports expose
feasibility flags so top-100/top-200 values are not mistaken for a true wider
candidate run. Reports now separate:

- gold absent from the candidate pool
- gold in top 50 or top 100 but not final top 5
- prompt ambiguity such as missing entity, metric, or period
- likely corpus/gold misalignment
- compression recall loss

The official workload context remains top 5. Wider candidate pools are used only
for diagnostics and reranker calibration.

The default local Qdrant configuration uses deterministic hash vectors so the
no-API/no-GPU report build remains reproducible. The Qdrant backend is still a
real persisted vector store, but this local fallback is not presented as a
sentence-transformer semantic retrieval claim.

## Calibrated Reranking

The reranker is labeled as `calibrated_linear`. It uses runtime-available
features only:

- Qdrant score
- BM25 score
- hybrid score
- lexical/title/text overlap
- company and ticker match
- metric synonym and XBRL concept match
- period, quarter, year, and section match
- vertical and section-type features

Gold evidence labels are allowed only offline for calibration and reporting.
They are not used during retrieval for a test record. Strict ablations continue
to reject direct source IDs, parent IDs, document IDs, filing IDs, accession
numbers, and answer-side evidence hints.

The optional cross-encoder backend remains disabled by default. It is only a
future config-gated path and is not required for CI.

## Evidence Selector

The final selector returns top 5 evidence records from the reranked candidate
pool, removes exact duplicate chunks, preserves provenance, and records why each
item was selected.

Supported selector labels:

- `calibrated_top5`
- `finance_calibrated_top5`
- `finance_diverse_metric_period_top5`
- `oracle_diagnostic_only`

The oracle strategy is diagnostic only and is never used for final benchmark
retrieval.

## Evidence Contract

Each selected evidence item is normalized into a retrieval-to-generation
contract with:

- evidence ID, source ID, parent ID, vertical, title, and section type
- company, ticker, metric, concept, and period when available
- retrieval score and rerank score
- selection reason
- evidence text
- citation label

The contract intentionally excludes gold labels.

## Quality Gate

The quality gate writes:

- `data/generated/context_engineering/retrieval_quality_gate_report.json`
- `data/generated/context_engineering/retrieval_quality_gate_summary.csv`

The gate reports `PASSED` only if all required thresholds are met. If any
threshold fails, the gate reports `BLOCKED` and lists the failed target, margin,
and recommended repair action.

## What Remains

If the gate remains blocked, Phase 4 inference scaling should not proceed as a
final benchmark. Acceptable next work is retrieval repair, corpus/chunking
redesign, or a small plumbing smoke that is clearly labeled as infrastructure
validation rather than benchmark inference.
