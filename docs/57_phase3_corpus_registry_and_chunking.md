# Phase 3 Corpus Registry And Chunking

Phase 3 Block 2 normalizes the promoted benchmark KB/evidence into context
records. It prepares context-engineering inputs for later memory modes without
running retrieval, embeddings, inference, GPU experiments, or harness changes.

The source benchmark data remains unchanged under:

```text
data/scaleup_2000_full/
```

Generated context-engineering outputs are written under:

```text
data/generated/context_engineering/
```

## Regeneration

Run:

```powershell
python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full --output-root data/generated/context_engineering
```

The command writes:

- `data/generated/context_engineering/corpus_registry.json`
- `data/generated/context_engineering/corpus_build_report.json`
- `data/generated/context_engineering/corpus_build_summary.csv`
- local JSONL corpora under `data/generated/context_engineering/corpora/`

The corpus JSONL files are generated artifacts. They are intentionally ignored
by Git unless a future promotion policy says otherwise.

## Corpus Registry

The registry maps each vertical to its evidence sources and chunk builder. Each
entry includes:

- `vertical`
- `corpus_id`
- `corpus_role`
- `input_path`
- `output_path`
- `chunk_builder`
- `notes`

The registry records three roles:

- `benchmark_kb`: promoted KB rows normalized into context records.
- `gold_linked_evidence`: gold/eval records used to flag required evidence.
- `full_retrieval_corpus`: future retrieval corpus inputs, currently Research
  AI full paper sections from the internal data-preparation export.

Block 2 builds normalized corpora from the promoted benchmark KB only.

## Vertical Strategies

### Airline

Strategy: policy-section chunking plus recursive fallback.

Airline records are policy/scenario based, so policy boundaries matter. The
builder keeps `doc_id`, policy title, document type, tags, source type, and
policy-family metadata such as `metadata.base_doc_id` when present. Long policy
text is split with a sentence-aware fallback.

### Healthcare Admin

Strategy: admin-procedure chunking plus safety-boundary metadata.

Healthcare Admin records separate administrative instructions from clinical
advice boundaries. The builder preserves procedure family IDs, tags, document
type, and boundary signals for privacy, identity, clinical triage, and
escalation when those signals are available.

### Retail

Strategy: parent-child product/category/review chunking.

Retail evidence needs product and category context. The parent context comes
from product/category metadata such as `parent_asin`, `asin`, `category`, and
`product_title`. Review evidence, review summaries, product metadata, and
support policy rows become child chunks while preserving rating and issue-term
metadata when available.

### Finance

Strategy: SEC/XBRL-aware structured chunking.

Finance does not default to semantic chunking because financial evidence needs
traceability, exact numeric provenance, and filing-aware boundaries. XBRL facts
are kept atomic. Filing events stay as filing metadata evidence. Narrative SEC
filing sections use sentence-window fallback for long sections.

The builder preserves ticker, company, form, filing date, report date,
accession number, concept, concepts, section title, and section type when
available. If expected finance metadata is missing, the build report writes
explicit warnings rather than failing silently.

### Research AI

Strategy: paper-section chunking plus sentence-window fallback.

Research papers already have strong section structure, so the builder preserves
paper and section metadata instead of blindly splitting into generic chunks.
The preserved fields include paper ID, paper title, section record ID, section
title, section type, topic, venue, and year when available.

Expected section types include:

- abstract
- introduction
- method
- experiments
- results
- limitations
- appendix

Long paper sections are split with a sentence-window fallback while retaining
paper and section metadata on every child chunk.

## How This Prepares Memory Modes

The normalized context records provide the shared evidence unit for future
memory modes:

- `mm0_no_context` can ignore context records while preserving gold links.
- `mm1_dense_top5` can later rank the same context records with dense retrieval.
- `mm2_hybrid_top5` can combine lexical and dense ranking over the same corpus.
- `mm3_compressed_hybrid_top5` can compress selected context while retaining
  provenance.
- `mm4_bounded_agentic` can use the same context IDs in a bounded workflow
  trace.

This keeps context construction modular and lets the existing mock, Hugging
Face, OpenAI-compatible, and OpenAI load runners continue to be reused when
inference begins.
