# Phase 2A Partial 1,000-Scale Dataset

This directory contains the promoted partial Phase 2A 1,000-scale dataset,
`phase2a_1000_partial`.

It contains 1,000 prompts per included vertical and 4,000 prompts total across:

- Airline
- Healthcare Admin
- Retail
- Finance

Research AI is intentionally excluded because its 1,000-scale generator is
pending. This is not the full 5,000-record 1,000-scale dataset.

## Layout

Each included vertical has three JSONL files:

- `<vertical>_prompts_1000.jsonl`
- `<vertical>_gold_1000.jsonl`
- `<vertical>_kb_1000.jsonl`

The manifest is `phase2a_1000_partial_manifest.json`.

## Scope

This checkpoint contains promoted deterministic data records only. It does not
include inference results, RAG indexes, retrieval artifacts, embeddings, prompt
assembly outputs, model calls, GPU runs, or benchmark logs.

## Next Step

Implement Research AI 1,000-scale generation, run full five-vertical QA, and
promote the full 5,000-record 1,000-scale dataset after review.
