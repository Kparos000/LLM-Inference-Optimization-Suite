# Phase 2A Full 2,000-Scale Dataset

This directory contains the promoted full Phase 2A 2,000-scale dataset,
`phase2a_2000_full`.

It contains 2,000 prompts per vertical and 10,000 prompts total across Airline,
Healthcare Admin, Retail, Finance, and Research AI.

## Layout

Each vertical has three JSONL files:

- `<vertical>_prompts_2000.jsonl`
- `<vertical>_gold_2000.jsonl`
- `<vertical>_kb_2000.jsonl`

The manifest is `phase2a_2000_full_manifest.json`.

## Scope

This checkpoint contains promoted deterministic data records only. It does not
include inference results, RAG indexes, retrieval artifacts, embeddings, prompt
assembly outputs, model calls, GPU runs, or benchmark logs.

## Next Step

Begin 2,000-per-vertical generator planning for the 10,000-record target.
