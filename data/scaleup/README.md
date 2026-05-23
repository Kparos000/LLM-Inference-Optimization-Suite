# Phase 2A 250-Scale Dataset

This directory contains the promoted Phase 2A 250-scale dataset,
`phase2a_250_scaleup`.

It is the first promoted scale-up checkpoint: 250 prompts per vertical and
1,250 prompts total across Airline, Healthcare Admin, Retail, Research AI, and
Finance.

## Layout

Each vertical has three JSONL files:

- `<vertical>_prompts_250.jsonl`
- `<vertical>_gold_250.jsonl`
- `<vertical>_kb_250.jsonl`

The manifest is `phase2a_250_manifest.json`.

## Scope

This is not the 1,000, 2,000, 4,000, or 5,000 per-vertical dataset. It contains
promoted deterministic data records only. It does not include inference results,
RAG indexes, retrieval artifacts, embeddings, prompt assembly outputs, model
calls, GPU runs, or benchmark logs.

## Next Step

The next planned work is extending the deterministic generators to the
1,000-per-vertical checkpoint after this 250-scale dataset is reviewed as the
baseline promoted set.
