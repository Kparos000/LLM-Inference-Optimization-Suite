# Phase 2A-16B Research AI Retrieval Corpus

Phase 2A-16B exports the full local Research AI section pool as a future
retrieval corpus for Phase 2B context engineering.

Command:

```powershell
python scripts/phase2/export_research_ai_retrieval_corpus.py --export-full-corpus
```

## Benchmark KB Vs Full Corpus

The promoted Research AI benchmark KB under `data/scaleup_2000_full/` contains
gold-linked evidence used for evaluation. It is intentionally smaller and more
controlled than the full paper-section source pool.

The full retrieval corpus contains every usable extracted paper section from the
local processed Research AI artifacts. It is meant to sit beside the benchmark
KB during Phase 2B retrieval and context experiments. It does not replace the
gold-linked benchmark KB.

Not all 2,590 extracted sections should be placed directly into every prompt.
The benchmark needs stable gold-linked evidence, while retrieval experiments
need a larger candidate corpus from which relevant context can later be selected.

Generated files stay local under
`data/generated/phase2a/retrieval_corpus/research_ai/`:

- `research_ai_full_sections_corpus.jsonl`
- `research_ai_full_sections_manifest.json`
- `research_ai_benchmark_kb_to_source_mapping.jsonl`
- `research_ai_retrieval_corpus_quality_report.json`

This export has no embeddings, no retrieval index, no model calls, no RAG, and
no inference.
