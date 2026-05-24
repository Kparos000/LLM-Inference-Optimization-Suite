# Phase 2A-16C 10,000-Record Dataset EDA

Phase 2A-16C generates comprehensive EDA for the promoted 10,000-record
Phase 2A dataset under `data/scaleup_2000_full/`.

Command:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report
```

Generated EDA outputs stay local under `data/generated/phase2a/eda/`:

- `phase2a_10000_dataset_inventory.json`
- `phase2a_10000_dataset_summary.csv`
- `phase2a_prompt_profile.json`
- `phase2a_kb_profile.json`
- `phase2a_gold_profile.json`
- `phase2a_alignment_report.json`
- `phase2a_evidence_reuse_report.json`
- `phase2a_safety_report.json`
- `phase2a_workload_shape_report.json`
- `word_views/*.txt`
- `plots/*.png` when plotting support is available

The EDA profiles prompt shape, gold/eval alignment, KB size, evidence reuse,
safety/domain hygiene, workload token buckets, and vertical-specific coverage.
It also reports whether the optional Research AI full retrieval corpus exists
and how it relates to the promoted benchmark KB.

This analysis supports Phase 2B context engineering and inference benchmark
planning. It does not build RAG, retrieval indexes, embeddings, model calls, GPU
runs, or inference outputs.
