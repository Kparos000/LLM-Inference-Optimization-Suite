# Phase 2A-16A Large-Scale Scaffolding

Phase 2A-16A defines future 4,000 and 5,000 prompts-per-vertical checkpoints.
It does not generate those datasets yet.

The planned future tiers are:

- `checkpoint_4000`: 4,000 prompts per vertical, 20,000 total prompts, GPU stress tier
- `checkpoint_5000`: 5,000 prompts per vertical, 25,000 total prompts, maximum expanded benchmark capacity

Command:

```powershell
python scripts/phase2/plan_phase2a_large_scale.py --write-report
```

Generated planning outputs stay local under `data/generated/phase2a/large_scale/`.

## KB Targets

- Airline: 600 to 900 rows at 4,000; 800 to 1,200 rows at 5,000
- Healthcare Admin: 600 to 900 rows at 4,000; 800 to 1,200 rows at 5,000
- Retail: 2,000 to 4,000 rows at 4,000; 2,500 to 5,000 rows at 5,000
- Finance: 2,500 to 4,500 rows at 4,000; 3,500 to 6,000 rows at 5,000
- Research AI: 1,600 to 2,800 rows at 4,000; 2,000 to 3,500 rows at 5,000

Research AI has a source pool of 60 approved papers and about 2,590 extracted
sections when local processed artifacts are present. The promoted benchmark KB
is a selected gold-linked subset, not the full retrieval corpus. EDA and
Phase 2B corpus checks should guide any future 4,000/5,000 generation.

No RAG, retrieval index, embeddings, inference, model calls, or GPU experiments
are part of this scaffolding step.
