# Phase 2A-11 250-Scale Dataset Promotion

Phase 2A-11 promotes the clean 250-scale candidate set into committed
`data/scaleup/` dataset files. This is a data promotion step only: it does not
build RAG, retrieval indexes, embeddings, prompt assembly, model calls, GPU
runs, or inference.

## Promotion Criteria

Promotion requires the Phase 2A-10 cross-vertical QA report to be clean:

- `promotion_ready: true`
- `critical_issue_count: 0`
- `warning_count: 0`
- 1,250 prompts
- 1,250 gold records
- all five verticals audited

If the QA report is missing or not clean, promotion exits non-zero and does not
copy candidate files.

## Command

```powershell
python scripts/phase2/promote_phase2a_scaleup_250.py --promote
```

The script reads generated candidates from `data/generated/phase2a/scaleup/`
and writes promoted files under `data/scaleup/`.

## Promoted Layout

Each vertical has prompt, gold, and KB files:

- `data/scaleup/airline/airline_prompts_250.jsonl`
- `data/scaleup/airline/airline_gold_250.jsonl`
- `data/scaleup/airline/airline_kb_250.jsonl`
- `data/scaleup/healthcare_admin/healthcare_admin_prompts_250.jsonl`
- `data/scaleup/healthcare_admin/healthcare_admin_gold_250.jsonl`
- `data/scaleup/healthcare_admin/healthcare_admin_kb_250.jsonl`
- `data/scaleup/retail/retail_prompts_250.jsonl`
- `data/scaleup/retail/retail_gold_250.jsonl`
- `data/scaleup/retail/retail_kb_250.jsonl`
- `data/scaleup/research_ai/research_ai_prompts_250.jsonl`
- `data/scaleup/research_ai/research_ai_gold_250.jsonl`
- `data/scaleup/research_ai/research_ai_kb_250.jsonl`
- `data/scaleup/finance/finance_prompts_250.jsonl`
- `data/scaleup/finance/finance_gold_250.jsonl`
- `data/scaleup/finance/finance_kb_250.jsonl`

The promoted dataset also includes:

- `data/scaleup/phase2a_250_manifest.json`
- `data/scaleup/README.md`

## Manifest

`phase2a_250_manifest.json` records:

- dataset name: `phase2a_250_scaleup`
- Phase 2A-11 promotion timestamp
- vertical names
- total prompt, gold, and KB counts
- per-vertical counts and file paths
- source candidate root
- promoted root
- QA report path
- quality summary
- scale-up notes

## Scope

This is the first promoted scale-up checkpoint. It is not the 1,000, 2,000,
4,000, or 5,000 per-vertical dataset. It contains promoted deterministic data
records only and includes no RAG, no inference, and no embeddings.

## Next Step

After this checkpoint is reviewed as the baseline promoted set, the next planned
work is extending deterministic generation to the 1,000-per-vertical checkpoint.
