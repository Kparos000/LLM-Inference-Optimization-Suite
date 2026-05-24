# Phase 2A-13A/13B/13C/13G 1,000-Scale Generator

Phase 2A-13 starts 1,000-scale local candidate generation. Airline and
Healthcare Admin were implemented first, followed by Retail and Finance once
their source-readiness checks passed. Research AI 1,000 generation is added
after the expanded paper source set reaches readiness.

This is deterministic local candidate generation only. It does not build RAG,
retrieval indexes, embeddings, prompt assembly, model calls, GPU runs, or
inference. In short, this is no RAG, no inference, and no embeddings.

## Commands

Generate Airline 1,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical airline --target-per-vertical 1000
```

Generate Healthcare Admin 1,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical healthcare_admin --target-per-vertical 1000
```

Generate Retail 1,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical retail --target-per-vertical 1000
```

Generate Finance 1,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical finance --target-per-vertical 1000
```

Generate Research AI 1,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical research_ai --target-per-vertical 1000
```

## Generated Files

Generated files stay local and ignored under `data/generated/phase2a/` until
they are reviewed and promoted in a later phase.

Airline:

- `data/generated/phase2a/scaleup/airline/airline_prompts_1000.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_gold_1000.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_kb_1000.jsonl`
- `data/generated/phase2a/scaleup_reports/airline_scaleup_1000_report.json`

Healthcare Admin:

- `data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_prompts_1000.jsonl`
- `data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_gold_1000.jsonl`
- `data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_kb_1000.jsonl`
- `data/generated/phase2a/scaleup_reports/healthcare_admin_scaleup_1000_report.json`

Retail:

- `data/generated/phase2a/scaleup/retail/retail_prompts_1000.jsonl`
- `data/generated/phase2a/scaleup/retail/retail_gold_1000.jsonl`
- `data/generated/phase2a/scaleup/retail/retail_kb_1000.jsonl`
- `data/generated/phase2a/scaleup_reports/retail_scaleup_1000_report.json`

Finance:

- `data/generated/phase2a/scaleup/finance/finance_prompts_1000.jsonl`
- `data/generated/phase2a/scaleup/finance/finance_gold_1000.jsonl`
- `data/generated/phase2a/scaleup/finance/finance_kb_1000.jsonl`
- `data/generated/phase2a/scaleup_reports/finance_scaleup_1000_report.json`

Research AI:

- `data/generated/phase2a/scaleup/research_ai/research_ai_prompts_1000.jsonl`
- `data/generated/phase2a/scaleup/research_ai/research_ai_gold_1000.jsonl`
- `data/generated/phase2a/scaleup/research_ai/research_ai_kb_1000.jsonl`
- `data/generated/phase2a/scaleup_reports/research_ai_scaleup_1000_report.json`

## Distribution Contracts

Airline generates 1,000 prompts with:

- status: 900 answer, 80 escalate, 20 spam_or_fraud
- output format: 760 text, 140 JSON, 100 markdown table
- expanded synthetic policy KB: 150 records

Healthcare Admin generates 1,000 prompts with:

- status: 880 answer, 80 escalate, 20 safety_boundary, 10 spam_or_fraud, 10 out_of_scope
- output format: 780 text, 140 JSON, 80 markdown table
- expanded synthetic admin policy KB: 150 records

Retail generates 1,000 prompts with:

- status: 890 answer, 35 insufficient_evidence, 35 escalate, 30 spam_or_low_quality, 10 out_of_scope
- output format: 740 text, 160 JSON, 100 markdown table
- expanded Retail KB: 500 to 1,000 records
- evidence from promoted Retail 250, local multi-category Amazon Reviews samples when available, and synthetic benchmark support policies that are not Amazon policy

Finance generates 1,000 prompts with:

- status: 920 answer, 40 insufficient_evidence, 40 escalate
- output format: 620 text, 200 JSON, 180 markdown table
- expanded SEC/XBRL filing-derived KB: 800 to 1,200 records
- evidence reuse spread across filing sections, XBRL facts, 8-K events, and promoted Finance 250 evidence

Research AI generates 1,000 prompts with:

- status: 900 answer, 40 insufficient_evidence, 40 escalate, 20 out_of_scope
- output format: 720 text, 140 JSON, 140 markdown table
- expanded paper-section KB: 800 to 1,200 records
- evidence from the approved paper registry, section manifest, and promoted Research AI 250 evidence

All implemented generators keep prompt/gold alignment, one gold record per
prompt, answerable evidence IDs, meaningful negative `must_not_include`
guardrails, and the linguistic variation quality gate.

## Clean Checkout Reproducibility

Finance and Research AI prefer rich local source artifacts when they are
available:

- Finance uses processed SEC filing sections and XBRL inventories.
- Research AI uses processed paper text and section manifests.

Those rich artifacts are generated locally and are not committed. In a clean CI
checkout, the 1,000 generators deterministically fall back to committed promoted
KB files under `data/scaleup_1000_full/`, `data/scaleup_1000_partial/`, and the
250-scale checkpoint under `data/scaleup/`. This keeps the 800 to 1,200 KB
target reproducible without committing raw PDFs, SEC raw files, generated
reports, RAG indexes, embeddings, or inference outputs.

## Scope Boundary

Larger 2,000, 4,000, and 5,000 target sizes remain planning-only until
explicitly implemented.

The next step after generating these local candidates is targeted review and QA
across all 1,000 files before any 2,000-scale extension. Promotion requires a
separate cross-vertical QA gate, first partial at 4,000 records and then full at
the full 5,000-record checkpoint.
