# Phase 2A-14 2,000-Scale Generator

Phase 2A-14 extends deterministic local candidate generation to 2,000 prompts
per vertical, or 10,000 prompts total across Airline, Healthcare Admin, Retail,
Finance, and Research AI.

This stage does not build RAG, retrieval indexes, embeddings, prompt assembly,
model calls, GPU runs, or inference. It is no RAG, no inference, and no
embeddings.

## Commands

Generate Airline 2,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical airline --target-per-vertical 2000
```

Generate Healthcare Admin 2,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical healthcare_admin --target-per-vertical 2000
```

Generate Retail 2,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical retail --target-per-vertical 2000
```

Generate Finance 2,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical finance --target-per-vertical 2000
```

Generate Research AI 2,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical research_ai --target-per-vertical 2000
```

## Targets

Each vertical generates:

- 2,000 prompts
- 2,000 gold records
- one matching gold record per prompt
- answerable evidence IDs and citations
- meaningful negative `must_not_include` guardrails
- linguistic variation rate of at least 0.60

KB target ranges:

- Airline: 300 to 500 rows
- Healthcare Admin: 300 to 500 rows
- Retail: 1,000 to 2,000 rows
- Finance: 1,500 to 2,500 rows
- Research AI: 1,600 to 2,000 rows

Finance and Research AI prefer rich local processed SEC, XBRL, paper text, and
paper-section artifacts when available. In clean checkout or CI, they fall back
to committed promoted 1,000-scale and 250-scale KB files, then deterministic
scale-up variants, so generation does not require ignored raw PDFs, SEC raw
files, or generated reports.

## Scope Boundary

This patch implements the 2,000-per-vertical checkpoint only. The 4,000 and
5,000 target tiers remain planning-only until explicitly implemented.

Promotion requires the Phase 2A-15 cross-vertical QA gate before files are
copied into `data/scaleup_2000_full/`.
