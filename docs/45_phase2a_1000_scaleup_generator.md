# Phase 2A-13A 1,000-Scale Generator

Phase 2A-13A starts 1,000-scale local candidate generation with the two
synthetic, policy-bound verticals: Airline and Healthcare Admin. This patch does
not generate Retail, Finance, or Research AI 1,000 records, and it does not
promote any generated files.

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

## Distribution Contracts

Airline generates 1,000 prompts with:

- status: 900 answer, 80 escalate, 20 spam_or_fraud
- output format: 760 text, 140 JSON, 100 markdown table
- expanded synthetic policy KB: 150 records

Healthcare Admin generates 1,000 prompts with:

- status: 880 answer, 80 escalate, 20 safety_boundary, 10 spam_or_fraud, 10 out_of_scope
- output format: 780 text, 140 JSON, 80 markdown table
- expanded synthetic admin policy KB: 150 records

Both generators keep prompt/gold alignment, one gold record per prompt,
answerable evidence IDs, meaningful negative `must_not_include` guardrails, and
the linguistic variation quality gate.

## Scope Boundary

Retail, Finance, and Research AI 1,000 generation come later. Retail needs the
multi-category generator extension, Finance needs its separate 1,000 generator
implementation, and Research AI remains blocked until the 40-paper or equivalent
section-coverage source expansion is complete.

The next step after generating these local candidates is targeted review and QA
for the Airline and Healthcare Admin 1,000 files before any promotion or
2,000-scale extension.
