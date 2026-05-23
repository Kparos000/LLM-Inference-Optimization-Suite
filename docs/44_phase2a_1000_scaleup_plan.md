# Phase 2A-12A 1,000-Scale Readiness Plan

Phase 2A-12A plans the next scale-up checkpoint: 1,000 prompts per vertical,
5,000 prompts total. This is a readiness and planning step only. It does not
generate the 5,000 records and does not build RAG, retrieval indexes,
embeddings, prompt assembly, model calls, GPU runs, or inference.
In short, this is no RAG, no inference, and no embeddings.

## Purpose

The promoted 250 dataset is the prerequisite for the 1,000-scale extension. The
planner reads `data/scaleup/phase2a_250_manifest.json`, confirms the promoted
250 checkpoint is complete, and writes readiness artifacts for extending each
vertical by another 750 prompts.

Command:

```powershell
python scripts/phase2/plan_phase2a_1000_scaleup.py --write-report
```

## Inputs

- `data/scaleup/phase2a_250_manifest.json`
- `data/sources/phase2a_scaleup_plan.json`
- optionally, the latest Phase 2A-10 QA report for context

The script fails if the promoted 250 manifest is missing or does not describe a
complete, promotion-ready 1,250-record dataset.

## Outputs

The planner writes local ignored readiness artifacts:

- `data/generated/phase2a/scaleup_reports/phase2a_1000_scaleup_readiness_report.json`
- `data/generated/phase2a/scaleup_reports/phase2a_1000_scaleup_matrix.csv`

These files are planning artifacts only and are not the 1,000-scale dataset.

## Per-Vertical Requirements

Airline can extend from the deterministic synthetic policy/ticket generator. It
still needs review at the 1,000 checkpoint, but no new real-world source
acquisition is required before the generator extension.

Healthcare Admin can extend from the deterministic synthetic administrative
generator. It must preserve the admin-only safety boundary and the urgent
clinical escalation boundary.

Retail needs a larger sampled review/metadata set and a category expansion plan
before full 1,000-scale generation. The planner expects expansion beyond the
current 250-scale source mix.

Research AI should expand from the current 20-paper 250 checkpoint toward about
40 papers, or otherwise increase section coverage enough to avoid repetitive
evidence use at 1,000 prompts.

Finance should use the current 8-company SEC/XBRL corpus first, while checking
evidence reuse to avoid repetitive prompts before committing to the full 1,000
generation.

## Review Subset Plan

For the 5,000-record checkpoint, review should be stratified by vertical,
task_type, status, output_format, difficulty, and evidence type.

Recommended review sizes:

- gold review subset: 500 to 1,000 records
- deep review subset: 150 to 300 records

## Why Not Generate Immediately

Generating 5,000 records without this readiness check would risk repeating the
same evidence, overusing source material that was acceptable only at 250 scale,
and skipping source expansion requirements for Retail and Research AI. The
readiness report makes those blockers explicit before implementation.

## Next Step

Implement 1,000 generation beginning with synthetic verticals while resolving
Retail and Research AI source-expansion blockers before full 5,000-record
generation.
