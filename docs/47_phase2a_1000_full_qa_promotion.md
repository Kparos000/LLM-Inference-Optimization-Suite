# Phase 2A-13G/13H Full 1,000-Scale QA and Promotion

Phase 2A-13G adds Research AI 1,000-scale generation using the expanded
Research AI paper-section source set. Phase 2A-13H audits and promotes the full
five-vertical 1,000-scale dataset if QA is clean.

This workflow is deterministic data generation, QA, and file promotion only. It
does not build RAG, retrieval indexes, embeddings, prompt assembly, model calls,
GPU runs, or inference. In short, this is no RAG, no inference, and no
embeddings.

## Commands

Generate Research AI 1,000 local candidates:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical research_ai --target-per-vertical 1000
```

Run full 1,000-scale QA:

```powershell
python scripts/phase2/audit_phase2a_scaleup_1000_full.py --run-audit
```

Run strict full QA:

```powershell
python scripts/phase2/audit_phase2a_scaleup_1000_full.py --run-audit --fail-on-critical
```

Promote the full 1,000-scale dataset after QA is clean:

```powershell
python scripts/phase2/promote_phase2a_scaleup_1000_full.py --promote
```

## Inputs

Full QA audits five verticals:

- Airline from `data/scaleup_1000_partial/airline/`
- Healthcare Admin from `data/scaleup_1000_partial/healthcare_admin/`
- Retail from `data/scaleup_1000_partial/retail/`
- Finance from `data/scaleup_1000_partial/finance/`
- Research AI from `data/generated/phase2a/scaleup/research_ai/`

Expected totals are 5,000 prompts and 5,000 gold records across five verticals.
Each vertical must have exactly 1,000 prompts and 1,000 gold records.

## QA Gates

The audit checks:

- File existence for all five verticals.
- Prompt/gold alignment and global prompt ID uniqueness.
- Evidence coverage for answerable records.
- Exact status and output format distributions.
- Linguistic variation: each vertical must keep `linguistic_variation_rate >= 0.60`.
- KB target ranges: Airline and Healthcare Admin 150-250, Retail 500-1,000, Finance 800-1,200, Research AI 800-1,200.
- Hygiene terms such as private paths, usernames, secrets, and raw user identifiers.
- Domain checks for Healthcare Admin, Finance, Retail, Research AI, and Airline.

Generated QA outputs stay local under `data/generated/phase2a/scaleup_reports/`:

- `phase2a_1000_full_qa_report.json`
- `phase2a_1000_full_qa_summary.csv`
- `phase2a_1000_full_issue_log.jsonl`

Promotion requires `promotion_ready: true`, `critical_issue_count: 0`, and
`warning_count: 0`.

## Promoted Layout

The promotion script writes the committed full checkpoint under
`data/scaleup_1000_full/`:

- `airline/`
- `healthcare_admin/`
- `retail/`
- `finance/`
- `research_ai/`
- `phase2a_1000_full_manifest.json`
- `README.md`

The manifest records `dataset_name: phase2a_1000_full`, 5,000 prompts, 5,000
gold records, per-vertical counts, and the QA quality summary.

## Next Step

Begin 2,000-per-vertical generator planning for the 10,000-record target.
