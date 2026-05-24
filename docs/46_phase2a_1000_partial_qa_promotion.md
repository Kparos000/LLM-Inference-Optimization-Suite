# Phase 2A-13D/13E Partial 1,000-Scale QA and Promotion

Phase 2A-13D audits the source-ready 1,000-scale candidates for four verticals:
Airline, Healthcare Admin, Retail, and Finance. Phase 2A-13E promotes those four
verticals only if the partial QA report is clean.

Research AI is excluded temporarily because its 1,000-scale generator is still
pending. The Research AI source expansion can be ready without this patch
promoting Research AI records.

This workflow is deterministic data QA and file promotion only. It does not
build RAG, retrieval indexes, embeddings, prompt assembly, model calls, GPU
runs, or inference.

## Commands

Run partial QA:

```powershell
python scripts/phase2/audit_phase2a_scaleup_1000_partial.py --run-audit
```

Run strict partial QA:

```powershell
python scripts/phase2/audit_phase2a_scaleup_1000_partial.py --run-audit --fail-on-critical
```

Promote the partial 1,000-scale dataset after QA is clean:

```powershell
python scripts/phase2/promote_phase2a_scaleup_1000_partial.py --promote
```

## Audited Files

The audit reads local generated candidates under `data/generated/phase2a/scaleup/`:

- `airline/airline_prompts_1000.jsonl`
- `airline/airline_gold_1000.jsonl`
- `airline/airline_kb_1000.jsonl`
- `healthcare_admin/healthcare_admin_prompts_1000.jsonl`
- `healthcare_admin/healthcare_admin_gold_1000.jsonl`
- `healthcare_admin/healthcare_admin_kb_1000.jsonl`
- `retail/retail_prompts_1000.jsonl`
- `retail/retail_gold_1000.jsonl`
- `retail/retail_kb_1000.jsonl`
- `finance/finance_prompts_1000.jsonl`
- `finance/finance_gold_1000.jsonl`
- `finance/finance_kb_1000.jsonl`

Expected totals are 4,000 prompts and 4,000 gold records across four verticals.
Each included vertical must have exactly 1,000 prompts and 1,000 gold records.

## QA Gates

The audit checks:

- File existence for all four generated verticals.
- Prompt/gold alignment and global prompt ID uniqueness.
- Evidence coverage for answerable records.
- Exact status distributions and output format distributions.
- Linguistic variation: each vertical must keep `linguistic_variation_rate >= 0.60`.
- KB target ranges: Airline and Healthcare Admin 150-250, Retail 500-1,000, Finance 800-1,200.
- Hygiene terms such as private paths, usernames, secrets, and raw user identifiers.
- Domain safety for Healthcare Admin, Finance, Retail, and Airline.

Generated QA outputs stay local under `data/generated/phase2a/scaleup_reports/`:

- `phase2a_1000_partial_qa_report.json`
- `phase2a_1000_partial_qa_summary.csv`
- `phase2a_1000_partial_issue_log.jsonl`

Promotion requires `promotion_ready: true`, `critical_issue_count: 0`, and
`warning_count: 0`.

## Promoted Layout

The promotion script writes the committed partial checkpoint under
`data/scaleup_1000_partial/`:

- `airline/`
- `healthcare_admin/`
- `retail/`
- `finance/`
- `phase2a_1000_partial_manifest.json`
- `README.md`

The manifest marks `partial_dataset: true`, lists Research AI in
`excluded_verticals`, and records the reason: Research AI 1,000 generator
pending.

## Next Step

Implement Research AI 1,000-scale generation, run full five-vertical 1,000 QA,
and promote the full 5,000-record dataset after review.
