# Phase 2A-15 2,000-Scale QA And Promotion

Phase 2A-15 audits and promotes the full 2,000-per-vertical dataset if, and
only if, QA is clean.

The target dataset is `phase2a_2000_full`:

- 5 verticals
- 2,000 prompts per vertical
- 10,000 prompts total
- 10,000 gold records total
- promoted output under `data/scaleup_2000_full/`

This QA and promotion stage does not build RAG, retrieval indexes, embeddings,
prompt assembly, model calls, GPU runs, or inference. It is no RAG, no
inference, and no embeddings.

## QA Command

Run the full 10,000-record audit:

```powershell
python scripts/phase2/audit_phase2a_scaleup_2000_full.py --run-audit
```

Run strict QA that exits non-zero on critical issues:

```powershell
python scripts/phase2/audit_phase2a_scaleup_2000_full.py --run-audit --fail-on-critical
```

Generated QA outputs stay local and ignored:

- `data/generated/phase2a/scaleup_reports/phase2a_2000_full_qa_report.json`
- `data/generated/phase2a/scaleup_reports/phase2a_2000_full_qa_summary.csv`
- `data/generated/phase2a/scaleup_reports/phase2a_2000_full_issue_log.jsonl`

## QA Criteria

The audit checks:

- all five generated vertical file groups exist
- each vertical has exactly 2,000 prompts and 2,000 gold records
- total prompt and gold counts are 10,000 each
- prompt IDs are globally unique
- prompt/gold records are aligned
- answerable records have evidence
- expected status distributions match the Phase 2A-14 contract
- expected output-format distributions match the Phase 2A-14 contract
- KB counts fall within target ranges
- linguistic variation rate is at least 0.60
- private paths, credentials, raw user IDs, TODO/FIXME-style hygiene issues are absent
- domain-specific safety checks pass for Airline, Healthcare Admin, Retail, Finance, and Research AI

## Promotion Command

Promote only after QA reports `promotion_ready: true`, `critical_issue_count: 0`,
and `warning_count: 0`:

```powershell
python scripts/phase2/promote_phase2a_scaleup_2000_full.py --promote
```

Promotion copies generated 2,000-scale candidate files into:

- `data/scaleup_2000_full/airline/`
- `data/scaleup_2000_full/healthcare_admin/`
- `data/scaleup_2000_full/retail/`
- `data/scaleup_2000_full/finance/`
- `data/scaleup_2000_full/research_ai/`

It also writes:

- `data/scaleup_2000_full/phase2a_2000_full_manifest.json`
- `data/scaleup_2000_full/README.md`

The next step after promotion is the public-facing 10,000-record dataset EDA
before Phase 2B context engineering.
