# Phase 2A-7 Cross-Vertical Data QA

Phase 2A-7 audits the committed seed data for all five Phase 2A verticals:
Finance, Airline Customer Support, Healthcare Administrative Support,
Research AI Assistant, and Retail / E-commerce Support.

This step is data QA only. It does not build RAG, retrieval, embeddings, prompt
assembly, model calls, GPU runs, benchmark inference, or the full 5,000-10,000
prompt datasets.

## Files Audited

The audit reads the committed prompt/source, KB/context, and gold/eval JSONL
files for each vertical:

- `data/real_world_samples/finance_sample.jsonl`
- `data/kb/finance/kb_sample.jsonl`
- `data/eval/gold/finance_gold_sample.jsonl`
- `data/real_world_samples/airline_sample.jsonl`
- `data/kb/airline/kb_sample.jsonl`
- `data/eval/gold/airline_gold_sample.jsonl`
- `data/real_world_samples/healthcare_admin_sample.jsonl`
- `data/kb/healthcare_admin/kb_sample.jsonl`
- `data/eval/gold/healthcare_admin_gold_sample.jsonl`
- `data/real_world_samples/research_ai_sample.jsonl`
- `data/kb/research_ai/kb_sample.jsonl`
- `data/eval/gold/research_ai_gold_sample.jsonl`
- `data/real_world_samples/retail_sample.jsonl`
- `data/kb/retail/kb_sample.jsonl`
- `data/eval/gold/retail_gold_sample.jsonl`

## QA Checks

The audit validates:

- required file existence
- prompt, KB, and gold counts
- prompt/gold alignment and orphan gold records
- answerable gold evidence coverage
- required KB document references
- expected status coverage, including negative and boundary records
- output format and task type diversity
- KB document type diversity
- public hygiene for private paths, credentials, raw identifiers, and unsupported
  benchmark claims
- Retail synthetic support policy labeling and generic product-title warnings
- Research AI reference-answer artifact checks
- Finance SEC/XBRL provenance signals
- Airline and Healthcare synthetic policy labeling

## Severity

`critical` issues block scale-up. Examples include missing files, prompt/gold
count mismatches, orphan gold records, raw identifiers, private paths, secrets,
and answerable gold records with no evidence.

`warning` issues should be reviewed before scale-up. Examples include generic
Retail product titles, missing explicit citation fields in older seed formats,
low KB diversity, or mechanical reference-answer phrasing.

`info` issues document expected seed-level limitations, such as raw generated
source files remaining local and ignored.

## Scale-Up Readiness

`ready_for_250_scale` is true only when a vertical has seed prompts, seed KB,
seed gold, no critical issues, at least one negative or boundary status, and
adequate answerable evidence coverage.

Warnings do not automatically block readiness, but they should be reviewed
before progressive scale-up.

## Outputs

Generated local outputs are ignored:

- `data/generated/phase2a/phase2a_cross_vertical_qa_report.json`
- `data/generated/phase2a/phase2a_cross_vertical_qa_summary.csv`
- `data/generated/phase2a/phase2a_issue_log.jsonl`

## Commands

Run the audit:

```text
python scripts/phase2/audit_phase2a_seed_data.py --run-audit
```

Run strict mode, which exits non-zero if critical issues are present:

```text
python scripts/phase2/audit_phase2a_seed_data.py --run-audit --fail-on-critical
```

Inspect the report:

```text
python -m json.tool data/generated/phase2a/phase2a_cross_vertical_qa_report.json
```

## Next Step

After reviewing the cross-vertical report and issue log, proceed to Phase 2A-8
progressive scale-up planning.
