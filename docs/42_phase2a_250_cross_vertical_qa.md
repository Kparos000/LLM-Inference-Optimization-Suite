# Phase 2A-10 250-Scale Cross-Vertical QA

Phase 2A-10 audits the full local 250-scale candidate set before any full
1,250-record promotion. This is a QA step only: it does not build RAG, retrieval
indexes, embeddings, prompt assembly, model calls, GPU runs, or inference.

Generated QA outputs remain under ignored `data/generated/phase2a/` paths until
they are reviewed. This patch does not promote Healthcare Admin, Retail,
Research AI, or Finance candidates.

## Purpose

The audit checks that all five verticals can stand together as one coherent
250-scale candidate set:

- Airline
- Healthcare Admin
- Retail
- Research AI
- Finance

The expected total is 1,250 prompts and 1,250 gold records: exactly 250 prompt
records and 250 gold records per vertical.

## Audited Files

The audit reads generated local candidate files by default:

- `data/generated/phase2a/scaleup/airline/airline_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_gold_250.jsonl`
- `data/generated/phase2a/scaleup/airline/airline_kb_250.jsonl`
- `data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_gold_250.jsonl`
- `data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_kb_250.jsonl`
- `data/generated/phase2a/scaleup/retail/retail_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/retail/retail_gold_250.jsonl`
- `data/generated/phase2a/scaleup/retail/retail_kb_250.jsonl`
- `data/generated/phase2a/scaleup/research_ai/research_ai_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/research_ai/research_ai_gold_250.jsonl`
- `data/generated/phase2a/scaleup/research_ai/research_ai_kb_250.jsonl`
- `data/generated/phase2a/scaleup/finance/finance_prompts_250.jsonl`
- `data/generated/phase2a/scaleup/finance/finance_gold_250.jsonl`
- `data/generated/phase2a/scaleup/finance/finance_kb_250.jsonl`

If a generated candidate file is missing, the audit fails and prints the exact
generation command to run, for example:

```powershell
python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical --vertical finance --target-per-vertical 250
```

## Commands

Run the normal audit:

```powershell
python scripts/phase2/audit_phase2a_scaleup_250.py --run-audit
```

Run the strict audit, which exits non-zero if critical issues are present:

```powershell
python scripts/phase2/audit_phase2a_scaleup_250.py --run-audit --fail-on-critical
```

## Checks

The audit validates:

- file existence for prompt, gold, and KB files
- exact per-vertical counts
- global prompt and gold counts
- prompt/gold alignment
- globally unique prompt IDs
- answerable evidence coverage
- negative/status-boundary coverage
- expected status distributions
- expected output format distributions
- linguistic variation metrics
- public hygiene
- domain-specific safety and boundary rules

Per-vertical status and output-format distributions must match the approved
250-scale plans exactly.

## Evidence Alignment

Answerable records must have evidence IDs. Where applicable, answerable records
should also include chunk IDs or citations. Required document IDs are checked
against the candidate KB for that vertical, and every prompt must have exactly
one matching gold record.

Negative records must include meaningful `must_not_include` values and must not
turn an unsupported or boundary request into a direct answer.

## Linguistic Variation

The audit requires the same linguistic variation gate used by the generators:
`linguistic_variation_rate` must be at least 0.60 and the dominant normalized
question template share must be at most 0.40. If generator report metrics are
available, the audit reads them; otherwise it recomputes metrics from prompt
questions.

## Domain-Specific Safety

Healthcare Admin checks keep generated answers administrative only and route
urgent or clinical requests to safety boundary or escalation.

Finance checks reject investment advice, buy/sell/hold recommendations, and
numeric claims without evidence.

Retail checks reject raw user IDs, generic product titles such as
`All_Beauty product <ASIN>`, and support policies that are not clearly labeled
as synthetic benchmark policy, not Amazon policy.

Research AI checks reject fabricated paper claims, general model memory answers,
and answerable rows without KB or paper evidence.

Airline checks reject unsupported compensation promises, verification bypasses,
and uncited policy exceptions.

## Reports

The audit writes:

- `data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_report.json`
- `data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_summary.csv`
- `data/generated/phase2a/scaleup_reports/phase2a_250_issue_log.jsonl`

The JSON report includes `promotion_ready`. If the set is clean,
`promotion_ready` is `true` and the next step is Phase 2A-11 promotion of the
full 250-scale dataset. If any issue remains, `promotion_ready` is `false` and
the candidate files should be fixed before promotion.
