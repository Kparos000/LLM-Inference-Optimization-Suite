# Phase 2A Progressive Scale-Up Plan

Phase 2A-8 defines the control framework for growing the reviewed seed datasets
from 40 prompts per vertical to the full Phase 2A dataset.

The approved cap is 10,000 total prompts across five verticals. It is not 10,000
prompts per vertical.

This step is planning only. It does not build RAG, retrieval, embeddings, prompt
assembly, model calls, GPU runs, benchmark inference, or full scaled prompt
datasets.

## Checkpoints

| Checkpoint | Prompts per vertical | Total prompts | Purpose |
| --- | ---: | ---: | --- |
| Seed | 40 | 200 | Completed reviewed seed |
| 250 | 250 | 1,250 | QA-scale deterministic dataset |
| 1,000 | 1,000 | 5,000 | Pilot benchmark dataset |
| 2,000 | 2,000 | 10,000 | Full Phase 2A dataset before Phase 2B/RAG |

The next generation target is the 250-record checkpoint.

## KB Expansion Targets

| Vertical | KB target at 250 | KB target at 1,000 | KB target at 2,000 |
| --- | --- | --- | --- |
| Finance | 250-400 | 800-1,200 | 1,500-2,500 |
| Airline | 75-120 | 150-250 | 300-500 |
| Healthcare Admin | 75-120 | 150-250 | 300-500 |
| Research AI | 300-500 | 800-1,200 | 1,000-1,800 |
| Retail | 150-250 | 500-1,000 | 1,000-2,000 |

## Vertical Strategy

Finance should scale from the current SEC/XBRL corpus by expanding section
coverage across 10-K, 10-Q, and 8-K records. The prompt mix should preserve
single-filing, multi-filing, XBRL numeric, table, trend, risk, compare-company,
insufficient-evidence, and escalation tasks.

Airline should expand the deterministic ticket generator and synthetic/public
inspired policy KB while preserving booking, refunds, changes, baggage, partner
airline, accessibility, travel documents, loyalty, disruption, escalation, and
spam/fraud behaviors.

Healthcare Admin should expand synthetic administrative tickets and policy KB
while preserving the admin-only safety boundary. It should cover privacy,
identity verification, billing, scheduling, records, referrals, portal,
telehealth, and urgent clinical boundary handling.

Research AI can use the current 20-paper corpus for 250 prompts if section
coverage remains good. Before the 1,000 and 2,000 checkpoints, expand to about
40 papers and preserve method, results, limitations, comparison, citation,
inference optimization, context/RAG, agentic systems, efficient model, and
evaluation tasks.

Retail can use the current All_Beauty sample for 250 prompts. Before 1,000
prompts, expand to three categories. Before 2,000 prompts, expand to five
categories. It should continue using metadata, reviews, and synthetic benchmark
policy overlay with no raw user IDs.

## Gold/Eval Strategy

Every generated prompt requires one deterministic gold/eval record. Prompt IDs
and gold prompt IDs must align, answerable records must include evidence IDs,
and negative-status records must include explicit `must_not_include` constraints.

At 10,000 total prompts, all 10,000 prompts should have deterministic gold/eval
records.

Create a 1,000-record gold-review subset stratified across:

- vertical
- expected status
- task type
- output format
- difficulty
- context length bucket

Create a 300-record deep-review subset emphasizing:

- finance numeric and multi-document prompts
- Research AI method, results, and comparison prompts
- Retail low-rating and policy reasoning prompts
- Airline escalation and fraud prompts
- Healthcare privacy and safety-boundary prompts

## Status Targets

Global 10,000-prompt target:

- answer: 88%-92%
- escalation, insufficient evidence, safety boundary, spam, and out-of-scope:
  8%-12%

Per-vertical targets are defined in
`data/sources/phase2a_scaleup_plan.json`.

## Quality Gates

Before every checkpoint:

- run Phase 2A-7 cross-vertical data QA
- keep prompt/gold one-to-one alignment
- keep evidence coverage for answerable records
- preserve negative and boundary status coverage
- keep raw/generated source data local unless intentionally curated
- do not add RAG, retrieval, embeddings, prompt assembly, model calls, or
  inference to this Phase 2A scale-up step

## Command

```text
python scripts/phase2/plan_phase2a_scaleup.py --write-report
```

Generated local outputs:

- `data/generated/phase2a/phase2a_scaleup_plan_report.json`
- `data/generated/phase2a/phase2a_scaleup_matrix.csv`

## Next Step

Proceed to Phase 2A-9 250-record scale-up generator foundation after reviewing
the scale-up plan report.
