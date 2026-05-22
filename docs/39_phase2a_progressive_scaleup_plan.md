# Phase 2A Progressive Scale-Up Plan

Phase 2A-8 defines the control framework for growing the reviewed seed datasets
from 40 prompts per vertical toward larger deterministic benchmark tiers.

The project is no longer capped at 10,000 total prompts. The near-term main
target is 10,000 total prompts, or 2,000 prompts per vertical. The plan also
scaffolds a 20,000 total prompt GPU stress tier and a 25,000 total prompt
expanded maximum capacity at 5,000 prompts per vertical.

This step is planning only. It does not build RAG, retrieval, embeddings, prompt
assembly, model calls, GPU runs, benchmark inference, or full scaled prompt
datasets.

## Checkpoints

| Checkpoint | Prompts per vertical | Total prompts | Purpose |
| --- | ---: | ---: | --- |
| Seed | 40 | 200 | Completed reviewed seed |
| 250 | 250 | 1,250 | QA-scale deterministic dataset |
| 1,000 | 1,000 | 5,000 | Pilot benchmark dataset |
| 2,000 | 2,000 | 10,000 | Near-term main Phase 2A dataset before Phase 2B/RAG |
| 4,000 | 4,000 | 20,000 | GPU-backed stress benchmark tier |
| 5,000 | 5,000 | 25,000 | Maximum expanded benchmark capacity |

The next generation target remains the 250-record checkpoint. We still progress
checkpoint by checkpoint and do not generate 20,000 or 25,000 prompts until QA
at smaller checkpoints passes.

## KB Expansion Targets

| Vertical | KB target at 250 | KB target at 1,000 | KB target at 2,000 | KB target at 4,000 | KB target at 5,000 |
| --- | --- | --- | --- | --- | --- |
| Finance | 250-400 | 800-1,200 | 1,500-2,500 | 2,500-4,500 | 3,500-6,000 |
| Airline | 75-120 | 150-250 | 300-500 | 600-900 | 800-1,200 |
| Healthcare Admin | 75-120 | 150-250 | 300-500 | 600-900 | 800-1,200 |
| Research AI | 300-500 | 800-1,200 | 1,000-1,800 | 1,600-2,800 | 2,000-3,500 |
| Retail | 150-250 | 500-1,000 | 1,000-2,000 | 2,000-4,000 | 2,500-5,000 |

## Vertical Strategy

Finance should scale from the current 8-company SEC/XBRL corpus first by
expanding section coverage across 10-K, 10-Q, and 8-K records. Add more
companies or filing years only if evidence reuse becomes repetitive. The prompt
mix should preserve single-filing, multi-filing, XBRL numeric, table, trend,
risk, compare-company, insufficient-evidence, and escalation tasks.

Airline should expand the deterministic synthetic ticket and synthetic/public
inspired policy generators with deterministic variations while preserving
booking, refunds, changes, baggage, partner airline, accessibility, travel
documents, loyalty, disruption, escalation, and spam/fraud behaviors.

Healthcare Admin should expand synthetic administrative tickets and policy KB
while preserving the admin-only safety boundary. It should cover privacy,
identity verification, billing, scheduling, records, referrals, portal,
telehealth, and urgent clinical boundary handling.

Research AI can use the current 20-paper corpus for 250 prompts if section
coverage remains good. Before the 1,000 checkpoint, expand to about 40 papers.
For the 4,000 and 5,000 prompt tiers, expand to roughly 60-80 papers while
preserving method, results, limitations, comparison, citation, inference
optimization, context/RAG, agentic systems, efficient model, and evaluation
tasks.

Retail can use the current All_Beauty sample for 250 prompts. Before 1,000+
scale, expand beyond All_Beauty to multiple categories. It should continue using
metadata, reviews, and synthetic benchmark policy overlay with no raw user IDs.

## Gold/Eval Strategy

Every generated prompt requires one deterministic gold/eval record. Prompt IDs
and gold prompt IDs must align, answerable records must include evidence IDs,
and negative-status records must include explicit `must_not_include`
constraints.

Gold review targets by tier:

| Checkpoint | Total prompts | Gold-review subset | Deep-review subset |
| --- | ---: | ---: | ---: |
| 2,000 per vertical | 10,000 | 1,000 | 300 |
| 4,000 per vertical | 20,000 | 1,500 | 500 |
| 5,000 per vertical | 25,000 | 2,000 | 750 |

Review subsets should be stratified across:

- vertical
- expected status
- task type
- output format
- difficulty
- context length bucket
- source document type

Deep-review subsets should emphasize:

- finance numeric and multi-document prompts
- Research AI method, results, and comparison prompts
- Retail low-rating and policy reasoning prompts
- Airline escalation and fraud prompts
- Healthcare privacy and safety-boundary prompts

## Status Targets

Global target at every checkpoint:

- answer: 88%-92%
- escalation, insufficient evidence, safety boundary, spam, and out-of-scope:
  8%-12%

Every checkpoint must include answer records, structured JSON records,
citation/evidence records, and at least one negative or boundary status class
where appropriate. No vertical may be 100% answer-only.

Per-vertical targets are defined in
`data/sources/phase2a_scaleup_plan.json`.

## Quality Gates

Before every checkpoint:

- run Phase 2A-7 cross-vertical data QA
- keep prompt/gold one-to-one alignment
- keep evidence coverage for answerable records
- preserve negative and boundary status coverage
- preserve structured JSON, citation/evidence lookup, and non-answer-only mixes
- keep raw/generated source data local unless intentionally curated
- do not add RAG, retrieval, embeddings, prompt assembly, model calls, GPU
  benchmark runs, or inference to Phase 2A scale-up planning

## Command

```text
python scripts/phase2/plan_phase2a_scaleup.py --write-report
```

Generated local outputs:

- `data/generated/phase2a/phase2a_scaleup_plan_report.json`
- `data/generated/phase2a/phase2a_scaleup_matrix.csv`

## Next Step

Proceed with Phase 2A-9 generator expansion, starting at 250 prompts per
vertical and scaffolding toward the 2,000, 4,000, and 5,000 prompts-per-vertical
checkpoints after QA passes at each smaller tier.
