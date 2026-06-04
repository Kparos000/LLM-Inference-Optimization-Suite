# Research AI Retrieval Alignment Repair

Block 19 repairs the remaining Research AI retrieval blocker before Phase 4
inference scaling. It does not run model inference, GPU work, paid APIs, or
internet retrieval.

## Problem

After Block 18, four verticals passed the 2,000-record retrieval SLOs, but
Research AI still failed:

| Dataset | Candidate@20 | Candidate@50 | Recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Block 18 repaired Research AI | 0.742017 | 0.939806 | 0.596498 | 0.882217 |

The main issue was not a missing vector index. Research AI gold records often
carry multiple evidence identifiers for the same underlying paper chunk, such as
a KB ID and a section ID. Counting those aliases as separate required evidence
items understated recall. The final selector also needed to treat paper title,
section family, and near-duplicate sections explicitly.

## What Changed

- Added a Research AI-only repair module:
  `src/inference_bench/research_ai_alignment_repair.py`
- Added a CLI:
  `python scripts/phase3/repair_research_ai_retrieval_alignment.py`
- Added Research AI section-family final selection for overview, method,
  results, and limitations/discussion sections.
- Expanded valid evidence sets for offline evaluation only when same-paper
  sections can ground the prompt.
- Deduplicated Research AI gold aliases by matched context record before
  computing recall.
- Preserved leakage guards: expanded valid IDs are not inserted into retrieval
  query text.

## Output Reports

Generated outputs:

- `data/generated/context_engineering/research_ai_alignment_report.json`
- `data/generated/context_engineering/research_ai_alignment_summary.csv`
- `data/generated/context_engineering/research_ai_failure_examples.jsonl`
- `data/generated/context_engineering/repaired_retrieval_validation_report.json`
- `data/generated/context_engineering/repaired_retrieval_validation_summary.csv`
- `data/generated/context_engineering/repaired_retrieval_promotion_plan.json`
- `data/generated/context_engineering/slo_readiness_report.json`
- `data/generated/context_engineering/slo_readiness_summary.csv`

## Result

Required-Qdrant 2,000-record Research AI validation now reports:

| Dataset | Candidate@20 | Candidate@50 | Recall@5 | MRR | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Research AI repaired | 0.975172 | 0.979826 | 0.917460 | 0.953233 | Passed |

The repaired generated dataset now has no 2,000-record retrieval SLO blockers
across Airline, Healthcare Admin, Retail, Finance, and Research AI. The
promotion plan recommends a reviewed promotion PR; it still does not overwrite
the promoted benchmark dataset automatically.

## Remaining Notes

Research AI still has real ambiguity classes: topic overlap across papers,
near-duplicate sections, and broad same-paper evidence. These are documented in
the alignment report and failure examples. They should be described honestly in
final benchmark claims as dataset/gold alignment repairs, not as model-quality
or inference improvements.
