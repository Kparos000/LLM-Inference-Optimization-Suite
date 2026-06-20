# B6R5 Finance And Research Quality Repair

Status: measured on June 20, 2026

B6R5 diagnosed and replayed the Finance and Research AI rows that blocked the
full B6R4 `model2_3b` 500-prompt gate. It used
`Qwen/Qwen2.5-3B-Instruct` through vLLM on the remote RTX 3070 with
`mm2_hybrid_top5`, streaming, temperature zero, and concurrency one.

No evaluator semantics, gold data, promoted retrieval source, B6R4 artifacts,
or workload-specific model routing changed.

## Failure Set

B6R5 selected the 40 failed B6R4 rows from the two blocking verticals:

- Finance: 20 rows
- Research AI: 20 rows

The replay input is:

```text
data/generated/phase4/b6r5_finance_research_failed_replay_input.jsonl
```

The primary root-cause labels were:

| Root Cause | Count |
| --- | ---: |
| `model_instruction_following_failure` | 40 |
| `likely_model_capacity_limitation` | 40 |
| `partial_multi_evidence_citation` | 39 |
| `wrong_evidence_selected` | 28 |
| `finance_metric_ambiguity` | 20 |
| `numeric_table_extraction_issue` | 20 |
| `research_synthesis_ambiguity` | 20 |
| `answer_semantically_underdeveloped` | 2 |
| `evidence_present_but_not_cited` | 1 |
| `json_contract_issue` | 1 |
| `truncation_issue` | 1 |

Finance failures were dominated by metric/period/table evidence selection.
Research AI failures were dominated by synthesis and multi-evidence citation
selection. In both verticals the gold evidence was in the rendered E1-E5
context for the targeted rows, so B6R5 did not diagnose a promoted retrieval
or gold-data failure.

## Strategies

B6R5 tested three surgical strategies over the same 40 rows:

| Strategy | JSON | Contract | Evidence | Grounded | Finance Evidence/Grounded | Research AI Evidence/Grounded | Safety | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_selection_preplan` | 100% | 100% | 80.0% | 80.0% | 90% / 90% | 70% / 70% | 0 | 0% |
| `vertical_specific_citation_reminder` | 100% | 100% | 17.5% | 17.5% | 15% / 15% | 20% / 20% | 0 | 0% |
| `output_budget_320` | 100% | 100% | 2.5% | 2.5% | 5% / 5% | 0% / 0% | 0 | 0% |

The selected strategy is:

```text
evidence_selection_preplan
```

Selection reason:

```text
best_evidence_groundedness_improvement_without_targeted_pass
```

## Decision

```text
B6R5_QUALITY_CAVEATED
```

The selected strategy repaired Finance failed-row quality above the 85%
targeted threshold, reaching 90% evidence match and 90% groundedness. Research
AI remained below threshold at 70% evidence match and 70% groundedness.

Because no targeted strategy passed both vertical thresholds, B6R5 did not
trigger a full 500-row rerun.

## Readiness

The refreshed full-run readiness audit reports:

```text
READY_WITH_QUALITY_CAVEAT
```

Detailed readiness is split:

- deployability readiness: `NOT_READY`
- benchmark execution readiness: `READY_WITH_QUALITY_CAVEAT`
- 1,000-prompt terminal baseline allowed: `true`

This allows a controlled 1,000-prompt terminal baseline as caveated benchmark
evidence only. It does not allow a deployability claim, concurrency sweep,
SGLang comparison, mm4 comparison, RunPod execution, 2,000-prompt run, or
10,000-prompt run.

## Artifacts

- `results/processed/b6r5_finance_research_failure_audit_report.json`
- `results/processed/b6r5_finance_research_failure_audit_summary.csv`
- `results/raw/b6r5_finance_research_targeted_replay_results.jsonl`
- `results/processed/b6r5_finance_research_targeted_replay_report.json`
- `results/processed/b6r5_finance_research_targeted_replay_summary.csv`
- `results/processed/b6r5_strategy_comparison.json`
- `results/processed/b6r4_vs_b6r5_comparison.json`
- `results/processed/b6_full_run_readiness_report.json`
- `results/processed/b6_full_run_readiness_summary.csv`

## Next Step

Run only a controlled 1,000-prompt terminal baseline if the objective is a
caveated benchmark capacity measurement. Keep deployability blocked until a
selected model path clears the Finance and Research AI vertical quality floors
without caveat.
