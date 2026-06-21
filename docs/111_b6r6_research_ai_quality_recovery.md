# B6R6 Research AI Quality Recovery

Status: measured on June 20, 2026

B6R6 restored the Research AI quality floor for `model2_3b`
(`Qwen/Qwen2.5-3B-Instruct`) on the remote RTX 3070 vLLM path. It used
`mm2_hybrid_top5`, streaming, temperature zero, concurrency one, the B6R4 full
500-row output as the baseline lock, and only the 20 failed B6R4 Research AI
rows for targeted strategy selection.

No evaluator semantics, SLO thresholds, gold data, promoted retrieval source,
workload-specific model routing, RunPod run, or concurrency increase was used.

## Baseline Lock

- B6R4 Research AI full-vertical floor: 80% evidence match and 80%
  groundedness.
- B6R4 failed-row replay count: 20 Research AI rows.
- Effective targeted floor: 80% evidence match and 80% groundedness.

Any strategy below that floor was automatically rejected.

## Failure Audit

Primary causes on the 20 failed Research AI rows:

| Root cause | Count |
| --- | ---: |
| `partial_multi_evidence_citation` | 20 |
| `model_capacity_limitation` | 20 |
| `prompt_context_mismatch` | 20 |
| `wrong_evidence_selected` | 9 |
| `synthesis_under_answer` | 2 |

The failures were not diagnosed as promoted retrieval, gold-data, or evaluator
failures.

## Targeted Strategies

| Strategy | JSON | Contract | Evidence | Grounded | Safety | Truncation | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `b6r4_original_behavior` | 100% | 100% | 0% | 0% | 0 | 0% | rejected |
| `b6r2_best_contract` | 100% | 70% | 65% | 65% | 0 | 0% | rejected |
| `evidence_whitelist` | 100% | 100% | 25% | 25% | 0 | 0% | rejected |
| `answer_skeleton` | 100% | 100% | 90% | 90% | 0 | 0% | selected |
| `output_budget_384` | 100% | 100% | 0% | 0% | 0 | 0% | rejected |

Selected strategy:

```text
answer_skeleton
```

Decision:

```text
B6R6_TARGETED_READY
```

The selected strategy exceeded the preferred 85% Research AI targeted floor,
so the full frozen 500-row rerun was triggered.

## Full 500 Gate

The full rerun used:

- Finance: B6R5 `evidence_selection_preplan`.
- Research AI: B6R6 `answer_skeleton`.
- Other verticals: B6R4-style baseline path with the same bounded retry/guard
  behavior.
- Model: `model2_3b`.
- Runtime: vLLM on remote RTX 3070.
- Concurrency: 1.

Overall result:

- Requests completed: 500/500
- JSON validity: 98.2%
- Contract validity: 97.8%
- Evidence match: 97.0%
- Groundedness: 96.6%
- Safety violations: 0
- Truncation: 1.8%
- Mean TTFT: 402.170 ms
- Mean TPOT: 17.161 ms
- Mean E2E latency: 2,160.616 ms
- Mean throughput: 723.256 tokens/s

Decision:

```text
B6R6_QUALITY_READY
```

## Per-Vertical Quality

| Vertical | JSON | Contract | Evidence | Grounded | Safety | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Airline | 93% | 93% | 93% | 93% | 0 | 7% |
| Healthcare Admin | 100% | 100% | 100% | 100% | 0 | 0% |
| Retail | 100% | 100% | 100% | 100% | 0 | 0% |
| Finance | 98% | 98% | 96% | 96% | 0 | 2% |
| Research AI | 100% | 98% | 96% | 94% | 0 | 0% |

Research AI recovered from B6R5's 70% failed-row evidence/groundedness to 90%
on the targeted failed rows, then reached 96% evidence match and 94%
groundedness on the full Research AI vertical.

## Readiness

The refreshed readiness audit reports:

```text
READY
```

Detailed readiness:

- deployability readiness: `READY`;
- benchmark execution readiness: `READY`;
- 1,000-prompt terminal baseline allowed: `true`.

RunPod cost claims remain blocked until hourly prices and throughput
multipliers are registered. The next allowed run is a controlled 1,000-prompt
terminal baseline at concurrency one. Do not run a concurrency sweep, SGLang
comparison, mm4 comparison, RunPod run, 2,000-prompt run, or 10,000-prompt run
from B6R6 alone.

## Artifacts

- `data/generated/phase4/b6r6_research_ai_failed_replay_input.jsonl`
- `results/processed/b6r6_research_ai_failure_audit_report.json`
- `results/processed/b6r6_research_ai_failure_audit_summary.csv`
- `results/raw/b6r6_research_ai_targeted_replay_results.jsonl`
- `results/processed/b6r6_research_ai_targeted_replay_report.json`
- `results/processed/b6r6_research_ai_targeted_replay_summary.csv`
- `results/processed/b6r6_strategy_comparison.json`
- `results/raw/b6r6_model2_3b_500_results.jsonl`
- `results/processed/b6r6_model2_3b_500_eval_report.json`
- `results/processed/b6r6_model2_3b_500_eval_summary.csv`
- `results/processed/b6_full_run_readiness_report.json`
- `results/processed/b6_full_run_readiness_summary.csv`

