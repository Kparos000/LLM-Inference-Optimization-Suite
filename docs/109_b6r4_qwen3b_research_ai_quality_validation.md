# B6R4 Qwen2.5-3B Research AI Quality Validation

Status: measured on June 20, 2026

B6R4 tested whether the corrected `model2_3b` alias, resolved as
`Qwen/Qwen2.5-3B-Instruct`, improves the Research AI blocker before any larger
or concurrent benchmark run.

The run used vLLM on the remote RTX 3070, `mm2_hybrid_top5`, streaming,
temperature zero, concurrency one, and the frozen B6/B6R1 Research AI failed
rows. No evaluator semantics, gold data, promoted retrieval source, B6/B6R1,
B6R2, or B6R3 artifacts changed.

## Targeted Replay

Input:

```text
data/generated/phase4/b6r1_research_ai_failed_replay_input.jsonl
```

Output artifacts:

- `results/raw/b6r4_model2_3b_research_ai_targeted_results.jsonl`
- `results/processed/b6r4_model2_3b_research_ai_targeted_report.json`
- `results/processed/b6r4_model2_3b_research_ai_targeted_summary.csv`
- `results/processed/b6r4_research_ai_model_capacity_comparison.json`

Decision:

```text
B6R4_TARGETED_MODEL2_3B_PASSED
```

Metrics:

- JSON validity: 100%
- contract validity: 100%
- evidence match: 88.46%
- groundedness: 88.46%
- safety violations: 0
- truncation: 0%
- mean TTFT: 484.168 ms
- mean TPOT: 17.062 ms
- mean E2E latency: 1,927.164 ms
- mean throughput: 725.960 tokens/s

The targeted gate passed, so the full frozen 500-row replay was triggered.

## Full 500 Gate

Output artifacts:

- `results/raw/b6r4_model2_3b_500_results.jsonl`
- `results/processed/b6r4_model2_3b_500_eval_report.json`
- `results/processed/b6r4_model2_3b_500_eval_summary.csv`
- `results/processed/b6_vs_b6r4_model2_3b_comparison.json`

Decision:

```text
B6R4_MODEL2_3B_500_BLOCKED
```

Overall metrics:

- requests completed: 500/500
- JSON validity: 98.4%
- contract validity: 98.4%
- evidence match: 90.6%
- groundedness: 90.6%
- safety violations: 0
- truncation: 1.6%
- mean TTFT: 690.308 ms
- mean TPOT: 17.249 ms
- mean E2E latency: 2,702.659 ms
- mean throughput: 623.403 tokens/s

The aggregate gate cleared JSON, contract, evidence, groundedness, safety, and
truncation thresholds. It failed the minimum vertical evidence and groundedness
thresholds.

Per-vertical evidence match and groundedness:

| Vertical | Evidence Match | Groundedness | Truncation |
| --- | ---: | ---: | ---: |
| Airline | 93% | 93% | 7% |
| Healthcare Admin | 100% | 100% | 0% |
| Retail | 100% | 100% | 0% |
| Finance | 80% | 80% | 1% |
| Research AI | 80% | 80% | 0% |

Finance and Research AI are the blocking verticals at 80%, below the 85%
minimum vertical threshold.

## Comparison

| Run | Scope | JSON | Contract | Evidence | Grounded | Truncation | Safety |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B6 Qwen2.5-1.5B | Research AI full vertical, 100 rows | 82.00% | 80.00% | 76.00% | 74.00% | 18.00% | 0 |
| B6R2 best Qwen2.5-1.5B contract | 26-row targeted replay | 96.15% | 96.15% | 80.77% | 80.77% | 0.00% | 0 |
| B6R3 model6 Llama 3.1 8B API | 26-row targeted replay | 100.00% | 100.00% | 96.15% | 96.15% | 0.00% | 0 |
| B6R4 Qwen2.5-3B | 26-row targeted replay | 100.00% | 100.00% | 88.46% | 88.46% | 0.00% | 0 |
| B6R4 Qwen2.5-3B | full 500 rows | 98.40% | 98.40% | 90.60% | 90.60% | 1.60% | 0 |

Qwen2.5-3B materially improves the frozen Research AI targeted set relative
to the best Qwen2.5-1.5B contract result and passes the targeted gate. It does
not clear the full 500 gate because Finance and Research AI remain below the
minimum vertical evidence and groundedness thresholds.

## Readiness

The full-run readiness status remains:

```text
NOT_READY
```

A 1,000-prompt terminal run is not allowed. Concurrency sweeps, SGLang/mm4
comparisons, RunPod execution, 2,000-prompt runs, and 10,000-prompt runs remain
blocked until a selected model path passes the full frozen 500-row gate and
RunPod price/calibration inputs are configured.

## Next Block

Recommended next block:

```text
B6R5_MODEL2_3B_FINANCE_RESEARCH_VERTICAL_REPAIR
```

Freeze B6R4 artifacts. Do not change evaluator semantics, gold data, or
promoted retrieval. Diagnose the Finance and Research AI full-500 failures on
the model2_3b run, then decide whether the next controlled comparison should
repair citation selection for Qwen2.5-3B, test model3_7b feasibility, or run a
full 500 API-provider model6 gate.
