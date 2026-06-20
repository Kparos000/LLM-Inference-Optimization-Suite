# Block B6R5 Finance And Research Quality Repair Summary

Status: measured on June 20, 2026

B6R5 replayed the 40 Finance and Research AI rows that blocked the full B6R4
`model2_3b` 500-prompt gate. It used Qwen2.5-3B through vLLM on the remote RTX
3070. No evaluator, gold data, promoted retrieval source, B6R4 artifact, or
workload-specific model routing changed.

## Result

Decision:

```text
B6R5_QUALITY_CAVEATED
```

- Failed replay rows: 40
- Finance rows: 20
- Research AI rows: 20
- Selected strategy: `evidence_selection_preplan`
- Full 500 rerun triggered: no

## Strategy Comparison

- `evidence_selection_preplan`: 100% JSON, 100% contract, 80% evidence match,
  80% groundedness, zero safety violations, zero truncation.
- `vertical_specific_citation_reminder`: 100% JSON, 100% contract, 17.5%
  evidence match, 17.5% groundedness, zero safety violations, zero truncation.
- `output_budget_320`: 100% JSON, 100% contract, 2.5% evidence match, 2.5%
  groundedness, zero safety violations, zero truncation.

The selected strategy repaired Finance failed-row quality to 90% evidence match
and 90% groundedness. Research AI remained at 70% evidence match and 70%
groundedness, so the targeted gate did not pass.

## Readiness

The refreshed full-run readiness audit is:

```text
READY_WITH_QUALITY_CAVEAT
```

Deployability remains `NOT_READY`. Benchmark execution is
`READY_WITH_QUALITY_CAVEAT`, and the deterministic audit allows a controlled
1,000-prompt terminal baseline as caveated benchmark evidence only. Concurrency,
SGLang, mm4, RunPod, 2,000-prompt, and 10,000-prompt runs remain blocked.
