# Block B6R6 Research AI Quality Recovery Summary

Status: measured on June 20, 2026

B6R6 restored the Research AI quality floor for `model2_3b` /
`Qwen/Qwen2.5-3B-Instruct` on the remote RTX 3070 vLLM path.

## Targeted Replay

- Failed Research AI rows from B6R4: 20
- Baseline lock: 80% evidence match and 80% groundedness
- Selected strategy: `answer_skeleton`
- Targeted JSON/contract validity: 100% / 100%
- Targeted evidence match/groundedness: 90% / 90%
- Safety violations: 0
- Truncation: 0%

Rejected strategies:

- `b6r4_original_behavior`: 0% evidence/groundedness
- `b6r2_best_contract`: 65% evidence/groundedness
- `evidence_whitelist`: 25% evidence/groundedness
- `output_budget_384`: 0% evidence/groundedness

## Full 500 Gate

- Requests completed: 500/500
- JSON validity: 98.2%
- Contract validity: 97.8%
- Evidence match: 97.0%
- Groundedness: 96.6%
- Safety violations: 0
- Truncation: 1.8%

Per-vertical evidence/groundedness:

- Airline: 93% / 93%
- Healthcare Admin: 100% / 100%
- Retail: 100% / 100%
- Finance: 96% / 96%
- Research AI: 96% / 94%

Decision:

```text
B6R6_QUALITY_READY
```

## Readiness

- Full-run readiness: `READY`
- Benchmark execution readiness: `READY`
- Deployability readiness: `READY`
- 1,000-prompt terminal baseline allowed: `true`
- RunPod cost claims: still blocked until price and throughput multipliers are
  registered

Next allowed block:

```text
CONTROLLED_1000_PROMPT_TERMINAL_BASELINE
```

Keep concurrency, SGLang, mm4, RunPod, 2,000-prompt, and 10,000-prompt runs
blocked until the 1,000-prompt terminal baseline is measured and reviewed.

