# Block B6R4 Qwen2.5-3B Research AI Quality Summary

Status: measured on June 20, 2026

B6R4 replayed the frozen 26 Research AI failed rows on `model2_3b`
(`Qwen/Qwen2.5-3B-Instruct`) through vLLM on the remote RTX 3070.

## Targeted Result

Decision:

```text
B6R4_TARGETED_MODEL2_3B_PASSED
```

- JSON validity: 100%
- Contract validity: 100%
- Evidence match: 88.46%
- Groundedness: 88.46%
- Safety violations: 0
- Truncation: 0%

The targeted pass triggered the full frozen 500-row run.

## Full 500 Result

Decision:

```text
B6R4_MODEL2_3B_500_BLOCKED
```

- JSON validity: 98.4%
- Contract validity: 98.4%
- Evidence match: 90.6%
- Groundedness: 90.6%
- Safety violations: 0
- Truncation: 1.6%

The aggregate metrics passed, but the full gate failed because Finance and
Research AI each reached only 80% evidence match and 80% groundedness, below
the 85% minimum vertical threshold.

## Readiness

The full-run readiness audit remains `NOT_READY`. A 1,000-prompt run is not
allowed. RunPod remains blocked until the full 500 gate passes and hourly
price/calibration inputs are configured.

## Next Block

```text
B6R5_MODEL2_3B_FINANCE_RESEARCH_VERTICAL_REPAIR
```

Freeze B6R4 artifacts and diagnose the Finance and Research AI full-500
failures without modifying gold data, evaluator semantics, or promoted
retrieval.
