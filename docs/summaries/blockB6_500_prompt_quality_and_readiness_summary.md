# Block B6 500-Prompt Quality And Readiness Summary

Status: measured on June 15, 2026

B6 ran a controlled 500-prompt scale gate at concurrency one with
Qwen2.5-1.5B served by vLLM on the remote RTX 3070. It used
`mm2_hybrid_top5`, streaming, temperature zero, a 160-token output cap, and
the B5 safety/planning/multi-evidence repair path.

## Preflight

- Runner input rows: 500
- Per vertical: 100
- All required gold evidence present in E1-E5: 500/500
- Finance all-required-gold-present: 100/100
- Partial/absent/unrecoverable rows: 0
- Canonical IDs exposed to the model: 0

## Quality

- Requests completed: 500/500
- JSON validity: 95.4%
- Contract validity: 94.8%
- Evidence match: 91.2%
- Groundedness: 90.8%
- Safety violations: 0
- Truncation: 4.6%
- Bounded retry attempts: 99
- Lexical guard repairs: 10

Decision:

```text
B6_QUALITY_IMPROVED_BUT_BLOCKED
```

The run cleared aggregate evidence and groundedness, but failed JSON,
contract, truncation, minimum vertical evidence, and minimum vertical
groundedness gates.

## Vertical Blocker

Research AI is the blocking vertical:

- JSON validity: 82%
- Contract validity: 80%
- Evidence match: 76%
- Groundedness: 74%
- Truncation: 18%

Finance is no longer the blocker in B6:

- JSON validity: 100%
- Contract validity: 100%
- Evidence match: 95%
- Groundedness: 95%
- Safety violations: 0
- Truncation: 0%

## Runtime

- Wall time: 871.876 seconds
- Mean TTFT: 141.543 ms
- Mean TPOT: 11.489 ms
- Mean E2E latency: 1,741.355 ms
- p95 E2E latency: 5,021.188 ms
- Mean throughput: 989.647 tokens/s
- Mean/peak GPU utilization: 81.33% / 100%
- Mean/peak GPU memory: 6,524.17 / 6,760 MB

At the measured concurrency-one rate, 10,000 prompts project to 4.844 RTX 3070
hours. RunPod cost projections remain blocked because hourly prices and
throughput multipliers are unset.

## Readiness

The full-run readiness audit status is:

```text
NOT_READY
```

The repository has the required dataset, context, runner safety, telemetry,
and SLO/diagnosis controls for continued controlled work. It is not ready for
larger or concurrent runs because B6 did not pass quality and RunPod cost
inputs are still missing.

## Next Block

Recommended next block:

```text
B6R1_RESEARCH_AI_TRUNCATION_AND_CONTRACT_REPAIR
```

Freeze B6 artifacts. Replay only failed, truncated, or invalid Research AI
rows first. Do not change gold data, evaluator semantics, or promoted
retrieval. Then rerun the same 500-row B6 gate before any concurrency sweep,
SGLang comparison, mm4 comparison, RunPod run, or 2,000/10,000-prompt run.
