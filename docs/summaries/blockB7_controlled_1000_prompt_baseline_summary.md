# Block B7 Controlled 1,000-Prompt Baseline Summary

Status: measured on June 21, 2026

B7 ran the first 1,000-prompt post-quality-gate baseline for `model2_3b` /
Qwen2.5-3B on the remote RTX 3070 vLLM path at concurrency one.

## Preflight

- Runner input rows: 1,000
- Per vertical: 200
- All required gold evidence present in E1-E5: 1,000/1,000
- Canonical IDs exposed to the model: 0
- Artifact sync dry run: passed
- Checkpoint/resume and manifest: enabled

## Result

The run produced 1,000 unique raw rows but only 663 successful requests. vLLM
failed at Finance prompt 17 with an EngineCore fatal CUDA/CUBLAS error, after
which 337 rows recorded request failures. The run resumed from partial raw
output and preserved the failed rows.

Decision:

```text
B7_CONTROLLED_1000_BASELINE_BLOCKED
```

## Quality

- JSON validity: 64.8%
- Contract validity: 64.8%
- Evidence match: 64.3%
- Groundedness: 64.3%
- Safety violations: 0
- Truncation: 1.2%

Per-vertical evidence match / groundedness:

- Airline: 93.0% / 93.0%
- Healthcare Admin: 100.0% / 100.0%
- Retail: 99.5% / 99.5%
- Finance: 7.5% / 7.5%
- Research AI: 21.5% / 21.5%

Finance and Research AI are dominated by serving failures after the vLLM engine
died, so B7 is an operational blocker rather than a clean model-quality
comparison.

## Runtime

- Accumulated request time: 1,542.353 seconds
- Mean TTFT: 429.823 ms
- Mean TPOT: 17.352 ms
- Mean E2E latency: 2,291.242 ms
- p95 E2E latency: 4,233.441 ms
- Requests/sec: 0.648
- Aggregate tokens/sec: 607.781

Projection from measured throughput:

- 2,000 prompts: 0.857 RTX 3070 hours
- 10,000 prompts: 4.284 RTX 3070 hours
- 40,000-prompt selected matrix: 17.137 RTX 3070 hours

## Telemetry And Sync

- GPU telemetry samples: 529
- Mean/peak GPU utilization: 84.45% / 100%
- Mean/peak memory: 7,381.60 / 7,770 MB
- Artifact backup completeness score: 1.0

## Next Block

Recommended next block:

```text
B7R1_VLLM_CUDA_FAILURE_ISOLATION
```

Do not run an API load probe, concurrency sweep, SGLang comparison, mm4
comparison, RunPod run, 2,000-prompt run, or 10,000-prompt run until the B7
serving failure is isolated and the same 1,000-row B7 input is rerun cleanly.
