# Controlled Final Simulation Summary

## What Ran

- Implemented the real `--run-full` execution path for
  `scripts/phase4/run_controlled_final_simulation.py`.
- Ran the final 25-config, 10,000-request controlled final-simulation baseline.
- Kept the frozen matrix unchanged and did not apply optimizations.
- Captured raw results, processed evaluation reports, SLO comparison, engine,
  memory-mode, concurrency, API-vs-self-hosted, model, cost, artifact-sync, GPU
  telemetry, and plotting-ready reports.

## Outcome

All gates passed before full execution:

- vLLM `model3_7b`: smoke-ready at `http://localhost:8000/v1`.
- SGLang `model3_7b`: smoke-ready at `http://localhost:30000/v1`.
- API `model6_gated`: smoke-ready with `.env` credentials.
- MM4: bounded LangGraph runner smoke-ready.

## Request Counts

- Planned requests: 10,000.
- Attempted requests: 10,000.
- Completed requests: 10,000.
- Failed requests: 0.
- Completed configs: 25.
- Failed configs: 0.

## Results

- Overall mean E2E latency: 2,359.66 ms.
- Overall mean TTFT: 205.04 ms.
- Overall mean TPOT: 14.01 ms.
- Overall mean tokens/sec: 72.99.
- vLLM beat SGLang on self-hosted latency and throughput in this baseline.
- Self-hosted concurrency 16 beat concurrency 32 on latency and throughput.
- API provider rows were faster than the self-hosted aggregate, with separate
  provider token cost and no GPU telemetry.
- Artifact sync and backup verification passed.
- Total measured cost estimate: `$0.873843`.

## Decision

The controlled final baseline is complete, but the final larger/deployability
experiment is not allowed yet. The run completed operationally, but quality SLOs
failed: JSON validity, generation-contract validity, evidence match, and
groundedness failed for every configuration. The next phase should focus on
generation-contract JSON repair, evidence alignment, groundedness repair, and
targeted MM4 quality repair before any larger run.

Exact SGLang startup command:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90 --context-length 4096 --max-running-requests 32 --chunked-prefill-size 8192
```
