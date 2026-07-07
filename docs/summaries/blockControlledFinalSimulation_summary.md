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

- Overall mean E2E latency: 1,911.71 ms.
- Overall mean TTFT: 462.48 ms.
- Overall mean TPOT: 46.89 ms.
- Overall mean tokens/sec: 487.61.
- vLLM beat SGLang on self-hosted latency and throughput in this baseline.
- Self-hosted concurrency 16 beat concurrency 32 on latency and throughput.
- API provider rows were faster than the self-hosted aggregate, with separate
  provider token cost and no GPU telemetry.
- Artifact sync and backup verification passed.
- Total measured cost estimate: `$0.793868`.

## Decision

The repaired controlled final baseline is complete. The SLO comparison reported
zero failed SLO fields across 25 configs, while aggregate raw-output evaluation
still shows contract, safety, evidence, and groundedness bottlenecks: 99.92%
JSON validity, 81.41% generation-contract validity, 61.92% evidence match,
60.26% groundedness, and 97 safety findings. The optimization phase can begin
from this baseline and should focus on final-answer contract normalization,
safety wording, evidence selection, groundedness, and concurrency-32
self-hosted efficiency.

Exact SGLang startup command:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90 --context-length 4096 --max-running-requests 32 --chunked-prefill-size 8192
```
