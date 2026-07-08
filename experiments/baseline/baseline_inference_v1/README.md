# Baseline_Inference_V1 Archive

Status: completed validation/baseline run from July 8, 2026.

This archive records the committed metadata and processed evidence for
`Baseline_Inference_V1`. This run was originally mislabeled as
`Main_Inference_V1`, but it was not the official Main_Inference run because it
covered 10,000 total requests across the 25-config matrix rather than 10,000
prompts per configuration.

It does not commit raw full results, telemetry, backups, or temporary
checkpoints. Checksums for those local artifacts are preserved in
`checksums/SHA256SUMS`.

## Run

- Run ID: `baseline_inference_v1`
- Matrix: 25 configs, 400 requests per config, 10,000 total requests
- Prompt sample: 80 prompts per vertical per config
- Scope: validation/baseline run, not the official 250,000-request
  `Main_Inference_V1`
- Self-hosted track: `model3_7b` / Qwen2.5-7B-Instruct on A100 SXM 80GB,
  vLLM and SGLang, mm0-mm4, concurrency 16 and 32
- API track: `model6_gated` / Llama 3.1 8B through API provider route, mm0-mm4,
  concurrency 4
- Started: `2026-07-08T15:45:53.937833+00:00`
- Completed: `2026-07-08T16:16:35.049544+00:00`
- Runtime: 1,900.585 seconds
- Requests: 10,000 attempted, 10,000 completed, 0 failed

## Results

- JSON validity: 99.93%
- Contract validity: 81.58%
- Evidence match: 62.26%
- Groundedness: 60.77%
- Safety findings: 96
- Mean TTFT: 428.389 ms
- Mean TPOT: 49.073 ms
- Mean E2E latency: 1,946.526 ms
- p95 / p99 E2E latency: 3,702.524 / 4,621.560 ms
- Mean / max GPU utilization: 50.05% / 100.00%
- Mean / max GPU memory: 73,999.019 / 74,419 MB
- GPU cost: `$0.786631`
- API cost: `$0.034694`
- Total cost: `$0.821325`

## SLO Verdict

- Benchmark execution: `COMPLETED`
- Runtime SLO: `PASS`
- Cost SLO: `PASS`
- Quality SLO: `FAIL`
- Safety SLO: `FAIL`
- Deployability: `NOT_DEPLOYABLE_SLO_FAILURES`

This is preserved as a completed 25-config validation/baseline run. The
official `Main_Inference_V1` remains pending and must run 25 configs x 10,000
prompts/config = 250,000 total requests before `Optimized_Inference_V1` can use
it as the before-optimization reference.
