# Main_Inference_V1

Status: official before-optimization reference run, completed July 9, 2026.

This archive records the corrected official Main_Inference_V1 run. It is the
full 25-config matrix with 10,000 prompts per config, for 250,000 total
requests. It supersedes the previously misnamed 10,000-total-request validation
run, which is now classified separately as Baseline_Inference_V1.

## Matrix

- Configs: 25.
- Prompts per config: 10,000.
- Total requests: 250,000.
- Verticals: Airline, Healthcare Admin, Retail, Finance, Research AI.
- Prompts per vertical per config: 2,000.
- Self-hosted model: `model3_7b` / `Qwen/Qwen2.5-7B-Instruct`.
- API model: `model6_gated` / `meta-llama/Llama-3.1-8B-Instruct`.
- Serving engines: vLLM, SGLang, API provider route.
- Memory modes: mm0-mm4.
- Self-hosted concurrency: 16 and 32.
- API concurrency: 4.
- GPU: A100 SXM 80GB.
- Hourly price used: `$1.49`.

## Result

- Requests completed: 250,000/250,000.
- Request failures: 0.
- Runtime: 42,538.856 seconds.
- Total cost: `$18.461297`.
- GPU cost: `$17.606360`.
- API cost: `$0.854937`.
- JSON validity: 99.822%.
- Contract validity: 80.5388%.
- Evidence match: 58.9724%.
- Groundedness: 56.7204%.
- Safety findings: 2,757.
- Deployability: `NOT_DEPLOYABLE_SLO_FAILURES`.

Runtime and cost SLOs passed. Quality and safety SLOs failed. This run is the
official before-optimization reference for Optimized_Inference_V1.

## Artifact Notes

The local archive directory includes processed reports, manifests, checkpoint,
GPU telemetry, progress logs, console logs, and checksums. The full compressed
raw results, matrix, and repaired runner input are preserved locally and under
`backups/main_inference_v1/`, but ignored by Git because each exceeds normal Git
hosting file-size limits.
