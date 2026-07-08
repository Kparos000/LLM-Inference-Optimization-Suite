# Official Baseline Inference Experiment

Baseline Version: `baseline_v1`

Date: 2026-07-08T07:19:03.329273+00:00

Git Commit: `188823fcf9bf8d5be7827de1ea33596157da33c8`

Git Branch: `main`

## Purpose

This archive freezes the official production baseline inference experiment. It is the reference baseline for later optimized reruns. The run used the frozen controlled-final 25-config matrix and did not tune prompts, retrieval, MM4, scheduling, batching, prefix caching, KV cache, concurrency, or SLO thresholds during execution.

## Dataset And Prompt Version

- Dataset version: `controlled_2000`
- Dataset workload hash: `e71ae1ed0411c2989cc68c2b9e2cfbe5322807f6b998d82e2afbd66437829662`
- Prompt version: `controlled_final_generation_contract_repaired_phase2_targeted_baseline`
- Matrix: 25 configs, 10,000 requests, 2,000 prompts per vertical.
- Verticals: Airline, Healthcare Admin, Retail, Finance, Research AI.

## Matrix Summary

- Self-hosted model: `model3_7b` / `Qwen/Qwen2.5-7B-Instruct`
- API model: `model6_gated` / `meta-llama/Llama-3.1-8B-Instruct`
- Serving engines: vLLM, SGLang, API provider route
- Memory modes: MM0, MM1, MM2, MM3, MM4
- Self-hosted concurrency: 16, 32
- API concurrency: 4
- Hardware: RunPod A100 SXM 80GB

## Environment

- GPU: NVIDIA A100-SXM4-80GB
- CUDA: Build cuda_12.4.r12.4/compiler.34097967_0
- Driver: 580.159.04
- Operating system: Linux-6.8.0-124-generic-x86_64-with-glibc2.35
- Python: 3.11.10

## Runtime

- Requests attempted: 10000
- Requests completed: 10000
- Failed requests: 0
- Wall runtime: 1891.030 seconds
- GPU hours: 0.525286
- Mean TTFT: 439.893 ms
- Mean TPOT: 48.010 ms
- Mean E2E latency: 1914.616 ms
- P50 E2E latency: 1760.566 ms
- P95 E2E latency: 3599.813 ms
- P99 E2E latency: 4405.132 ms
- Mean total tokens/sec: 473.726

## Cost

- Total cost: `$0.817373`
- GPU cost: `$0.782676`
- API cost: `$0.034697`
- Cost/request: `$0.00008174`
- Cost/1000 requests: `$0.081737`

## Quality And Safety

- JSON validity: 99.93%
- Contract validity: 81.54%
- Format validity: 81.54%
- Evidence match: 62.30%
- Groundedness: 60.73%
- Safety findings: 103
- Safety finding rate: 1.03%
- Hallucination proxy: ungrounded answer rate, `39.27%`
- Citation correctness proxy: evidence match rate, 62.30%

## SLO Verdict

- Runtime SLO: `PASS`
- Quality SLO: `FAIL`
- Safety SLO: `FAIL`
- Cost SLO: `PASS`
- Benchmark execution: `COMPLETED`
- Deployability: `NOT_DEPLOYABLE_SLO_FAILURES`

## GPU Telemetry

- GPU samples: 1622
- Mean GPU utilization: 51.26%
- Max GPU utilization: 100.00%
- Mean power: 224.42 W
- Max power: 467.79 W
- Mean VRAM used: 73999.04 MB
- Max VRAM used: 74419.00 MB
- Max temperature: 66.00 C

## Artifact Inventory

- Raw compressed results: `raw/final_10000_baseline_v1_results.jsonl.gz`
- GPU telemetry: `raw/final_10000_baseline_v1_gpu_telemetry.jsonl`
- Manifest: `manifest.json` and `raw/final_10000_baseline_v1_manifest.json`
- Checkpoint: `raw/final_10000_baseline_v1_checkpoint.json`
- Evaluation report and summary: `processed/final_10000_baseline_v1_eval_report.json`, `processed/final_10000_baseline_v1_eval_summary.csv`
- Runtime report: `processed/final_10000_baseline_v1_runtime_report.json`
- Cost report: `processed/final_10000_baseline_v1_cost_report.json`
- SLO report and summary: `processed/final_10000_baseline_v1_slo_report.json`, `processed/final_10000_baseline_v1_slo_summary.csv`
- Engine, memory, concurrency, model, vertical, API, and API-vs-self-hosted comparisons: `processed/`
- Plotting CSV and JSON: `processed/final_10000_baseline_v1_plotting_dataset.csv`, `processed/final_10000_baseline_v1_plotting_dataset.json`
- Progress log: `logs/final_10000_progress.jsonl`
- Artifact sync report: `processed/final_10000_baseline_v1_artifact_sync_report.json`
- Metadata: `metadata.json`
- Checksums: `SHA256SUMS.txt`

## Directory Structure

```text
baseline_v1/
  baseline-readme.md
  metadata.json
  manifest.json
  SHA256SUMS.txt
  raw/
  processed/
  figures/
  logs/
```

## Reproduction

Run from the repository root at commit `188823fcf9bf8d5be7827de1ea33596157da33c8` with vLLM on `http://localhost:8000/v1`, SGLang on `http://localhost:30000/v1`, API credentials available from `.env`, and A100 SXM pricing set to `$1.49/hr`:

```bash
python scripts/phase4/run_controlled_final_simulation.py --run-full   --raw-results-path results/raw/final_10000_baseline_v1_results.jsonl   --manifest-path results/raw/final_10000_baseline_v1_manifest.json   --gpu-telemetry-path results/raw/final_10000_baseline_v1_gpu_telemetry.jsonl   --checkpoint-path results/checkpoints/final_10000_baseline_v1_checkpoint.json   --progress-log-path results/processed/final_10000_progress.jsonl
```

The full command with all output path overrides is recorded in `manifest.json`.

## Known Caveats

- Raw results are archived as gzip-compressed JSONL to avoid committing a single file above common Git hosting limits.
- This baseline is operationally complete but not deployable: quality and safety SLOs failed.
- Standalone figure images were not generated; plotting-ready CSV and JSON artifacts are included.

## Checksum Verification

From `experiments/baseline_v1/` run:

```bash
sha256sum -c SHA256SUMS.txt
```
