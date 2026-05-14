# Phase 1 Experiment Inventory

## Purpose

Phase 1 established a reproducible inference benchmarking foundation covering local mock runs, Hugging Face baseline execution, vLLM serving, concurrency/load tests, chunking, checkpoints, logs, metadata, and curated artifact preservation.

This inventory records the experiments represented by committed documentation, configuration, scripts, tests, and curated sample artifacts. It does not depend on uncommitted raw RunPod files.

## Experiment Table

| Experiment ID | Stage | Backend | Model | Workload(s) | Prompt Count | Concurrency | Main Artifacts | Primary Learning |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| EXP-001 | Pipeline validation | Mock | Mock model | `smoke` | 3 configured | 1 | `configs/experiments.yaml`, `tests/test_mock_runner.py` | Benchmark schemas, CSV writing, and CLI plumbing were validated without model downloads. |
| EXP-002 | Local smoke | Hugging Face | `Qwen/Qwen2.5-0.5B-Instruct` | `smoke` | sample CSV present | 1 | `results/samples/raw/hf_smoke_results_sample.csv`, HF smoke figures | Local HF execution path and basic metric capture were validated. |
| EXP-003 | Structured-output smoke | Hugging Face | `Qwen/Qwen2.5-0.5B-Instruct` | `structured_output_smoke` | 3 | 1 | `results/samples/raw/hf_structured_output_results_sample.csv`, `results/samples/raw/hf_structured_output_generations_sample.jsonl` | Structured-output traces can be captured and scored, but correctness evaluation remains limited. |
| EXP-004 | Expanded local baseline | Hugging Face | `Qwen/Qwen2.5-0.5B-Instruct` | `short_chat`, `code_helpdesk`, `long_context`, `shared_prefix` | 3 to 5 per workload | 1 | `results/samples/raw/hf_workload_comparison_sample.csv`, `docs/08_hf_baseline_findings.md` | Local CPU-oriented HF baseline showed long-context TTFT was the largest latency driver. |
| EXP-005 | vLLM smoke | vLLM OpenAI-compatible | `Qwen/Qwen2.5-0.5B-Instruct` | `smoke` | 1 | 1 | `results/samples/raw/vllm_smoke_results_sample.csv`, `results/samples/raw/vllm_smoke_generations_sample.jsonl` | vLLM server/client integration worked through the OpenAI-compatible runner. |
| EXP-006 | vLLM expanded baseline | vLLM OpenAI-compatible | `Qwen/Qwen2.5-0.5B-Instruct` | Five calibration workloads | 3 to 5 per workload | 1 | `results/samples/raw/vllm_workload_comparison_sample.csv`, `docs/12_experiment_log.md` | vLLM produced much lower TPOT than local HF calibration, while traces showed quality and truncation still need review. |
| EXP-007 | HF vs vLLM calibration | Hugging Face and vLLM | `Qwen/Qwen2.5-0.5B-Instruct` | Five workload families where available | 3 to 5 per workload | 1 | `docs/13_hf_vs_vllm_calibration_comparison.md` | The comparison is useful as an architecture baseline, not as a hardware-equal benchmark. |
| EXP-008 | Long-context concurrency sweep | vLLM OpenAI-compatible | `Qwen/Qwen2.5-0.5B-Instruct` | `long_context` synthetic scaled workload | 1,000 per concurrency level | 1, 4, 8, 16, 32 | `results/samples/processed/vllm_long_context_1000_concurrency_comparison_sample.csv`, metadata samples | Aggregate throughput increased with concurrency, while TTFT and p99 latency increased at higher concurrency. |
| EXP-009 | Chunked/resumable long-context run | vLLM OpenAI-compatible | `Qwen/Qwen2.5-0.5B-Instruct` | `long_context` synthetic scaled workload | 1,000 represented by curated sample | 32 sample plus broader checkpoint/log artifacts | `results/samples/logs/vllm_long_context_1000_conc32_chunked_sample.log`, `results/samples/checkpoints/vllm_long_context_1000_conc32_chunked_checkpoint_sample.json`, `docs/15_resumable_benchmarking_plan.md` | Chunking, checkpoints, logs, and resume behavior are represented by curated run artifacts and tested without real server calls. |
| EXP-010 | 5,000-prompt all-workloads concurrency run | vLLM OpenAI-compatible | `Qwen/Qwen2.5-0.5B-Instruct` | Five synthetic workload families | 5,000 per workload/configuration; 75,000 total requests represented | 8, 16, 32 | `results/samples/processed/vllm_qwen0_5b_all_workloads_5000_concurrency_comparison_sample.csv`, 5,000-prompt metadata, logs, checkpoints | Aggregate throughput increased with concurrency while TTFT and p99 latency rose, showing the throughput/tail-latency tradeoff. |

## EXP-001: Mock Smoke Benchmark

### Objective

Validate the benchmark pipeline without downloading models or using GPU resources.

### Configuration

- Backend: mock
- Workload: `smoke`
- Prompt limit: 3 configured in `configs/experiments.yaml`
- Concurrency: 1

### What Was Measured

The mock runner exercises result schemas, deterministic metric generation, output writing, CLI behavior, and reporting integration.

### Result Summary

Curated runtime sample metrics are not promoted for this run. The experiment is represented by configuration and tests.

### Key Learning

The mock path made it possible to validate the benchmark harness before spending GPU time or relying on optional inference dependencies.

### Limitations

Mock metrics do not represent model behavior, output quality, or serving throughput.

### Follow-up Action

Keep mock runs as CI-safe regression coverage.

## EXP-002: Hugging Face Smoke Benchmark

### Objective

Confirm local Hugging Face execution and metric capture for a small model.

### Configuration

- Backend: Hugging Face
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workload: `smoke`
- Concurrency: 1

### What Was Measured

TTFT, TPOT, end-to-end latency, throughput, generated text traces, and report-ready figures.

### Result Summary

Curated artifacts include `hf_smoke_results_sample.csv` and HF smoke latency/throughput figures. The smoke run validated the local execution path rather than final performance.

### Key Learning

Local HF inference was sufficient to calibrate instrumentation and output handling before vLLM.

### Limitations

Small prompt count, small model, local environment, and no concurrency.

### Follow-up Action

Use HF results as local baseline context for later backend comparisons.

## EXP-003: Hugging Face Structured-Output Benchmark

### Objective

Validate that structured-output prompts and generated JSONL traces can be captured and scored.

### Configuration

- Backend: Hugging Face
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workload: `structured_output_smoke`
- Prompt count: 3 in curated sample

### What Was Measured

CSV latency metrics, generated text traces, and structured-output validity checks.

### Result Summary

The curated structured-output sample contains 3 successful rows. Structured-output scoring exists, but it is not yet a comprehensive correctness evaluation.

### Key Learning

Format validation is useful, but correctness evaluation must go beyond valid JSON and required fields.

### Limitations

Small sample size and no deterministic answer-quality scoring.

### Follow-up Action

Add correctness evaluation for structured and non-structured tasks.

## EXP-004: Expanded Hugging Face Baseline

### Objective

Establish local baseline behavior across expanded workload categories.

### Configuration

- Backend: Hugging Face
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workloads: `short_chat`, `code_helpdesk`, `long_context`, `shared_prefix`
- Concurrency: 1

### What Was Measured

Average TTFT, TPOT, end-to-end latency, throughput, success counts, and workload comparison rows.

### Result Summary

| workload | rows | success | avg TTFT ms | avg TPOT ms | avg latency ms | avg throughput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_chat` | 5 | 5 | 1,697.43 | 135.96 | 8,087.57 | 8.41 |
| `code_helpdesk` | 5 | 5 | 1,655.61 | 133.69 | 12,217.50 | 8.35 |
| `long_context` | 3 | 3 | 5,995.48 | 129.11 | 16,375.99 | 10.46 |
| `shared_prefix` | 5 | 5 | 2,696.97 | 128.57 | 12,854.01 | 9.58 |

### Key Learning

TTFT and TPOT measure different bottlenecks. The local HF baseline showed that long-context latency was driven primarily by TTFT/prefill cost, while TPOT was comparatively stable across workloads.

### Limitations

Local CPU-oriented results should not be generalized to GPU serving.

### Follow-up Action

Use the same workload families for vLLM calibration and later controlled backend comparisons.

## EXP-005: vLLM Smoke Benchmark

### Objective

Confirm that a vLLM OpenAI-compatible server can be benchmarked by the project client.

### Configuration

- Backend: vLLM OpenAI-compatible server
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workload: `smoke`
- Environment: RunPod Linux GPU pod with NVIDIA L40S, as documented in `docs/12_experiment_log.md`

### What Was Measured

CSV latency metrics and prompt-level JSONL generation traces through `inference-bench openai-compatible-run`.

### Result Summary

Curated smoke artifacts are present for metrics and generations. This run validated integration, not final performance.

### Key Learning

vLLM is useful for serving/load tests because it exposes an OpenAI-compatible endpoint and supports high-throughput serving behavior.

### Limitations

Single prompt smoke run and no concurrency.

### Follow-up Action

Run expanded workloads and scaled concurrency tests.

## EXP-006: vLLM Expanded Baseline

### Objective

Measure a first vLLM baseline across the expanded workload set.

### Configuration

- Backend: vLLM
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Optimization: `vllm_baseline`
- Workloads: `short_chat`, `code_helpdesk`, `long_context`, `shared_prefix`, `structured_output_smoke`
- Concurrency: 1

### What Was Measured

TTFT, TPOT, end-to-end latency, throughput, success counts, and generation traces.

### Result Summary

| workload | rows | success | avg TTFT ms | avg TPOT ms | avg latency ms | avg throughput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_chat` | 5 | 5 | 52.53 | 2.55 | 114.32 | 488.46 |
| `code_helpdesk` | 5 | 5 | 59.09 | 2.80 | 225.97 | 385.07 |
| `long_context` | 3 | 3 | 79.88 | 2.45 | 218.23 | 656.55 |
| `shared_prefix` | 5 | 5 | 54.21 | 2.61 | 210.60 | 484.19 |
| `structured_output_smoke` | 3 | 3 | 74.48 | 4.40 | 168.09 | 419.94 |

### Key Learning

vLLM produced much lower TPOT than the local HF CPU baseline. Prompt traces also showed that speed metrics alone are insufficient because truncation and response quality still require review.

### Limitations

Small model, small prompt counts, single concurrency, and manual quality review.

### Follow-up Action

Add scaled workloads, concurrency sweeps, latency percentiles, and correctness evaluation.

## EXP-007: HF vs vLLM Calibration Comparison

### Objective

Compare the local HF baseline with the RunPod vLLM baseline as an architecture/integration calibration.

### Configuration

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- HF environment: local baseline
- vLLM environment: RunPod L40S
- Scope: calibration, not hardware-equal benchmark

### What Was Measured

Average latency, TTFT, TPOT, and throughput across available curated samples.

### Result Summary

The comparison showed a large directional difference between local HF CPU-oriented execution and GPU-backed vLLM serving. Because hardware differed, the result is not a controlled backend-only comparison.

### Key Learning

Moving to vLLM/RunPod changed the performance profile enough to justify serving-oriented concurrency testing.

### Limitations

Different hardware environments, small prompt counts, and no correctness scoring.

### Follow-up Action

Repeat comparisons with larger workloads and controlled assumptions where practical.

## EXP-008: 1,000-Prompt vLLM Long-Context Concurrency Sweep

### Objective

Measure how long-context serving behavior changes across concurrency levels.

### Configuration

- Backend: vLLM
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workload: synthetic `long_context_1000`
- Concurrency levels: 1, 4, 8, 16, 32
- Prompt count: 1,000 per concurrency level

### What Was Measured

Per-request latency, TTFT, TPOT, p95/p99 latency, success/failure counts, and run-level aggregate throughput metadata.

### Result Summary

| concurrency | requests | failures | avg latency ms | p99 latency ms | avg TTFT ms | p99 TTFT ms | aggregate req/s | aggregate output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 0 | 209.39 | 221.21 | 12.45 | 14.70 | 4.77 | 344.41 |
| 4 | 1,000 | 0 | 235.94 | 250.10 | 14.77 | 23.12 | 16.87 | 1,220.35 |
| 8 | 1,000 | 0 | 244.55 | 270.25 | 17.51 | 39.36 | 32.33 | 2,332.59 |
| 16 | 1,000 | 0 | 261.79 | 345.56 | 27.38 | 103.82 | 59.63 | 4,298.81 |
| 32 | 1,000 | 0 | 321.79 | 394.33 | 59.92 | 127.60 | 93.46 | 6,726.45 |

### Key Learning

Higher concurrency improved aggregate throughput but increased tail latency and TTFT. p95/p99 matter because averages understate the user-visible tail behavior under load.

Aggregate throughput differs from per-request throughput because aggregate metrics measure completed requests or output tokens across wall-clock run time, while per-request throughput is computed within each request's own latency window.

### Limitations

Single workload family, small 0.5B model, one hardware environment, synthetic workload, and no correctness evaluation.

### Follow-up Action

Extend concurrency sweeps across all workload families and add correctness and memory measurements.

## EXP-009: Chunked/Resumable 1,000-Prompt Long-Context Run

### Objective

Make long-running OpenAI-compatible load benchmarks safer through chunking, checkpoints, logs, and resume support.

### Configuration

- Runner: `openai-load-run`
- Backend: vLLM OpenAI-compatible
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workload represented by curated sample: `long_context`
- Prompt count represented by curated sample: 1,000
- Concurrency represented by curated sample: 32
- Support: chunking, append-safe writes, checkpoint file, progress log, resume mode
- Test coverage: `tests/test_openai_load_runner_resumable.py`

### What Was Measured

Checkpoint writing, progress logging, resume skipping, append-safe CSV behavior, JSONL append behavior, and run completion metadata.

### Result Summary

Curated checkpoint and log samples are present for the 1,000-prompt long-context concurrency 32 run:

- `results/samples/logs/vllm_long_context_1000_conc32_chunked_sample.log`
- `results/samples/checkpoints/vllm_long_context_1000_conc32_chunked_checkpoint_sample.json`

Additional committed checkpoint and log artifacts cover 1,000-prompt workload/concurrency combinations. The implemented runner foundation is represented by source code, docs, tests, and curated samples.

### Key Learning

Chunking/checkpoints/logs are necessary for long GPU runs because failures such as pod shutdown, server crash, network disconnect, or credit exhaustion can otherwise lose all end-of-run outputs.

### Limitations

Stricter checkpoint configuration validation and failed-prompt retry mode remain future work.

### Follow-up Action

Use the resumable path for future 1,000+ and 5,000+ prompt GPU runs.

## EXP-010: 5,000-Prompt Qwen 0.5B vLLM Concurrency Run Across Five Workloads

### Objective

Run a larger synthetic workload baseline across five workload families and concurrency levels 8, 16, and 32.

### Configuration

- Backend: vLLM
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Workloads: `short_chat`, `code_helpdesk`, `long_context`, `shared_prefix`, `structured_output`
- Concurrency levels: 8, 16, 32
- Prompt count: 5,000 per workload/configuration
- Total benchmark configurations represented: 15
- Total requests represented: 75,000

### What Was Measured

Per-request latency, TTFT, TPOT, throughput, success/failure counts, aggregate run-level throughput metadata, chunked progress logs, and checkpoints.

### Result Summary

The curated comparison CSV `results/samples/processed/vllm_qwen0_5b_all_workloads_5000_concurrency_comparison_sample.csv` is present. It represents 15 benchmark configurations across five workloads and three concurrency levels. Each configuration contains 5,000 successful requests and 0 recorded failures.

| concurrency | configs | total requests | failures | avg aggregate req/s | avg aggregate output tok/s | avg request latency ms | max p99 latency ms | avg TTFT ms | max p99 TTFT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5 | 25,000 | 0 | 34.12 | 2,152.78 | 227.86 | 284.41 | 20.20 | 37.57 |
| 16 | 5 | 25,000 | 0 | 59.22 | 3,737.48 | 245.46 | 349.48 | 33.39 | 108.99 |
| 32 | 5 | 25,000 | 0 | 87.44 | 5,589.52 | 298.07 | 410.49 | 65.68 | 162.41 |

Aggregate throughput increased from concurrency 8 to 32, while average request latency, p99 latency, average TTFT, and p99 TTFT also rose. This matters because it shows the benchmark can expose an inference bottleneck tradeoff: higher concurrent load improves total served work, but the tail of first-token and end-to-end latency becomes materially worse.

### Key Learning

The 5,000-prompt run shows why concurrency sweeps need both aggregate throughput and latency percentiles. Throughput improved as concurrency increased, but the p99 latency and TTFT growth indicate that a deployment decision cannot be based on throughput alone.

### Limitations

This remains a synthetic workload baseline, not a real-world-data benchmark. Results are tied to the specific RunPod/L40S conditions, small Qwen 0.5B model, selected prompt templates, and single-run sample artifacts.

### Follow-up Action

Use this run as source material for Phase 1 reporting, then add correctness evaluation, memory profiling, repeated runs, and real-world workloads before drawing broader model-serving conclusions.

## Cross-Experiment Lessons

- TTFT and TPOT measure different bottlenecks. TTFT captures first-token/prefill behavior, while TPOT captures decode speed after generation begins.
- Higher concurrency improves aggregate throughput but can increase tail latency.
- Chunking/checkpoints/logs are necessary for long GPU runs.
- vLLM is useful for serving/load tests because it exposes an OpenAI-compatible endpoint and supports high-throughput serving behavior.
- Synthetic prompts are useful for controlled experiments, but Phase 2 needs real-world data and correctness evaluation.
- Correctness must be added before serious model-scale conclusions.

## Phase 1 Limitations

- Synthetic workloads only.
- Limited model scale so far.
- No deterministic correctness evaluation yet.
- No memory profiling integrated yet.
- No quantization, prefix-caching, or speculative-decoding experiments yet.
- GPU results are from specific RunPod/L40S conditions and should not be generalized without repeated runs.

## What Phase 2 Should Build Next

- Deterministic correctness evaluation.
- Real-world corpus schema.
- Dataset provenance and license tracking.
- Real-world prompt builders.
- Model sweep support.
- Memory profiling.
- Before/after optimization loop.

## Artifact Coverage Notes

- Curated raw samples are stored under `results/samples/raw`.
- Curated processed samples include 1,000-prompt concurrency comparisons and the 5,000-prompt Qwen 0.5B all-workloads concurrency comparison.
- Curated figure samples include HF smoke and structured-output latency/throughput plots.
- Curated logs and checkpoints are present for chunked 1,000-prompt and 5,000-prompt vLLM runs.
