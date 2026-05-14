# Phase 1 Project Report: LLM Inference Optimization Suite

## Executive Summary

Phase 1 built a reproducible inference benchmarking suite for measuring LLM serving performance across synthetic workloads, local baselines, Hugging Face execution, vLLM GPU serving, concurrency testing, chunked/resumable execution, logs, checkpoints, and curated artifact preservation.

The project moved from scaffold and mock validation to real GPU-backed vLLM concurrency experiments. The benchmark now captures TTFT, TPOT, end-to-end latency, throughput, latency percentiles, aggregate throughput, success/failure counts, run metadata, checkpoints, and logs. The current experiments are synthetic-workload baselines, not yet real-world data benchmarks.

The largest committed Phase 1 run represents 75,000 inference requests across 15 benchmark configurations: 5 workloads, 5,000 prompts per configuration, and concurrency levels 8, 16, and 32. Those samples show a clear serving tradeoff: higher concurrency improved aggregate throughput, while TTFT and p99 latency increased.

## Phase 1 Objectives

Phase 1 had six practical objectives:

- Build a professional inference benchmark harness with stable schemas, metrics, workload loading, result writing, and CLI commands.
- Avoid paid GPU usage until the local harness, CI, configs, sample workloads, reporting, and dry-run plan were stable.
- Establish reproducible metrics for latency, throughput, output traces, system metadata, and curated artifacts.
- Add Hugging Face execution as a local model baseline.
- Add vLLM GPU serving through an OpenAI-compatible endpoint.
- Run scaled concurrency experiments and preserve curated artifacts after the GPU pod was stopped.

## System Architecture Built

The repository now contains the following benchmark components:

| component | purpose |
| --- | --- |
| Workload loaders | Load JSONL prompt records into repeatable benchmark runs. |
| Synthetic workload generator | Generate deterministic scaled prompt files from templates using count, seed, workload selection, and output directory. |
| Schema/result models | Standardize model, workload, experiment, and result fields. |
| Metrics modules | Compute latency, throughput, cost, and memory-related helper values. |
| Reporting summaries | Summarize CSV results with averages and percentiles. |
| Plot generation | Produce baseline latency and throughput plots for curated samples. |
| Hugging Face runner | Execute local Hugging Face model runs where optional dependencies are installed. |
| OpenAI-compatible runner | Run single-request vLLM/OpenAI-compatible serving baselines. |
| Async concurrency load runner | Run concurrent OpenAI-compatible requests and record per-request metrics. |
| Chunked/resumable execution | Flush results after chunks, write checkpoints, write logs, and resume completed prompt IDs. |
| Metadata capture | Preserve run-level metadata including aggregate requests/sec and aggregate output tokens/sec. |
| Sample artifact promotion | Promote reviewed samples into `results/samples` while keeping arbitrary raw outputs out of version control. |
| Public-content audit | Check public repo content for prohibited public-facing phrases and potential secret patterns. |
| CI/test suite | Validate configuration, schemas, runners, reporting, quality helpers, workflow scripts, and documentation. |

## Workloads Used in Phase 1

Phase 1 used five synthetic workload families. Synthetic prompts were used first because they are deterministic, inspectable, free of private data, and suitable for measuring serving behavior before introducing external datasets.

| workload | what it simulates | why it matters | likely bottleneck |
| --- | --- | --- | --- |
| `short_chat` | Short professional writing, summarization, rewrites, confirmations, and explanations. | Represents common low-latency assistant interactions. | Scheduling overhead, TTFT, and small-response throughput. |
| `code_helpdesk` | Debugging, Git, CLI, Python, dependency, environment, and troubleshooting prompts. | Tests technical support behavior and longer generated answers. | Decode length, truncation, TPOT, and output quality. |
| `long_context` | Synthetic passages about platform operations, support processes, data pipelines, incidents, rollout notes, and documentation updates. | Exercises prompt prefill and context handling. | TTFT, prefill cost, queueing, and p95/p99 latency. |
| `shared_prefix` | Repeated internal IT-support prefix with varied user requests. | Models repeated instructions and future prefix-caching opportunities. | Prefix reuse potential, policy/quality review, and cache behavior. |
| `structured_output` | Prompts requesting valid JSON with `category`, `answer`, and `confidence` fields. | Tests format adherence and downstream parseability. | Structured-output correctness, validation failures, and generation truncation. |

## Metrics Explained

| metric | meaning | why it matters |
| --- | --- | --- |
| TTFT | Time To First Token. | Helps identify prefill, queueing, cold-start, and first-token latency issues. |
| TPOT | Time Per Output Token after generation begins. | Helps identify decode-path bottlenecks, model speed, and runtime efficiency. |
| End-to-end latency | Total request duration. | Captures user-visible request time. |
| Throughput tokens/sec | Per-request output token rate. | Useful for comparing decode speed at request level. |
| Aggregate requests/sec | Completed requests divided by wall-clock run time. | Shows system-level serving capacity under load. |
| Aggregate output tokens/sec | Total output tokens divided by wall-clock run time. | Shows total generation capacity across concurrent requests. |
| p50/p90/p95/p99 latency | Latency percentiles across requests. | Shows median and tail behavior that averages can hide. |
| Success/failure rate | Completed versus failed requests. | Confirms whether throughput came from successful serving or masked failures. |

Averages alone are insufficient because concurrent serving can look healthy on mean latency while a meaningful tail of users experiences slow first-token or end-to-end response times. p95 and p99 are especially important for production-style inference because they expose queueing and scheduling pressure.

Aggregate throughput differs from per-request throughput. Per-request throughput describes one request's token rate inside its own latency window. Aggregate throughput describes how much total work the serving system completed across the full wall-clock run.

## Tooling and Environment Decisions

The project started with mock and local validation so schemas, CLI commands, result writing, and reports could be tested without model downloads or GPU cost. Hugging Face was added first because it provides a direct local model execution path and a useful baseline for prompt traces and metric capture.

vLLM was added next because Phase 1 needed a serving-oriented GPU backend. The OpenAI-compatible endpoint made it possible to benchmark vLLM through the same client shape used by many production inference systems. RunPod/L40S was used for practical GPU access after the local benchmark foundation was stable.

Linux RunPod scripts were added because GPU execution happened in a Linux environment. The scripts keep repeated smoke, expanded baseline, and artifact promotion workflows consistent instead of relying on manually pasted commands.

Environment files are handled carefully: `.env` is ignored, `.env.example` documents expected variables, and public scripts use placeholders rather than credentials. Curated samples are committed because they support reproducibility review and reporting. Large raw outputs and arbitrary generated files are not committed by default because they can be large, noisy, and require review before public inclusion.

## Experiment Inventory

The source of truth for Phase 1 experiment coverage is `docs/19_phase1_experiment_inventory.md`. This section summarizes the completed and represented experiments.

| experiment | objective | setup | artifacts | result summary | key learning | limitation |
| --- | --- | --- | --- | --- | --- | --- |
| Mock smoke benchmark | Validate the benchmark pipeline without models or GPU. | Mock backend, `smoke`, configured prompt limit 3. | `configs/experiments.yaml`, `tests/test_mock_runner.py`. | Validated schemas, CSV writing, CLI plumbing, and reporting path. | CI-safe validation is useful before real inference. | Mock metrics do not represent model behavior. |
| Hugging Face smoke benchmark | Confirm local HF execution and metric capture. | Hugging Face, `Qwen/Qwen2.5-0.5B-Instruct`, `smoke`, concurrency 1. | `results/samples/raw/hf_smoke_results_sample.csv`, HF smoke figures. | Validated local execution and basic metric capture. | Local HF was a practical first real-model baseline. | Small prompt count and no concurrency. |
| Hugging Face structured-output benchmark | Validate structured-output traces and scoring. | Hugging Face, structured-output smoke workload, 3 curated rows. | `results/samples/raw/hf_structured_output_results_sample.csv`, `results/samples/raw/hf_structured_output_generations_sample.jsonl`. | Three successful rows are present in the curated sample. | Format validation is useful but not full correctness evaluation. | Small sample and no deterministic answer-quality scoring. |
| vLLM smoke benchmark | Confirm vLLM server/client integration. | vLLM OpenAI-compatible server, RunPod L40S, smoke prompt. | `results/samples/raw/vllm_smoke_results_sample.csv`, `results/samples/raw/vllm_smoke_generations_sample.jsonl`. | Integration worked through `openai-compatible-run`. | OpenAI-compatible serving path is functional. | Single-prompt smoke run. |
| vLLM expanded baseline | Measure first GPU-backed baseline across five workloads. | vLLM, Qwen 0.5B, five calibration workloads, concurrency 1. | `results/samples/raw/vllm_workload_comparison_sample.csv`, `docs/12_experiment_log.md`. | vLLM showed much lower TPOT than local HF calibration samples. | Speed metrics need prompt-level quality review. | Small prompt counts and one model size. |
| 100-prompt validation scale | Define a first scaled validation level. | Documented by the scaled workload methodology. | `docs/14_scaled_workload_generation.md`, `configs/scaled_workloads.yaml`. | No separate curated 100-prompt result artifact is committed. | The validation scale exists for future repeatable checks. | Metrics are unavailable without a committed run artifact. |
| 1,000-prompt vLLM concurrency sweep | Measure serving under increasing concurrency. | vLLM, Qwen 0.5B, long-context synthetic workload, concurrency 1/4/8/16/32. | `results/samples/processed/vllm_long_context_1000_concurrency_comparison_sample.csv`, metadata samples. | Aggregate throughput increased while TTFT and p99 latency also increased. | Concurrency exposes throughput/tail-latency tradeoffs. | Single workload family in the summary sample. |
| Chunked/resumable 1,000-prompt long-context run | Validate failure-recovery mechanics. | `openai-load-run`, chunked writes, checkpoint, log, resume support. | `results/samples/logs/vllm_long_context_1000_conc32_chunked_sample.log`, `results/samples/checkpoints/vllm_long_context_1000_conc32_chunked_checkpoint_sample.json`. | Checkpoint/log artifacts are present for the curated sample. | Long GPU runs need periodic persistence. | Retry mode for failed prompts remains future work. |
| 5,000-prompt Qwen 0.5B vLLM concurrency run | Establish a stronger synthetic GPU serving baseline. | vLLM, Qwen 0.5B, five workloads, concurrency 8/16/32, 5,000 prompts per configuration. | `results/samples/processed/vllm_qwen0_5b_all_workloads_5000_concurrency_comparison_sample.csv`, metadata, logs, checkpoints. | 15 configurations and 75,000 requests are represented, with 0 recorded failures. | Throughput improved with concurrency, but TTFT and p99 latency rose. | Synthetic data, one model size, one environment sample. |

## Results Summary

### Hugging Face Baseline

The Hugging Face baseline tested local execution for the Qwen 0.5B model across smoke, structured-output, and expanded workload samples. It proved that the harness could run real model inference, capture TTFT/TPOT/end-to-end latency, preserve JSONL generation traces, and produce curated reports and figures.

The expanded Hugging Face baseline showed long-context prompts had much higher TTFT than short prompts in the curated local samples. This aligned with the expected prefill cost of longer inputs. The HF results are useful as a local calibration baseline, not as a hardware-equal comparison to vLLM.

### vLLM Smoke and Expanded Baseline

The vLLM smoke and expanded baseline runs validated the OpenAI-compatible serving path on RunPod L40S with `Qwen/Qwen2.5-0.5B-Instruct`. The expanded vLLM baseline covered `short_chat`, `code_helpdesk`, `long_context`, `shared_prefix`, and `structured_output_smoke`.

Curated samples in `results/samples/raw/vllm_workload_comparison_sample.csv` showed all rows succeeded across the five calibration workloads. The experiment log records that vLLM integration captured TTFT, TPOT, throughput, CSV metrics, and JSONL traces successfully. Prompt-level observations also found incomplete/truncated responses and quality concerns in some outputs, reinforcing that speed is not enough.

### 1,000-Prompt Concurrency Sweep

The 1,000-prompt long-context concurrency sweep tested vLLM at concurrency levels 1, 4, 8, 16, and 32. Each represented configuration in the curated comparison had 1,000 requests and 0 recorded failures.

| concurrency | requests | failures | avg latency ms | p99 latency ms | avg TTFT ms | p99 TTFT ms | aggregate req/s | aggregate output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | 0 | 209.39 | 221.21 | 12.45 | 14.70 | 4.77 | 344.41 |
| 4 | 1,000 | 0 | 235.94 | 250.10 | 14.77 | 23.12 | 16.87 | 1,220.35 |
| 8 | 1,000 | 0 | 244.55 | 270.25 | 17.51 | 39.36 | 32.33 | 2,332.59 |
| 16 | 1,000 | 0 | 261.79 | 345.56 | 27.38 | 103.82 | 59.63 | 4,298.81 |
| 32 | 1,000 | 0 | 321.79 | 394.33 | 59.92 | 127.60 | 93.46 | 6,726.45 |

Concurrency improved aggregate throughput but increased latency and tail latency. That pattern is central to serving analysis: the best concurrency value depends on throughput targets and latency objectives.

### Chunked/Resumable Run

The chunked/resumable runner uses chunk size 100 by default. It prints progress feedback, writes result rows after each chunk, writes generation JSONL rows after each chunk when enabled, saves checkpoints after chunks, and can skip completed prompt IDs in resume mode.

Committed checkpoint and log samples show why this matters for failure recovery. Long GPU runs can be interrupted by pod shutdown, network disconnect, server crash, disk issues, or credit exhaustion. Chunking, checkpoints, logs, metadata, and resume support reduce the chance that a long run loses all useful outputs.

### 5,000-Prompt Qwen 0.5B Run

The largest committed Phase 1 sample is `results/samples/processed/vllm_qwen0_5b_all_workloads_5000_concurrency_comparison_sample.csv`. It covers:

- 5 workloads: `short_chat`, `code_helpdesk`, `long_context`, `shared_prefix`, `structured_output`.
- 3 concurrency levels: 8, 16, 32.
- 15 benchmark configurations.
- 5,000 successful requests per configuration.
- 75,000 total requests represented.
- 0 recorded failures in the curated comparison.

| concurrency | configs | total requests | failures | avg aggregate req/s | avg aggregate output tok/s | avg request latency ms | max p95 latency ms | max p99 latency ms | avg TTFT ms | max p95 TTFT ms | max p99 TTFT ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5 | 25,000 | 0 | 34.12 | 2,152.78 | 227.86 | 278.45 | 284.41 | 20.20 | 32.08 | 37.57 |
| 16 | 5 | 25,000 | 0 | 59.22 | 3,737.48 | 245.46 | 299.22 | 349.48 | 33.39 | 48.75 | 108.99 |
| 32 | 5 | 25,000 | 0 | 87.44 | 5,589.52 | 298.07 | 387.99 | 410.49 | 65.68 | 125.39 | 162.41 |

This is a stronger synthetic baseline than the smoke tests because it covers multiple workload families, substantially more prompts, higher concurrency, run-level metadata, chunked logs, and checkpoints. It still should not be treated as a real-world data benchmark or a final model-scale conclusion.

## Bottleneck Analysis

### TTFT Bottlenecks

TTFT bottlenecks can come from long prompts, prefill cost, queueing under concurrency, cold cache effects, and prefixes that are not reused. The Hugging Face local baseline and vLLM concurrency samples both show why first-token latency must be measured separately from decode speed.

Possible fixes include prefix caching, prompt compression, shorter context windows where acceptable, separate queues for long and short requests, and larger or faster GPU capacity.

### TPOT Bottlenecks

TPOT bottlenecks can come from a slow decode path, a model that is too large for the serving target, memory bandwidth constraints, and inefficient kernels. Phase 1 showed that TPOT changed dramatically between local HF calibration and GPU-backed vLLM calibration, although that comparison is not hardware-equal.

Possible fixes include quantization, speculative decoding, a better GPU, a smaller model, or a more optimized runtime configuration.

### Tail Latency Bottlenecks

Tail latency bottlenecks appear as high p95/p99 latency under concurrency. They can be caused by mixed workload lengths, queueing pressure, scheduling behavior, and saturation at high concurrency.

Possible fixes include concurrency tuning, admission control, workload routing, batching configuration changes, and replicas or data parallelism.

### Reliability Bottlenecks

Reliability bottlenecks include pod crash, credit exhaustion, interrupted SSH session, vLLM crash, and disk issues. Phase 1 implemented chunking, checkpointing, logs, metadata, resume support, and curated artifact promotion so long runs can survive interruptions and still produce reviewable outputs.

## Engineering Decisions and Justifications

- Synthetic workloads first: deterministic synthetic prompts made it possible to test serving mechanics without private data or external dataset licensing concerns.
- GPU delayed until harness stability: the project avoided GPU spend until configs, CI, runners, reporting, and artifact policy were in place.
- vLLM chosen for serving/load tests: vLLM exposes a production-relevant serving pattern and an OpenAI-compatible endpoint.
- OpenAI-compatible interface used: this reduced backend coupling and made the same client shape useful for vLLM and future compatible servers.
- RunPod used for practical GPU access: RunPod/L40S provided a workable GPU environment for vLLM calibration and load testing.
- Concurrency sweep used instead of one fixed concurrency: serving behavior changes under load, so one concurrency level cannot identify the throughput/tail-latency curve.
- 8/16/32 selected for the larger run: earlier 1/4/8/16/32 experiments established the curve shape, and the larger run focused on higher-load serving behavior.
- Large generated prompt files ignored by default: deterministic generation preserves reproducibility without bloating the public repository.
- Curated artifacts committed: reviewed samples support reporting and reproducibility after the GPU environment is gone.
- Raw generations not committed by default: generated outputs can be large and require review for safety, content quality, and public suitability.

## What Phase 1 Proves

Phase 1 proves that the project can run reproducible inference benchmarks, collect useful serving metrics, compare concurrency settings, run GPU-backed vLLM experiments, preserve artifacts after the GPU pod is gone, and support future before/after optimization studies.

It also proves that the benchmark can distinguish request-level and run-level behavior: TTFT, TPOT, end-to-end latency, percentiles, aggregate requests/sec, aggregate output tokens/sec, success/failure counts, metadata, checkpoints, and logs are all part of the measurement surface.

## What Phase 1 Does Not Yet Prove

- It does not yet prove real-world data performance.
- It does not yet prove answer correctness at scale.
- It does not yet compare multiple larger models.
- It does not yet include integrated memory profiling.
- It does not yet test prefix caching, quantization, or speculative decoding.
- It does not yet benchmark SGLang or TGI.
- It does not yet include cost-normalized model comparisons.

## Phase 2 Direction

Phase 2 should add deterministic correctness evaluation and real-world benchmark data. The next benchmark layer should define a real-world corpus schema, dataset provenance and licensing records, and prompt builders for practical domains such as customer support, airline operations, developer support, and finance-style analysis.

The optimization loop should then compare before/after changes under the same workload and metric framework. Priority areas include memory measurement, model sweep support, prefix caching, quantization, and speculative decoding.

## Generated Plot Artifacts

Phase 1 plot artifacts are generated from the committed 5,000-prompt Qwen 0.5B vLLM comparison sample and written under `results/samples/figures/phase1`.

Aggregate throughput plots:

- `results/samples/figures/phase1/aggregate_requests_per_second_by_concurrency.png`
- `results/samples/figures/phase1/aggregate_output_tokens_per_second_by_concurrency.png`

Latency plots:

- `results/samples/figures/phase1/avg_latency_by_concurrency.png`
- `results/samples/figures/phase1/p95_latency_by_concurrency.png`
- `results/samples/figures/phase1/p99_latency_by_concurrency.png`

TTFT plots:

- `results/samples/figures/phase1/avg_ttft_by_concurrency.png`
- `results/samples/figures/phase1/p95_ttft_by_concurrency.png`
- `results/samples/figures/phase1/p99_ttft_by_concurrency.png`

TPOT plots:

- `results/samples/figures/phase1/avg_tpot_by_concurrency.png`
- `results/samples/figures/phase1/p95_tpot_by_concurrency.png`
- `results/samples/figures/phase1/p99_tpot_by_concurrency.png`

Workload comparison plots:

- `results/samples/figures/phase1/workload_avg_latency_at_conc32.png`
- `results/samples/figures/phase1/workload_aggregate_requests_at_conc32.png`
- `results/samples/figures/phase1/workload_p99_latency_at_conc32.png`
- `results/samples/figures/phase1/workload_p99_ttft_at_conc32.png`

Trade-off and reliability plots:

- `results/samples/figures/phase1/throughput_vs_avg_latency.png`
- `results/samples/figures/phase1/throughput_vs_p99_latency.png`
- `results/samples/figures/phase1/aggregate_requests_vs_p99_ttft.png`
- `results/samples/figures/phase1/failure_count_by_workload_concurrency.png`
- `results/samples/figures/phase1/success_count_by_workload_concurrency.png`

Manifest:

- `results/samples/figures/phase1/plot_manifest.json`

Future plot needs:

- Future quality vs latency chart.
- Future model size vs throughput chart.

## Interview Talking Points

- Inference bottlenecks can be separated into first-token latency, decode speed, tail latency, aggregate throughput, and reliability.
- TTFT and TPOT are different because prefill/queueing and decode are different parts of the serving path.
- Concurrency sweeps matter because they reveal the tradeoff between total served work and tail latency.
- Checkpointing matters because long GPU runs can be interrupted, and losing end-of-run outputs wastes both time and budget.
- Correctness must be added because fast responses are not useful if they are incomplete, malformed, or wrong.
- This project can create business value by turning model-serving choices into measurable tradeoffs around latency, throughput, reliability, and quality.

## Appendix A: Artifact Index

- `results/samples/processed/`: curated comparison CSVs, including 1,000-prompt and 5,000-prompt concurrency summaries.
- `results/samples/raw/`: curated result CSVs, generation JSONL traces, system metadata, run metadata, and per-configuration samples.
- `results/samples/logs/`: curated progress logs for chunked vLLM runs.
- `results/samples/checkpoints/`: curated checkpoint JSON files for resumable vLLM runs.
- `docs/19_phase1_experiment_inventory.md`: source inventory for completed Phase 1 experiments.
- `docs/15_resumable_benchmarking_plan.md`: failure-recovery and checkpointing plan.
- `docs/13_hf_vs_vllm_calibration_comparison.md`: HF vs vLLM calibration comparison.

## Appendix B: Reproduction Commands

Representative validation command:

```text
inference-bench validate-config
```

Representative scaled workload generation command:

```text
inference-bench generate-workloads --count 100 --output-dir data/prompts/scaled --seed 42
```

Representative vLLM load benchmark command:

```text
inference-bench openai-load-run --workload-path data/prompts/scaled/long_context_1000.jsonl --output-path results/raw/vllm_long_context_1000_conc32_chunked_results.csv --generation-output-path results/raw/vllm_long_context_1000_conc32_generations.jsonl --run-metadata-path results/raw/vllm_long_context_1000_conc32_metadata.json --checkpoint-path results/raw/vllm_long_context_1000_conc32_checkpoint.json --log-path results/raw/vllm_long_context_1000_conc32.log --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --run-id vllm-long-context-1000-conc32 --backend vllm --optimization vllm_baseline --concurrency 32 --max-new-tokens 96 --stream --chunk-size 100 --progress-interval 100
```

Representative comparison command:

```text
inference-bench compare-results --input-csv results/raw/vllm_short_chat_results.csv --input-csv results/raw/vllm_code_helpdesk_results.csv --output-csv results/processed/vllm_workload_comparison.csv
```

Representative artifact promotion command:

```text
bash scripts/promote_sample_artifacts.sh
```

## Appendix C: Glossary

- TTFT: Time To First Token, or the delay before the first generated token arrives.
- TPOT: Time Per Output Token, or decode time per generated token after the first token.
- Prefill: Processing the input prompt before generation starts.
- Decode: Generating output tokens after prefill.
- KV cache: Cached key/value attention state used to avoid recomputing previous tokens.
- Concurrency: Number of requests allowed in flight at once.
- Throughput: Amount of work completed per unit time, measured here as requests/sec or output tokens/sec.
- p95/p99 latency: Tail latency values where 95% or 99% of requests are at or below the reported latency.
- Checkpoint: A saved progress record used to resume long benchmark runs.
- vLLM: A high-throughput LLM serving runtime used in Phase 1 GPU experiments.
- OpenAI-compatible endpoint: An HTTP API shape compatible with OpenAI-style chat/completion clients.
