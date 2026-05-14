# Scaled Benchmark And Concurrency Stress Plan

## Purpose

The benchmark suite has already validated local calibration runs, but final inference-engineering conclusions require larger workloads, concurrency stress, backend comparisons, and optimization variants.

## Why Smoke Tests Are Not Enough

Runs with 3 to 5 prompts validate instrumentation only. They are useful for confirming loaders, runners, metrics, traces, summaries, plots, and metadata. They are not enough for final performance claims.

Serious results require larger prompt sets, repeated runs, documented hardware, and concurrency tests.

## Serious Benchmark Dimensions

- Backend/runtime
- Model size
- Workload type
- Concurrency level
- Optimization strategy
- Hardware environment
- Output quality / structured validity

## Workload Scale Targets

| scale | prompts per workload | purpose |
| --- | ---: | --- |
| Calibration | 3 to 5 prompts | Validate instrumentation only |
| Small benchmark | 50 prompts | First meaningful local/backend comparison |
| Medium benchmark | 100 to 500 prompts | Stable comparison and plotting |
| Large benchmark | 1,000+ prompts where feasible | Serious serving stress test, likely GPU/cloud required |

Deterministic synthetic workloads can be generated with `inference-bench generate-workloads`. The 100-prompt files are intended for first concurrency validation, while 1,000-prompt files are intended for serious GPU benchmark runs. Larger generated files may be produced on RunPod and should not be committed unless they are intentionally curated.

## Concurrency Targets

- Concurrency 1
- Concurrency 4
- Concurrency 8
- Concurrency 16
- Concurrency 32

Concurrency stress is especially important for vLLM because serving engines are designed to improve throughput under concurrent load. Single-request latency and multi-request throughput are different evaluation problems. Concurrency results should be interpreted with averages and tail latency percentiles, including p50, p90, p95, and p99.

Concurrency testing will use the `openai-load-run` command against an OpenAI-compatible endpoint. For vLLM, this provides the next measurement layer after single-request baseline runs while keeping the benchmark client independent from the server startup workflow.

## Backend Comparison Plan

Phase A: Hugging Face baseline.

Phase B: vLLM OpenAI-compatible server/client baseline.

Phase C: vLLM optimization variants.

Phase D: Optional SGLang comparison after vLLM is stable.

The OpenAI-compatible runner and `openai-load-run` will be used for vLLM concurrency and backend comparison experiments once a reviewed vLLM server environment is available.

## Optimization Comparison Plan

- Hugging Face baseline: establishes the local model-execution reference point.
- vLLM baseline: tests serving runtime behavior before enabling optimization variants.
- Quantization: tests whether reduced precision improves memory use, throughput, or deployability while preserving output quality.
- Prefix caching: tests whether repeated prompt prefixes reduce prefill cost for shared-prefix workloads.
- Speculative decoding: tests whether draft-model assisted decoding improves generation speed.
- Combined optimized configuration: tests the practical combined effect of compatible optimizations.

## Model Scale Plan

- `Qwen/Qwen2.5-0.5B-Instruct` for local smoke.
- `Qwen/Qwen2.5-1.5B-Instruct` for small model baseline.
- `Qwen/Qwen2.5-7B-Instruct` for first serious GPU benchmark.
- `Qwen/Qwen2.5-32B-Instruct` for scale comparison if compute allows.
- Larger placeholder model only if budget/hardware permits.

## Metrics To Collect

- TTFT
- TPOT
- End-to-end latency
- Throughput
- p50 latency
- p90 latency
- p95 latency
- p99 latency
- Input tokens
- Output tokens
- Success rate
- Failure rate
- Structured-output validity
- Peak memory
- Estimated cost
- System metadata

## Repetition And Warmup Policy

Warmup requests should be excluded from final summaries when feasible. Repeated runs reduce noise, and final reports should state the number of runs and sample sizes. First-run model download/load time should not be mixed with per-request inference timing.

## Stress-Test Risks

- Local CPU results are not representative of production GPU serving.
- Windows may not be ideal for vLLM execution.
- GPU results are hardware-specific.
- Concurrency can increase tail latency.
- Quantization may affect quality and format adherence.
- Large models may require tensor parallelism or larger GPUs.

## Success Criteria

A successful scaled benchmark should produce:

- Reproducible configs
- Documented hardware
- Workload-specific result CSVs
- Prompt-level JSONL traces
- Comparison CSVs
- Plots
- Structured-output validity summaries
- Clear limitations
- Enough evidence to compare HF baseline, vLLM baseline, and optimization variants
