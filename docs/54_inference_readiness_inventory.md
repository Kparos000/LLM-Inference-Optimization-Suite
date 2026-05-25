# Inference Readiness Inventory

This inventory audits the repository after the promoted 10,000-record benchmark
dataset and public EDA layer. It is a planning artifact only: no new inference
code, GPU experiments, external API calls, embeddings, vector indexes, or RAG
execution are introduced here.

The current public EDA location is `data/generated/dataset_10000/`. The older
`data/generated/eda/dataset_10000/` path is legacy and should not be used for
Phase 2B planning.

## 1. What Has Already Been Built

- A promoted five-vertical benchmark dataset under `data/scaleup_2000_full/`.
- Dataset totals: 10,000 prompts, 10,000 gold/eval records, and 4,740 promoted
  KB rows.
- Public EDA outputs under `data/generated/dataset_10000/`, including dashboard,
  interactive charts, static plots, word clouds, word views, per-vertical pages,
  JSON reports, and CSV summaries.
- A Finance convenience EDA entry point under `data/generated/finance/`.
- Research AI full retrieval corpus export artifacts under
  `data/generated/phase2a/retrieval_corpus/research_ai/`.
- A Python benchmark harness with YAML model/workload/experiment configs.
- Mock, Hugging Face, OpenAI-compatible, and concurrent OpenAI-compatible load
  runner foundations.
- Result CSV and generation JSONL schemas.
- Client-side TTFT, TPOT, end-to-end latency, token throughput, token counts, and
  cost placeholder fields.
- Summary, comparison, and plotting utilities for benchmark CSVs.
- System metadata capture for platform, Python, torch/CUDA availability, CUDA
  device names, and transformers version.
- Curated sample Phase 1 vLLM artifacts showing concurrency runs and aggregate
  throughput metadata.

## 2. Datasets And Workloads Available

The promoted benchmark dataset has five verticals:

| vertical | prompts | gold/evals | KB rows |
| --- | ---: | ---: | ---: |
| airline | 2,000 | 2,000 | 300 |
| healthcare_admin | 2,000 | 2,000 | 300 |
| retail | 2,000 | 2,000 | 1,000 |
| finance | 2,000 | 2,000 | 1,540 |
| research_ai | 2,000 | 2,000 | 1,600 |

The existing synthetic harness workloads remain available for calibration:
`smoke`, `structured_output_smoke`, `short_chat`, `code_helpdesk`,
`long_context`, and `shared_prefix`.

The EDA workload-shape report ranks context pressure as:

1. `research_ai`: high pressure, about 452.7 estimated input tokens per prompt.
2. `finance`: high pressure, about 179.9 estimated input tokens per prompt.
3. `retail`: medium pressure, about 143.0 estimated input tokens per prompt.
4. `airline`: medium pressure, about 130.3 estimated input tokens per prompt.
5. `healthcare_admin`: low pressure, about 91.2 estimated input tokens per prompt.

## 3. What Can Currently Be Measured

Current result rows can capture:

- `ttft_ms` when streaming is enabled.
- `tpot_ms`.
- `end_to_end_latency_ms`.
- `throughput_tokens_per_second` per request.
- `input_tokens` and `output_tokens`.
- `peak_memory_mb` as a schema field, although current runners usually leave it
  blank.
- `estimated_cost_usd` as a schema field, currently zero for local runs.
- request success/failure and error messages.

Current aggregate tooling can calculate:

- p50/p90/p95/p99 latency, TTFT, TPOT, and throughput from result CSVs.
- aggregate request throughput and output-token throughput for the concurrent
  OpenAI-compatible load runner metadata.
- row counts, success counts, failure counts, backend labels, model labels,
  optimization labels, and workload labels.
- structured JSON validity for generation JSONL files.

Current EDA can measure:

- dataset inventory and manifest alignment.
- prompt, gold, and KB length distributions.
- estimated prompt, KB, and expected output tokens.
- evidence coverage and reuse concentration.
- safety/domain-boundary flags in the source dataset.
- per-vertical task, status, and output-format mix.

## 4. What Cannot Currently Be Measured

The current repo cannot yet measure these in a production-quality way:

- Server-side queue time.
- Actual server batch size or continuous-batching composition over time.
- Prefill start/end and decode start/end as separate backend events.
- Native prefill/decode token throughput.
- GPU utilization, power draw, temperature, memory bandwidth, or SM occupancy.
- GPU memory headroom under live load.
- KV cache allocated blocks, used blocks, evictions, fragmentation, or hit rate.
- PagedAttention block-table behavior.
- Prefix cache hit rate, prefix-cache memory use, or prefix-aware routing benefit.
- Estimated MFU tied to hardware peak FLOPs and observed token throughput.
- Tensor/pipeline/data/expert parallel efficiency.
- vLLM-vs-SGLang comparisons.
- Retrieval recall@k, citation recall, groundedness, or answer correctness against
  the promoted gold/eval records.
- Bounded agentic workflow cost/latency per step.

## 5. Needed Before The First GPU Run

Before running a GPU experiment, implement or document:

- A Phase 2B context-record schema that joins prompt, gold, selected KB context,
  context mode, retrieval metadata, and token estimates.
- Deterministic no-context and gold-context modes for the 10,000-record dataset.
- A small smoke subset, a calibration subset, and a stratified first GPU subset.
- Exact tokenizer-based token counts for each model under test.
- A result schema extension for context mode, retrieval mode, context tokens,
  queue time, server timing fields, concurrency, and run hardware ID.
- A run manifest that records backend version, model, dtype, quantization,
  max context, max batch/concurrency, GPU type, driver, CUDA, and container image.
- A telemetry collection plan using `nvidia-smi dmon` or DCGM/NVML sampling.
- Backend metrics capture for vLLM and, later, SGLang.
- A no-paid-GPU checklist confirming that local validation, expected outputs, and
  artifact paths are reviewed.

## 6. Hardware Telemetry Needed

Minimum telemetry per run:

- GPU name, count, driver, CUDA runtime, memory capacity, and interconnect.
- GPU utilization percentage over time.
- GPU memory used, free, reserved, and headroom over time.
- Power draw and power limit.
- Temperature and thermal throttling indicators.
- CPU utilization and host RAM use.
- Disk read/write pressure if large context files are streamed.
- Network throughput if the client and server run on separate machines.
- Wall-clock timestamps aligned with request-level result timestamps.

Preferred telemetry:

- DCGM fields for SM activity, tensor-core activity, memory copy utilization,
  PCIe/NVLink throughput, ECC errors, and clock throttling reasons.
- Backend-exported metrics for scheduler queues, batch sizes, KV cache blocks,
  prefix cache, and token throughput.

## 7. vLLM Metrics Needed

For vLLM, collect:

- Request count, running requests, waiting requests, and finished requests.
- Scheduler queue time.
- Iteration-level batch size and token counts.
- Prompt/prefill tokens processed per second.
- Decode tokens processed per second.
- TTFT, TPOT, inter-token latency, and end-to-end latency.
- KV cache block usage, free blocks, evictions, and allocation failures.
- Prefix cache hit rate, if prefix caching is enabled.
- PagedAttention block pressure and memory utilization.
- GPU memory utilization and configured `gpu_memory_utilization`.
- Chunked prefill setting, max model length, max batched tokens, max sequences,
  tensor parallel size, pipeline parallel size, and dtype.

## 8. SGLang Metrics Needed

For SGLang, collect the comparable metrics:

- Request queue depth and queue time.
- Running batch size and token mix.
- Prefill and decode token throughput.
- TTFT, TPOT, inter-token latency, and end-to-end latency.
- KV cache pool usage and eviction behavior.
- Radix/prefix cache hit rate, if enabled.
- Memory pool usage and out-of-memory events.
- Speculative decoding metrics if used later.
- Tensor parallel settings, data parallel settings, expert parallel settings for
  MoE models, max running requests, context length, and dtype.

## 9. How To Estimate MFU

MFU should be reported as an estimate until profiler-backed FLOP accounting is
available.

Use a dense-model decode approximation:

```text
estimated_decode_MFU =
  (2 * parameter_count * generated_tokens_per_second)
  / hardware_peak_flops_per_second
```

For total workload MFU, also estimate prompt/prefill work:

```text
estimated_total_MFU =
  estimated_model_flops_for_prompt_and_decode
  / (wall_clock_seconds * hardware_peak_flops_per_second)
```

Required inputs:

- model parameter count and architecture notes.
- dtype and expected tensor-core peak FLOPs for the GPU.
- measured generated tokens/sec and prompt tokens/sec.
- batch/concurrency level.
- context tokens and output tokens.

Limitations:

- Attention cost, long-context prefill, MoE sparsity, quantization, speculative
  decoding, tensor parallel communication, and CPU/network overhead make this an
  estimate rather than a precise utilization measure.
- Report MFU with hardware, model, dtype, backend, and measurement window.

## 10. How To Separate Prefill And Decode

Client-side streaming gives:

- request start.
- first token timestamp.
- request end.

That is enough for TTFT and TPOT, but not enough to isolate queue, prefill, and
decode. For Phase 2C, capture:

- client request start/end.
- server receive timestamp.
- scheduler enqueue/dequeue timestamps.
- prefill start/end.
- first decode token timestamp.
- decode end.
- prompt tokens, generated tokens, and cache-hit status.

Then compute:

- `queue_time = prefill_start - server_receive`.
- `prefill_time = prefill_end - prefill_start`.
- `decode_time = decode_end - first_decode_token`.
- `ttft = first_token_client_observed - request_start`.
- `server_ttft = first_decode_token - server_receive`.

## 11. How To Test Batching And Continuous Batching

Use the existing `openai-load-run` runner as the client foundation, then add
backend-native batch telemetry.

Initial grid:

- context modes: `no_context`, `gold_context`.
- concurrency: 1, 4, 8, 16, 32.
- prompt counts: 100 for smoke, 500 for calibration, 2,000 for vertical-level
  runs.
- verticals: start with healthcare_admin, finance, and research_ai to cover low,
  high structured, and high context pressure.

Measure:

- p50/p95/p99 TTFT, TPOT, and end-to-end latency.
- aggregate requests/sec and tokens/sec.
- backend iteration batch size.
- queue depth and queue time.
- GPU utilization and memory headroom.

Continuous batching is ready to evaluate only when server-side batch composition
is captured, not just client concurrency.

## 12. How To Test Variable-Length Request Behavior

Build stratified subsets from the EDA token profiles:

- short prompt / short context / short output.
- short prompt / long context.
- long prompt / short output.
- long prompt / long expected output.
- multi-evidence prompts.
- high context-pressure research_ai prompts.
- finance calculation and structured-output prompts.

Run mixed and bucketed schedules:

- mixed natural ordering.
- sorted by estimated context tokens.
- bucketed by context length.
- adversarial mix with many long-context prompts inserted into short traffic.

Compare TTFT, TPOT, queue time, batch size, KV memory, and p99 latency tails.

## 13. How To Test KV Cache And Memory Pressure

Create context-token sweeps:

- no_context.
- gold_context.
- top-k retrieval context with k = 1, 3, 5, 8.
- fixed context budgets such as 512, 1,024, 2,048, 4,096, and 8,192 tokens.

Increase concurrency until one of these occurs:

- queue time grows sharply.
- p99 TTFT grows sharply.
- GPU memory headroom drops below a safety threshold.
- backend reports KV cache allocation pressure or evictions.
- request failures or OOM errors appear.

Record KV cache usage, PagedAttention block usage, memory headroom, and whether
the bottleneck is prefill-bound, decode-bound, scheduler-bound, or memory-bound.

## 14. How To Test Prefix Caching And Prefix-Aware Routing

Use existing `shared_prefix` synthetic workloads and new Phase 2B prompt
templates with repeated system instructions.

Test modes:

- prefix caching disabled.
- prefix caching enabled.
- prefix-aware ordering or routing by template/system prefix.
- mixed-prefix traffic.

Required measurements:

- prefix cache hit rate.
- TTFT improvement on repeated prefixes.
- memory overhead of prefix cache.
- whether prefix-aware routing improves throughput or harms fairness.

Do not claim prefix caching benefit from client-side latency alone; require
backend prefix-cache metrics or an equivalent controlled A/B design.

## 15. Model Sharding Strategy Plan

Start single-GPU before sharding:

- Qwen2.5 0.5B or 1.5B for smoke.
- Qwen2.5 7B for first serious GPU benchmark if memory permits.

Sharding options to plan:

- Tensor parallelism: first choice for models that fit across multiple GPUs but
  need fast interconnect.
- Pipeline parallelism: useful when layers must be split and batch sizes can keep
  stages busy, but it adds scheduling bubbles.
- Data parallelism: useful for independent replicas and routing, not a single
  request that exceeds one GPU.
- Expert parallelism: relevant only for MoE models; not a first-run requirement.

For each sharding run, record:

- model size, dtype, quantization, context length, and memory estimate.
- GPU count, interconnect, tensor/pipeline/data/expert parallel sizes.
- throughput scaling, latency impact, memory headroom, and communication
  overhead.

## 16. Phase 2B Implementation Roadmap

1. Define context-record and retrieval-result schemas.
2. Create deterministic dataset-to-workload conversion for the 10,000-record
   benchmark.
3. Implement `no_context` and `gold_context` prompt assembly.
4. Normalize KB records into a shared chunk/citation schema.
5. Implement lexical retrieval first, then dense/hybrid/reranked modes.
6. Add context budgeting and truncation policies.
7. Add retrieval metrics: recall@k, required evidence coverage, citation recall.
8. Add generation evaluation: groundedness, format validity, must-include,
   must-not-include, safety/domain-boundary checks.
9. Add bounded agentic workflow fixtures without autonomous tool execution.
10. Produce small committed samples and larger ignored generated outputs.

## 17. Phase 2C Implementation Roadmap

1. Extend result schemas for context mode, retrieval mode, queue time, batch
   metadata, telemetry IDs, and backend metrics.
2. Add a run manifest schema for GPU/backend configuration.
3. Add telemetry collectors and align telemetry timestamps with request results.
4. Add vLLM metrics adapter.
5. Add SGLang metrics adapter.
6. Add MFU estimator using model metadata, hardware peak FLOPs, and observed
   prompt/decode throughput.
7. Add prefill/decode disaggregation fields.
8. Run first GPU smoke with no-context and gold-context modes.
9. Run batching, variable-length, context-length, KV-cache, and prefix-cache
   sweeps.
10. Add bottleneck diagnosis summaries and compare vLLM against SGLang.

## 18. First Inference Experiment Design

Purpose: validate Phase 2B context assembly and Phase 2C instrumentation before
large GPU spend.

Candidate design:

- Backend: vLLM OpenAI-compatible server.
- Model: Qwen2.5 0.5B for smoke, then Qwen2.5 1.5B or 7B after telemetry is
  stable.
- Dataset subset: 250 prompts stratified across five verticals, task types,
  output formats, evidence counts, and estimated context-token buckets.
- Context modes: `no_context` and `gold_context`.
- Retrieval mode: none for first smoke; lexical retrieval only after the context
  schema is validated.
- Concurrency: 1, 4, 8, 16.
- Max new tokens: use vertical/task defaults derived from gold answer lengths.
- Required outputs: result CSV, generation JSONL, run manifest JSON, telemetry
  time series, backend metrics snapshot, and evaluation report.
- Stop condition: any OOM, critical safety/eval schema failure, or telemetry
  capture failure.

## 19. Risks And Blockers

- The promoted benchmark is ready, but Phase 2B context assembly is not yet
  implemented.
- Current token counts are mixed: exact tokenizer counts exist for HF runs, while
  OpenAI-compatible runs use whitespace counts.
- Current load runner captures concurrency, but not server batch composition.
- Current system metadata is not enough for GPU telemetry or MFU.
- Current `peak_memory_mb` fields are usually blank.
- vLLM/SGLang backend metrics are not yet captured.
- Retrieval recall and groundedness are not implemented.
- Research AI full corpus exists, but no retrieval index or retrieval evaluation
  mode has been built.
- Agentic workflows need strict bounds before they can be safely benchmarked.

## 20. Exact Next Implementation Steps

1. Add Phase 2B schemas for context records, retrieval results, assembled prompts,
   and evaluation records.
2. Add a dataset converter that turns `data/scaleup_2000_full/` into generated
   workload JSONL files with `vertical`, `task_type`, `expected_status`,
   `expected_output_format`, `context_mode`, `retrieval_mode`, `gold_id`, and
   evidence metadata.
3. Implement `no_context` and `gold_context` assembly only.
4. Add exact tokenizer counting for planned models before GPU runs.
5. Extend benchmark result rows with `context_tokens`, `queue_time_ms`,
   `prefill_ms`, `decode_ms`, `batch_size`, `concurrency`, `telemetry_run_id`,
   `context_mode`, and `retrieval_mode`.
6. Add a run manifest writer.
7. Add local telemetry schema and GPU telemetry collection adapters, without
   running them yet.
8. Add vLLM metrics ingestion from server logs or metrics endpoint.
9. Add SGLang metrics ingestion once SGLang is selected for comparison.
10. Add retrieval recall@k and groundedness evaluators.
11. Create the first GPU experiment config and dry-run all artifact paths.
12. Only then run the first GPU smoke experiment.
