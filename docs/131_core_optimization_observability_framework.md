# Core Optimization Observability Framework

Status: implemented as a planning and instrumentation framework; no inference
executed; no GPU required; no core optimization result claimed.

## Purpose

Core optimization cannot start with a toggle. It has to start with evidence.
For every engineer-applied inference optimization, the project now records the
same reasoning chain:

```text
Problem -> Mechanism -> Instrumentation -> One-factor experiment -> Measured result -> Decision
```

This framework tells the future product UI and future experiment runner what
must be measured before an optimization can be called successful. Missing
engine counters remain missing. Static estimates remain estimates.

## Source Files

Implementation:

- `src/inference_bench/core_optimization_observability.py`
- `scripts/phase4/build_core_optimization_observability.py`
- `tests/test_core_optimization_observability.py`

Authoritative config:

- `configs/core_optimization_observability.yaml`
- `configs/core_optimization_scenario_registry.yaml`

Generated planning artifacts:

- `experiments/main/main_inference_v1/processed/core_optimization_observability_registry.json`
- `experiments/main/main_inference_v1/processed/core_optimization_observability_readiness.json`
- `experiments/main/main_inference_v1/processed/core_optimization_observability_readiness.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_observability_inventory.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_observability_inventory.csv`
- `experiments/main/main_inference_v1/processed/core_optimization_event_schema.json`
- `experiments/main/main_inference_v1/processed/core_optimization_ui_observability_contract.json`
- `experiments/main/main_inference_v1/processed/core_optimization_missing_instrumentation.json`
- `experiments/main/main_inference_v1/processed/core_optimization_adapter_coverage.json`
- `experiments/main/main_inference_v1/processed/coreopt_prefix_layout_static_v1_instrumentation_plan.json`
- `experiments/main/main_inference_v1/processed/coreopt_scheduler_batch_vllm_v1_instrumentation_plan.json`
- `experiments/main/main_inference_v1/processed/coreopt_prefix_cache_vllm_v1_instrumentation_plan.json`
- `experiments/main/main_inference_v1/processed/coreopt_chunked_prefill_sglang_v1_instrumentation_plan.json`
- `experiments/main/main_inference_v1/processed/coreopt_prefix_layout_static_v1_prefix_opportunity_analysis.json`
- `experiments/main/main_inference_v1/processed/coreopt_prefix_layout_static_v1_prefix_opportunity_analysis.csv`

## Audit Findings

The saved Main_Inference_V1 artifacts already provide aggregate run,
evaluation, cost, comparison, progress, and GPU telemetry fields. They do not
provide enough serving-engine internals to prove cache, scheduler, KV-cache,
chunked-prefill, quantization, speculation, kernel, or distributed-serving
effects.

| Field family | Captured now | Derivable now | Missing for measured optimization claim |
| --- | --- | --- | --- |
| Run identity | manifest fields | config/workload hashes where present | per-scenario changed-factor hash for future runs |
| Request lifecycle | completion progress checkpoints | coarse replay progress | per-request queue, schedule, prefill, decode, first-token events |
| Performance | TTFT, TPOT, E2E, aggregate throughput | percentile summaries and SLO rows | per-engine-iteration batch metrics |
| Resources | GPU utilization, VRAM, power, temperature, process names | telemetry summaries | KV-cache blocks, model memory split, per-engine idle intervals |
| Quality/safety | JSON, contract, format, evidence, groundedness, safety, truncation | SLO failure rows and diagnosis inputs | none for current aggregate quality gates |
| Cost | GPU/API/total cost | cost per request and projections | route-level optimized comparison cost |
| Prefix layout | rendered workload prompts | exact prefix-family estimates | actual engine prefix-cache hits |
| Prefix cache | no hit/miss counters | static prefix opportunity only | hit/miss counts, hit tokens, occupancy, evictions |
| Scheduler/batch | configured concurrency | aggregate concurrency comparisons | queue wait, active batch size, scheduled tokens |
| KV cache | GPU memory aggregates | formula-based future estimates | block counts, occupancy, evictions, preemptions |
| Chunked prefill | aggregate latency by config | length-bucket planning | chunk counts and engine iteration composition |
| Speculation | not run | none | draft/accept/reject counters |

All public UI fields are safe because they come from saved artifacts,
configuration, deterministic static workload analysis, or explicit unavailable
states. No `.env` values, provider secrets, or raw full response payloads are
exposed.

## Registry Architecture

The observability registry covers the 15 authoritative core optimization IDs
from `configs/core_optimization_taxonomy.yaml` and excludes all deployability
repair IDs. Each entry records:

- optimization ID and display name;
- optimization domain and difficulty tier;
- problem statement;
- mechanism;
- hypothesis;
- primary and secondary metrics;
- protected quality/safety metrics;
- required, optional, existing, and missing instrumentation;
- derivation method;
- sampling frequency and temporal resolution;
- aggregation levels;
- engine-, model-, and device-specific fields;
- external profiler requirements;
- experiment type;
- UI visualization contract;
- acceptance and rejection evidence;
- instrumentation readiness state.

Readiness states are:

- `ready_existing`
- `ready_derivable`
- `requires_runner_instrumentation`
- `requires_engine_metrics`
- `requires_external_profiler`
- `unsupported_current_runtime`
- `future_architecture`

Domain and difficulty are separate axes. For example, `quantization` is a
domain, while its difficulty tier remains a planning attribute.

## Metric Envelopes

The framework defines common envelopes so future one-factor experiments use
consistent field names:

- Run identity: run, scenario, optimization, model, engine, backend, memory
  mode, concurrency, hardware, precision, git commit, hashes, timestamps, and
  status.
- Request lifecycle: request/prompt/config IDs, arrival, queue, schedule,
  prefill, first token, decode, completion, retry/failure, and token counts.
- Performance: TTFT, queue wait, prefill latency, TPOT, ITL, E2E,
  requests/sec, tokens/sec, successful requests/sec, and batch throughput.
- Resources: GPU utilization, VRAM, power, temperature, CPU/RAM, model memory,
  KV-cache memory, telemetry timestamp, and active engine/process.
- Quality/safety: JSON, contract, format, evidence ID presence, evidence
  match, groundedness, safety, truncation, completion/failure rate,
  escalation correctness, insufficient-evidence correctness, and MM4 bounds.
- Cost: GPU cost, API cost, total cost, cost/request, cost/1,000 requests,
  cost/successful request, tokens/GPU dollar, and tokens/API dollar where
  meaningful.

Backends do not have to emit fields they cannot truthfully expose.

## Prefix Opportunity

`coreopt_prefix_layout_static_v1` is the first recommended core task because
it is CPU-only. The new static analyzer scans:

```text
data/workloads/final_10000/prompt_plus_metadata/mm2_hybrid_top5.jsonl
```

It produced a planning estimate over 10,000 rows:

- rows scanned: 10,000;
- prefix family count: 1;
- mean reusable-token ratio estimate: 0.07598;
- tokenization: `repo_regex_tokenizer_v1`;
- historical cache reuse claimed: false;
- semantic similarity counted: false.

The result is intentionally not a cache-hit measurement. Prefix reuse means
token-identical leading tokens only. Because historical engine tokenization and
cache counters were not captured in Main_Inference_V1, this analysis is
labeled `estimated`.

The first full static scenario is now complete under:

```text
experiments/optimizations/coreopt_prefix_layout_static_v1/
```

It scans 40,000 authoritative rendered workload rows across mm0-mm3
`prompt_plus_metadata` workloads, compares `baseline_prompt_layout_v1` with
`prefix_optimized_prompt_layout_v1`, and stores hashes/metrics without raw
prompt text. It is `measured_static_analysis`, not a latency experiment. The
derived mean exact common prefix increased from 29 to 358 tokens, with total
input tokens unchanged. The scenario decision is `MISSING_CONFIGURATION`
because no acceptance threshold is configured for promoting static prefix
layout into engine validation. See
`docs/132_coreopt_prefix_layout_static_v1.md`.

## Event Schema

The unified event schema is saved at:

```text
experiments/main/main_inference_v1/processed/core_optimization_event_schema.json
```

It validates discriminated event payloads for:

`run_started`, `config_started`, `request_arrived`, `request_queued`,
`request_scheduled`, `prefill_started`, `prefill_chunk_completed`,
`prefix_cache_lookup`, `prefix_cache_hit`, `prefix_cache_miss`,
`kv_cache_allocated`, `kv_cache_evicted`, `batch_iteration`, `decode_token`,
`request_completed`, `request_failed`, `telemetry_sample`,
`quality_evaluation`, `optimization_decision`, `prompt_layout_rendered`,
`prefix_family_assigned`, `static_metric_computed`, and `run_completed`.

Each event carries schema version, timestamp, run/scenario/config identity,
engine, model, optimization ID, source, measurement type, event type, and a
typed payload. Event payloads reject unrelated fields.

## Adapter Coverage

Implemented adapters are read-only:

- Main_Inference manifest adapter;
- progress log adapter;
- GPU telemetry adapter;
- evaluation report adapter;
- cost report adapter;
- vLLM/SGLang engine metrics unavailable adapter;
- static prefix-layout adapter.

The saved artifacts parse without live engines. Engine metrics that were not
captured are reported as missing, including cache hit counts, active batch
size, KV block counters, chunk count, and speculative acceptance rate.

## Scenario Readiness

The first scenario is complete as static analysis. Runtime one-factor
scenarios remain planned:

| Scenario | Optimization | Result type | Readiness | GPU needed now |
| --- | --- | --- | --- | --- |
| `coreopt_prefix_layout_static_v1` | prompt prefix layout | `measured_static_analysis` | `ready_derivable` | No |
| `coreopt_scheduler_batch_vllm_v1` | scheduler/batch tuning | `planned` | `requires_runner_instrumentation` | Not until instrumentation exists |
| `coreopt_prefix_cache_vllm_v1` | prefix-cache verification | `planned` | `requires_engine_metrics` | Not until engine metrics exist |
| `coreopt_chunked_prefill_sglang_v1` | chunked prefill | `planned` | `requires_engine_metrics` | Not until engine metrics exist |

`Optimized_Inference_V1` is still `missing_not_created`; no champion is
selected.

## API And UI Contract

FastAPI exposes read-only planned endpoints:

- `/api/optimizations/observability/registry`
- `/api/optimizations/observability/readiness`
- `/api/optimizations/observability/inventory`
- `/api/optimizations/observability/prefix-opportunity`
- `/api/optimizations/observability/event-schema`
- `/api/optimizations/observability/missing-instrumentation`
- `/api/optimizations/observability/cards`
- `/api/optimizations/coreopt-prefix-layout-static-v1`
- `/api/optimizations/coreopt-prefix-layout-static-v1/summary`
- `/api/optimizations/coreopt-prefix-layout-static-v1/layouts`
- `/api/optimizations/coreopt-prefix-layout-static-v1/prefix-metrics`
- `/api/optimizations/coreopt-prefix-layout-static-v1/equivalence`
- `/api/optimizations/coreopt-prefix-layout-static-v1/decision`
- `/api/optimizations/coreopt-prefix-layout-static-v1/story`

The Next.js Optimization Lab renders a compact observability card set. Each
card shows:

```text
Problem -> Mechanism -> Instrumentation -> Experiment -> Result -> Decision
```

The UI must distinguish measured, derived, estimated, unavailable, planned,
and future values. Missing telemetry must never render as zero.

## Current Limitations

- No live vLLM/SGLang metrics endpoint scrape exists for Main_Inference_V1.
- No per-request queue/scheduler timeline exists in the saved baseline.
- No actual prefix-cache hit/miss or KV-cache block counters exist.
- No speculative decoding run exists.
- No quantized or TensorRT-LLM run exists.
- No optimized artifacts exist.
- No static acceptance threshold exists for
  `minimum_reusable_token_ratio_delta_for_engine_validation`.

## Exact Next Task

Configure the missing static acceptance threshold, review
`coreopt_prefix_layout_static_v1`, and design
`coreopt_prefix_layout_engine_validation_v1`. The next experiment must run a
one-factor engine validation with protected quality/safety gates and actual
latency or cache metrics before any runtime improvement claim is accepted.
