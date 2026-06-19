# Production Runtime Registry

Status: implemented June 19, 2026

Phase 1B adds a typed production runtime registry at
`configs/runtime_engines.yaml` and `src/inference_bench/runtime_registry.py`.
It is a configuration and selection guard only; it does not start servers,
call APIs, allocate GPU resources, or run inference.

## Architecture Layering

The project now separates execution planning into four layers:

1. Runtime: Hugging Face Transformers, vLLM, SGLang, API provider route, and
   TensorRT-LLM planned placeholder.
2. Infrastructure: developer workstation, remote RTX 3070, RunPod GPU, or
   provider-managed API infrastructure.
3. Tooling: runners, manifests, telemetry, checkpointing, pricing, and result
   schemas.
4. Evaluation: deterministic JSON/contract, evidence, groundedness, safety,
   latency, throughput, and cost gates.

This keeps model capability, serving engine, hardware ownership, and evaluator
semantics separate.

## Runtime Status

| Runtime | Engine | Backend type | Status | Live selectable |
| --- | --- | --- | --- | --- |
| `huggingface_transformers` | `huggingface` | `local_compute` | `ready` | yes |
| `vllm` | `vllm` | `self_hosted_gpu` | `ready` | yes |
| `sglang` | `sglang` | `self_hosted_gpu` | `ready` | yes |
| `api_provider_route` | `api_provider` | `api_provider` | `ready` | yes |
| `tensorrt_llm` | `tensorrt_llm` | `self_hosted_gpu` | `planned` | no |

Supported status values are `ready`, `dry_run_ready`, `planned`, and
`deprecated`.

## Compatibility Rules

- API aliases must use `api_provider_route` with `provider_managed` hardware.
- Open-weight aliases may use Hugging Face Transformers, vLLM, or SGLang when
  the model registry and runtime registry both allow the pairing.
- Self-hosted GPU runtimes must not be selected for provider API aliases.
- API provider runs must not claim provider GPU telemetry.
- TensorRT-LLM is registered only as a planned engine. It cannot be selected
  for live runs unless a future smoke test changes `smoke_tested` and
  `live_run_supported` intentionally.

## Result Metadata

Benchmark results, generation traces, and result-track rows now reserve:

- `runtime`;
- `engine`;
- `backend_type`;
- `hardware`;
- `provider`.

The result-track join key includes these fields so API-provider and
self-hosted GPU results cannot be merged accidentally.

## TensorRT-LLM Guard

TensorRT-LLM is deliberately absent from runnable experiment and stress
matrices. It is kept under `planned_engines_not_runnable` in
`configs/stress_plan.yaml` and under `tensorrt_llm` in the runtime registry.
The selector rejects live TensorRT-LLM requests while it remains planned and
unsmoked.
