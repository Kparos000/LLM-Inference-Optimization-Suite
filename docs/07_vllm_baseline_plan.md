# vLLM Baseline Preparation Plan

## Purpose

This document defines the preparation plan for the first vLLM baseline phase. It documents intent, constraints, decisions, and readiness checks before any vLLM runtime execution is added.

## Why vLLM Is Next After Hugging Face Baseline

The Hugging Face runner validates local model loading, prompt execution, metric capture, trace preservation, and reporting with the smallest practical integration surface. vLLM is the next backend because it introduces a serving-oriented runtime that can be compared against the Hugging Face baseline using the same workloads and metric definitions.

## What vLLM Will Let Us Test

- Serving runtime overhead
- Continuous batching
- PagedAttention and KV cache efficiency
- OpenAI-compatible server/client workflow
- Later quantization experiments
- Later prefix caching experiments
- Later speculative decoding experiments

## Initial Model Strategy

- `Qwen/Qwen2.5-0.5B-Instruct` for the first vLLM smoke run if supported
- `Qwen/Qwen2.5-1.5B-Instruct` as the next small benchmark candidate
- `Qwen/Qwen2.5-7B-Instruct` as the first serious GPU benchmark candidate

## Initial Workloads

- `short_chat`
- `code_helpdesk`
- `long_context`
- `shared_prefix`
- `structured_output_smoke`

## Metrics To Compare Against HF

- TTFT
- TPOT
- End-to-end latency
- Throughput
- Memory
- Structured-output validity

## Local Constraints

The Windows local environment may not be ideal for vLLM. vLLM is usually Linux/GPU-oriented, so the first vLLM run may require WSL2, Linux, or cloud GPU infrastructure.

## Execution Environment Decision

The vLLM execution environment decision is documented in [vLLM execution environment decision](10_vllm_environment_decision.md). vLLM execution is deferred until the environment checklist is satisfied.

## OpenAI-Compatible Runner Plan

The project will use an OpenAI-compatible runner to benchmark a running vLLM server through the server/client workflow. vLLM execution remains deferred until the environment decision is made. The runner can support streaming TTFT measurement when the server streams responses.

## Smoke-Test Procedure

The first vLLM smoke-test procedure is documented in [vLLM smoke-test procedure](11_vllm_smoke_test.md). It should be executed only after the environment decision and readiness checklist are satisfied.

## No Paid GPU Rule

vLLM should not be run on paid GPU until configs, commands, baseline expectations, and artifact policies are documented. The purpose of this phase is readiness, not execution.

## Readiness Checklist Before vLLM Execution

- Hugging Face baseline workflow is documented and stable.
- Expanded workloads load and pass mock-run validation.
- System metadata capture is available.
- Result promotion rules are documented.
- Planned vLLM commands are reviewed before execution.
- Expected output paths are under `results/`.
- No generated benchmark outputs are committed by default.
- Hardware target and operating system are selected.
- Model support and memory requirements are checked before runtime use.
- Comparison metrics are aligned with the Hugging Face baseline.

## Risks And Mitigations

- Runtime compatibility risk: validate the target operating system and Python environment before installing vLLM.
- GPU memory risk: start with the smallest supported Qwen model and preserve larger models for later.
- Cost risk: keep paid GPU execution blocked until the readiness checklist is complete.
- Comparison drift: reuse existing workloads, metrics, and report commands wherever possible.
- Artifact sprawl: keep outputs under `results/` and promote only reviewed samples.
