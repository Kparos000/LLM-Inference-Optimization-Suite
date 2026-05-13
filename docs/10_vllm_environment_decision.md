# vLLM Execution Environment Decision

## Purpose

This document records the vLLM execution environment decision before installing or running vLLM. The goal is to keep the project compute-efficient, reproducible, and clear about where serious serving benchmarks should run.

## Decision Summary

- Local Windows remains the development and benchmark-client environment.
- vLLM execution should target Linux, preferably a cloud GPU instance when the benchmark harness is ready.
- WSL2 may be used as an optional bridge for Linux command rehearsal, but not as the primary serious benchmark environment unless GPU/CUDA support is confirmed.
- No paid GPU run should happen until vLLM commands, workload scale, concurrency plan, and expected artifacts are documented.

## Current Local Environment

The local machine is useful for repo development, CLI validation, Hugging Face calibration, report generation, and OpenAI-compatible client development.

Current local results should not be treated as production GPU-serving results.

## Candidate Environments Considered

- Native Windows
- WSL2 on Windows
- Local Linux GPU machine
- Cloud GPU Linux instance

## Recommended Environment

Use Linux cloud GPU as the primary target for serious vLLM benchmark execution. Use WSL2 only as optional preparation if needed. Keep Windows as the development/client environment.

## Why Not Native Windows For Serious vLLM Benchmarking

vLLM is primarily Linux/GPU-oriented for production-style serving. Native Windows support is not the safest baseline for reproducible serious benchmarks. Using Linux reduces environment risk and makes results easier to explain.

## Role Of WSL2

WSL2 can be useful for rehearsing Linux commands, validating shell workflows, and preparing environment setup notes. It should not become the primary serious benchmark environment unless GPU/CUDA support is confirmed and the environment can be documented cleanly.

## Role Of Cloud GPU

Cloud GPU execution enables realistic GPU inference measurements, vLLM serving, concurrency stress tests, 7B-class model benchmarking, and future quantization, prefix caching, and speculative decoding experiments.

Cloud GPU should be used only after dry-run readiness checks pass.

## Minimum Readiness Before Paid GPU

- Repo clean
- CI passing
- Public-content audit passing
- OpenAI-compatible runner implemented
- Workload/concurrency plan documented
- vLLM server command reviewed
- Expected artifacts documented
- Budget/timebox defined
- First run uses small model and limited prompts

## Initial Cloud GPU Target

Start with a small/affordable NVIDIA GPU instance where possible. The first vLLM smoke should use `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`.

The first serious benchmark candidate remains `Qwen/Qwen2.5-7B-Instruct`. Larger models should wait until the 7B workflow is stable.

## Risks And Mitigations

- Environment drift: capture system metadata and preserve command history for promoted results.
- GPU cost: use a limited prompt count and explicit timebox for the first paid run.
- Model compatibility: start with small Qwen models before moving to 7B.
- CUDA or driver mismatch: verify the instance image and vLLM requirements before installing dependencies.
- Benchmark inconsistency: reuse checked-in workloads, config files, and result artifact paths.

## Next Execution Step

The next execution step is to rehearse the OpenAI-compatible client command against a reviewed vLLM server plan, then run the smallest vLLM smoke benchmark only after the readiness checklist is satisfied.
