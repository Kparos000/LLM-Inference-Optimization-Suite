# LLM Inference Optimization Suite

A reproducible AI inference engineering project for learning, measuring, and explaining LLM inference optimization techniques.

## Project Goal

This project benchmarks and explains how modern LLM inference optimizations affect:

- Time to First Token
- Time Per Output Token
- End-to-end latency
- Throughput
- Memory usage
- Cost per token
- Output quality

## Engineering Principle

Measure ? Understand ? Optimize ? Scale

Paid GPU will not be used until the local harness, CI/CD, metrics, workload loader, and dry-run experiment plan are correct.

## Current Status

- Project scaffold and CI are complete.
- Benchmark foundation schemas and workload/result utilities are being added.
- Metric utilities for latency, throughput, cost, and memory are part of the benchmark foundation.
- A deterministic mock benchmark runner is available for validating the benchmark pipeline without model downloads or GPU.
- Reporting utilities can summarize benchmark CSVs and generate basic plots.
- YAML configuration files define models, workloads, and experiments.
- The Hugging Face runner foundation is available; real model execution is optional and requires installing the `hf` extra.
- Hugging Face runs can preserve generated text in JSONL artifacts for later quality analysis.
- The Hugging Face runner supports optional streaming TTFT measurement.
- Generation JSONL artifacts provide full prompt-level traces for later analysis.
- Structured-output smoke workloads and JSON validation utilities are available for future quality checks.
- Hardware and system metadata capture is available as a generated reproducibility artifact.
- A controlled local Hugging Face baseline script is available for smoke runs with metrics, traces, system info, and plots.
- Curated sample artifact promotion rules are available for selected reviewed outputs.
- Expanded workload categories are available for short chat, code/helpdesk, long context, and shared-prefix inference testing.
- vLLM baseline planning has started, with execution intentionally deferred until readiness checks are complete.

## Documentation

- [Project scope](docs/00_project_scope.md)
- [Reproducibility](docs/01_reproducibility.md)
- [Dry-run plan](docs/02_dry_run_plan.md)
- [Decision log](docs/03_decision_log.md)
- [Publication notes](docs/04_publication_notes.md)
- [Hugging Face smoke test](docs/05_hf_smoke_test.md)
- [Result promotion policy](docs/06_result_promotion_policy.md)
- [vLLM baseline preparation plan](docs/07_vllm_baseline_plan.md)

## Environment Variables

Copy `.env.example` to `.env` for local secrets. Never commit `.env`. `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` may be used for Hugging Face model access. Real Hugging Face execution requires installing the `hf` extra.

## Initial Development Model

The default development model is:

```text
Qwen/Qwen2.5-0.5B-Instruct

