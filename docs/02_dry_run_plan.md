# Dry-Run Experiment Plan

## Purpose

This document defines the no-GPU validation path that must pass before real model inference is added.

## Current No-GPU Validation Path

The current validation path uses the mock backend to exercise workload loading, metric calculation, CSV writing, summary reporting, and plot generation without downloading models or requiring GPU access.

## Current Command Sequence For Local Validation

```text
ruff format .
ruff check .
ruff format --check .
mypy src tests
pytest
inference-bench doctor
inference-bench validate-config
inference-bench system-info --output-path results/raw/system_info.json
inference-bench mock-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/mock_results.csv
inference-bench report-summary --input-csv results/raw/mock_results.csv
inference-bench make-plots --input-csv results/raw/mock_results.csv --output-dir results/figures
```

## First Real Model Smoke-Test Plan

The first real model run should use the smallest configured development model, the smoke workload, a single backend, a single optimization label, and a small prompt limit. The purpose is to validate integration correctness, not performance claims.

## Initial Model Strategy

- Qwen/Qwen2.5-0.5B-Instruct for lightweight smoke testing
- Qwen/Qwen2.5-1.5B-Instruct as the next small benchmark candidate
- Qwen/Qwen2.5-7B-Instruct as the first serious GPU benchmark candidate
- 32B and large-model placeholders reserved for future scale-comparison experiments

## Backend Strategy

- Mock backend first
- Hugging Face runner next
- vLLM after the Hugging Face baseline is stable
- SGLang as an optional later extension after vLLM

## Planned Workload Progression

- smoke
- structured_output_smoke
- short_chat
- code_helpdesk
- long_context
- shared_prefix

Structured-output quality checks will validate generated JSON and required-field completeness for workloads that declare JSON output expectations.

## Planned Optimization Progression

- baseline
- vLLM serving
- quantization
- prefix caching
- speculative decoding
- combined optimized configuration

## Exit Criteria Before Paid GPU

- Formatting, linting, typing, and tests pass locally.
- Default YAML configs validate.
- Mock benchmark run writes a result CSV.
- Summary reporting works against the mock result CSV.
- Plot generation writes PNG files from the mock result CSV.
- The first real-model command plan is documented before execution.

## Risks And Mitigations

- Backend integration drift: keep backend-specific logic isolated behind runners.
- Inconsistent benchmark inputs: use checked-in YAML configs and workload files.
- Compute waste: validate the full no-GPU path before paid GPU use.
- Misleading early results: label smoke tests clearly and avoid treating them as performance conclusions.
