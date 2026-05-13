# Benchmark Methodology And Experimental Design

## Purpose

This document describes the benchmark design, current calibration baseline, and planned inference optimization experiments for the LLM Inference Optimization Suite.

## Benchmark Design Principles

- Use small local runs to validate instrumentation before spending GPU resources.
- Separate measurement infrastructure from model and runtime execution.
- Keep workload, model, backend, optimization, and hardware metadata explicit.
- Prefer reproducible config-driven experiments over ad hoc commands.
- Preserve both aggregate metrics and prompt-level traces.

## Current Benchmark Harness Capabilities

The current harness supports JSONL workload loading, YAML-backed model/workload/experiment configuration, deterministic mock runs, local Hugging Face runs, CSV metric output, prompt-level generation traces, summary reporting, plot generation, system metadata capture, structured-output validation utilities, and multi-result comparison CSVs.

The mock backend validates the pipeline without model downloads or GPU access. The Hugging Face runner provides the current local model execution path for calibration and baseline comparison.

## Calibration Baseline

Calibration baseline: the current Hugging Face local run set validates the benchmark harness and establishes a local reference baseline. These runs are not presented as final optimization results.

Final optimization analysis will compare backends and optimization variants under documented workload, model, hardware, and concurrency conditions. The calibration baseline provides a reference point for understanding whether later runtime changes improve TTFT, TPOT, end-to-end latency, throughput, memory behavior, or output quality.

## Workload Categories

- `smoke`: small pipeline validation workload.
- `structured_output_smoke`: JSON-format workload for structured-output validation.
- `short_chat`: short conversational tasks.
- `code_helpdesk`: technical support and code-assistance tasks.
- `long_context`: moderate context summarization and extraction tasks.
- `shared_prefix`: repeated instruction-prefix workload for later prefix-caching evaluation.

## Metrics

- TTFT: time to first generated token or first non-empty streamed text chunk.
- TPOT: average time per output token after generation begins.
- End-to-end latency: total request duration from start to completion.
- Throughput: tokens processed per second for the measured request.
- Token counts: input and output token counts used to contextualize latency and throughput.
- Cost estimate: estimated cost field for runs where cost accounting is available.
- Memory/system metadata: hardware, runtime, and library context captured separately from result rows.
- Structured-output validity: JSON validity and required-field completeness for structured-output workloads.

## Hardware And System Metadata

System metadata is captured as a generated artifact so benchmark results can be interpreted with hardware and runtime context. Metadata includes platform, Python version, CPU/RAM details when available, optional torch/CUDA information, and optional transformers version.

Hardware context is important because latency, throughput, and memory behavior are not portable across machines, operating systems, accelerator types, or library versions.

## Result Artifacts

Benchmark runs write raw CSV metrics and optional JSONL generation traces under `results/`. Summary reports, comparison CSVs, plots, and selected sample artifacts can be generated from those raw outputs.

Raw generated outputs remain ignored by default. Reviewed sample artifacts may be promoted under `results/samples` when they support reports, README content, paper notes, or reproducibility examples.

## Planned Benchmark Progression

- Hugging Face local baseline
- vLLM baseline
- Concurrency stress tests
- Quantization experiments
- Prefix caching experiments
- Speculative decoding experiments
- Model-scale comparison if compute allows
- Optional SGLang comparison after vLLM is stable

## Limitations

- Current real-model runs use a small local model.
- Current workload sizes are intentionally small.
- Current calibration results are local-environment measurements.
- vLLM and SGLang execution paths are not active yet.
- Memory measurement is not yet integrated into per-result rows.
- Structured-output quality evaluation is lightweight and will need broader scoring for final comparisons.

## How To Interpret Current Results

Current Hugging Face results should be interpreted as calibration measurements for the harness and workload set. They are useful for validating metric capture, comparing workload shape, and establishing a reference before serving-runtime experiments.

They should not be treated as final optimization findings. Final comparisons will require consistent workload definitions, documented concurrency settings, captured system metadata, comparable output artifacts, and clear separation between backend behavior and optimization-specific behavior.

## How Final Optimization Results Will Be Reported

Final reports should compare each backend and optimization variant across the same workload categories, using consistent model identifiers, prompt limits, concurrency settings, and hardware metadata. Results should include aggregate metrics, prompt-level traces where useful, structured-output validity for relevant workloads, comparison CSVs, and selected plots.

Reported findings should distinguish latency, throughput, memory, and quality trade-offs rather than reducing performance to a single metric.
