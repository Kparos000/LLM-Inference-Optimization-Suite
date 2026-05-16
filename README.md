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
- Multiple benchmark CSV files can be compared in one summary table.
- Initial Hugging Face baseline findings have been documented across expanded workloads.
- Benchmark methodology and experimental design are documented.
- Scaled workload and concurrency stress-test planning is documented before vLLM execution.
- An OpenAI-compatible runner foundation is available for future vLLM server benchmarking; vLLM execution is still intentionally deferred.
- The vLLM execution environment decision is documented.
- The vLLM smoke-test procedure is documented but not executed.
- vLLM client workflow scripts are available for smoke and expanded workload baselines, with execution deferred until the environment/server is ready.
- Linux/RunPod workflow scripts are available for vLLM smoke, expanded baseline, and curated sample promotion workflows.
- The first vLLM baseline experiment log is documented for the RunPod L40S calibration run.
- An early HF-vs-vLLM calibration comparison is documented with scope and limitations.
- An OpenAI-compatible concurrency load runner foundation is available for future vLLM load testing.
- Scaled synthetic workloads can be generated with `inference-bench generate-workloads --count 100`.
- Reporting includes latency percentiles and aggregate throughput metadata for concurrency runs.
- `openai-load-run` supports chunking, checkpointing, resume mode, and progress logs for long-running benchmarks.

## Documentation

### Project setup and reproducibility

- [Project scope](docs/00_project_scope.md)
- [Reproducibility](docs/01_reproducibility.md)
- [Dry-run plan](docs/02_dry_run_plan.md)
- [Decision log](docs/03_decision_log.md)
- [Result promotion policy](docs/06_result_promotion_policy.md)

### Benchmark methodology

- [Benchmark methodology](docs/08_benchmark_methodology.md)
- [Scaled workload generation](docs/14_scaled_workload_generation.md)

### vLLM/GPU execution

- [Hugging Face smoke test](docs/05_hf_smoke_test.md)
- [vLLM baseline preparation plan](docs/07_vllm_baseline_plan.md)
- [vLLM execution environment decision](docs/10_vllm_environment_decision.md)
- [vLLM smoke-test procedure](docs/11_vllm_smoke_test.md)
- [Resumable benchmarking plan](docs/15_resumable_benchmarking_plan.md)

### Experiment results

- [Experiment log](docs/12_experiment_log.md)
- [HF vs vLLM calibration comparison](docs/13_hf_vs_vllm_calibration_comparison.md)
- [Hugging Face baseline findings](docs/16_hf_baseline_findings.md)

### Phase 1 reporting

- [Phase 1 experiment inventory](docs/19_phase1_experiment_inventory.md)
- [Phase 1 project report](docs/20_phase1_project_report.md)
- [Phase 1 plot interpretation](docs/21_phase1_plot_interpretation.md)

### Phase 2 planning

- [Publication notes](docs/04_publication_notes.md)
- [Scaled benchmark plan](docs/09_scaled_benchmark_plan.md)
- [Project handover source pack](docs/24_project_handover_source_pack.md)
- [Phase 2 master plan](docs/27_phase2_master_plan.md)
- [Project handover: Phase 2 start](docs/28_project_handover_phase2.md)
- [Phase 2 data strategy](docs/29_phase2_data_strategy.md)
- [Phase 2 data source validation matrix](docs/30_phase2_data_source_validation_matrix.md)
- [Phase 2 vertical data contracts](docs/31_phase2_vertical_data_contracts.md)
- [Data directory policy](data/README.md)

## Environment Variables

Copy `.env.example` to `.env` for local secrets. Never commit `.env`. `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` may be used for Hugging Face model access. Real Hugging Face execution requires installing the `hf` extra.

## Quality Checks

The repository includes `scripts/audit_repo_public_content.py` for lightweight public-content and secrets review.

## Initial Development Model

The default development model is:

```text
Qwen/Qwen2.5-0.5B-Instruct

