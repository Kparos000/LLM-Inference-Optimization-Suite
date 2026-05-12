# Reproducibility

## Purpose

This document defines the reproducibility expectations for benchmark runs in the LLM Inference Optimization Suite.

## Reproducibility Principles

- Benchmark definitions should come from checked-in configuration files, not ad hoc command edits.
- Workloads, metrics, result schemas, and report generation should be validated before real model inference is introduced.
- Raw benchmark results should be traceable to a run ID, workload, model, backend, optimization setting, and timestamp.
- No paid GPU run should happen until the benchmark harness, configs, metrics, reporting, and dry-run commands pass locally.

## What Is Controlled In Each Benchmark

- Model configuration and identifier
- Backend name
- Optimization label
- Workload file and prompt identifiers
- Output result path
- Metric definitions
- Reporting and plot generation commands

## What Must Be Recorded For Every Run

- Run ID and UTC timestamp
- Backend, model name, optimization, and workload name
- Prompt ID
- Input and output token counts
- TTFT, TPOT, end-to-end latency, and throughput
- Peak memory when available
- Estimated cost when available
- Success status and error message when applicable

## Hardware And System Metadata

Hardware and system metadata should be captured for benchmark runs so result files have enough context for reproducibility checks. Metrics without hardware context should not be treated as portable across machines, runtimes, or library versions.

System info artifacts remain generated outputs unless they are deliberately promoted for documentation, publication, or regression comparison.

## Raw Output Commit Policy

Generated raw outputs are not all committed by default because benchmark CSVs, logs, and plots can be regenerated, can grow quickly, and may vary by hardware or backend version. The repository should prioritize source, configuration, workload definitions, selected representative outputs, and reproducibility notes.

## Promoting Selected Outputs

Selected final or sample outputs may later be promoted into the repository when they support documentation, publication, or regression comparison. Promoted outputs should be small, clearly named, and tied to documented commands and configuration files.

## CI/CD Expectations

CI should remain lightweight and should validate formatting, linting, typing, tests, config loading, and no-GPU benchmark pipeline behavior. CI must not require model downloads, GPU access, or paid services at this stage.

## GPU Spending Rule

Paid GPU resources are deferred until the local benchmark harness, configs, metrics, reporting, and dry-run commands pass consistently.
