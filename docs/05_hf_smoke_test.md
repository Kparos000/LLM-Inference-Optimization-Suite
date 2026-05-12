# Hugging Face Smoke Test

## Purpose

This document describes the local Hugging Face smoke test workflow for validating the first real-model runner path after the no-GPU benchmark pipeline is stable.

## No Paid GPU Scope

The smoke test is intended for local execution and does not add paid GPU automation. It should be used to validate runner integration with a small model and a small prompt limit before any larger benchmark run.

## Local Environment File

Create a local `.env` file from the committed template:

```text
Copy-Item .env.example .env
```

Add local credentials only to `.env` if needed. Do not commit `.env`.

## Install HF Extra

```text
python -m pip install -e ".[hf,dev]"
```

The same command is available in `scripts/install_hf_extra.ps1`.

## Smoke Test Command

```text
inference-bench hf-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/hf_smoke_results.csv --model-id Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 32 --max-prompts 1
```

## Follow-Up Reporting

```text
inference-bench report-summary --input-csv results/raw/hf_smoke_results.csv
```

Generated result files remain ignored unless deliberately promoted into the repository later as selected reproducibility artifacts.
