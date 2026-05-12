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
inference-bench hf-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/hf_smoke_results.csv --generation-output-path results/raw/hf_smoke_generations.jsonl --model-id Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 32 --max-prompts 1
```

## Optional Streaming Smoke Test

```text
inference-bench hf-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/hf_streaming_smoke_results.csv --generation-output-path results/raw/hf_streaming_smoke_generations.jsonl --model-id Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 32 --max-prompts 1 --use-streaming
```

Non-streaming mode measures end-to-end latency but leaves TTFT blank. Streaming mode attempts to capture TTFT by recording when the first generated text chunk arrives.

## Follow-Up Reporting

```text
inference-bench report-summary --input-csv results/raw/hf_smoke_results.csv
```

CSV files store benchmark metrics. JSONL generation files store generated text for later qualitative review.

Generation JSONL files now preserve full prompt-level traces, including prompt text, generated text, latency metrics, throughput, cost estimate, and success/error status.

Generated result artifacts remain ignored by Git unless deliberately promoted into the repository later as selected reproducibility artifacts.
