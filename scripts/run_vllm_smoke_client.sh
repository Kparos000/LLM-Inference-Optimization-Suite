#!/usr/bin/env bash
set -euo pipefail

echo "Assuming a vLLM OpenAI-compatible server is already running at http://localhost:8000/v1."
echo "This script does not start the server."

echo "Capturing system metadata..."
inference-bench system-info --output-path results/raw/system_info_vllm_client.json

echo "Running vLLM smoke client benchmark..."
inference-bench openai-compatible-run \
  --workload-path data/prompts/smoke_workload.jsonl \
  --output-path results/raw/vllm_smoke_results.csv \
  --generation-output-path results/raw/vllm_smoke_generations.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --run-id vllm-smoke \
  --backend vllm \
  --optimization vllm_baseline \
  --max-new-tokens 32 \
  --max-prompts 1 \
  --stream

echo "Summarizing vLLM smoke results..."
inference-bench report-summary --input-csv results/raw/vllm_smoke_results.csv
