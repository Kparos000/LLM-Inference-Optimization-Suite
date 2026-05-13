$ErrorActionPreference = "Stop"

Write-Host "Assuming a vLLM OpenAI-compatible server is already running at http://localhost:8000/v1."
Write-Host "This script does not start the server."

Write-Host "Capturing system metadata..."
inference-bench system-info --output-path results/raw/system_info_vllm_client.json

Write-Host "Running vLLM short_chat baseline..."
inference-bench openai-compatible-run --workload-path data/prompts/short_chat.jsonl --output-path results/raw/vllm_short_chat_results.csv --generation-output-path results/raw/vllm_short_chat_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --run-id vllm-short-chat --backend vllm --optimization vllm_baseline --max-new-tokens 48 --max-prompts 5 --stream
inference-bench report-summary --input-csv results/raw/vllm_short_chat_results.csv

Write-Host "Running vLLM code_helpdesk baseline..."
inference-bench openai-compatible-run --workload-path data/prompts/code_helpdesk.jsonl --output-path results/raw/vllm_code_helpdesk_results.csv --generation-output-path results/raw/vllm_code_helpdesk_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --run-id vllm-code-helpdesk --backend vllm --optimization vllm_baseline --max-new-tokens 80 --max-prompts 5 --stream
inference-bench report-summary --input-csv results/raw/vllm_code_helpdesk_results.csv

Write-Host "Running vLLM long_context baseline..."
inference-bench openai-compatible-run --workload-path data/prompts/long_context.jsonl --output-path results/raw/vllm_long_context_results.csv --generation-output-path results/raw/vllm_long_context_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --run-id vllm-long-context --backend vllm --optimization vllm_baseline --max-new-tokens 96 --max-prompts 3 --stream
inference-bench report-summary --input-csv results/raw/vllm_long_context_results.csv

Write-Host "Running vLLM shared_prefix baseline..."
inference-bench openai-compatible-run --workload-path data/prompts/shared_prefix.jsonl --output-path results/raw/vllm_shared_prefix_results.csv --generation-output-path results/raw/vllm_shared_prefix_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --run-id vllm-shared-prefix --backend vllm --optimization vllm_baseline --max-new-tokens 80 --max-prompts 5 --stream
inference-bench report-summary --input-csv results/raw/vllm_shared_prefix_results.csv

Write-Host "Running vLLM structured_output_smoke baseline..."
inference-bench openai-compatible-run --workload-path data/prompts/structured_output_smoke.jsonl --output-path results/raw/vllm_structured_output_results.csv --generation-output-path results/raw/vllm_structured_output_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --run-id vllm-structured-output --backend vllm --optimization vllm_baseline --max-new-tokens 96 --max-prompts 3 --stream
inference-bench report-summary --input-csv results/raw/vllm_structured_output_results.csv

Write-Host "Writing vLLM workload comparison..."
inference-bench compare-results --input-csv results/raw/vllm_short_chat_results.csv --input-csv results/raw/vllm_code_helpdesk_results.csv --input-csv results/raw/vllm_long_context_results.csv --input-csv results/raw/vllm_shared_prefix_results.csv --input-csv results/raw/vllm_structured_output_results.csv --output-csv results/processed/vllm_workload_comparison.csv

Write-Host "Scoring vLLM structured-output generations..."
inference-bench score-structured-jsonl --input-jsonl results/raw/vllm_structured_output_generations.jsonl --required-fields category,answer,confidence
