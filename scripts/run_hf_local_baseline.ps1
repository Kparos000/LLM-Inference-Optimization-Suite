$ErrorActionPreference = "Stop"

Write-Host "Capturing system metadata..."
inference-bench system-info --output-path results/raw/system_info.json

Write-Host "Running Hugging Face streaming smoke workload..."
inference-bench hf-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/hf_smoke_results.csv --generation-output-path results/raw/hf_smoke_generations.jsonl --model-id Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 32 --max-prompts 3 --use-streaming

Write-Host "Running Hugging Face structured-output streaming smoke workload..."
inference-bench hf-run --workload-path data/prompts/structured_output_smoke.jsonl --output-path results/raw/hf_structured_output_results.csv --generation-output-path results/raw/hf_structured_output_generations.jsonl --model-id Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 96 --max-prompts 3 --use-streaming

Write-Host "Summarizing free-form smoke results..."
inference-bench report-summary --input-csv results/raw/hf_smoke_results.csv

Write-Host "Summarizing structured-output smoke results..."
inference-bench report-summary --input-csv results/raw/hf_structured_output_results.csv

Write-Host "Scoring structured-output JSON traces..."
inference-bench score-structured-jsonl --input-jsonl results/raw/hf_structured_output_generations.jsonl --required-fields category,answer,confidence

Write-Host "Generating free-form smoke plots..."
inference-bench make-plots --input-csv results/raw/hf_smoke_results.csv --output-dir results/figures/hf_smoke

Write-Host "Generating structured-output smoke plots..."
inference-bench make-plots --input-csv results/raw/hf_structured_output_results.csv --output-dir results/figures/hf_structured_output

Write-Host "Controlled local Hugging Face baseline complete."
