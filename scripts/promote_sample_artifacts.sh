#!/usr/bin/env bash
set -euo pipefail

raw_sample_dir="results/samples/raw"
figure_sample_dir="results/samples/figures"

mkdir -p "${raw_sample_dir}" "${figure_sample_dir}"

copy_artifact() {
  local source_path="$1"
  local destination_path="$2"

  if [[ -f "${source_path}" ]]; then
    cp -f "${source_path}" "${destination_path}"
    echo "Copied ${source_path} -> ${destination_path}"
  else
    echo "Missing optional artifact: ${source_path}"
  fi
}

copy_artifact "results/raw/system_info.json" "results/samples/raw/system_info_sample.json"
copy_artifact "results/raw/hf_smoke_results.csv" "results/samples/raw/hf_smoke_results_sample.csv"
copy_artifact "results/raw/hf_structured_output_results.csv" "results/samples/raw/hf_structured_output_results_sample.csv"
copy_artifact "results/raw/hf_structured_output_generations.jsonl" "results/samples/raw/hf_structured_output_generations_sample.jsonl"
copy_artifact "results/raw/hf_short_chat_results.csv" "results/samples/raw/hf_short_chat_results_sample.csv"
copy_artifact "results/raw/hf_code_helpdesk_results.csv" "results/samples/raw/hf_code_helpdesk_results_sample.csv"
copy_artifact "results/raw/hf_long_context_results.csv" "results/samples/raw/hf_long_context_results_sample.csv"
copy_artifact "results/raw/hf_shared_prefix_results.csv" "results/samples/raw/hf_shared_prefix_results_sample.csv"
copy_artifact "results/raw/hf_short_chat_generations.jsonl" "results/samples/raw/hf_short_chat_generations_sample.jsonl"
copy_artifact "results/raw/hf_code_helpdesk_generations.jsonl" "results/samples/raw/hf_code_helpdesk_generations_sample.jsonl"
copy_artifact "results/raw/hf_long_context_generations.jsonl" "results/samples/raw/hf_long_context_generations_sample.jsonl"
copy_artifact "results/raw/hf_shared_prefix_generations.jsonl" "results/samples/raw/hf_shared_prefix_generations_sample.jsonl"
copy_artifact "results/processed/hf_workload_comparison.csv" "results/samples/raw/hf_workload_comparison_sample.csv"
copy_artifact "results/raw/vllm_smoke_results.csv" "results/samples/raw/vllm_smoke_results_sample.csv"
copy_artifact "results/raw/vllm_short_chat_results.csv" "results/samples/raw/vllm_short_chat_results_sample.csv"
copy_artifact "results/raw/vllm_code_helpdesk_results.csv" "results/samples/raw/vllm_code_helpdesk_results_sample.csv"
copy_artifact "results/raw/vllm_long_context_results.csv" "results/samples/raw/vllm_long_context_results_sample.csv"
copy_artifact "results/raw/vllm_shared_prefix_results.csv" "results/samples/raw/vllm_shared_prefix_results_sample.csv"
copy_artifact "results/raw/vllm_structured_output_results.csv" "results/samples/raw/vllm_structured_output_results_sample.csv"
copy_artifact "results/raw/vllm_smoke_generations.jsonl" "results/samples/raw/vllm_smoke_generations_sample.jsonl"
copy_artifact "results/raw/vllm_short_chat_generations.jsonl" "results/samples/raw/vllm_short_chat_generations_sample.jsonl"
copy_artifact "results/raw/vllm_code_helpdesk_generations.jsonl" "results/samples/raw/vllm_code_helpdesk_generations_sample.jsonl"
copy_artifact "results/raw/vllm_long_context_generations.jsonl" "results/samples/raw/vllm_long_context_generations_sample.jsonl"
copy_artifact "results/raw/vllm_shared_prefix_generations.jsonl" "results/samples/raw/vllm_shared_prefix_generations_sample.jsonl"
copy_artifact "results/raw/vllm_structured_output_generations.jsonl" "results/samples/raw/vllm_structured_output_generations_sample.jsonl"
copy_artifact "results/processed/vllm_workload_comparison.csv" "results/samples/raw/vllm_workload_comparison_sample.csv"
copy_artifact "results/figures/hf_smoke/latency_by_optimization.png" "results/samples/figures/hf_smoke_latency_by_optimization.png"
copy_artifact "results/figures/hf_smoke/throughput_by_optimization.png" "results/samples/figures/hf_smoke_throughput_by_optimization.png"
copy_artifact "results/figures/hf_structured_output/latency_by_optimization.png" "results/samples/figures/hf_structured_latency_by_optimization.png"
copy_artifact "results/figures/hf_structured_output/throughput_by_optimization.png" "results/samples/figures/hf_structured_throughput_by_optimization.png"

echo "Sample artifact promotion complete. Review promoted files before committing."
