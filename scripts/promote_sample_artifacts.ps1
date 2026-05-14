$ErrorActionPreference = "Stop"

$rawSampleDir = "results/samples/raw"
$figureSampleDir = "results/samples/figures"

New-Item -ItemType Directory -Force -Path $rawSampleDir | Out-Null
New-Item -ItemType Directory -Force -Path $figureSampleDir | Out-Null

$artifacts = @(
    @{
        Source = "results/raw/system_info.json"
        Destination = "results/samples/raw/system_info_sample.json"
    },
    @{
        Source = "results/raw/hf_smoke_results.csv"
        Destination = "results/samples/raw/hf_smoke_results_sample.csv"
    },
    @{
        Source = "results/raw/hf_structured_output_results.csv"
        Destination = "results/samples/raw/hf_structured_output_results_sample.csv"
    },
    @{
        Source = "results/raw/hf_structured_output_generations.jsonl"
        Destination = "results/samples/raw/hf_structured_output_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/hf_short_chat_results.csv"
        Destination = "results/samples/raw/hf_short_chat_results_sample.csv"
    },
    @{
        Source = "results/raw/hf_code_helpdesk_results.csv"
        Destination = "results/samples/raw/hf_code_helpdesk_results_sample.csv"
    },
    @{
        Source = "results/raw/hf_long_context_results.csv"
        Destination = "results/samples/raw/hf_long_context_results_sample.csv"
    },
    @{
        Source = "results/raw/hf_shared_prefix_results.csv"
        Destination = "results/samples/raw/hf_shared_prefix_results_sample.csv"
    },
    @{
        Source = "results/raw/hf_short_chat_generations.jsonl"
        Destination = "results/samples/raw/hf_short_chat_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/hf_code_helpdesk_generations.jsonl"
        Destination = "results/samples/raw/hf_code_helpdesk_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/hf_long_context_generations.jsonl"
        Destination = "results/samples/raw/hf_long_context_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/hf_shared_prefix_generations.jsonl"
        Destination = "results/samples/raw/hf_shared_prefix_generations_sample.jsonl"
    },
    @{
        Source = "results/processed/hf_workload_comparison.csv"
        Destination = "results/samples/raw/hf_workload_comparison_sample.csv"
    },
    @{
        Source = "results/raw/vllm_smoke_results.csv"
        Destination = "results/samples/raw/vllm_smoke_results_sample.csv"
    },
    @{
        Source = "results/raw/vllm_short_chat_results.csv"
        Destination = "results/samples/raw/vllm_short_chat_results_sample.csv"
    },
    @{
        Source = "results/raw/vllm_code_helpdesk_results.csv"
        Destination = "results/samples/raw/vllm_code_helpdesk_results_sample.csv"
    },
    @{
        Source = "results/raw/vllm_long_context_results.csv"
        Destination = "results/samples/raw/vllm_long_context_results_sample.csv"
    },
    @{
        Source = "results/raw/vllm_shared_prefix_results.csv"
        Destination = "results/samples/raw/vllm_shared_prefix_results_sample.csv"
    },
    @{
        Source = "results/raw/vllm_structured_output_results.csv"
        Destination = "results/samples/raw/vllm_structured_output_results_sample.csv"
    },
    @{
        Source = "results/raw/vllm_smoke_generations.jsonl"
        Destination = "results/samples/raw/vllm_smoke_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/vllm_short_chat_generations.jsonl"
        Destination = "results/samples/raw/vllm_short_chat_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/vllm_code_helpdesk_generations.jsonl"
        Destination = "results/samples/raw/vllm_code_helpdesk_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/vllm_long_context_generations.jsonl"
        Destination = "results/samples/raw/vllm_long_context_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/vllm_shared_prefix_generations.jsonl"
        Destination = "results/samples/raw/vllm_shared_prefix_generations_sample.jsonl"
    },
    @{
        Source = "results/raw/vllm_structured_output_generations.jsonl"
        Destination = "results/samples/raw/vllm_structured_output_generations_sample.jsonl"
    },
    @{
        Source = "results/processed/vllm_workload_comparison.csv"
        Destination = "results/samples/raw/vllm_workload_comparison_sample.csv"
    },
    @{
        Source = "results/figures/hf_smoke/latency_by_optimization.png"
        Destination = "results/samples/figures/hf_smoke_latency_by_optimization.png"
    },
    @{
        Source = "results/figures/hf_smoke/throughput_by_optimization.png"
        Destination = "results/samples/figures/hf_smoke_throughput_by_optimization.png"
    },
    @{
        Source = "results/figures/hf_structured_output/latency_by_optimization.png"
        Destination = "results/samples/figures/hf_structured_latency_by_optimization.png"
    },
    @{
        Source = "results/figures/hf_structured_output/throughput_by_optimization.png"
        Destination = "results/samples/figures/hf_structured_throughput_by_optimization.png"
    }
)

foreach ($artifact in $artifacts) {
    if (Test-Path -Path $artifact.Source -PathType Leaf) {
        Copy-Item -Path $artifact.Source -Destination $artifact.Destination -Force
        Write-Host "Copied $($artifact.Source) -> $($artifact.Destination)"
    }
    else {
        Write-Host "Missing optional artifact: $($artifact.Source)"
    }
}

Write-Host "Sample artifact promotion complete. Review promoted files before committing."
