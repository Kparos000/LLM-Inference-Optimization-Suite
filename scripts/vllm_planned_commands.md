# Planned vLLM Commands

These commands are placeholders for review before execution. They are not an executable script and should not be run until the vLLM readiness checklist is complete.

## Environment Setup Placeholder

```text
# Review target OS, Python version, CUDA driver, and vLLM installation requirements first.
# Install commands intentionally omitted until the target environment is selected.
```

## vLLM OpenAI-Compatible Server Placeholder

```text
# Example shape only:
# python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

## Benchmark Client Placeholder

```text
# Planned client command shape:
# inference-bench vllm-run --workload-path data/prompts/short_chat.jsonl --output-path results/raw/vllm_short_chat_results.csv --generation-output-path results/raw/vllm_short_chat_generations.jsonl
```

## Report Summary Placeholder

```text
# inference-bench report-summary --input-csv results/raw/vllm_short_chat_results.csv
```

## Make Plots Placeholder

```text
# inference-bench make-plots --input-csv results/raw/vllm_short_chat_results.csv --output-dir results/figures/vllm_short_chat
```
