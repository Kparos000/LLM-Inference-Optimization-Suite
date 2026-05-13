# Planned vLLM Commands

These commands are placeholders for review before execution. They are not an executable script and should not be run until the vLLM readiness checklist is complete.

These commands are intended for Linux, WSL2, or cloud GPU review. Do not run them on the local Windows base environment unless the environment has been intentionally prepared.

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

## Future OpenAI-Compatible Client Workflow

These commands are planned placeholders and should be reviewed before execution.

Install OpenAI client extra:

```text
python -m pip install -e ".[openai,dev]"
```

Future vLLM server placeholder:

```text
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

Future benchmark client placeholder:

```text
inference-bench openai-compatible-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/vllm_smoke_results.csv --generation-output-path results/raw/vllm_smoke_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --max-new-tokens 32 --max-prompts 1 --stream
```

## Report Summary Placeholder

```text
# inference-bench report-summary --input-csv results/raw/vllm_short_chat_results.csv
```

## Make Plots Placeholder

```text
# inference-bench make-plots --input-csv results/raw/vllm_short_chat_results.csv --output-dir results/figures/vllm_short_chat
```
