# vLLM Smoke-Test Procedure

## Purpose

The vLLM smoke test verifies that a vLLM OpenAI-compatible server can be started and benchmarked by this project's OpenAI-compatible runner. It is an integration test, not a final performance benchmark.

## Scope

- One small model
- One small workload
- Limited prompts
- Streaming enabled
- CSV metrics output
- JSONL prompt trace output
- System metadata capture

## Prerequisites

- Linux/WSL2/cloud GPU environment selected
- Python environment created
- Repo cloned
- Dependencies installed
- Hugging Face token configured if needed
- No secrets committed
- GPU budget/timebox confirmed if using paid GPU

## Environment Assumptions

This procedure is intended for the selected Linux, WSL2, or cloud GPU environment. It should not be run in the local Windows base environment unless that environment has been intentionally prepared for Linux-compatible vLLM testing.

## Model Choice

- First smoke candidate: `Qwen/Qwen2.5-0.5B-Instruct`
- Alternative smoke candidate: `Qwen/Qwen2.5-1.5B-Instruct`
- First serious benchmark candidate: `Qwen/Qwen2.5-7B-Instruct`
- Larger models wait until the vLLM workflow is stable

## Planned Install Command

Install the project OpenAI client extra:

```text
python -m pip install -e ".[openai,dev]"
```

Planned vLLM installation placeholder:

```text
python -m pip install vllm
```

vLLM should be installed only in the chosen Linux/WSL2/cloud environment. Do not install vLLM in CI. Do not install vLLM in the local Windows base environment unless intentionally testing WSL/Linux compatibility.

## Planned Server Command

```text
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 --dtype auto --api-key EMPTY
```

This starts an OpenAI-compatible server. The command should be reviewed against the installed vLLM version before execution. For some models, chat template or generation config behavior may need review.

## Server Health Check

```text
curl http://localhost:8000/v1/models
```

The endpoint should return the served model before running the benchmark client.

## Planned Benchmark Client Command

```text
inference-bench openai-compatible-run --workload-path data/prompts/smoke_workload.jsonl --output-path results/raw/vllm_smoke_results.csv --generation-output-path results/raw/vllm_smoke_generations.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --max-new-tokens 32 --max-prompts 1 --stream
```

## Expected Artifacts

- `results/raw/system_info.json`
- `results/raw/vllm_smoke_results.csv`
- `results/raw/vllm_smoke_generations.jsonl`
- `report-summary` output
- Optional figures after plotting

## Smoke-Test Success Criteria

- Server starts
- Model endpoint is reachable
- One prompt completes successfully
- TTFT is populated for streaming
- TPOT is populated
- End-to-end latency is populated
- JSONL trace includes prompt and generated text
- `git status` remains clean because raw outputs are ignored

## Failure Handling

- If server fails to start, check vLLM install, CUDA/GPU availability, model support, and memory.
- If client fails, check base URL, API key placeholder, model name, and server logs.
- If out of memory, reduce model size or max model length.
- If model download fails, check Hugging Face token and network access.

## Cleanup Notes

Stop the server process after the smoke test. Preserve command history, system metadata, result CSVs, and JSONL traces under `results/` for review. Promote only reviewed sample artifacts when needed.

## What This Smoke Test Does Not Prove

- Not final performance
- Not concurrency performance
- Not optimization comparison
- Not cost analysis
- Not model-scale analysis

## Next Step After Smoke Test

After a successful smoke test, run a small benchmark with the expanded workload set and then proceed to the documented scaled benchmark and concurrency stress plan.
