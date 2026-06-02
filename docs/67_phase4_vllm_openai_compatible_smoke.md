# Phase 4 vLLM OpenAI-Compatible Smoke

This block prepares the existing OpenAI-compatible runner path for a tiny Phase
4 smoke workload. It does not run a large benchmark, paid API call, SGLang run,
or retrieval rebuild.

## What This Validates

The smoke uses runner-adapted Phase 3 `WorkloadItem` JSONL and sends it through
the OpenAI-compatible client shape used by vLLM. The script preserves:

- `prompt_id`
- `workload_id`
- `vertical`
- `memory_mode`
- `ablation_mode`
- prompt text
- generated text
- latency
- token counts when the endpoint returns usage
- success and error fields
- run manifest metadata

Dry-run mode writes fixture output without contacting a server. Real mode first
checks that the OpenAI-compatible server is reachable and probes `/models` when
the endpoint supports it.

## Start A Local vLLM Server

Use the same model alias as the local HF smoke:

```powershell
python -m vllm.entrypoints.openai.api_server `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --host 0.0.0.0 `
  --port 8000
```

The smoke assumes the server exposes:

```text
http://localhost:8000/v1
```

Local vLLM can use `EMPTY` as the API key. Do not use this script for paid API
providers; API-priced models use the separate paid-smoke path with explicit cost
guards.

## Dry-Run Smoke

```powershell
python scripts/phase4/run_openai_compatible_smoke.py `
  --input-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl `
  --output-path results/raw/phase4_openai_compatible_smoke_results.jsonl `
  --model-alias model1_0_5b `
  --model-name Qwen/Qwen2.5-0.5B-Instruct `
  --base-url http://localhost:8000/v1 `
  --api-key EMPTY `
  --limit 5 `
  --max-new-tokens 64 `
  --dry-run
```

Dry-run mode should complete even when no vLLM server is running.

## Real Server Smoke

After the local vLLM server is running, remove `--dry-run`:

```powershell
python scripts/phase4/run_openai_compatible_smoke.py `
  --input-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl `
  --output-path results/raw/phase4_openai_compatible_smoke_results.jsonl `
  --model-alias model1_0_5b `
  --model-name Qwen/Qwen2.5-0.5B-Instruct `
  --base-url http://localhost:8000/v1 `
  --api-key EMPTY `
  --limit 5 `
  --max-new-tokens 64
```

If the server is missing, the script fails before issuing generation requests
and prints the exact startup requirement.

## Evaluate Smoke Output

```powershell
python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_openai_compatible_smoke_results.jsonl `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed
```

The evaluator writes:

```text
results/processed/phase4_openai_compatible_smoke_eval_report.json
results/processed/phase4_openai_compatible_smoke_eval_summary.csv
```

## Why This Prepares GPU Smoke

The OpenAI-compatible path is the client side of the planned vLLM GPU workflow.
Once this tiny smoke works against a local server, Phase 4 can move to a small
real vLLM server test, then the planned 500-prompt GPU smoke. SGLang remains out
of scope until the vLLM path is stable.
