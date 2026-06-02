# Block 10 vLLM OpenAI-Compatible Smoke Summary

## Files Changed

- `scripts/phase4/run_openai_compatible_smoke.py`
- `scripts/phase4/evaluate_generation_outputs.py`
- `tests/test_phase4_openai_compatible_smoke.py`
- `docs/67_phase4_vllm_openai_compatible_smoke.md`
- `docs/summaries/block10_vllm_openai_compatible_smoke_summary.md`

## Server And Model

- Base URL used for dry-run metadata: `http://localhost:8000/v1`
- API key value in manifests: redacted
- Model alias: `model1_0_5b`
- Model name: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend path: OpenAI-compatible client, intended for local vLLM
- Paid API calls: none
- Real server calls: none in the required dry-run

## Workload

- Runner input:
  `data/generated/phase4/smoke_500_mm2_runner_input.jsonl`
- Raw dry-run output:
  `results/raw/phase4_openai_compatible_smoke_results.jsonl`
- Split: `smoke_500`
- Memory mode: `mm2_hybrid_top5`
- Ablation mode: `prompt_plus_metadata`
- Prompt count: 5

## Dry-Run Metrics

- Rows written: 5
- Successful dry-run rows: 5
- Min latency: 0.0003 ms
- Mean latency: 0.00064 ms
- Max latency: 0.0015 ms
- Total input tokens: 1,512
- Mean input tokens: 302.4
- Total output tokens: 45
- Mean output tokens: 9.0

These latency values are dry-run fixture latencies and are not benchmark
measurements.

## Evaluator Summary

- Rows evaluated: 5
- Joined to gold records: 5
- Joined rate: 1.0
- Format-valid rows: 5
- Format-valid rate: 1.0

## Server Readiness

The script skips readiness checks in `--dry-run`. In real mode it probes:

```text
http://localhost:8000/v1/models
```

If the endpoint is missing, it fails before generation with a message explaining
that the vLLM OpenAI-compatible server must be started.

## Real vLLM Command

After a local vLLM server is available, run:

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

## Commands Run

- `pytest tests/test_phase4_workload_adapter.py`
- `pytest tests/test_phase4_smoke_plumbing.py`
- `pytest tests/test_phase4_openai_compatible_smoke.py`
- `python scripts/phase4/run_openai_compatible_smoke.py --input-path data/generated/phase4/smoke_500_mm2_runner_input.jsonl --output-path results/raw/phase4_openai_compatible_smoke_results.jsonl --model-alias model1_0_5b --model-name Qwen/Qwen2.5-0.5B-Instruct --base-url http://localhost:8000/v1 --api-key EMPTY --limit 5 --max-new-tokens 64 --dry-run`
- `python scripts/phase4/evaluate_generation_outputs.py --results-path results/raw/phase4_openai_compatible_smoke_results.jsonl --dataset-root data/scaleup_2000_full --output-root results/processed`
- `python scripts/audit_repo_public_content.py`
- `inference-bench doctor`
- `inference-bench validate-config`
- `ruff check src/inference_bench scripts/phase4 tests/test_phase4_openai_compatible_smoke.py`
- `ruff format --check src/inference_bench scripts/phase4 tests/test_phase4_openai_compatible_smoke.py`

## Commit Hash

The final pushed commit hash is reported in the Codex final response. It cannot
be embedded self-referentially in the same commit without changing the commit
hash.
