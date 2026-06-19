# Block 25 Phase 4 Pre-GPU Readiness Summary

## Result

Status: `PRE_GPU_PLUMBING_READY`

Required pre-GPU plumbing checks pass. Live GPU smoke readiness remains false
because no GPU has been provisioned, no serving endpoint is active, and actual
RunPod cost values have not been entered.

## Ready

- Promoted retrieval manifest and repaired retrieval SLOs
- Grounded generation contract
- Local Hugging Face smoke and dry-run path
- OpenAI-compatible vLLM wrapper
- Request telemetry schema
- Backend capability matrix
- SGLang OpenAI-compatible dry-run scaffold
- RunPod cost-input schema and elapsed-hour calculation
- Git conflict-artifact check

## Not Available Yet

- live TTFT, TPOT, latency, and throughput;
- GPU utilization, memory, power, and hardware metadata;
- actual RunPod GPU type, region, hourly price, and elapsed cost;
- live vLLM and SGLang server validation.

These items are reported as `NOT_AVAILABLE`, not failed.

## Backend Status

| Key | Status | Notes |
| --- | --- | --- |
| `hf_local` | `ready` | Local in-process correctness plumbing |
| `openai_compatible_vllm` | `dry_run_ready` | Requires a live GPU server |
| `sglang_openai_compatible_future` | `ready` | Dry-run scaffold and live validation complete |
| `tensorrt_llm_future` | `planned` | Registry-visible only; not runnable until smoke-tested |

## SGLang Dry-Run

Five records completed without a server, GPU, model call, or paid API call.
Workload IDs, prompt IDs, verticals, memory modes, ablation modes, generation
contract fields, and the shared result schema were preserved.

Raw dry-run output and its manifest remain ignored under `results/raw/`.

## GPU Cost Configuration

`configs/gpu_costs.yaml` uses RunPod placeholders and does not invent a price.
The tested formula is:

```text
elapsed_hours * hourly_price_usd
```

The calculator refuses incomplete live-run values.

## Next GPU Smoke

1. Provision a suitable GPU for `model1_0_5b`.
2. Record the exact RunPod listing and hourly price.
3. Start vLLM and verify `/v1/models`.
4. Run five `mm2_hybrid_top5` requests at concurrency 1.
5. Evaluate generation-contract validity and evidence matching.
6. Capture latency, TTFT, TPOT, throughput, GPU memory/utilization, and cost.
7. Stop if the five-request run has schema, evaluation, or telemetry failures.

## Artifacts

- `src/inference_bench/phase4_readiness.py`
- `scripts/phase4/check_phase4_readiness.py`
- `scripts/phase4/run_sglang_compatible_smoke.py`
- `configs/backend_matrix.yaml`
- `configs/gpu_costs.yaml`
- `data/generated/phase4/phase4_readiness_report.json`
- `data/generated/phase4/phase4_readiness_summary.csv`
- `docs/84_phase4_pre_gpu_readiness.md`

## Commands Run

```powershell
pytest tests/test_phase4_readiness.py
pytest tests/test_phase4_backend_matrix.py
pytest tests/test_phase4_sglang_scaffold.py
pytest tests/test_gpu_cost_config.py

python scripts/phase4/check_phase4_readiness.py `
  --output-root data/generated/phase4

python scripts/phase4/run_sglang_compatible_smoke.py `
  --dry-run `
  --input-path data/generated/phase4/generation_contract_runner_input.jsonl `
  --output-path results/raw/phase4_sglang_dry_run.jsonl `
  --limit 5
```

## Verification

- New targeted tests: `9 passed`
- Full test suite: `881 passed`
- Mypy: no issues in 168 source files
- Ruff check: passed
- Ruff format check: 217 files already formatted
- Public-content audit: passed
- `inference-bench doctor`: passed without requiring a GPU
- `inference-bench validate-config`: passed
