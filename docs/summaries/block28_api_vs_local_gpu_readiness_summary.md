# Block 28 API Versus Local GPU Readiness Summary

## Decision

`READY_FOR_SMALL_GPU_SMOKE`

All required readiness criteria pass. This decision authorizes the next
five-request GPU serving smoke, not a scaled benchmark.

## Comparison

| Metric | Local Qwen 0.5B | API Llama 3.1 8B |
| --- | ---: | ---: |
| JSON validity | 100% | 100% |
| Contract validity | 80% | 100% |
| Evidence-ID presence | 100% | 100% |
| Evidence match | 40% | 60% |
| Groundedness | 20% | 60% |
| Safety violations | 0% | 0% |
| Input tokens | 7,456 | 6,243 |
| Output tokens | 574 | 403 |
| Mean latency | 146,223.348 ms | 1,282.893 ms |
| Median latency | 172,320.627 ms | 1,270.200 ms |
| Cost per request | Not measured | `$0.000029002` |
| Cost per successful answer | Not measured | `$0.000029002` |
| Cost per grounded answer | Not measured | `$0.000048337` |

Local infrastructure cost is unknown, not zero.

## Readiness Checks

- Retrieval SLO pass: PASS
- Generation contract works: PASS
- API smoke successful: PASS
- Cost accounting works: PASS
- Local HF smoke successful: PASS
- Telemetry exists: PASS
- GPU cost config exists: PASS

Remaining repository blockers: none.

Execution prerequisites:

- provision and identify the GPU;
- record the actual hourly price and region;
- start vLLM and verify `/v1/models`;
- capture TTFT, TPOT, throughput, utilization, memory, power, and cost;
- evaluate five requests before any scale increase.

## Comparison Limitation

The result sets share prompt IDs, verticals, `mm2_hybrid_top5`, and the
generation evaluator. Their rendered prompt text and citation alias maps are
not byte-identical because Block 24 used an older renderer. Latency and token
deltas are directional rather than controlled model-only results.

## Generated Reports

- `results/processed/phase4_api_vs_local_comparison_report.json`
- `results/processed/phase4_api_vs_local_comparison_summary.csv`
- `docs/87_phase4_api_vs_local_comparison.md`
- `docs/summaries/block28_api_vs_local_gpu_readiness_summary.md`

## Exact Next Command Plan

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype auto --api-key EMPTY
```

```powershell
python scripts/phase4/run_openai_compatible_smoke.py `
  --input-path data/generated/phase4/api_priced_contract_runner_input.jsonl `
  --output-path results/raw/phase4_vllm_gpu_smoke_results.jsonl `
  --model-alias model1_0_5b `
  --model-name Qwen/Qwen2.5-0.5B-Instruct `
  --base-url http://localhost:8000/v1 `
  --api-key EMPTY `
  --limit 5 `
  --max-new-tokens 256
```

Run evaluation and inspect telemetry before increasing to the 500-record smoke.

## Commit

Commit message: `Compare API and local smoke for GPU readiness`

The final commit hash is reported after verification and push.

## Verification

- Mypy: no issues in 174 source files
- Pytest: 894 passed
- Ruff check: passed
- Ruff format check: 228 files formatted
- Public-content audit: passed
- `inference-bench doctor`: passed
- `inference-bench validate-config`: passed
