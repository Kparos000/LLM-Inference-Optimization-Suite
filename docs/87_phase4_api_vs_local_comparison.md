# Phase 4 API Versus Local Comparison

Block 28 compares the completed local Qwen 0.5B generation-contract smoke with
the completed API-priced Llama 3.1 8B smoke and applies the explicit small-GPU
readiness gate.

No inference, GPU work, vLLM, SGLang, or additional paid API call was run for
this block. The comparison reads existing Block 24 and Block 27 artifacts.

## Compared Runs

| Run | Model | Execution | Records | Memory mode |
| --- | --- | --- | ---: | --- |
| Local | `Qwen/Qwen2.5-0.5B-Instruct` | Transformers on local CPU | 5 | `mm2_hybrid_top5` |
| API | `meta-llama/Llama-3.1-8B-Instruct` | HF router, Novita | 5 | `mm2_hybrid_top5` |

Both result sets contain the same five prompt IDs, one per vertical, and use
the same evaluator contract.

## Comparability Caveat

This is an ID-aligned comparison, not a perfectly controlled model comparison.

The Block 24 and Block 27 prompt text hashes and citation alias maps differ.
Block 24 used an older prompt renderer, while Block 27 used the promoted
`prompt_plus_metadata` runner input. Hardware and execution architecture also
differ substantially.

Quality values remain useful smoke evidence. Token and latency deltas are
directional and must not be presented as a controlled model-only benchmark.
The first GPU smoke should use one frozen exported runner input for every
backend.

## Quality Comparison

| Metric | Local Qwen 0.5B | API Llama 3.1 8B | Delta |
| --- | ---: | ---: | ---: |
| JSON validity | 100% | 100% | 0 points |
| Contract validity | 80% | 100% | +20 points |
| Evidence-ID presence | 100% | 100% | 0 points |
| Full evidence match | 40% | 60% | +20 points |
| Deterministic groundedness | 20% | 60% | +40 points |
| Safety violations | 0% | 0% | 0 points |

The API model improved contract adherence, evidence match, and groundedness on
the five-prompt smoke. Airline and Healthcare Admin still omitted one required
evidence family in the API run.

## Runtime And Tokens

| Metric | Local Qwen 0.5B | API Llama 3.1 8B |
| --- | ---: | ---: |
| Input tokens | 7,456 | 6,243 |
| Output tokens | 574 | 403 |
| Mean latency | 146,223.348 ms | 1,282.893 ms |
| Median latency | 172,320.627 ms | 1,270.200 ms |

The API path was much faster in this smoke, but the result combines model,
hardware, provider-serving, token-count, and prompt-rendering differences.

## Cost Comparison

| Metric | Local Qwen 0.5B | API Llama 3.1 8B |
| --- | ---: | ---: |
| Cost per request | Not measured | `$0.000029002` |
| Cost per successful answer | Not measured | `$0.000029002` |
| Cost per grounded answer | Not measured | `$0.000048337` |

Local cost is not reported as zero. The local smoke did not measure machine
energy, amortized hardware, or infrastructure cost. The API values use actual
provider token counts and the captured Novita rate.

## GPU Readiness Gate

Decision: `READY_FOR_SMALL_GPU_SMOKE`

| Criterion | Status |
| --- | --- |
| Promoted retrieval SLOs pass | PASS |
| Generation contract works | PASS |
| API-priced smoke successful | PASS |
| API cost accounting works | PASS |
| Local HF smoke successful | PASS |
| Telemetry schema exists | PASS |
| GPU cost configuration exists | PASS |

There are no remaining repository-readiness blockers for a five-request GPU
smoke.

The following are execution prerequisites, not failed readiness checks:

- provision a GPU with sufficient VRAM;
- record its actual type, region, and hourly price;
- start vLLM and verify `/v1/models`;
- capture streaming TTFT/TPOT and GPU telemetry;
- stop after five requests and evaluate before increasing scale.

## Exact Next GPU Smoke Plan

1. Copy `configs/gpu_costs.yaml` to a run-specific configuration and fill the
   actual GPU type, region, hourly price, instance ID, and start time.
2. Start vLLM:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --api-key EMPTY
```

3. Verify the server:

```bash
curl http://localhost:8000/v1/models
```

4. Run exactly five promoted records:

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

5. Evaluate the output:

```powershell
python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_vllm_gpu_smoke_results.jsonl `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed `
  --report-name phase4_vllm_gpu_smoke_eval_report.json `
  --summary-name phase4_vllm_gpu_smoke_eval_summary.csv
```

6. Record the end time and calculate elapsed GPU cost. Do not start concurrency
   or the 500-record smoke until generation-contract evaluation and telemetry
   are complete.

## Outputs

- `results/processed/phase4_api_vs_local_comparison_report.json`
- `results/processed/phase4_api_vs_local_comparison_summary.csv`
- `docs/87_phase4_api_vs_local_comparison.md`
- `docs/summaries/block28_api_vs_local_gpu_readiness_summary.md`

The processed JSON and CSV remain local and ignored according to repository
policy. They can be regenerated with:

```powershell
python scripts/phase4/compare_api_vs_local.py
```
