# Phase 4 API-Priced Gated Model Smoke

Block 27 validates the promoted retrieval and generation-contract stack through
a real Hugging Face Inference Provider request path.

## Execution Decision

The requested decision order was:

1. `model5_gated`: `meta-llama/Llama-3.2-3B-Instruct`
2. `model6_gated`: `meta-llama/Llama-3.1-8B-Instruct`

The official router metadata exposed no provider for `model5_gated` with both
input and output token prices. The run did not estimate or invent those rates.

`model6_gated` had a live Novita route with:

- input: `$0.02` per 1 million tokens;
- output: `$0.05` per 1 million tokens;
- source:
  `https://router.huggingface.co/v1/models/meta-llama/Llama-3.1-8B-Instruct`;
- snapshot timestamp: `2026-06-06T01:22:14.834095+00:00`.

The selected request model was
`meta-llama/Llama-3.1-8B-Instruct:novita`, which forces the provider used by
the captured price.

## Workload

The smoke used exactly five records, one per vertical:

- Airline
- Healthcare Admin
- Retail
- Finance
- Research AI

Every record used the promoted retrieval baseline:

- `mm2_hybrid_top5`;
- `prompt_plus_metadata`;
- `qdrant_vector`;
- `qdrant_local`;
- no direct source hints.

No retrieval behavior or promoted dataset record was changed.

## Measured Result

All five API requests completed successfully.

| Metric | Result |
| --- | ---: |
| JSON validity | 5/5 (100%) |
| Contract validity | 5/5 (100%) |
| Evidence-ID presence | 5/5 (100%) |
| Full evidence match | 3/5 (60%) |
| Deterministic groundedness | 3/5 (60%) |
| Safety violations | 0/5 |
| Mean latency | 1,282.893 ms |
| Median latency | 1,270.200 ms |
| Mean output throughput | 63.166 tokens/s |
| Input tokens | 6,243 |
| Output tokens | 403 |
| Total tokens | 6,646 |

TTFT was unavailable because the smoke used a non-streaming chat-completion
request. No TTFT value was inferred.

Airline and Healthcare Admin each cited one of two required evidence families.
Retail, Finance, and Research AI matched all required evidence identifiers.

## Cost

Measured token cost:

| Metric | Result |
| --- | ---: |
| Input cost | `$0.00012486` |
| Output cost | `$0.00002015` |
| Total cost | `$0.00014501` |
| Cost per request | `$0.000029002` |
| Cost per successful answer | `$0.000029002` |
| Cost per grounded answer | `$0.000048337` |

These values use provider-reported token counts and the captured Novita price.
They are not infrastructure-cost estimates.

## Qwen 0.5B Comparison

The comparison uses the hardened five-prompt Qwen 0.5B generation-contract
report.

| Metric | Qwen 0.5B | Llama 3.1 8B API | Delta |
| --- | ---: | ---: | ---: |
| JSON validity | 100% | 100% | 0 points |
| Contract validity | 80% | 100% | +20 points |
| Evidence-ID presence | 100% | 100% | 0 points |
| Evidence match | 40% | 60% | +20 points |
| Groundedness | 20% | 60% | +40 points |

The stronger model improved contract adherence, full evidence match, and
deterministic groundedness on this smoke sample. Five records are insufficient
for a benchmark-quality model claim.

## Safety Gates

The runner:

- requires `HF_TOKEN`;
- never records or prints the token;
- requires `--allow-paid-api-call`;
- refuses missing pricing;
- checks gated repository access;
- uses the fallback only when the primary lacks a valid execution prerequisite;
- runs exactly five promoted records;
- records the selected provider and pricing source.

Because a credential was displayed in an interactive development context
during this work, it should be rotated after validation.

## Commands

Capture live fallback pricing:

```powershell
python scripts/phase3/snapshot_hf_inference_pricing.py `
  --models model6_gated `
  --output configs/api_pricing.yaml `
  --report data/generated/context_engineering/hf_api_pricing_snapshot_report.json
```

Run the explicitly authorized five-request smoke:

```powershell
python scripts/phase4/run_api_priced_model_smoke.py `
  --output-path results/raw/phase4_api_priced_smoke_results.jsonl `
  --readiness-report results/processed/phase4_api_priced_readiness_report.json `
  --pricing-config configs/api_pricing.yaml `
  --primary-model-alias model5_gated `
  --fallback-model-alias model6_gated `
  --max-new-tokens 256 `
  --allow-paid-api-call
```

Evaluate output and calculate measured cost:

```powershell
python scripts/phase4/evaluate_api_priced_smoke.py `
  --results-path results/raw/phase4_api_priced_smoke_results.jsonl `
  --dataset-root data/scaleup_2000_full
```

## Generated Reports

- `results/raw/phase4_api_priced_smoke_results.jsonl`
- `results/processed/phase4_api_priced_readiness_report.json`
- `results/processed/phase4_api_priced_smoke_eval_report.json`
- `results/processed/phase4_api_priced_smoke_eval_summary.csv`
- `results/processed/phase4_api_priced_cost_report.json`
- `results/processed/phase4_api_priced_cost_summary.csv`
- `data/generated/context_engineering/hf_api_pricing_snapshot_report.json`

Raw and processed execution results remain local and ignored according to
repository policy. The compact pricing snapshot report is safe to commit.

## Phase Decision

API-priced gated-model validation is complete for the five-prompt smoke scope.

Before GPU benchmarking, the remaining work is:

- validate streaming TTFT and TPOT capture;
- improve multi-evidence citation completeness;
- run the existing local/OpenAI-compatible path against a real GPU server;
- execute the controlled 500-prompt smoke with concurrency and hardware
  telemetry.

No GPU, vLLM, SGLang, or retrieval modification was used in Block 27.
