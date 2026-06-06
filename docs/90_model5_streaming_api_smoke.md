# Model5 Streaming API Smoke

Block 30B attempted the five-prompt `model5_gated` streaming smoke using:

- `meta-llama/Llama-3.2-3B-Instruct`;
- `mm2_hybrid_top5`;
- the promoted Qdrant-backed retrieval baseline;
- one prompt per vertical;
- the shared generation contract;
- required streaming;
- a maximum of 128 output tokens per prompt.

No GPU, vLLM, SGLang, retrieval change, or evaluator weakening was used.

## Execution Result

Status: `BLOCKED_PRE_EXECUTION`

Model5 did not run. The live route audit passed:

- `HF_TOKEN` valid;
- gated model access granted;
- live provider `featherless-ai`;
- chat completion documented;
- streaming documented.

Execution was blocked because neither complete live input/output token pricing
nor an enabled audited manual token-price override exists. The official
Featherless source describes flat monthly subscription pricing, which cannot be
converted into per-token rates without assumptions.

No chat-completion request was sent. The model6 fallback was not used because
the primary model did not reach execution. Treating a pricing preflight failure
as an execution failure would violate the fallback requirement and could create
an unrequested model6 charge.

## Metrics

All model5 generation, latency, token, cost, and quality metrics are unavailable
and represented as null, not zero.

| Metric | Model5 Block 30B |
| --- | ---: |
| Executed prompts | 0 |
| JSON validity | unavailable |
| Contract validity | unavailable |
| Evidence match | unavailable |
| Groundedness | unavailable |
| TTFT | unavailable |
| ITL p50/p95/p99 | unavailable |
| TPOT | unavailable |
| End-to-end latency | unavailable |
| Input/output tokens | unavailable |
| Total cost | unavailable |
| Cost per request | unavailable |
| Cost per grounded answer | unavailable |

The total cost is not reported as `$0`. No request was made, but the missing
provider pricing remains a separate fact from incurred cost.

## Baseline Comparison

| Metric | Model5 Block 30B | Model6 Block 29 | Local Qwen 0.5B Block 24 |
| --- | ---: | ---: | ---: |
| Execution | blocked | 5/5 | 5/5 |
| Streaming | not attempted | 5/5 | no |
| JSON validity | unavailable | 100% | 100% |
| Contract validity | unavailable | 100% | 80% |
| Evidence match | unavailable | 60% | 40% |
| Groundedness | unavailable | 60% | 20% |
| Mean TTFT | unavailable | 836.431 ms | unavailable |
| Mean TPOT | unavailable | 5.081 ms | unavailable |
| Mean E2E latency | unavailable | 1,205.995 ms | 146,223 ms |
| Input tokens | unavailable | 6,243 | 7,456 |
| Output tokens | unavailable | 366 | 574 |
| Total cost | unavailable | `$0.00014316` | unmeasured local cost |

The model6 and local measurements are prior results. Block 30B made no
additional paid call.

## Fallback Semantics

The new orchestrator separates:

- preflight failure: token, access, provider, or pricing gate fails;
- execution failure: a primary generation request is actually attempted and
  produces no successful result.

Model6 is eligible only for the second case. It is not selected because model5
lacks pricing.

## Outputs

- `results/raw/phase4_model5_streaming_smoke_results.jsonl`
- `results/processed/phase4_model5_streaming_eval_report.json`
- `results/processed/phase4_model5_streaming_eval_summary.csv`
- `results/processed/phase4_model5_streaming_cost_report.json`
- `results/processed/phase4_model5_streaming_latency_report.json`

The raw file contains a clearly labeled preflight-status record. The processed
reports have `metrics_available: false` and null metrics. Generated results
remain local and ignored.

## Command

```powershell
python scripts/phase4/run_model5_streaming_smoke.py `
  --input-path data/generated/phase4/api_priced_contract_runner_input.jsonl `
  --output-path results/raw/phase4_model5_streaming_smoke_results.jsonl `
  --processed-root results/processed `
  --dataset-root data/scaleup_2000_full `
  --pricing-config configs/api_pricing.yaml `
  --limit 5 `
  --max-new-tokens 128 `
  --allow-paid-api-call
```

The command exits with code `2` for the current expected preflight block.

## Final Experiment Decision

Keep model5 in the model registry as a conditional candidate, but do not include
it in the active costed final experiment until exact input/output token rates
are available from live router metadata or an authoritative audited source.
Model6 remains the valid API-priced comparison model.
