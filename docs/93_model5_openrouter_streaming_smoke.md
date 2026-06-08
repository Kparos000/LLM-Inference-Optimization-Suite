# Model5 OpenRouter Streaming Smoke

Block 31B validated `model5_gated` through OpenRouter with real, paid, streaming
generation. The run used five prompts, one per vertical, with
`mm2_hybrid_top5`, the promoted retrieval baseline, and the shared generation
contract. No GPU, vLLM, SGLang, or retrieval change was involved.

## Execution

- Alias: `model5_gated`
- Model: `mistralai/ministral-3b-2512`
- Provider: OpenRouter
- Route: `https://openrouter.ai/api/v1/chat/completions`
- Requests: 5/5 successful
- Streaming: 5/5
- Maximum output: 128 tokens
- Credential: loaded locally from `OPENROUTER_API_KEY`; never written to output

The prompt-specified `generation_contract_runner_input.jsonl` was rejected
before any API call because it had `ablation_mode=none`. The guarded runner
requires the promoted `prompt_plus_metadata` baseline. The successful run used
`data/generated/phase4/api_priced_contract_runner_input.jsonl`, which contains
the same five verticals and `mm2_hybrid_top5`.

## Quality

| Metric | Model5 |
| --- | ---: |
| JSON validity | 80% |
| Contract validity | 80% |
| Evidence-ID presence | 80% |
| Evidence match | 40% |
| Groundedness | 40% |
| Safety violations | 0% |

Retail and Research AI fully matched required evidence. Airline and Healthcare
Admin cited only part of the required evidence family. Finance reached the
128-token limit and returned truncated JSON. The strict evaluator was not
changed.

## Streaming Latency

| Metric | Mean |
| --- | ---: |
| TTFT | 605.19 ms |
| ITL p50 | 1.72 ms |
| ITL p95 | 19.47 ms |
| ITL p99 | 37.16 ms |
| TPOT | 5.62 ms |
| End-to-end latency | 1,153.49 ms |

## Tokens And Cost

- Input tokens: 6,990
- Output tokens: 496
- Total tokens: 7,486
- Total cost: `$0.00074860`
- Cost per request: `$0.00014972`
- Cost per successful answer: `$0.00014972`
- Cost per grounded answer: `$0.00037430`

Pricing was read from `configs/api_pricing.yaml`: `$0.10` per million input
tokens and `$0.10` per million output tokens.

## Baseline Comparison

The five prompt IDs align across model5, model6, and local Qwen smoke artifacts.
This remains a directional plumbing comparison, not a statistically meaningful
model ranking.

| Metric | Model5 Ministral 3B | Model6 Llama 3.1 8B | Local Qwen 0.5B |
| --- | ---: | ---: | ---: |
| Contract validity | 80% | 100% | 80% |
| Evidence match | 40% | 60% | 40% |
| Groundedness | 40% | 60% | 20% |
| Mean TTFT | 605.19 ms | 836.43 ms | unavailable |
| Mean E2E | 1,153.49 ms | 1,205.99 ms | 146,223.35 ms |
| Cost/request | `$0.00014972` | `$0.00002863` | unmeasured |

Model5 had 231.24 ms lower mean TTFT and 52.51 ms lower mean end-to-end latency
than model6. Model6 had 20 percentage points higher evidence match and
groundedness and cost about one fifth as much per request.

## Recommendation

Retain model5 in the final experiment as a distinct 3B OpenRouter route and
provider comparison. Do not treat it as the preferred quality or cost model
based on this smoke. Model6 is the stronger current API baseline.

## Artifacts

- `results/raw/phase4_model5_openrouter_streaming_smoke_results.jsonl`
- `results/processed/phase4_model5_openrouter_streaming_eval_report.json`
- `results/processed/phase4_model5_openrouter_streaming_eval_summary.csv`
- `results/processed/phase4_model5_openrouter_streaming_cost_report.json`
- `results/processed/phase4_model5_openrouter_streaming_latency_report.json`
- `results/processed/phase4_model5_vs_model6_api_comparison_report.json`
- `results/processed/phase4_model5_vs_model6_api_comparison_summary.csv`

Raw and processed smoke outputs remain local under repository output policy.

