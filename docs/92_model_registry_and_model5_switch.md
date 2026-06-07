# Model Registry and Model5 Switch

Block 31A replaces the active `model5_gated` route with a cost-auditable
sub-4B model. No generation request, GPU work, or paid API call was made in this
block.

## Why Model5 Changed

The previous active route used `meta-llama/Llama-3.2-3B-Instruct` through
Featherless AI. Model access worked, but the available route exposed flat
subscription pricing rather than complete input/output token rates. That made
the model unsuitable for request-level cost comparisons.

The old canonical record remains in `configs/models.yaml` and resolves through
the deprecated alias `old_model5_llama_3_2_3b`. It is no longer the active
`model5_gated` target.

## Public Alias Registry

| Alias | Canonical key | Model ID | Execution |
| --- | --- | --- | --- |
| `model1_0_5b` | `qwen2_5_0_5b_instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | local or self-hosted |
| `model2_1_5b` | `qwen2_5_1_5b_instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | local or self-hosted |
| `model3_7b` | `qwen2_5_7b_instruct` | `Qwen/Qwen2.5-7B-Instruct` | self-hosted GPU |
| `model4_32b` | `qwen2_5_32b_instruct` | `Qwen/Qwen2.5-32B-Instruct` | later self-hosted GPU |
| `model5_gated` | `ministral_3b_2512_api` | `mistralai/ministral-3b-2512` | OpenRouter API |
| `model6_gated` | `llama_3_1_8b_instruct_api` | `meta-llama/Llama-3.1-8B-Instruct` | HF Inference Provider |
| `model7_large_placeholder` | `future_large_model_placeholder` | `placeholder/large-model` | future |

The generated, machine-readable table is in:

- `results/processed/phase4_model_registry_report.json`
- `results/processed/phase4_model_registry_summary.csv`

## Ministral Pricing

On June 7, 2026, both the public OpenRouter models API and model page reported:

- input: `$0.10` per 1 million tokens;
- output: `$0.10` per 1 million tokens;
- context length: 131,072 tokens.

Sources:

- <https://openrouter.ai/mistralai/ministral-3b-2512>
- <https://openrouter.ai/api/v1/models>

The checked-in pricing status is `detected_or_manual_verified`. The audit
compares the checked-in rates with public metadata and does not silently
estimate missing prices.

Cost is calculated as:

```text
input_cost = input_tokens / 1,000,000 * input_rate
output_cost = output_tokens / 1,000,000 * output_rate
total_cost = input_cost + output_cost
```

## OpenRouter Route

OpenRouter uses an OpenAI-compatible API:

- base URL: `https://openrouter.ai/api/v1`;
- chat route: `https://openrouter.ai/api/v1/chat/completions`;
- credential: `OPENROUTER_API_KEY`;
- model ID: `mistralai/ministral-3b-2512`;
- streaming: supported through server-sent events;
- token usage: read from the provider usage payload when available.

The existing generation contract, evaluator, streaming telemetry, and cost
calculator are reused. Provider routing is selected from model metadata:
model5 uses OpenRouter, while model6 continues to use the Hugging Face router
with `HF_TOKEN`.

Missing `OPENROUTER_API_KEY` does not break configuration or tests. It blocks
only a live OpenRouter request with a clear error.

## Audit Command

```powershell
python scripts/phase4/audit_api_pricing.py `
  --models model5_gated model6_gated `
  --pricing-config configs/api_pricing.yaml `
  --output-root results/processed
```

This command reads public metadata and writes:

- `results/processed/phase4_model_registry_report.json`
- `results/processed/phase4_model_registry_summary.csv`
- `results/processed/phase4_model5_switch_report.json`
- `results/processed/phase4_model5_switch_summary.csv`

It does not send a generation request.

## Future Tiny Smoke

A future explicitly authorized paid smoke can use the existing guarded runner:

```powershell
$env:OPENROUTER_API_KEY = "<local secret>"
python scripts/phase4/run_api_priced_smoke.py `
  --input-path data/generated/phase4/api_priced_contract_runner_input.jsonl `
  --output-path results/raw/phase4_model5_openrouter_smoke.jsonl `
  --model-alias model5_gated `
  --fallback-model-alias model6_gated `
  --limit 5 `
  --max-new-tokens 128 `
  --stream `
  --require-streaming `
  --allow-paid-api-call
```

The runner selects the OpenRouter route automatically. This command was not run
in Block 31A.
