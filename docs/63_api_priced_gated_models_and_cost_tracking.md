# API-Priced Gated Models and Cost Tracking

Block 6 adds API-priced gated model support so the project can compare two
different cost surfaces:

- self-hosted GPU infrastructure cost for local/vLLM/SGLang-style runs,
- API token cost for Hugging Face Inference Provider runs.

This block does not run the full benchmark, GPU experiments, vLLM, SGLang, or
expensive API calls.

## Models Added

The public model alias order is now:

| Alias | Model ID | Execution target |
|---|---|---|
| `model1_0_5b` | `Qwen/Qwen2.5-0.5B-Instruct` | local or self-hosted |
| `model2_3b` | `Qwen/Qwen2.5-3B-Instruct` | local or self-hosted |
| `model3_7b` | `Qwen/Qwen2.5-7B-Instruct` | self-hosted GPU |
| `model4_32b` | `Qwen/Qwen2.5-32B-Instruct` | later self-hosted GPU |
| `model5_gated` | `mistralai/ministral-3b-2512` | OpenRouter API |
| `model6_gated` | `meta-llama/Llama-3.1-8B-Instruct` | HF Inference Provider API |
| `model7_gated` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | HF Inference Provider API, pricing pending |

Deprecated aliases such as `model2_1_5b`, `model7_large_placeholder`,
`model5_large_placeholder`, and `old_model5_llama_3_2_3b` are retained for
historical reports and frozen experiment configs.

## Why API-Priced Models Were Added

Self-hosted open-weight models have no per-token vendor API price, but they are
not free. Their cost comes from GPU rental, runtime, failed runs, and
engineering overhead. API-priced models have explicit input/output token prices,
which makes cost-per-request and cost-per-answer easier to measure.

The benchmark needs both views:

- API token cost: useful for direct hosted-provider comparisons.
- GPU infrastructure cost: useful for vLLM/SGLang/self-hosted optimization.

## HF_TOKEN Safety

`HF_TOKEN` is required for gated Hugging Face models and must never be committed
or printed. The tiny smoke script reads it only from the environment and never
writes it to logs, JSON, CSV, or stdout.

Set `HF_TOKEN` only in your local shell or secret manager. Do not put real
tokens in committed files, examples, reports, logs, JSON, or CSV.

## Pricing Snapshot

Pricing is captured from Hugging Face router model metadata:

```powershell
python scripts/phase3/snapshot_hf_inference_pricing.py `
  --models model5_gated model6_gated model7_gated `
  --output configs/api_pricing.yaml `
  --report data/generated/context_engineering/hf_api_pricing_snapshot_report.json
```

The script captures:

- model alias and model ID,
- provider and provider status,
- input/output dollars per 1M tokens,
- context length when available,
- latency and throughput metadata when available,
- tool and structured-output support when available,
- snapshot timestamp and pricing source URL.

The script does not hard-code fake prices. If Hugging Face metadata does not
expose pricing for a target model, the script fails clearly and writes a report
with providers found, missing pricing fields, and candidate priced alternatives
when discoverable.

## Tiny Paid API Smoke

The smoke script is intentionally gated:

```powershell
python scripts/phase3/hf_api_tiny_smoke.py `
  --model model5_gated `
  --max-new-tokens 32 `
  --allow-paid-api-call
```

It refuses to run unless:

- `HF_TOKEN` exists,
- `--allow-paid-api-call` is passed,
- `max_new_tokens` is between 1 and 32,
- pricing exists in `configs/api_pricing.yaml`.

Results are written under:

```text
results/raw/hf_api_tiny_smoke/
```

Captured fields include model alias, model ID, provider, input/output tokens,
input/output/total API cost, latency, success, and error type.

## Cost Formula

API token cost is calculated as:

```text
input_cost_usd =
  input_tokens / 1_000_000 * input_cost_per_1m_tokens_usd

output_cost_usd =
  output_tokens / 1_000_000 * output_cost_per_1m_tokens_usd

total_api_cost_usd =
  input_cost_usd + output_cost_usd
```

Later Phase 4 and Phase 5 reporting can derive:

- cost per 1,000 requests,
- cost per 1M tokens,
- cost per successful answer,
- cost per grounded correct answer.

## Final Cost Comparison

For final reporting, keep API token cost separate from self-hosted GPU cost.

`hf_inference_provider` rows should use API token pricing. vLLM and SGLang rows
should use infrastructure cost based on GPU hourly price, runtime, total
requests, total tokens, successful answers, and grounded correct answers.
