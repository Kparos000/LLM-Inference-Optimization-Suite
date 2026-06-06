# Model5 Pricing and Provider Routing

Block 30A audits `model5_gated`
(`meta-llama/Llama-3.2-3B-Instruct`) without model generation, GPU work, or
retrieval changes.

## Audit Result

The June 6, 2026 audit found:

| Check | Result |
| --- | --- |
| Local `HF_TOKEN` valid | Yes, HTTP 200 |
| Gated model repository access | Granted, HTTP 200 |
| Live inference provider | `featherless-ai` |
| Chat completion | Documented as supported |
| Streaming | Supported by the Hugging Face chat API; not live-tested in this block |
| Complete live input/output token pricing | Unavailable |
| Active manual token-price override | No |
| Costed model5 smoke allowed | No |

The public Hugging Face router metadata exposes Featherless as the live route
but does not expose separate input and output token prices.

## Manual Override Policy

`configs/api_pricing.yaml` contains a disabled model5 override template with:

- provider and model ID;
- input and output USD per 1 million tokens;
- source URL;
- `manual_override` status;
- review date and notes.

The token-rate fields remain null and `enabled` is false. The cited official
Featherless source describes flat monthly plans rather than per-token input and
output rates. Converting a monthly plan into token rates would require workload
and utilization assumptions, so it is not a valid manual token-price override.

To activate an override, a reviewer must provide exact input and output
USD-per-million-token rates from an authoritative source and set `enabled:
true`. Complete live router pricing still takes precedence over a manual
override.

## Decision Rules

The route resolver applies this order:

1. Use complete live pricing marked `detected`.
2. Otherwise use a complete, enabled, audited `manual_override`.
3. Otherwise block costed execution with an explicit reason.

Null rates, disabled templates, and monthly subscription prices never silently
enable a token-costed run.

## Provider Capability Evidence

- Hugging Face documents Featherless as a chat-completion provider:
  <https://huggingface.co/docs/inference-providers/main/providers/featherless-ai>
- Hugging Face documents streaming in its chat-completion API:
  <https://huggingface.co/docs/inference-providers/en/tasks/chat-completion>
- The live model route is:
  <https://router.huggingface.co/v1/models/meta-llama/Llama-3.2-3B-Instruct>
- The audited Featherless pricing explanation is:
  <https://featherless.ai/blog/llm-api-pricing-comparison-2026-complete-guide-inference-costs>

Streaming capability was established from provider/API documentation, not by a
paid model request.

## Run the Audit

```powershell
python scripts/phase4/audit_model5_pricing_route.py `
  --model-alias model5_gated `
  --pricing-config configs/api_pricing.yaml `
  --output-root results/processed
```

The command performs identity, gated repository access, and router metadata GET
requests. It does not call chat completion.

Generated local reports:

- `results/processed/phase4_model5_pricing_route_report.json`
- `results/processed/phase4_model5_pricing_route_summary.csv`

These reports remain ignored under repository output policy.

## Current Decision

Model5 is accessible and routable, but it cannot run a defensibly costed smoke
under the current per-token accounting contract. Model6 remains the priced API
route until model5 exposes complete live rates or an exact audited token-price
source becomes available.
