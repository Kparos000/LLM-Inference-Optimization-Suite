# Block 30A Model5 Pricing Routing Summary

## Status

`ROUTE_AVAILABLE_PRICING_BLOCKED`

No model generation, paid request, GPU work, retrieval change, or evaluator
change was performed.

## Live Audit

- Model: `model5_gated` (`meta-llama/Llama-3.2-3B-Instruct`)
- `HF_TOKEN`: valid, HTTP 200
- Gated repository access: granted, HTTP 200
- Live provider: `featherless-ai`
- Chat completion: documented
- Streaming: documented through the Hugging Face chat API, not live-tested here
- Live input/output token pricing: unavailable
- Manual override template: present
- Manual override active: no
- Costed smoke: blocked

## Pricing Decision

The official Featherless source describes flat monthly pricing. It does not
provide exact input and output USD-per-million-token rates for this model.
Therefore:

- no monthly subscription price was converted into a token rate;
- no zero-cost assumption was made;
- the checked-in override is disabled and contains null token rates;
- model5 remains blocked until complete live rates or an authoritative manual
  token-price source exists.

Resolution precedence is verified:

1. complete live pricing;
2. complete enabled manual override;
3. explicit block.

## Reports

- `results/processed/phase4_model5_pricing_route_report.json`
- `results/processed/phase4_model5_pricing_route_summary.csv`

Generated reports remain local and ignored.

## Files Changed

- `configs/api_pricing.yaml`
- `src/inference_bench/api_pricing.py`
- `src/inference_bench/model5_pricing_routing.py`
- `scripts/phase4/audit_model5_pricing_route.py`
- `tests/test_phase4_model5_pricing_routing.py`
- `docs/89_model5_pricing_and_provider_routing.md`
- `docs/summaries/block30A_model5_pricing_routing_summary.md`
- `README.md`

## Verification

- `pytest tests/test_phase4_model5_pricing_routing.py`: 6 passed
- `pytest tests/test_phase4_api_pricing.py`: 4 passed
- `mypy src tests`: passed, 181 source files checked
- `pytest`: passed, 910 tests
- `ruff check .`: passed
- `ruff format --check .`: passed, 239 files formatted
- `python scripts/audit_repo_public_content.py`: passed
- `inference-bench doctor`: passed
- `inference-bench validate-config`: passed
