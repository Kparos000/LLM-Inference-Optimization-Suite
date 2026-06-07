# Block 31A Model5 Ministral Switch Summary

## Status

`IMPLEMENTED_NO_PAID_CALL`

`model5_gated` now resolves to:

- canonical key: `ministral_3b_2512_api`;
- model ID: `mistralai/ministral-3b-2512`;
- provider: OpenRouter;
- credential: `OPENROUTER_API_KEY`;
- input price: `$0.10` per 1M tokens;
- output price: `$0.10` per 1M tokens.

`model6_gated` remains:

- canonical key: `llama_3_1_8b_instruct_api`;
- model ID: `meta-llama/Llama-3.1-8B-Instruct`;
- provider route: Hugging Face router / Novita;
- input price: `$0.02` per 1M tokens;
- output price: `$0.05` per 1M tokens.

The old Llama 3.2 3B route remains available as
`old_model5_llama_3_2_3b`, but is inactive and has no usable per-token price.

## Alias Order

1. `model1_0_5b`
2. `model2_1_5b`
3. `model3_7b`
4. `model4_32b`
5. `model5_gated`
6. `model6_gated`
7. `model7_large_placeholder`

## Reports

- `results/processed/phase4_model_registry_report.json`
- `results/processed/phase4_model_registry_summary.csv`
- `results/processed/phase4_model5_switch_report.json`
- `results/processed/phase4_model5_switch_summary.csv`

The report status is `READY_FOR_EXPLICIT_TINY_OPENROUTER_SMOKE`. No paid
OpenRouter call was made.

## Files Changed

- `configs/models.yaml`
- `configs/api_pricing.yaml`
- `.env.example`
- `src/inference_bench/api_routes.py`
- `src/inference_bench/api_priced_validation.py`
- `src/inference_bench/api_pricing.py`
- `src/inference_bench/config.py`
- `src/inference_bench/model_registry.py`
- `src/inference_bench/openrouter_api.py`
- `src/inference_bench/phase3_readiness.py`
- `src/inference_bench/streaming_metrics.py`
- `scripts/phase4/audit_api_pricing.py`
- `scripts/phase4/run_api_priced_smoke.py`
- `scripts/phase4/run_grounding_repair_smoke.py`
- model/pricing/config/public-content tests
- `docs/92_model_registry_and_model5_switch.md`
- `README.md`

## Verification

- requested focused tests: 9 passed
- full test suite: 926 passed
- `mypy src tests`: passed across 191 source files
- `ruff check .`: passed
- `ruff format --check .`: passed across 251 files
- `python scripts/audit_repo_public_content.py`: passed
- `inference-bench doctor`: passed
- `inference-bench validate-config`: passed
- pricing audit: both model5 and model6 runnable with complete registered rates
