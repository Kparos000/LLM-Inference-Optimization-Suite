# Block 26 Stronger-Model Contract Summary

## Status

`DRY_RUN_ONLY`

The stronger-model quality comparison is not yet measured.

## Execution Decision

- `model2_1_5b` was requested first.
- `Qwen/Qwen2.5-1.5B-Instruct` was not present in the local Hugging Face cache.
- The local-only attempt failed clearly and did not download weights.
- `HF_TOKEN` was absent.
- `configs/api_pricing.yaml` did not contain a live pricing snapshot.
- `model5_gated` therefore ran through the API-route dry-run path only.
- No paid API call, GPU work, vLLM work, or model inference occurred.

## Workload Validation

Five records were selected, one per vertical. Every record used:

- `mm2_hybrid_top5`;
- `prompt_plus_metadata`;
- `qdrant_vector`;
- `qdrant_local`;
- `source_hints_used: false`.

## Comparison

| Metric | Block 24 | Block 26 |
| --- | ---: | ---: |
| JSON validity | 100% | Not measured |
| Contract validity | 80% | Not measured |
| Evidence-ID presence | 100% | Not measured |
| Evidence match | 40% | Not measured |
| Groundedness | 20% | Not measured |
| Mean latency | 146.223 seconds | Not measured |
| Median latency | 172.321 seconds | Not measured |
| Input tokens | 7,456 | Not measured |
| Output tokens | 574 | Not measured |

No conclusion can yet be made about whether the 0.5B model caused the Block 24
failures.

## Files Changed

- `src/inference_bench/stronger_model_validation.py`
- `scripts/phase4/run_stronger_model_contract_smoke.py`
- `scripts/phase4/evaluate_stronger_model_contract.py`
- `scripts/phase4/run_local_hf_smoke.py`
- `scripts/phase3/hf_api_tiny_smoke.py`
- `tests/test_phase4_stronger_model_contract.py`
- `tests/test_phase4_generation_contract_hardening.py`
- `docs/85_phase4_stronger_model_contract_validation.md`
- `README.md`

## Generated Local Outputs

- `results/raw/phase4_stronger_model_contract_smoke.jsonl`
- `results/processed/phase4_stronger_model_contract_eval_report.json`
- `results/processed/phase4_stronger_model_contract_eval_summary.csv`

These outputs remain ignored according to repository policy.

## Next Action

Cache `model2_1_5b` deliberately on a machine with sufficient free RAM and
rerun the five-prompt local path. If local execution remains impractical,
snapshot valid provider pricing and run the gated API path only with explicit
paid-call authorization.

## Verification

- Focused tests: `33 passed`
- Full test suite: `885 passed`
- Mypy: no issues in 170 source files
- Ruff check: passed
- Ruff format check: 221 files already formatted
- Public-content audit: passed
- `inference-bench doctor`: passed without requiring a GPU
- `inference-bench validate-config`: passed
