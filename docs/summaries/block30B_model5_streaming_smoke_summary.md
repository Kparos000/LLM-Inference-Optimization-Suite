# Block 30B Model5 Streaming Smoke Summary

## Status

`BLOCKED_PRE_EXECUTION`

## Required Answers

1. **Did model5 run?** No.
2. **Exact reason:** no complete live token pricing or enabled audited manual
   token-price override exists.
3. **Was fallback used?** No. Model6 fallback is allowed only after an actual
   model5 execution failure; model5 was stopped by preflight.
4. **Streaming metrics:** unavailable because no request was sent.
5. **Cost metrics:** unavailable and stored as null. No price or cost was
   fabricated.
6. **Quality metrics:** unavailable because no generation output exists.
7. **Final experiment:** keep model5 as a conditional registry candidate, but
   exclude it from the active costed experiment until authoritative token rates
   are available.

## Preflight Result

- token valid: yes
- gated access: yes
- provider: `featherless-ai`
- chat completion: documented
- streaming: documented, not executed
- planned prompts: 5, one per vertical
- executed prompts: 0
- paid API calls: 0
- fallback calls: 0

## Prior Baselines

| Metric | Model6 Block 29 | Local Qwen 0.5B Block 24 |
| --- | ---: | ---: |
| JSON validity | 100% | 100% |
| Contract validity | 100% | 80% |
| Evidence match | 60% | 40% |
| Groundedness | 60% | 20% |
| Mean E2E latency | 1,205.995 ms | 146,223 ms |
| Total cost | `$0.00014316` | unmeasured |

No new model5 comparison value can be calculated from a blocked run.

## Outputs

- `results/raw/phase4_model5_streaming_smoke_results.jsonl`
- `results/processed/phase4_model5_streaming_eval_report.json`
- `results/processed/phase4_model5_streaming_eval_summary.csv`
- `results/processed/phase4_model5_streaming_cost_report.json`
- `results/processed/phase4_model5_streaming_latency_report.json`

Generated results remain local and ignored.

## Files Changed

- `src/inference_bench/model5_pricing_routing.py`
- `src/inference_bench/model5_streaming_validation.py`
- `scripts/phase4/audit_model5_pricing_route.py`
- `scripts/phase4/run_model5_streaming_smoke.py`
- `scripts/phase4/finalize_api_streaming_smoke.py`
- `tests/test_phase4_model5_streaming_smoke.py`
- `docs/90_model5_streaming_api_smoke.md`
- `docs/summaries/block30B_model5_streaming_smoke_summary.md`
- `README.md`

## Verification

- `pytest tests/test_phase4_api_pricing.py`: 4 passed
- `pytest tests/test_phase4_streaming_metrics.py`: 3 passed
- `pytest tests/test_phase4_model5_streaming_smoke.py`: 2 passed
- `pytest tests/test_phase4_model5_pricing_routing.py`: 6 passed
- `mypy src tests`: passed, 183 source files checked
- `pytest`: passed, 912 tests
- `ruff check .`: passed
- `ruff format --check .`: passed, 242 files formatted
- `python scripts/audit_repo_public_content.py`: passed
- `inference-bench doctor`: passed
- `inference-bench validate-config`: passed
- custom streaming finalizer output names: validated
