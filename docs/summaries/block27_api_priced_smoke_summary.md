# Block 27 API-Priced Smoke Summary

## Status

`COMPLETE`

API-priced gated-model validation completed for five promoted records.

## Required Answers

1. **Did the model execute?** Yes. Five of five requests succeeded.
2. **Was pricing detected?** Yes, for `model6_gated` through Novita. Complete
   pricing was not available for `model5_gated`, so no price was estimated.
3. **What was total token usage?** 6,646 tokens: 6,243 input and 403 output.
4. **What was total cost?** `$0.00014501`.
5. **Did quality improve over Qwen 0.5B?** Yes for contract validity,
   evidence match, and groundedness; JSON validity and evidence-ID presence
   remained at 100%.
6. **Did groundedness improve?** Yes, from 20% to 60%.
7. **Did evidence match improve?** Yes, from 40% to 60%.
8. **Is API-priced validation complete?** Yes for the five-prompt smoke scope.
9. **What is the next blocker before GPU benchmarking?** Streaming TTFT/TPOT
   telemetry and a real GPU-serving smoke at controlled concurrency. Airline
   and Healthcare multi-evidence citation completeness also remains imperfect.

## Selected Model And Price

- Alias: `model6_gated`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Provider: `novita`
- Input price: `$0.02` per 1 million tokens
- Output price: `$0.05` per 1 million tokens
- Pricing source:
  `https://router.huggingface.co/v1/models/meta-llama/Llama-3.1-8B-Instruct`
- Pricing snapshot: `2026-06-06T01:22:14.834095+00:00`

`model5_gated` was not executed because the official router metadata did not
provide both input and output pricing for its available provider.

## Quality

| Metric | Qwen 0.5B | API-priced 8B | Delta |
| --- | ---: | ---: | ---: |
| JSON validity | 100% | 100% | 0 points |
| Contract validity | 80% | 100% | +20 points |
| Evidence-ID presence | 100% | 100% | 0 points |
| Evidence match | 40% | 60% | +20 points |
| Groundedness | 20% | 60% | +40 points |
| Safety violations | 0% | 0% | 0 points |

Retail, Finance, and Research AI fully matched required evidence. Airline and
Healthcare Admin each omitted one required evidence family.

## Runtime And Cost

- Requests: 5
- Successful requests: 5
- Mean latency: 1,282.893 ms
- Median latency: 1,270.200 ms
- Mean output throughput: 63.166 tokens/s
- TTFT: unavailable from the non-streaming response
- Input tokens: 6,243
- Output tokens: 403
- Total tokens: 6,646
- Input cost: `$0.00012486`
- Output cost: `$0.00002015`
- Total cost: `$0.00014501`
- Cost per request: `$0.000029002`
- Cost per grounded answer: `$0.000048337`

## Files Changed

- `configs/api_pricing.yaml`
- `src/inference_bench/api_priced_validation.py`
- `scripts/phase4/run_api_priced_model_smoke.py`
- `scripts/phase4/evaluate_api_priced_smoke.py`
- `tests/test_phase4_api_priced_smoke.py`
- `docs/86_phase4_api_priced_model_smoke.md`
- `docs/summaries/block27_api_priced_smoke_summary.md`
- `data/generated/context_engineering/hf_api_pricing_snapshot_report.json`
- `README.md`

## Generated Reports

- `results/raw/phase4_api_priced_smoke_results.jsonl`
- `results/processed/phase4_api_priced_readiness_report.json`
- `results/processed/phase4_api_priced_smoke_eval_report.json`
- `results/processed/phase4_api_priced_smoke_eval_summary.csv`
- `results/processed/phase4_api_priced_cost_report.json`
- `results/processed/phase4_api_priced_cost_summary.csv`
- `data/generated/context_engineering/hf_api_pricing_snapshot_report.json`

Raw and processed result files remain local and ignored. No credential is
stored in a generated report.

## Commands Run

- live HF router pricing snapshot for `model5_gated` and `model6_gated`;
- successful pricing snapshot for `model6_gated`;
- sanitized HF token and gated repository access checks;
- five-request guarded API smoke;
- generation-contract evaluation;
- measured token-cost aggregation;
- focused pytest, mypy, and Ruff validation;
- full repository verification commands listed in the task.

## Commit

Commit message: `Validate API-priced gated model smoke`

The pushed commit hash is reported in the final execution result.
