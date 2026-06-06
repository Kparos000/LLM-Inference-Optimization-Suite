# Block 29 Streaming API, Pricing, And Grounding Summary

## Status

`COMPLETE`

The five-request API stream completed without GPU, vLLM, SGLang, or retrieval
changes.

## Pricing

- `model5_gated`: `unavailable`
  - provider found: Featherless AI
  - complete input/output pricing: unavailable
  - manual override: absent
  - costed run: blocked
- `model6_gated`: `detected`
  - provider: Novita
  - input: `$0.02` per 1 million tokens
  - output: `$0.05` per 1 million tokens
  - costed run: allowed

The run selected `model6_gated` because it was the first model with complete
non-fabricated pricing and gated-model access.

## Streaming Metrics

Streaming worked for 5/5 requests.

| Metric | Mean | Median |
| --- | ---: | ---: |
| TTFT | 836.431 ms | 846.518 ms |
| ITL p50 | 3.783 ms | 3.921 ms |
| ITL p95 | 12.839 ms | 11.804 ms |
| ITL p99 | 30.627 ms | 25.940 ms |
| TPOT | 5.081 ms | 4.993 ms |
| End-to-end latency | 1,205.995 ms | 1,214.199 ms |

Provider usage supplied all token counts:

- input tokens: 6,243
- output tokens: 366
- total tokens: 6,609

## Cost

- total cost: `$0.00014316`
- cost per request: `$0.000028632`
- cost per successful answer: `$0.000028632`
- cost per grounded answer: `$0.000047720`

## Quality

- JSON validity: 100%
- contract validity: 100%
- evidence-ID presence: 100%
- evidence match: 60%
- groundedness: 60%
- safety violations: 0%

## Grounding Failure Classes

- `missing_required_evidence_id`: 2
- `cited_partial_evidence_only`: 2
- `multi_evidence_under_citation`: 2
- `semantic_under_answer`: 1

Airline and Healthcare Admin each emitted two short labels that resolved to one
canonical evidence family, leaving the second required family uncited.

## GPU Comparison Status

`READY_FOR_GPU_COMPARISON`

TTFT, ITL, TPOT, end-to-end latency, provider token usage, cost accounting,
generation quality, and grounding diagnostics are now available. The remaining
quality caveat is multi-evidence citation completeness.

## Reports

- `results/raw/phase4_api_streaming_smoke_results.jsonl`
- `results/processed/phase4_api_pricing_audit_report.json`
- `results/processed/phase4_api_pricing_audit_summary.csv`
- `results/processed/phase4_api_streaming_smoke_eval_report.json`
- `results/processed/phase4_api_streaming_smoke_eval_summary.csv`
- `results/processed/phase4_api_streaming_cost_report.json`
- `results/processed/phase4_api_streaming_cost_summary.csv`
- `results/processed/phase4_api_streaming_latency_report.json`
- `results/processed/phase4_grounding_failure_report.json`
- `results/processed/phase4_grounding_failure_summary.csv`

Generated raw and processed reports remain local and ignored.

## Files Changed

- `configs/api_pricing.yaml`
- `src/inference_bench/api_pricing.py`
- `src/inference_bench/api_priced_validation.py`
- `src/inference_bench/streaming_metrics.py`
- `src/inference_bench/grounding_diagnostics.py`
- `scripts/phase3/snapshot_hf_inference_pricing.py`
- `scripts/phase4/audit_api_pricing.py`
- `scripts/phase4/run_api_priced_smoke.py`
- `scripts/phase4/finalize_api_streaming_smoke.py`
- `tests/test_phase4_api_pricing.py`
- `tests/test_phase4_streaming_metrics.py`
- `tests/test_phase4_grounding_diagnostics.py`
- `docs/88_phase4_streaming_api_pricing_and_grounding.md`
- `README.md`

## Commit

Commit message:
`Add streaming API metrics pricing registry and grounding diagnostics`

The final commit hash is reported after verification and push.

## Verification

- Focused Block 29 tests: 10 passed
- Full pytest suite: 904 passed
- Mypy: no issues in 179 source files
- Ruff check: passed
- Ruff format check: 236 files formatted
- Public-content audit: passed
- `inference-bench doctor`: passed
- `inference-bench validate-config`: passed
