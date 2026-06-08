# Block 31B Summary

## Result

- Model5 executed: yes, 5/5 requests.
- Streaming worked: yes, 5/5 responses.
- Model/provider: `mistralai/ministral-3b-2512` through OpenRouter.
- Workload: five verticals, `mm2_hybrid_top5`, `prompt_plus_metadata`.
- GPU/vLLM/SGLang: not run.

## Model5 Metrics

| Metric | Result |
| --- | ---: |
| JSON validity | 80% |
| Contract validity | 80% |
| Evidence-ID presence | 80% |
| Evidence match | 40% |
| Groundedness | 40% |
| Safety violations | 0% |
| Mean TTFT | 605.19 ms |
| Mean ITL p50 / p95 / p99 | 1.72 / 19.47 / 37.16 ms |
| Mean TPOT | 5.62 ms |
| Mean E2E | 1,153.49 ms |
| Input / output / total tokens | 6,990 / 496 / 7,486 |
| Total cost | `$0.00074860` |
| Cost/request | `$0.00014972` |
| Cost/grounded answer | `$0.00037430` |

Failures were two multi-evidence under-citations and one Finance response
truncated at 128 output tokens.

## Comparison

Model6 produced 100% contract validity, 60% evidence match, and 60%
groundedness at `$0.00002863` per request. Model5 was 231.24 ms faster in mean
TTFT and 52.51 ms faster in mean E2E, but cost 5.23 times more per request and
was 20 percentage points lower on evidence match and groundedness.

Local Qwen 0.5B produced 80% contract validity, 40% evidence match, and 20%
groundedness. Its measured CPU mean latency was 146.22 seconds; local
infrastructure cost and streaming metrics were unavailable.

Recommendation: retain model5 as a 3B OpenRouter/provider comparison, not as the
current quality or cost leader.

## Controlled Inference Readiness

Status: `NOT_READY`

Already built:

- promoted retrieval source of truth and passing retrieval SLOs;
- deterministic workload splits and prompt-to-gold evaluation;
- memory modes `mm0` through `mm3`, with `mm4` contract-only;
- vertical chunking, canonical retrieval keys, Qdrant, and compression;
- checkpoint/resume, chunked persistence, timeouts, and failure rows;
- run manifests, request telemetry, backend/model/memory/error metadata;
- API pricing, GPU cost formulas, paid-call guard, and request limits;
- local HF and API smoke paths plus vLLM/SGLang dry-run adapters.

Remaining GPU blockers:

- select the RunPod GPU and model;
- fill region and hourly price;
- freeze the reviewed five-prompt vLLM smoke matrix.

Exact next block: Block 32A, a guarded concurrency-1 five-prompt vLLM GPU smoke
with live hardware telemetry and measured infrastructure cost.

## Files

- `src/inference_bench/controlled_run_readiness.py`
- `src/inference_bench/model_smoke_comparison.py`
- `scripts/phase4/audit_controlled_inference_readiness.py`
- `scripts/phase4/compare_model5_baselines.py`
- `tests/test_phase4_openrouter_smoke.py`
- `tests/test_phase4_controlled_run_readiness.py`
- `docs/93_model5_openrouter_streaming_smoke.md`
- `docs/94_controlled_inference_readiness_audit.md`
- `data/generated/phase4/controlled_inference_readiness_report.json`
- `data/generated/phase4/controlled_inference_readiness_summary.csv`

Measured raw and processed result files remain local and ignored.

## Commands

```powershell
pytest tests/test_phase4_openrouter_smoke.py
pytest tests/test_phase4_controlled_run_readiness.py
python scripts/phase4/run_api_priced_smoke.py ... --allow-paid-api-call
python scripts/phase4/evaluate_generation_outputs.py ...
python scripts/phase4/finalize_api_streaming_smoke.py ...
python scripts/phase4/compare_model5_baselines.py
python scripts/phase4/audit_controlled_inference_readiness.py
```

