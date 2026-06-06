# Phase 4 Streaming API, Pricing, And Grounding Diagnostics

Block 29 closes three API-smoke caveats before the first GPU serving run:

- the API path now measures streaming TTFT and inter-chunk latency;
- pricing is stored in an auditable registry with explicit unavailable and
  manual-override states;
- non-grounded outputs receive deterministic failure classifications.

No GPU, vLLM, SGLang, retrieval change, or evaluator weakening was used.

## Pricing Registry

`configs/api_pricing.yaml` now records:

- provider and model ID;
- input/output USD per 1 million tokens;
- pricing source and URL;
- last-checked timestamp;
- `detected`, `manual_override`, or `unavailable` status;
- explanatory notes.

Detected complete pricing takes precedence over a manual override. A manual
override is accepted only when detected pricing is incomplete and both rates,
provider, source URL, and review timestamp are explicitly supplied. If neither
path is complete, costed execution is blocked.

## Pricing Audit

The live audit on June 6, 2026 found:

| Alias | Model | Status | Costed smoke |
| --- | --- | --- | --- |
| `model5_gated` | Llama 3.2 3B Instruct | `unavailable` | Blocked |
| `model6_gated` | Llama 3.1 8B Instruct | `detected` | Runnable |

For `model5_gated`, Featherless AI was exposed as a live provider, but complete
input/output token pricing was not exposed. No manual override was configured,
so no price was estimated.

For `model6_gated`, Novita exposed:

- input: `$0.02` per 1 million tokens;
- output: `$0.05` per 1 million tokens.

The smoke therefore used `model6_gated`.

## Streaming Contract

The runner supports:

- `--stream`;
- `--no-stream`;
- `--require-streaming`.

With streaming enabled, the runner requests server-sent events and records:

- TTFT: request start to first non-empty content event;
- inter-token-latency percentiles over content-event arrival intervals;
- TPOT: post-first-token duration divided by remaining output tokens;
- end-to-end latency;
- provider-reported input and output tokens;
- total token cost.

Inter-token latency is measured at stream content-event granularity. Providers
may emit more than one tokenizer token per event, so ITL is an observed stream
delivery metric, not a claim about internal decode scheduling.

If no content events arrive, `streaming_available` is false and TTFT/ITL/TPOT
remain null. `--require-streaming` then fails the run.

## Five-Prompt Streaming Smoke

The measured run used:

- `model6_gated`;
- Novita through the Hugging Face router;
- five prompts, one per vertical;
- promoted `prompt_plus_metadata` retrieval;
- `mm2_hybrid_top5`;
- Qdrant-backed promoted retrieval;
- maximum 128 new tokens.

Streaming succeeded for all five requests. Provider usage counts were returned
for every request.

### Latency

| Metric | Mean | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 836.431 ms | 846.518 ms | 720.255 ms | 909.087 ms |
| ITL p50 | 3.783 ms | 3.921 ms | 1.465 ms | 4.966 ms |
| ITL p95 | 12.839 ms | 11.804 ms | 10.696 ms | 18.002 ms |
| ITL p99 | 30.627 ms | 25.940 ms | 17.539 ms | 53.337 ms |
| TPOT | 5.081 ms | 4.993 ms | 4.494 ms | 5.550 ms |
| End-to-end | 1,205.995 ms | 1,214.199 ms | 1,111.636 ms | 1,278.559 ms |

### Tokens And Cost

| Metric | Result |
| --- | ---: |
| Input tokens | 6,243 |
| Output tokens | 366 |
| Total tokens | 6,609 |
| Input cost | `$0.00012486` |
| Output cost | `$0.00001830` |
| Total cost | `$0.00014316` |
| Cost per request | `$0.000028632` |
| Cost per successful answer | `$0.000028632` |
| Cost per grounded answer | `$0.000047720` |

## Generation Quality

| Metric | Result |
| --- | ---: |
| JSON validity | 100% |
| Contract validity | 100% |
| Evidence-ID presence | 100% |
| Full evidence match | 60% |
| Deterministic groundedness | 60% |
| Safety violations | 0% |

The streaming path preserved the Block 27 quality result. Streaming did not fix
the remaining evidence-selection behavior.

## Grounding Diagnostics

Two outputs were not grounded:

- Airline cited `E1` and `E3`, which both map to the same
  `CA-POL-012` evidence family, and omitted required `CA-POL-013`.
- Healthcare Admin cited `E1` and `E3`, which both map to the same
  `MCH-POL-001` evidence family, and omitted required `MCH-POL-020`.

Failure counts:

| Failure class | Count |
| --- | ---: |
| `missing_required_evidence_id` | 2 |
| `cited_partial_evidence_only` | 2 |
| `multi_evidence_under_citation` | 2 |
| `semantic_under_answer` | 1 |

No malformed contracts, wrong-only citation sets, or insufficient-evidence
misuse occurred. The diagnostic is deterministic and does not claim semantic
entailment beyond evaluator signals.

## Commands

Audit pricing:

```powershell
python scripts/phase4/audit_api_pricing.py `
  --models model5_gated model6_gated `
  --pricing-config configs/api_pricing.yaml `
  --output-root results/processed
```

Run the explicitly authorized stream:

```powershell
python scripts/phase4/run_api_priced_smoke.py `
  --input-path data/generated/phase4/api_priced_contract_runner_input.jsonl `
  --output-path results/raw/phase4_api_streaming_smoke_results.jsonl `
  --model-alias model5_gated `
  --fallback-model-alias model6_gated `
  --limit 5 `
  --max-new-tokens 128 `
  --stream `
  --require-streaming `
  --pricing-config configs/api_pricing.yaml `
  --allow-paid-api-call
```

Evaluate and finalize:

```powershell
python scripts/phase4/evaluate_generation_outputs.py `
  --results-path results/raw/phase4_api_streaming_smoke_results.jsonl `
  --dataset-root data/scaleup_2000_full `
  --output-root results/processed `
  --report-name phase4_api_streaming_smoke_eval_report.json `
  --summary-name phase4_api_streaming_smoke_eval_summary.csv

python scripts/phase4/finalize_api_streaming_smoke.py `
  --results-path results/raw/phase4_api_streaming_smoke_results.jsonl `
  --eval-report results/processed/phase4_api_streaming_smoke_eval_report.json `
  --output-root results/processed
```

## Generated Outputs

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

These generated execution outputs remain local and ignored.

## GPU Comparison Decision

The API smoke is ready for a controlled GPU comparison. Streaming latency,
token usage, API cost, evaluator quality, and deterministic grounding
diagnostics are now available.

The remaining quality caveat is multi-evidence citation completeness. The first
GPU smoke should freeze the same promoted runner input and compare whether
local vLLM generation reproduces or improves the 60% evidence-match and
groundedness result.
