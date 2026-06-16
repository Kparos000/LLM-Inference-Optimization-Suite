# B6 500-Prompt Quality Scale Gate

Status: measured on June 15, 2026

## Scope

B6 ran the first post-B5 scale gate:

- hardware: `remote_rtx3070`;
- engine: vLLM;
- model: `model2_1_5b` / `Qwen/Qwen2.5-1.5B-Instruct`;
- memory mode: `mm2_hybrid_top5`;
- split: balanced 500 prompts, 100 per vertical;
- concurrency: 1;
- streaming: enabled;
- temperature: 0;
- maximum output: 160 tokens;
- B5 safety, planning, and multi-evidence citation repairs active.

No gold data, promoted retrieval data, or evaluator semantics were modified.

## Artifacts

- Config: `configs/experiments/b6_remote_rtx3070_vllm_1_5b_500_quality_gate.yaml`
- Runner input: `data/generated/phase4/b6_context_aligned_500_runner_input.jsonl`
- Preflight report: `results/processed/b6_context_alignment_preflight_report.json`
- Raw results: `results/raw/b6_vllm_1_5b_500_results.jsonl`
- Evaluation report: `results/processed/b6_vllm_1_5b_500_eval_report.json`
- Summary CSV: `results/processed/b6_vllm_1_5b_500_eval_summary.csv`
- Latency CSV: `results/processed/b6_vllm_1_5b_500_latency_summary.csv`
- GPU telemetry: `results/processed/b6_vllm_1_5b_500_gpu_telemetry_summary.json`
- B5/B6 comparison: `results/processed/b6_b5_vs_b6_comparison.json`
- Runtime projections: `results/processed/b6_runtime_projection_report.json`

## Preflight

The offline context-alignment preflight passed:

| Vertical | Rows | All Required Evidence Present | Partial | Absent | Leaked Canonical IDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Airline | 100 | 100 | 0 | 0 | 0 |
| Healthcare Admin | 100 | 100 | 0 | 0 | 0 |
| Retail | 100 | 100 | 0 | 0 | 0 |
| Finance | 100 | 100 | 0 | 0 | 0 |
| Research AI | 100 | 100 | 0 | 0 | 0 |
| All | 500 | 500 | 0 | 0 | 0 |

The preflight status was `PREFLIGHT_PASSED_B6_CONTEXT_ALIGNMENT`.

## Result

B6 completed operationally:

- 500 of 500 requests completed;
- JSON validity: 95.4%;
- contract validity: 94.8%;
- evidence match: 91.2%;
- groundedness: 90.8%;
- safety violations: 0;
- truncation rate: 4.6%;
- bounded retry attempts: 99;
- lexical guard repairs: 10.

The quality gate decision is:

```text
B6_QUALITY_IMPROVED_BUT_BLOCKED
```

B6 passed the aggregate evidence and groundedness thresholds and kept safety at
zero. It failed JSON validity, contract validity, truncation, minimum vertical
evidence match, and minimum vertical groundedness.

## Per-Vertical Quality

| Vertical | JSON | Contract | Evidence | Grounded | Safety | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Airline | 95% | 95% | 91% | 91% | 0 | 5% |
| Healthcare Admin | 100% | 100% | 100% | 100% | 0 | 0% |
| Retail | 100% | 99% | 94% | 94% | 0 | 0% |
| Finance | 100% | 100% | 95% | 95% | 0 | 0% |
| Research AI | 82% | 80% | 76% | 74% | 0 | 18% |

Finance is no longer the blocking vertical in B6. Research AI is the blocking
vertical, mainly through truncation and invalid JSON/contract outputs at the
160-token cap.

## Latency And GPU

Overall measured latency:

- mean TTFT: 141.543 ms;
- p50/p95/p99 TTFT: 140.612 / 196.082 / 213.050 ms;
- mean TPOT: 11.489 ms;
- mean ITL p50/p95/p99: 11.245 / 15.442 / 20.095 ms;
- mean E2E latency: 1,741.355 ms;
- p50/p95/p99 E2E latency: 1,397.336 / 5,021.188 / 5,771.729 ms;
- mean throughput: 989.647 tokens/s.

GPU telemetry:

- mean/peak utilization: 81.33% / 100%;
- mean/peak memory used: 6,524.17 / 6,760 MB;
- mean/peak power: 124.93 / 137.66 W;
- mean/peak temperature: 64.80 / 69 C;
- telemetry samples: 527.

## B5 Versus B6

Compared with the B5 full frozen 100 rerun:

- JSON validity decreased from 99.0% to 95.4%;
- contract validity decreased from 99.0% to 94.8%;
- evidence match decreased from 96.0% to 91.2%;
- groundedness decreased from 96.0% to 90.8%;
- truncation increased from 1.0% to 4.6%;
- safety violations stayed at zero.

The larger prompt set preserved aggregate evidence/groundedness above 90%, but
the distribution exposed a Research AI failure mode hidden by the 100-prompt
gate.

## Runtime Projection

B6 wall time was 871.876 seconds for 500 prompts, or 0.573 requests/second.
At the same concurrency-one rate:

- 500 prompts: 0.242 hours;
- 2,500 prompts: 1.211 hours;
- 5,000 prompts: 2.422 hours;
- 10,000 prompts: 4.844 hours.

A matrix with 8, 16, or 32 full 10,000-prompt configurations would project to
38.750, 77.500, or 155.000 RTX 3070 hours. RunPod GPU cost projections remain
`price_missing_and_throughput_multiplier_missing` because the configured
RunPod price file has no reviewed hourly prices or throughput multipliers.

## Decision

Do not run concurrency 2/4, SGLang, mm4, RunPod, 2,000-prompt, or 10,000-prompt
benchmarks from this state. B6 is operationally successful but quality-blocked
for scaling.

Recommended next repair block:

```text
B6R1_RESEARCH_AI_TRUNCATION_AND_CONTRACT_REPAIR
```

Freeze the B6 500-prompt artifacts. Replay only failed, truncated, or invalid
Research AI rows first. Do not change gold data, evaluator semantics, or
promoted retrieval. Compare either a Research-AI-specific output budget
increase or a stricter concise-answer renderer against the same B6 rows, and
accept only if JSON and contract validity reach at least 97%, truncation is no
more than 2%, Research AI evidence and groundedness are at least 85%, and
safety remains zero.
