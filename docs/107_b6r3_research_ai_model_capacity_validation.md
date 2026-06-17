# B6R3 Research AI Model Capacity Validation

Status: measured on June 17, 2026

B6R3 replayed the same frozen 26 Research AI rows from B6/B6R1/B6R2 through
`model6_gated`, resolved as `meta-llama/Llama-3.1-8B-Instruct` through the
existing Hugging Face provider route with Novita pricing.

No gold data, evaluator semantics, promoted retrieval source, B6/B6R1/B6R2
artifacts, self-hosted vLLM/SGLang servers, RunPod inputs, or benchmark scale
changed.

## Purpose

B6R2 showed that vertical-specific Research AI contracts improved output
control but did not make Qwen2.5-1.5B pass the targeted Research AI gate. B6R3
therefore tested whether a stronger model clears the same frozen failed-row
set without changing retrieval, context, or evaluation.

This is a model-capacity validation, not a full benchmark replacement.

## Artifacts

- Loader: `src/inference_bench/research_ai_capacity_validation.py`
- Runner: `scripts/phase4/run_b6r3_model6_research_ai_capacity.py`
- Input: `data/generated/phase4/b6r1_research_ai_failed_replay_input.jsonl`
- Raw results: `results/raw/b6r3_model6_research_ai_capacity_results.jsonl`
- Manifest: `results/raw/b6r3_model6_research_ai_capacity_manifest.json`
- Report: `results/processed/b6r3_model6_research_ai_capacity_report.json`
- Summary: `results/processed/b6r3_model6_research_ai_capacity_summary.csv`
- Comparison: `results/processed/b6r3_model6_vs_b6r2_comparison.json`

The runner supports dry-run, streaming, paid-call gating, bounded retry,
incremental JSONL writes, and resume by completed `prompt_id`.

## Result

Decision:

```text
B6R3_MODEL6_CAPACITY_PASSED
```

Model6 completed the 26-row targeted replay with no request failures:

- JSON validity: 100%
- contract validity: 100%
- evidence match: 96.15%
- groundedness: 96.15%
- safety violations: 0
- truncation: 0%
- total cost: `$0.00077462`
- cost per request: approximately `$0.00002979`

Gate thresholds were JSON at least 97%, contract at least 97%, evidence match
at least 85%, groundedness at least 85%, safety violations equal to zero, and
truncation no more than 2%.

## Runtime

- Provider/backend: Hugging Face provider route, Novita
- Streaming: enabled and required
- Maximum new tokens: 320
- Mean TTFT: 857.338 ms
- Mean TPOT: 7.083 ms
- Mean ITL p50/p95/p99: 6.177 / 17.066 / 27.411 ms
- Mean E2E latency: 1,498.687 ms
- Input/output/total tokens: 32,831 / 2,360 / 35,191

These timings include provider/network behavior and are not hardware-equal with
the self-hosted RTX 3070 vLLM runs.

## Comparison

| Run | Scope | JSON | Contract | Evidence | Grounded | Truncation | Safety |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B6 | Research AI full vertical, 100 rows | 82.00% | 80.00% | 76.00% | 74.00% | 18.00% | 0 |
| B6R1 best | 26-row targeted replay | 92.31% | 84.62% | 73.08% | 65.38% | 7.69% | 0 |
| B6R2 best | 26-row targeted replay | 96.15% | 96.15% | 80.77% | 80.77% | 0.00% | 0 |
| B6R3 model6 | 26-row targeted replay | 100.00% | 100.00% | 96.15% | 96.15% | 0.00% | 0 |

B6R3 is the first targeted Research AI replay to pass the targeted gate.

## Residual Failure

One row still failed evidence match and groundedness:

- `research_ai_scaleup_2000_0099`
- JSON valid: yes
- contract valid: yes
- safety violation: no
- truncation: no
- issue: the model cited abstract evidence plus an unrelated supplied label,
  but omitted the required introduction evidence.

The failure is a citation-selection miss, not a JSON, truncation, retrieval, or
safety failure.

## Interpretation

B6R3 makes Qwen2.5-1.5B model capacity the likely blocker for the Research AI
failed-row subset. The same retrieved evidence and evaluator pass with
model6, while the strongest Qwen2.5-1.5B vertical-contract replay did not.

This does not prove the full 500-row gate would pass with model6, and it does
not authorize a 1,000-prompt terminal run, concurrency sweep, SGLang/mm4
comparison, RunPod run, or 2,000/10,000-prompt benchmark.

## Next Block

Recommended next block:

```text
B6R4_STRONGER_MODEL_PATH_AND_500_GATE_DECISION
```

Decide whether the benchmark should proceed on the API-provider track with
model6, find a stronger feasible self-hosted model path, or explicitly
document Qwen2.5-1.5B as quality-limited for Research AI. After that decision,
rerun the same frozen 500-row gate only with the selected model path and
unchanged evaluator/gold/retrieval controls.
