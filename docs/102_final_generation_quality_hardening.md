# Final Generation Quality Hardening

Status: `QUALITY_READY_FOR_FROZEN_100`

Phase B5 implemented the safety and citation-selection repair block after B4.
It kept the B4 matrix frozen:

- Qwen2.5-1.5B-Instruct;
- vLLM on the remote RTX 3070;
- `mm2_hybrid_top5`;
- prompt-plus-metadata ablation;
- concurrency one;
- temperature zero through the existing runner path;
- 160 maximum new tokens;
- unchanged gold data, evaluator semantics, promoted retrieval source, model,
  engine, memory mode, and hardware.

## What Changed

Implemented modules:

- `src/inference_bench/safety_generation_repair.py`
- `src/inference_bench/multi_evidence_selector.py`
- `src/inference_bench/answer_planning.py`

Implemented CLI:

- `scripts/phase4/run_b5_targeted_generation_repair.py`

The repair does three things:

- uses stable safety rule IDs instead of repeating prohibited wording in repair
  prompts;
- applies a lexical guard to valid JSON answer and citation-note fields while
  preserving evidence labels, confidence, and insufficiency state;
- injects an internal E-label answer plan so the model sees which supplied
  `E1`-style labels must support the final JSON.

The internal plan exposes only short labels such as `E1` and `E3`. It does not
expose canonical gold IDs to the model or alter evaluator inputs.

## Targeted Replay

B5 first replayed only the 25 failed B4 prompt IDs.

Targeted result:

| Metric | B4 failed subset | B5 targeted |
| --- | ---: | ---: |
| JSON validity | 88% | 100% |
| Contract validity | 88% | 100% |
| Evidence match | 4% | 92% |
| Groundedness | 4% | 92% |
| Safety violations | 2 | 0 |
| Truncation | 12% | 0% |

Targeted per-vertical evidence match and groundedness:

- Airline: 87.5%, with zero safety violations.
- Healthcare Admin: 100%.
- Retail: 100%.
- Finance: 83.33%.
- Research AI: 100%.

The targeted replay used two lexical-guard safety repairs and four missing-label
retries. It passed the B5 target gate:

- JSON at least 97%;
- contract at least 97%;
- evidence match at least 85%;
- groundedness at least 85%;
- safety violations equal zero.

## Full Frozen 100 Rerun

Because the targeted replay passed, B5 reran the full frozen 100-prompt matrix.

Full result:

| Metric | B4 full 100 | B5 full 100 |
| --- | ---: | ---: |
| JSON validity | 97% | 99% |
| Contract validity | 97% | 99% |
| Evidence match | 76% | 96% |
| Groundedness | 76% | 96% |
| Safety violations | 2 | 0 |
| Truncation | 3% | 1% |

Full per-vertical evidence match and groundedness:

- Airline: 95%.
- Healthcare Admin: 100%.
- Retail: 100%.
- Finance: 90%.
- Research AI: 95%.

The full frozen 100 run completed 100 of 100 requests. Mean latency was:

- TTFT: 142.102 ms;
- TPOT: 10.718 ms;
- E2E: 1,473.156 ms.

## Residual Failures

B5 is not a perfect run. The full frozen 100 still has four residual failures:

- one Airline citation miss;
- two Finance citation misses;
- one Research AI truncated JSON output.

The configured B5 quality gate still passed because aggregate JSON, contract,
evidence, groundedness, and safety targets were met. These residual failures
must remain visible before scaling.

## Artifacts

Generated local artifacts:

- `results/processed/b5_targeted_repair_report.json`
- `results/processed/b5_targeted_repair_summary.csv`
- `results/processed/b5_b4_vs_b5_comparison.json`
- `results/processed/b5_failed_prompt_replay.jsonl`
- `results/processed/b5_targeted_repair_latency_summary.csv`
- `results/processed/b5_targeted_repair_gpu_telemetry.csv`
- `results/processed/b5_targeted_repair_gpu_telemetry_summary.json`
- `results/processed/b5_full_frozen_100_report.json`
- `results/processed/b5_full_frozen_100_summary.csv`
- `results/processed/b5_full_frozen_100_latency_summary.csv`
- `results/raw/b5_full_frozen_100_replay.jsonl`

The generated raw and processed run artifacts are local benchmark outputs and
are not committed.

## SLO Diagnosis

The deterministic SLO diagnosis was run over the B5 targeted replay. Because
the diagnosis scope is the 25-row B4 failure subset, its recommendations are
not a substitute for a balanced full-workload SLO report. It reported five
failed selected SLOs across targeted vertical slices and primary catalog
recommendations of `use_stronger_model` for two slices. The observed B5 repair
result shows that prompt-side evidence planning and safety lexical repair fixed
the immediate frozen-matrix blocker without changing models.

## Decision

```text
QUALITY_READY_FOR_FROZEN_100
NOT_A_FINAL_SCALE_BENCHMARK
```

B5 clears the frozen 100-prompt generation-quality gate. It does not make a
cost claim, does not prove concurrency behavior, and does not replace the need
for backend-native queue, batch, prefix-cache, and KV-cache telemetry.

Recommended next block:

```text
B6_CONTROLLED_SCALE_AND_CONCURRENCY_GATE
```

Run a controlled 500-prompt scale gate at concurrency one first, then a bounded
concurrency 2/4 sweep only if quality remains above gate and safety remains
zero.
