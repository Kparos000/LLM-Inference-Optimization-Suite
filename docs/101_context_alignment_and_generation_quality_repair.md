# Phase B4 Context Alignment And Generation Quality Repair

Status: `QUALITY_BLOCKED`

Date: June 15, 2026

## Objective

Phase B4 executed the B3 recommended repair block:

```text
B3R1_FROZEN_WORKLOAD_CONTEXT_ALIGNMENT_REPAIR
```

The goal was to isolate B1 quality failures by repairing the frozen
workload-to-rendered-context alignment before any larger benchmark, concurrency
sweep, model change, evaluator change, or promoted retrieval change.

B4 used the same 100 B1 prompt IDs, Qwen2.5-1.5B, vLLM, the remote RTX 3070,
`mm2_hybrid_top5`, concurrency 1, streaming, temperature zero, and a bounded
`max_new_tokens` increase from 128 to 160 because B3 found six truncations at
128 tokens.

## Implementation

New implementation:

- `src/inference_bench/context_alignment_repair.py`;
- `src/inference_bench/generation_prompt_repair.py`;
- `scripts/phase4/repair_b1_context_alignment.py`;
- `scripts/phase4/run_b4_vllm_1_5b_context_aligned_smoke.py`;
- `scripts/phase4/diagnose_b4_quality_and_slos.py`.

Updated implementation:

- `src/inference_bench/generation_contract.py` now supports hiding canonical
  citation aliases from the model, adding Finance metadata lines, and enabling
  a stricter citation checklist for B4.

Tests:

- `tests/test_context_alignment_repair.py`;
- `tests/test_finance_context_rendering.py`;
- `tests/test_generation_prompt_repair.py`.

## Generated Local Artifacts

Context alignment:

- `data/generated/phase4/b4_context_aligned_runner_input.jsonl`;
- `results/processed/b4_context_alignment_report.json`;
- `results/processed/b4_context_alignment_summary.csv`;
- `results/processed/b4_finance_context_alignment_examples.jsonl`.

Live B4 smoke:

- `results/raw/b4_vllm_1_5b_context_aligned_results.jsonl`;
- `results/raw/b4_vllm_1_5b_context_aligned_manifest.json`;
- `results/processed/b4_vllm_1_5b_context_aligned_eval_report.json`;
- `results/processed/b4_vllm_1_5b_context_aligned_eval_summary.csv`;
- `results/processed/b4_vllm_1_5b_context_aligned_latency_summary.csv`;
- `results/processed/b4_vllm_1_5b_context_aligned_gpu_telemetry.csv`;
- `results/processed/b4_vllm_1_5b_context_aligned_gpu_telemetry_summary.json`;
- `results/processed/b4_b1_vs_b4_comparison.json`;
- `results/processed/b4_b1_vs_b4_comparison.csv`.

Post-run diagnosis:

- `results/processed/b4_generation_quality_audit_report.json`;
- `results/processed/b4_generation_quality_audit_summary.csv`;
- `results/processed/b4_finance_failure_examples.jsonl`;
- `results/processed/b4_quality_failure_examples.jsonl`;
- `results/processed/b4_slo_diagnosis_report.json`;
- `results/processed/b4_slo_diagnosis_summary.csv`;
- `results/processed/b4_optimization_recommendation_summary.csv`.

These are generated artifacts and remain ignored unless separately promoted.

## Context Alignment Preflight

B4 did not modify gold data, evaluator semantics, or the promoted retrieval
source of truth. It rebuilt the frozen 100 runner inputs from the promoted
repaired retrieval dataset and selected final E1-E5 context so every expected
gold evidence ID had a private evaluator alias.

Canonical source IDs were not exposed to the model. Finance prompts included
visible business metadata such as ticker, company, form, period, fiscal year,
metric, concept, section, and source type where available.

| Vertical | B1 all gold present | B4 all gold present | Delta |
| --- | ---: | ---: | ---: |
| All | 48/100 | 100/100 | +52 pp |
| Airline | 9/20 | 20/20 | +55 pp |
| Healthcare Admin | 13/20 | 20/20 | +35 pp |
| Retail | 8/20 | 20/20 | +60 pp |
| Finance | 2/20 | 20/20 | +90 pp |
| Research AI | 16/20 | 20/20 | +20 pp |

Preflight status:

```text
PREFLIGHT_PASSED_CONTEXT_ALIGNMENT_IMPROVED
```

## Live B4 Result

B4 completed 100/100 requests.

| Metric | B1 | B4 | Delta |
| --- | ---: | ---: | ---: |
| JSON validity | 93% | 97% | +4 pp |
| Contract validity | 92% | 97% | +5 pp |
| Evidence match | 35% | 76% | +41 pp |
| Groundedness | 35% | 76% | +41 pp |
| Safety violations | 2 | 2 | 0 |
| Truncation rate | 6% | 3% | -3 pp |

B4 passed the JSON, contract, evidence, and groundedness thresholds for its
temporary B4 gate, but failed the safety threshold:

```text
QUALITY_BLOCKED
```

The gate remains blocked because safety violations must be zero. The B4 result
is a real quality improvement, not readiness for scale.

## Per-Vertical Quality

| Vertical | JSON | Contract | Evidence match | Groundedness | Safety | Truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Airline | 100% | 100% | 65% | 65% | 2 | 0 |
| Healthcare Admin | 95% | 95% | 75% | 75% | 0 | 1 |
| Retail | 100% | 100% | 95% | 95% | 0 | 0 |
| Finance | 95% | 95% | 70% | 70% | 0 | 1 |
| Research AI | 95% | 95% | 75% | 75% | 0 | 1 |

Retail now meets the temporary B4 quality thresholds. Airline remains blocked
by safety and citation misses. Finance improved sharply but remains below the
production quality SLO.

## Latency And GPU Impact

| Metric | B1 | B4 | Delta |
| --- | ---: | ---: | ---: |
| Mean TTFT | 185.529 ms | 137.966 ms | -47.563 ms |
| Mean TPOT | 11.341 ms | 11.280 ms | -0.061 ms |
| Mean E2E latency | 1,269.874 ms | 1,701.776 ms | +431.902 ms |
| Mean total tokens/sec | 1,259.139 | 1,006.374 | -252.765 |
| Mean input tokens | 1,482.69 | 1,553.80 | +71.11 |
| Mean output tokens | 96.52 | 138.09 | +41.57 |

B4 improved TTFT and held TPOT roughly flat, but E2E latency increased because
the run used a longer output cap and 31 bounded repair attempts.

GPU telemetry:

- mean GPU utilization: 83.02%;
- peak GPU utilization: 100%;
- peak sampled memory: 6,602 MB of 8,192 MB;
- mean power: 124.86 W;
- peak temperature: 67 C.

## Post-B4 Failure Audit

The B4 offline audit found 25 failed rows.

| Failure class | Failed rows |
| --- | ---: |
| Evidence present but not cited | 24 |
| Model instruction-following failure | 25 |
| Answer semantically underdeveloped | 15 |
| Partial multi-evidence citation | 12 |
| Invalid JSON | 3 |
| Invalid contract | 3 |
| Truncation | 3 |
| Safety violation | 2 |
| Context ordering issue | 1 |
| Required gold absent from E1-E5 | 0 |
| Finance metric/period missing | 0 |
| Incorrect insufficient-evidence use | 0 |

This is the key B4 isolation result: the dominant B1 context-absence problem
has been removed from the frozen 100-row run. The remaining failures are mostly
model citation selection and instruction following over available evidence.

## Finance Diagnosis

B4 Finance moved from 5% evidence match and groundedness in B1 to 70% in B4.

Finance failure audit:

- failed rows: 6 of 20;
- all required gold evidence present in E1-E5 for all 6 failed rows;
- evidence present but ignored: 6;
- wrong evidence cited: 5;
- invalid JSON/contract: 1;
- truncation: 1;
- Finance metric/period missing: 0;
- Finance safety violations: 0;
- investment/advice/projection wording matches: 0.

Finance is no longer a retrieval-context availability problem on the B4 frozen
input. It is now primarily a model citation-selection and instruction-following
problem, with one truncation. It is not a Finance safety problem in B4.

## B2 SLO Diagnosis On B4

The B2 failed-SLO-only diagnosis engine was run on B4 artifacts. It used the
same deterministic SLO profile and catalogs as B2; no LLM was used.

| Vertical | Failed SLOs | Bottlenecks | Primary catalog recommendation |
| --- | ---: | --- | --- |
| Airline | 3 | low evidence match, low groundedness, safety violations | `use_stronger_model` |
| Healthcare Admin | 2 | low evidence match, low groundedness | `use_stronger_model` |
| Retail | 0 | none | none |
| Finance | 2 | low evidence match, low groundedness | `use_stronger_model` |
| Research AI | 2 | low evidence match, low groundedness | `use_stronger_model` |

The B2 recommender is allowed to propose a stronger model now because B4 has
zero remaining gold-absent failed rows. It should not be interpreted as the
only next action. The directly observed B4 failure modes still justify smaller
one-factor repairs before spending a larger benchmark.

## Exact Recommended Repair Block

Next block:

```text
B4R1_SAFETY_AND_CITATION_SELECTION_REPAIR
```

Actions:

1. Keep the same 100 B4 prompt IDs, gold data, evaluator, promoted retrieval
   source, model, engine, hardware, memory mode, concurrency, and temperature.
2. Repair the safety retry prompt so it does not include previous unsafe text
   or source snippets containing prohibited phrases. Supply only safe
   paraphrased task context, allowed E labels, and the violated safety class.
3. Add a pre-generation or repair-time lexical guard that instructs the model
   to paraphrase prohibited terms instead of repeating them. Do not change the
   evaluator or `must_not_include` checks.
4. Improve evidence presentation for multi-evidence tasks by making each E
   block's role shorter and more explicit, without exposing canonical gold IDs.
5. Add an offline citation-selection audit that checks whether each required
   E label is visibly distinguishable before running inference.
6. Rerun only the same 100-prompt B4 matrix. Stop if safety remains nonzero.
7. Only after safety reaches zero and citation/grounding remain above the B4
   gate should a controlled stronger-model comparison or concurrency sweep be
   considered.

Do not run a larger benchmark from B4. The current decision remains:

```text
READY_FOR_SMALL_MODEL_SERVING_EXPERIMENTS
QUALITY_BLOCKED_FOR_SCALE
```
