# Coreopt Prefix Layout Static V1

Status: completed as CPU-only static analysis on August 2, 2026.

This is the first core inference optimization experiment after the capability
and observability audit. It does not run inference, start vLLM or SGLang, use
API models, mutate `Main_Inference_V1`, or create `Optimized_Inference_V1`.

Artifact root:

```text
experiments/optimizations/coreopt_prefix_layout_static_v1/
```

## Purpose

`coreopt_prefix_layout_static_v1` tests whether the authoritative runner prompt
layout can be rearranged to improve exact leading-prefix reuse potential.

The optimization ID is:

```text
prompt_prefix_layout_optimization
```

The hypothesis is that stable, repeated instructions should appear before
request-specific context and user questions. That gives future prefix-cache
engine validation a longer exact token-identical prefix to reuse.

## What Changed

Baseline layout:

```text
SYSTEM
MEMORY MODE
RETRIEVED EVIDENCE
USER QUESTION
OUTPUT CONTRACT
```

Candidate layout:

```text
SYSTEM
MEMORY MODE
OUTPUT CONTRACT
RETRIEVED EVIDENCE
USER QUESTION
```

The candidate preserves section bytes and changes only section order. Evidence
content, evidence order, aliases, memory-mode labels, generation contract text,
safety instructions, model metadata, workload rows, prompt IDs, gold contracts,
and evaluator semantics are held constant.

## Authoritative Rendering Path

The experiment uses the same runner-facing rendering path as the full
experiment:

- `src/inference_bench/workload_adapter.py`
- `src/inference_bench/generation_contract.py`

The raw workload `messages` field is not treated as the final prompt. The
runner converts workload rows through `workload_record_to_runner_item`, which
calls the generation-contract renderer. `baseline_prompt_layout_v1` is defined
as that exact authoritative output.

## Workload Scope

The static analysis scanned 40,000 workload rows:

- `data/workloads/final_10000/prompt_plus_metadata/mm0_no_context.jsonl`
- `data/workloads/final_10000/prompt_plus_metadata/mm1_dense_top5.jsonl`
- `data/workloads/final_10000/prompt_plus_metadata/mm2_hybrid_top5.jsonl`
- `data/workloads/final_10000/prompt_plus_metadata/mm3_compressed_hybrid_top5.jsonl`

Each row was rendered twice:

- once as `baseline_prompt_layout_v1`;
- once as `prefix_optimized_prompt_layout_v1`.

No raw prompt text is stored in the committed analysis artifacts.

## Tokenizer

The analysis attempted to use the local cached tokenizer for:

```text
Qwen/Qwen2.5-7B-Instruct
```

The tokenizer was not available locally, so the artifact records a deterministic
regex fallback tokenizer. This is acceptable for static planning and section
comparison, but it is not sufficient for latency, cache-hit, or cost claims.

Tokenizer metadata is stored at:

```text
experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_tokenizer_report.json
```

## Measured Static Result

Source:

```text
experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_prefix_summary.json
```

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Rendered prompts | 40,000 | 40,000 |
| Total input tokens | 32,893,355 | 32,893,355 |
| Mean input tokens | 822.333875 | 822.333875 |
| Median input tokens | 904 | 904 |
| p95 input tokens | 1,208 | 1,208 |
| p99 input tokens | 1,519 | 1,519 |
| Prefix families | 4 | 4 |
| Mean exact common prefix tokens | 29 | 358 |
| Mean reusable-token ratio | 0.041903 | 0.517288 |

Static delta:

```text
+329 mean exact common prefix tokens
+0.475385 mean reusable-token ratio
0 total input-token change
```

This means the candidate creates a much larger exact leading prefix opportunity
without increasing total prompt size. It does not prove lower TTFT.

## Equivalence

Source:

```text
experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_equivalence_report.json
```

Equivalence status:

```text
PASS
```

Rows checked:

```text
40000
```

The report records:

- all sections present;
- section content byte-equivalent;
- evidence content byte-equivalent;
- evidence order fixed;
- citation aliases preserved;
- schema instruction unchanged;
- safety instruction unchanged;
- memory instruction unchanged;
- max output settings unchanged;
- no forbidden gold leakage detected.

The report also flags `instruction_priority_risk: true`, because moving
sections can change how the model follows instructions. That risk requires a
real inference validation run before accepting the candidate.

## Decision

Source:

```text
experiments/optimizations/coreopt_prefix_layout_static_v1/coreopt_prefix_layout_static_v1_decision.json
```

Decision:

```text
MISSING_CONFIGURATION
```

Reason:

The static analysis completed and found a positive derived prefix-reuse delta,
but the repository has no configured minimum threshold for accepting a static
prefix-layout change into engine validation.

Missing configuration:

```text
minimum_reusable_token_ratio_delta_for_engine_validation
```

The next required experiment is:

```text
coreopt_prefix_layout_engine_validation_v1
```

## Claims Not Allowed

This experiment must not be described as proving:

- TTFT improvement;
- latency improvement;
- cache-hit improvement;
- cost improvement;
- deployability improvement.

Those claims require a one-factor engine validation run with cache metrics,
latency metrics, protected quality gates, and no regression in safety or
contract behavior.

## UI Contract

The product platform exposes the static experiment through read-only endpoints:

- `/api/optimizations/coreopt-prefix-layout-static-v1`
- `/api/optimizations/coreopt-prefix-layout-static-v1/summary`
- `/api/optimizations/coreopt-prefix-layout-static-v1/layouts`
- `/api/optimizations/coreopt-prefix-layout-static-v1/prefix-metrics`
- `/api/optimizations/coreopt-prefix-layout-static-v1/equivalence`
- `/api/optimizations/coreopt-prefix-layout-static-v1/decision`
- `/api/optimizations/coreopt-prefix-layout-static-v1/story`

The Optimization Lab should present this as:

1. problem: stable contract instructions were late in the prompt;
2. mechanism: move stable content before dynamic context/question sections;
3. instrumentation: tokenize, hash, assign prefix families, compute static
   reusable-prefix metrics;
4. result: longer derived exact common prefix;
5. decision: missing threshold and engine validation required.

## Files

Key artifacts:

- `coreopt_prefix_layout_static_v1_manifest.json`
- `layouts/baseline_prompt_layout_v1.json`
- `layouts/prefix_optimized_prompt_layout_v1.json`
- `coreopt_prefix_layout_static_v1_held_constants.json`
- `coreopt_prefix_layout_static_v1_tokenizer_report.json`
- `coreopt_prefix_layout_static_v1_prefix_families.csv`
- `coreopt_prefix_layout_static_v1_prefix_summary.json`
- `coreopt_prefix_layout_static_v1_prefix_summary.csv`
- `coreopt_prefix_layout_static_v1_per_vertical_memory.csv`
- `coreopt_prefix_layout_static_v1_prompt_section_analysis.csv`
- `coreopt_prefix_layout_static_v1_equivalence_report.json`
- `coreopt_prefix_layout_static_v1_decision.json`
- `coreopt_prefix_layout_static_v1_plotting_dataset.json`
- `coreopt_prefix_layout_static_v1_ui_story.json`
- `checksums/SHA256SUMS.txt`

Code:

- `src/inference_bench/coreopt_prefix_layout_static.py`
- `scripts/phase4/build_coreopt_prefix_layout_static_v1.py`
- `tests/test_coreopt_prefix_layout_static_v1.py`

## Current State

`coreopt_prefix_layout_static_v1` is now a measured static-analysis scenario,
not a deployed optimization and not a champion. It is ready for review and for
an engineer-approved engine validation plan.
