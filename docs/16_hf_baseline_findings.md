# Hugging Face Baseline Findings

## Purpose

This document summarizes the initial local Hugging Face baseline results across expanded workloads. It is intended to preserve early findings before adding a vLLM comparison path.

## Experimental Context

The baseline used the local Hugging Face runner against checked-in workload files. The run captured workload-level CSV metrics and prompt-level traces under `results/`, with generated artifacts ignored by default unless deliberately promoted as reviewed samples.

## Model And Backend

- Backend: Hugging Face local runner
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Execution context: local CPU-oriented environment
- Runtime mode: baseline local generation

## Workloads Compared

- `short_chat`
- `code_helpdesk`
- `long_context`
- `shared_prefix`

## Summary Table

| workload | rows | success_count | avg_ttft_ms | avg_tpot_ms | avg_end_to_end_latency_ms | avg_throughput_tokens_per_second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short_chat | 5 | 5 | 1697.43 | 135.96 | 8087.57 | 8.41 |
| code_helpdesk | 5 | 5 | 1655.61 | 133.69 | 12217.50 | 8.35 |
| long_context | 3 | 3 | 5995.48 | 129.11 | 16375.99 | 10.46 |
| shared_prefix | 5 | 5 | 2696.97 | 128.57 | 12854.01 | 9.58 |

## Key Findings

- All workloads completed successfully.
- `long_context` had the highest TTFT.
- TPOT stayed relatively close across workloads.
- This suggests the workload difference is mostly driven by prefill/input-context cost rather than decode speed.
- `shared_prefix` is important for later prefix caching experiments.
- These are local CPU-oriented results and should not be generalized to GPU serving.

## Interpretation

The largest latency difference appears in TTFT rather than TPOT. That pattern is consistent with longer or more instruction-heavy inputs increasing prefill work before decoding begins. The relatively close TPOT values suggest decode speed was more stable across these small workloads than initial context processing.

The `shared_prefix` workload creates a controlled path for evaluating prefix caching later because each prompt repeats the same instruction prefix while varying the user request.

## Limitations

- Small model only.
- Small workload sizes.
- Local CPU environment.
- No vLLM comparison yet.
- No memory measurement yet beyond metadata.
- Structured output quality still needs more systematic scoring.

## Implications For vLLM Phase

The Hugging Face baseline establishes a reference point before introducing vLLM. The vLLM phase should preserve the same workload set and compare TTFT, TPOT, end-to-end latency, throughput, memory, and structured-output validity under a serving-oriented runtime.

The `long_context` result makes prefill behavior a priority for vLLM comparison. The `shared_prefix` workload should be retained for later prefix caching evaluation once the baseline serving path is stable.

## Notes For Future Paper/Report

- Use these values as local baseline context, not as general hardware-independent performance claims.
- Highlight that workload shape affects TTFT materially even before serving optimizations are introduced.
- Preserve comparison CSVs and system metadata alongside any promoted figures or sample traces.
- Separate smoke-test findings from larger-scale benchmark claims.
