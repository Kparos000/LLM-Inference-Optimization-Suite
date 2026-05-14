# Experiment Log

## Purpose

This log preserves curated experiment observations that are useful for engineering review, future comparisons, and public-facing writeups. It records benchmark context, measured metrics, output-quality observations, limitations, and next actions without committing raw generated benchmark outputs.

## Experiment Log Format

Each experiment entry should include:

- Experiment name
- Environment
- Model and backend
- Workloads executed
- Metric summary
- Structured-output result, when applicable
- Qualitative output observations
- Key engineering lessons
- Limitations
- Next actions

## Experiment: vLLM Baseline Smoke and Expanded Workload Run

Experiment name: `vLLM RunPod L40S baseline calibration`

Optimization label: `vllm_baseline`

This run established the first GPU-backed vLLM baseline for the benchmark harness. The goal was to confirm that the OpenAI-compatible runner could capture latency, throughput, CSV metrics, and JSONL prompt traces against a live vLLM server before expanding into larger models, concurrency testing, or optimization variants.

## Environment

- RunPod Linux GPU pod
- NVIDIA L40S
- vLLM OpenAI-compatible server
- Benchmark client: `inference-bench openai-compatible-run`

## Model and Backend

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Backend: `vLLM`
- Optimization label: `vllm_baseline`

## Workloads Executed

- `short_chat`
- `code_helpdesk`
- `long_context`
- `shared_prefix`
- `structured_output_smoke`

## Metric Summary

| workload | rows | success_count | avg_end_to_end_latency_ms | avg_ttft_ms | avg_tpot_ms | avg_throughput_tokens_per_second |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_chat` | 5 | 5 | 114.32 | 52.53 | 2.55 | 488.46 |
| `code_helpdesk` | 5 | 5 | 225.97 | 59.09 | 2.80 | 385.07 |
| `long_context` | 3 | 3 | 218.23 | 79.88 | 2.45 | 656.55 |
| `shared_prefix` | 5 | 5 | 210.60 | 54.21 | 2.61 | 484.19 |
| `structured_output_smoke` | 3 | 3 | 168.09 | 74.48 | 4.40 | 419.94 |

## Structured-Output Result

| metric | value |
| --- | ---: |
| total_records | 3 |
| valid_json_count | 3 |
| required_fields_count | 3 |
| invalid_json_count | 0 |
| required_fields_rate | 1.000 |

## Qualitative Output Observations

- Several short-chat and summarization outputs were acceptable.
- Some code/helpdesk outputs were incomplete or truncated due to limited max-new-tokens.
- Some shared-prefix IT-support outputs were weak or policy-risky, showing that speed metrics alone are insufficient.
- Structured-output format checks passed, but stricter raw-JSON enforcement may be useful later because one response included markdown code fencing.
- First prompt TTFT was noticeably higher than later prompts in several workloads, suggesting warmup/cache effects should be handled explicitly in future benchmarks.

## Key Engineering Lessons

- vLLM integration with the benchmark harness is functional.
- TTFT, TPOT, throughput, CSV metrics, and JSONL traces are being captured successfully.
- GPU serving dramatically changes performance characteristics compared with local CPU calibration.
- Prompt-level traces are necessary because aggregate latency metrics do not capture answer quality.
- Future benchmark phases should include warmup handling, latency distributions, truncation detection, and quality scoring.

## Limitations

- Small 0.5B model only.
- Small prompt counts.
- Single concurrency only.
- No vLLM optimization variants yet.
- No 7B model yet.
- Quality review is manual at this stage.
- Results are hardware- and environment-specific.

## Next Actions

- Use the Linux `.sh` workflow scripts for future RunPod executions instead of manually pasted benchmark client commands.
- Compare HF baseline vs vLLM baseline.
- Add concurrency/load testing.
- Add latency distribution metrics.
- Add quality scoring/truncation detection.
- Move to 7B once the workflow is stable.
