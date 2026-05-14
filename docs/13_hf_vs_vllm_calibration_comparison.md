# Hugging Face vs vLLM Calibration Comparison

## Purpose

This document summarizes the first calibration comparison between the local Hugging Face baseline and the RunPod vLLM baseline using curated sample artifacts. The purpose is to preserve an architecture and integration checkpoint before larger controlled benchmark phases.

## Scope

This is a calibration comparison, not a final scaled benchmark.

Hugging Face results were local baseline results. vLLM results were collected on RunPod L40S. Because the hardware differs, this comparison should be interpreted as an architecture/integration baseline, not a controlled hardware-equal benchmark.

The comparison uses small prompt counts and the `Qwen/Qwen2.5-0.5B-Instruct` model. It is useful for validating the benchmark harness, metric capture, artifact promotion path, and early backend behavior, but it should not be treated as a final performance claim.

## Compared Systems

| system | backend | environment | model | optimization label |
| --- | --- | --- | --- | --- |
| Local Hugging Face baseline | Hugging Face Transformers | Local CPU baseline environment | `Qwen/Qwen2.5-0.5B-Instruct` | `hf_baseline` |
| RunPod vLLM baseline | vLLM OpenAI-compatible server | RunPod L40S GPU pod | `Qwen/Qwen2.5-0.5B-Instruct` | `vllm_baseline` |

## Workloads

- `short_chat`
- `code_helpdesk`
- `long_context`
- `shared_prefix`
- `structured_output_smoke`

## Metric Summary

Values are averages from curated sample artifacts in `results/samples/raw`. Latency, TTFT, and TPOT are reported in milliseconds. Throughput is reported in tokens per second.

| workload | HF rows | HF avg latency | HF avg TTFT | HF avg TPOT | HF avg throughput | vLLM rows | vLLM avg latency | vLLM avg TTFT | vLLM avg TPOT | vLLM avg throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `short_chat` | 5 | 8,087.57 | 1,697.43 | 135.96 | 8.41 | 5 | 114.32 | 52.53 | 2.55 | 488.46 |
| `code_helpdesk` | 5 | 12,217.50 | 1,655.61 | 133.69 | 8.35 | 5 | 225.97 | 59.09 | 2.80 | 385.07 |
| `long_context` | 3 | 16,375.99 | 5,995.48 | 129.11 | 10.46 | 3 | 218.23 | 79.88 | 2.45 | 656.55 |
| `shared_prefix` | 5 | 12,854.01 | 2,696.97 | 128.57 | 9.58 | 5 | 210.60 | 54.21 | 2.61 | 484.19 |
| `structured_output_smoke` | 3 | 20,650.04 | 4,274.47 | 172.37 | 7.05 | 3 | 168.09 | 74.48 | 4.40 | 419.94 |

## Key Observations

- vLLM integration succeeded through the OpenAI-compatible runner.
- vLLM baseline showed much lower TPOT than the local HF CPU baseline in these calibration samples.
- vLLM baseline showed high throughput even on small workloads.
- Prompt-level traces revealed that quality and truncation still need evaluation.
- The first prompt in some workloads showed higher TTFT than later prompts, suggesting warmup/cache effects.

## Interpretation

The results show that the benchmark harness can compare Hugging Face and vLLM outputs using the same workload families, CSV metrics, and JSONL prompt traces. The vLLM path also confirms that the OpenAI-compatible runner can capture TTFT, TPOT, throughput, and prompt-level generations from a GPU serving backend.

The large TPOT and throughput differences are directionally consistent with moving from local CPU inference to GPU-backed vLLM serving. However, the systems are not hardware-equivalent, so the numbers should be used as calibration evidence rather than as a controlled backend comparison.

Quality remains part of the benchmark surface. Faster token generation does not by itself establish better benchmark outcomes when responses may be incomplete, truncated, or unsuitable for the prompt.

## Limitations

- Small 0.5B model.
- Small prompt counts.
- Different hardware environments.
- No concurrency yet.
- No 7B model yet.
- No optimization variants yet.
- Quality review is still manual.

## Implications for Next Benchmark Phase

- Add warmup handling before measured prompts.
- Add latency distribution metrics beyond averages.
- Add concurrency and load testing.
- Add truncation detection and quality scoring.
- Compare Hugging Face and vLLM under more controlled hardware assumptions where practical.
- Move to a 7B model after the workflow remains stable on the smaller model.
