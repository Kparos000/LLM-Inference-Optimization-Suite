# Phase 1 Plot Interpretation

## Purpose

This document explains how to read the Phase 1 benchmark plots generated from committed curated sample artifacts. The plots are based on the 5,000-prompt Qwen 0.5B vLLM synthetic benchmark and should not be generalized beyond that setup without repeated runs, larger models, and real-world data.

## Source Artifacts

Primary input:

- `results/samples/processed/vllm_qwen0_5b_all_workloads_5000_concurrency_comparison_sample.csv`

Supporting metadata:

- `results/samples/raw/*5000*metadata*.json`

Generated figures:

- `results/samples/figures/phase1/`

## Throughput Plots

Throughput plots show serving capacity. Aggregate requests/sec reports how many requests completed per wall-clock second. Aggregate output tokens/sec reports how many output tokens were produced per wall-clock second.

These plots should be read as system-level capacity views, not single-request quality measures.

## Latency Plots

Latency plots show user-visible delay. Average latency is useful for broad comparison, while p95 and p99 latency show tail behavior under concurrent load.

If aggregate throughput increases while p99 latency also increases, the benchmark is showing a throughput/tail-latency tradeoff.

## TTFT Plots

TTFT plots show first-token behavior. TTFT is influenced by prompt prefill, queueing, warmup effects, scheduling, and concurrency pressure.

High p95 or p99 TTFT means some requests waited noticeably longer before receiving the first generated token.

## TPOT Plots

TPOT plots show decode behavior after generation begins. TPOT is useful for identifying generation-path bottlenecks such as model size, runtime efficiency, memory bandwidth, or kernel performance.

TPOT should be interpreted separately from TTFT because prefill and decode stress different parts of the serving path.

## Workload Comparison Plots

Workload comparison plots show how workload families behave at concurrency 32. These figures help identify whether a specific prompt family is driving latency, TTFT, or throughput differences.

The workload plots are especially useful for comparing short prompts, long-context prompts, code/helpdesk prompts, shared-prefix prompts, and structured-output prompts under the same concurrency level.

## Trade-Off Plots

Trade-off plots show throughput versus latency pressure. They are designed to answer whether extra throughput is coming with higher average latency, p99 latency, or p99 TTFT.

These plots are useful for selecting a concurrency target because production serving usually needs both capacity and latency objectives.

## Failure And Success Plots

Failure and success plots confirm reliability. A high-throughput run is not useful if it produces failures. In the committed 5,000-prompt sample, success/failure plots should be read alongside latency and throughput plots to confirm that throughput did not come from dropped or failed requests.

## Interpretation Boundaries

These plots summarize a synthetic benchmark for one model size, one serving backend, and one GPU environment. They support Phase 1 engineering analysis, but they do not yet prove real-world data performance, answer correctness, larger-model behavior, or optimization wins from prefix caching, quantization, or speculative decoding.
