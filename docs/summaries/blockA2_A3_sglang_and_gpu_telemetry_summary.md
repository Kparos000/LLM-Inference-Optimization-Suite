# Block A2/A3 SGLang And GPU Telemetry Summary

## Decision

```text
SGLANG_LIVE_SMOKE_PASS
KEEP_AS_SECONDARY_ENGINE
VLLM_REMAINS_DEFAULT_RTX3070_BACKEND
NOT_READY_FOR_QUALITY_SCALE
```

SGLang 0.5.13 started on `zeever-gpu`, loaded
`Qwen/Qwen2.5-0.5B-Instruct`, exposed the OpenAI-compatible API on port 30000,
and completed all 50 frozen prompts. The run used the same prompt IDs, model,
memory mode, generation settings, hardware, evaluator, and concurrency as A1.

## SGLang Metrics

| Metric | Result |
| --- | ---: |
| Requests completed | 50/50 |
| JSON validity | 100% |
| Contract validity | 58% |
| Evidence-ID presence | 90% |
| Evidence match | 36% |
| Deterministic groundedness | 24% |
| Safety violations | 2/50 |
| Mean TTFT | 135.971 ms |
| p95 TTFT | 112.068 ms |
| Mean TPOT | 24.202 ms |
| p95 TPOT | 32.831 ms |
| Mean E2E | 1,066.357 ms |
| p95 E2E | 1,378.222 ms |
| Mean throughput | 673.905 tokens/s |

## GPU Telemetry

| Metric | SGLang | vLLM |
| --- | ---: | ---: |
| Samples | 34 | 27 |
| Mean utilization | 33.38% | 37.15% |
| Peak utilization | 64% | 74% |
| Mean memory used | 6,548.65 MB | 6,303.33 MB |
| Peak memory used | 6,551 MB | 6,372 MB |
| Mean power | 68.35 W | 68.31 W |
| Peak temperature | 47 C | 51 C |

The enhanced sampler records interval, requested duration, start/end
timestamps, process names, peak memory, mean/peak utilization, mean power, and
peak temperature. The A2 manifest links the telemetry CSV and summary JSON.

## Comparison

SGLang improved mean TTFT by 11.887 ms and evidence match by six percentage
points. It regressed mean TPOT by 2.200 ms/token, mean E2E by 185.861 ms,
throughput by 206.670 tokens/s, contract validity by 14 percentage points,
groundedness by four points, and peak memory by 179 MB.

HF and API rows are included only as contextual five-prompt baselines. They are
not hardware- or workload-equal to the 50-prompt GPU pair.

## Remaining Blockers

- The 0.5B model remains below generation quality SLOs on both GPU engines.
- A controlled vLLM concurrency 2/4 study is still required before scaling
  request count.
- Stronger-model feasibility is constrained by 8 GB VRAM.
- No hourly infrastructure price is registered.
- Queue, batch, prefix/radix-cache, and KV-cache time series remain absent.
- Semantic claim-level groundedness is not implemented.

## Artifacts

- `configs/experiments/a2_remote_rtx3070_sglang_smoke.yaml`
- `results/raw/a2_remote_rtx3070_sglang_smoke_results.jsonl`
- `results/raw/a2_remote_rtx3070_sglang_smoke_manifest.json`
- `results/processed/a2_remote_rtx3070_sglang_eval_report.json`
- `results/processed/a2_remote_rtx3070_sglang_eval_summary.csv`
- `results/processed/a2_remote_rtx3070_sglang_latency_summary.csv`
- `results/processed/a2_remote_rtx3070_sglang_gpu_telemetry.csv`
- `results/processed/a2_remote_rtx3070_sglang_gpu_telemetry_summary.json`
- `results/processed/a2_vllm_vs_sglang_comparison_report.json`
- `results/processed/a2_vllm_vs_sglang_comparison_summary.csv`
- `docs/96_remote_rtx3070_sglang_smoke.md`

No `PROJECT_STATE` file exists in the repository, so no such file was updated.
