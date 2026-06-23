# B7R1 vLLM CUDA Stability Repair

Status: measured on June 22, 2026 local time (June 23, 2026 UTC)

B7R1 reran the exact frozen B7 1,000-row input for `model2_3b` /
`Qwen/Qwen2.5-3B-Instruct` through vLLM on the remote RTX 3070. It did not
change retrieval, gold data, evaluator semantics, prompt IDs, memory mode, or
the B6R6 Finance/Research repairs.

## Failure Audit

The B7 audit found an operational serving collapse:

- first failed row: `finance_scaleup_2000_0017`;
- first failure type: vLLM EngineCore HTTP 500;
- fatal engine errors: 1;
- failed rows: 337;
- affected failed verticals: Finance 184, Research AI 153;
- peak sampled VRAM: 7,770 MB of 8,192 MB;
- primary diagnosis: `serving_stability_failure`.

This is not a retrieval, gold, evaluator, or semantic-quality failure.

## Safe Profile

The first intended profile, `gpu_memory_utilization=0.78` with
`max_model_len=4096`, could not initialize vLLM 0.23.0 because no KV-cache
blocks were available. Lowering `max_model_len` to 3,584 at 0.78 still failed
to allocate KV-cache blocks.

The loadable safe profile is:

- profile: `remote_rtx3070_qwen3b_safe_v1`;
- engine: vLLM;
- model alias: `model2_3b`;
- hardware: `remote_rtx3070`;
- `gpu_memory_utilization`: 0.82;
- `max_model_len`: 3,584;
- `max_num_seqs`: 1;
- `max_num_batched_tokens`: 3,584;
- `enforce_eager`: true;
- `disable_custom_all_reduce`: true.

It remains materially below the unstable B7 baseline allocation and loaded
successfully before the rerun.

## Preflight

- Runner input rows: 1,000
- Per vertical: 200
- Required gold evidence present in E1-E5: 1,000/1,000
- Canonical IDs exposed to the model: 0
- Runtime registry allowed vLLM on `remote_rtx3070`
- Artifact sync dry run: passed
- Checkpoint/resume: enabled
- Manifest: enabled
- Safe serving profile: passed

Preflight status:

```text
PREFLIGHT_PASSED_B7R1_VLLM_STABILITY_REPAIR
```

## Execution Result

- Completed prompts: 1,000/1,000
- Successful requests: 1,000
- Failed requests: 0
- Fatal engine errors: 0
- Serving restarts: 0
- Resumed from checkpoint: false
- Artifact sync backup completeness: 1.0

Stability decision:

```text
B7R1_STABILITY_READY
```

## Quality

Overall:

- JSON validity: 98.5%
- contract validity: 98.3%
- evidence match: 96.1%
- groundedness: 95.9%
- safety violations: 0
- truncation: 1.2%

Per vertical:

| Vertical | JSON | Contract | Evidence | Grounded | Safety | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Airline | 95.0% | 95.0% | 94.0% | 94.0% | 0 | 5.0% |
| Healthcare Admin | 100.0% | 100.0% | 100.0% | 100.0% | 0 | 0.0% |
| Retail | 100.0% | 100.0% | 99.0% | 99.0% | 0 | 0.0% |
| Finance | 99.0% | 99.0% | 94.0% | 94.0% | 0 | 1.0% |
| Research AI | 98.5% | 97.5% | 93.5% | 92.5% | 0 | 0.0% |

The quality gate passed.

## Runtime

- Wall time: 2,308.381 seconds
- Requests/sec: 0.433
- Aggregate tokens/sec: 647.267
- Input tokens: 1,394,010
- Output tokens: 100,128
- Total tokens: 1,494,138
- Mean TTFT: 475.941 ms
- p95 TTFT: 593.148 ms
- Mean TPOT: 17.608 ms
- Mean E2E latency: 2,254.364 ms
- p50 E2E latency: 1,955.434 ms
- p95 E2E latency: 4,131.577 ms

The stable profile traded some request throughput for reliability. B7R1 ran
longer than B7 because B7's measured wall time included many zero-token failed
requests after the engine collapse.

## Workload Shape

Input sequence length distribution:

- 0-512: 0
- 512-1,024: 257
- 1,024-2,048: 661
- 2,048-4,096: 69
- 4,096-8,192: 13
- 8,192+: 0

Output sequence length distribution:

- 0-64: 151
- 64-128: 713
- 128-256: 113
- 256-512: 23
- 512-1,024: 0
- 1,024+: 0

Cache-readiness:

- repeated prefix tokens: 19,980
- shared context percentage: 0.0%
- prefix reuse potential: 0.014347
- KV-cache pressure estimate: 0.364780
- cacheability score: 0.071413
- estimated prefix-cache benefit: 0.060630

## GPU Telemetry

- Samples: 727
- Mean/peak GPU utilization: 84.53% / 100%
- Peak sampled VRAM: 7,404 MB
- Mean sampled VRAM: 7,045 MB
- Mean power: 168.46 W
- Peak temperature: 74 C

Peak VRAM stayed below the 7,600 MB safe-profile threshold.

## Runtime Projection

Concurrency-one linear projection from the measured run:

- 2,000 prompts: 1.282 RTX 3070 hours
- 10,000 prompts: 6.412 RTX 3070 hours
- selected 40,000-prompt matrix: 25.649 RTX 3070 hours

This is not a RunPod cost claim. RunPod price and throughput multiplier inputs
remain unset.

## Artifacts

- `results/processed/b7_vllm_cuda_failure_audit_report.json`
- `results/processed/b7_vllm_cuda_failure_audit_summary.csv`
- `results/processed/b7r1_preflight_report.json`
- `results/raw/b7r1_model2_3b_1000_results.jsonl`
- `results/raw/b7r1_model2_3b_1000_manifest.json`
- `results/raw/b7r1_model2_3b_1000_gpu_telemetry.jsonl`
- `results/processed/b7r1_model2_3b_1000_eval_report.json`
- `results/processed/b7r1_model2_3b_1000_eval_summary.csv`
- `results/processed/b7_vs_b7r1_comparison.json`
- `results/processed/b7r1_artifact_sync_report.json`
- `results/processed/b7r1_runtime_projection.json`
- `results/processed/b7r1_readiness_report.json`

## Readiness

B7R1 clears the B7 serving-stability blocker for the RTX 3070 Qwen3B vLLM
track at concurrency one. The next step can be an API load probe, provided it
is treated as the API-provider track and not a RunPod/GPU readiness claim.

Do not run a concurrency sweep, SGLang comparison, mm4 comparison, RunPod run,
2,000-prompt run, or 10,000-prompt run until the B7R1 result is reviewed and
the next experiment is explicitly selected. RunPod remains blocked by missing
hourly price and throughput calibration inputs.
