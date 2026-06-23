# B7 Controlled 1,000-Prompt Baseline

Status: measured on June 21, 2026

B7 ran the first post-B6R6 scale baseline for `model2_3b` /
`Qwen/Qwen2.5-3B-Instruct` through vLLM on the remote RTX 3070. The run used
`mm2_hybrid_top5`, streaming, temperature zero, concurrency one, the B6R5
Finance evidence preplan, the B6R6 Research AI answer skeleton, artifact sync,
checkpoint/resume, manifest recording, and nvidia-smi GPU telemetry.

## Preflight

- Runner input rows: 1,000
- Per vertical: 200
- Required gold evidence present in E1-E5: 1,000/1,000
- Canonical IDs exposed to the model: 0
- Model alias: `model2_3b`
- Model ID: `Qwen/Qwen2.5-3B-Instruct`
- Runtime: vLLM on `remote_rtx3070`
- Artifact sync dry run: passed
- Checkpoint/resume: enabled
- Manifest: enabled

Preflight status:

```text
PREFLIGHT_PASSED_B7_CONTROLLED_1000_BASELINE
```

## Execution

The raw file contains 1,000 unique prompt rows. The run resumed from partial
raw output after an interrupted checkpoint write and skipped the 953 prompt IDs
already present in `results/raw/b7_model2_3b_1000_results.jsonl`.

Request-level outcome:

- successful requests: 663
- failed requests: 337
- request success rate: 66.3%
- duplicate prompt IDs: 0

The first serving failure occurred at Finance prompt 17. The vLLM container log
shows an EngineCore fatal error ending in:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR
```

After that fatal engine failure, the remaining Finance and then Research AI
requests in the first pass recorded connection failures. This makes B7 an
operational serving-stability failure, not a clean model-quality scale result.
The failed rows are retained in the raw artifact and were not rewritten into
successful rows.

## Quality

Overall:

- JSON validity: 64.8%
- contract validity: 64.8%
- evidence match: 64.3%
- groundedness: 64.3%
- safety violations: 0
- truncation: 1.2%

Per-vertical evidence match / groundedness:

- Airline: 93.0% / 93.0%
- Healthcare Admin: 100.0% / 100.0%
- Retail: 99.5% / 99.5%
- Finance: 7.5% / 7.5%
- Research AI: 21.5% / 21.5%

Finance and Research AI metrics are dominated by request failures after the
engine died. They must not be read as clean semantic quality estimates for
Qwen2.5-3B.

Decision:

```text
B7_CONTROLLED_1000_BASELINE_BLOCKED
```

Failed gate metrics:

- JSON validity;
- contract validity;
- evidence match;
- groundedness;
- minimum vertical evidence match;
- minimum vertical groundedness.

## Runtime

- Accumulated request time: 1,542.353 seconds
- Mean TTFT: 429.823 ms
- p50/p95/p99 TTFT: 421.228 / 513.607 / 571.570 ms
- Mean TPOT: 17.352 ms
- p50/p95/p99 TPOT: 17.302 / 18.144 / 18.584 ms
- Mean E2E latency: 2,291.242 ms
- p50/p95/p99 E2E latency: 1,924.165 / 4,233.441 / 9,062.385 ms
- Requests/sec: 0.648
- Aggregate tokens/sec: 607.781
- Aggregate output tokens/sec: 45.635

The runtime projection is a linear concurrency-one estimate from the measured
artifact. It is not a guarantee and is not a RunPod cost claim:

- 2,000 prompts: 0.857 RTX 3070 hours
- 10,000 prompts: 4.284 RTX 3070 hours
- selected 40,000-prompt matrix: 17.137 RTX 3070 hours

RunPod cost projection remains blocked because reviewed hourly prices and
throughput multipliers are not registered.

## Workload Shape

Input sequence length distribution:

- 0-512: 337
- 512-1,024: 182
- 1,024-2,048: 411
- 2,048-4,096: 68
- 4,096-8,192: 2
- 8,192+: 0

Output sequence length distribution:

- 0-64: 399
- 64-128: 493
- 128-256: 88
- 256-512: 20
- 512-1,024: 0
- 1,024+: 0

Cache-readiness metrics:

- repeated prefix tokens: 19,980
- shared context percentage: 0.0%
- prefix reuse potential: 0.023067
- KV-cache pressure estimate: 0.228860
- cacheability score: 0.089801
- estimated prefix-cache benefit: 0.079061

## GPU Telemetry

- Samples: 529
- Mean/peak GPU utilization: 84.45% / 100%
- Mean/peak GPU memory: 7,381.60 / 7,770 MB
- Mean/peak power: 169.09 / 201.29 W
- Mean/peak temperature: 70.81 / 75 C

Telemetry captured the vLLM EngineCore process while the server was alive.

## Artifact Sync

Artifact sync passed at run end:

- backup root: `backups/`
- backup completeness score: 1.0
- raw JSONL hash verified: yes
- manifest hash verified: yes
- telemetry hash verified: yes
- processed reports hash verified: yes

Primary reports:

- `data/generated/phase4/b7_model2_3b_1000_runner_input.jsonl`
- `results/raw/b7_model2_3b_1000_results.jsonl`
- `results/raw/b7_model2_3b_1000_manifest.json`
- `results/raw/b7_model2_3b_1000_gpu_telemetry.jsonl`
- `results/processed/b7_model2_3b_1000_eval_report.json`
- `results/processed/b7_model2_3b_1000_eval_summary.csv`
- `results/processed/b7_model2_3b_1000_runtime_projection.json`
- `results/processed/b7_model2_3b_1000_artifact_sync_report.json`
- `results/processed/b7_model2_3b_1000_readiness_report.json`

## Next Step

B7R1 has now superseded this operational blocker. It audited the vLLM
EngineCore failure, found that `gpu_memory_utilization=0.78` could not
initialize KV-cache blocks, loaded a safe profile at
`gpu_memory_utilization=0.82` with `max_model_len=3584`, and reran the same
frozen 1,000-row input successfully.

See `docs/113_b7r1_vllm_cuda_stability_repair.md` for the repaired result.
