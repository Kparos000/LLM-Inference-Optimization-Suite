# Controlled Final-Experiment Simulation

## Scope

This block ran the controlled final 10,000-request baseline simulation on the
RunPod A100 SXM pod. It did not change the benchmark matrix, optimize prompts,
weaken evaluators, or modify gold data.

The executed matrix was:

- Dataset: five verticals, 80 prompts per vertical per configuration.
- Self-hosted GPU track: `model3_7b` / `Qwen/Qwen2.5-7B-Instruct` on A100 SXM.
- Self-hosted engines: vLLM and SGLang.
- Self-hosted memory modes: `mm0_no_context`, `mm1_dense_top5`,
  `mm2_hybrid_top5`, `mm3_compressed_hybrid_top5`, `mm4_bounded_agentic`.
- Self-hosted concurrency: 16 and 32.
- API track: `model6_gated` / Llama 3.1 8B API route.
- API memory modes: `mm0_no_context`, `mm1_dense_top5`,
  `mm2_hybrid_top5`, `mm3_compressed_hybrid_top5`, `mm4_bounded_agentic`.
- API concurrency: 4.

Matrix cardinality:

| Track | Configs | Requests/config | Planned requests |
| --- | ---: | ---: | ---: |
| Self-hosted GPU | 20 | 400 | 8,000 |
| API provider | 5 | 400 | 2,000 |
| Total | 25 | 400 | 10,000 |

## Safety Gate Result

All full-run gates passed before execution:

| Track | Status | Reason |
| --- | --- | --- |
| vLLM `model3_7b` | smoke-ready | `/v1/models` at `http://localhost:8000/v1` listed `Qwen/Qwen2.5-7B-Instruct`. |
| SGLang `model3_7b` | smoke-ready | Runtime registry allows SGLang for `model3_7b` on `a100_sxm_80gb`, and `/v1/models` at `http://localhost:30000/v1` listed `Qwen/Qwen2.5-7B-Instruct`. |
| API `model6_gated` | smoke-ready | `.env` credentials and provider route were visible to the runner. |
| MM4 | smoke-ready | The bounded LangGraph MM4 runner was importable and included in the full matrix. |

The exact A100 SGLang startup command for this run is:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90 --context-length 4096 --max-running-requests 32 --chunked-prefill-size 8192
```

The SGLang health check is `GET http://localhost:30000/v1/models`.

## Baseline Result

The run completed all planned requests:

| Metric | Value |
| --- | ---: |
| Requests attempted | 10,000 |
| Requests completed | 10,000 |
| Requests failed | 0 |
| Configs completed | 25 |
| Configs failed | 0 |
| Wall-clock runtime | 2,076.93 seconds |
| A100 GPU cost estimate | `$0.859619` |
| API token cost estimate | `$0.014224` |
| Total measured cost estimate | `$0.873843` |

Quality summary:

| Metric | Value |
| --- | ---: |
| Joined rate | 100.00% |
| Format-valid rate | 100.00% |
| JSON-valid rate | 0.00% |
| Generation-contract-valid rate | 0.00% |
| Evidence-match rate | 3.97% |
| Grounded rate | 3.97% |
| Safety violations | 12 |
| Truncations | 0 |

The benchmark execution completed, but the baseline is not deployable because
generation-contract JSON validity, evidence matching, and groundedness failed
SLOs across every configuration.

## Runtime Results

Overall runtime metrics:

| Metric | Value |
| --- | ---: |
| Mean E2E latency | 2,359.66 ms |
| P50 E2E latency | 2,497.38 ms |
| P95 E2E latency | 2,859.67 ms |
| P99 E2E latency | 3,149.24 ms |
| Mean TTFT | 205.04 ms |
| P95 TTFT | 863.86 ms |
| P99 TTFT | 1,004.70 ms |
| Mean TPOT | 14.01 ms |
| Mean tokens/sec | 72.99 |

Engine comparison on the self-hosted track:

| Engine | Mean latency | Mean tokens/sec | Result |
| --- | ---: | ---: | --- |
| vLLM | 2,427.63 ms | 73.51 | Latency and throughput winner |
| SGLang | 2,535.34 ms | 69.79 | Slower in this baseline |

Concurrency comparison:

| Track | Concurrency | Mean latency | Mean tokens/sec |
| --- | ---: | ---: | ---: |
| Self-hosted | 16 | 2,372.65 ms | 74.78 |
| Self-hosted | 32 | 2,590.32 ms | 68.53 |
| API provider | 4 | 1,872.36 ms | 78.32 |

Memory-mode latency ranking:

| Rank | Memory mode | Mean latency | Mean tokens/sec |
| ---: | --- | ---: | ---: |
| 1 | `mm2_hybrid_top5` | 2,351.21 ms | 73.17 |
| 2 | `mm1_dense_top5` | 2,352.28 ms | 73.20 |
| 3 | `mm4_bounded_agentic` | 2,360.68 ms | 73.02 |
| 4 | `mm3_compressed_hybrid_top5` | 2,362.79 ms | 72.80 |
| 5 | `mm0_no_context` | 2,371.33 ms | 72.74 |

Quality did not meaningfully separate memory modes in this baseline; each
memory-mode aggregate had 3.75% evidence match and groundedness.

## GPU Telemetry

GPU telemetry was captured for self-hosted configurations only. API-provider
rows did not report GPU telemetry.

| Metric | Value |
| --- | ---: |
| Telemetry samples | 1,871 |
| Mean GPU utilization | 48.60% |
| Max GPU utilization | 100.00% |
| Mean VRAM used | 69,569.63 MiB |
| Max VRAM used | 70,065 MiB |
| Mean power draw | 194.78 W |
| Mean temperature | 41.67 C |
| Max temperature | 60 C |

## Reports

The full run wrote the requested generated artifacts:

- `results/raw/controlled_final_simulation_results.jsonl`
- `results/raw/controlled_final_simulation_manifest.json`
- `results/raw/controlled_final_simulation_gpu_telemetry.jsonl`
- `results/processed/controlled_final_simulation_eval_report.json`
- `results/processed/controlled_final_simulation_eval_summary.csv`
- `results/processed/controlled_final_simulation_engine_comparison.csv`
- `results/processed/controlled_final_simulation_memory_mode_comparison.csv`
- `results/processed/controlled_final_simulation_concurrency_comparison.csv`
- `results/processed/controlled_final_simulation_api_track_comparison.csv`
- `results/processed/controlled_final_simulation_api_vs_self_hosted_comparison.csv`
- `results/processed/controlled_final_simulation_model_comparison.csv`
- `results/processed/controlled_final_simulation_slo_report.json`
- `results/processed/controlled_final_simulation_slo_summary.csv`
- `results/processed/controlled_final_simulation_cost_report.json`
- `results/processed/controlled_final_simulation_artifact_sync_report.json`
- `results/processed/controlled_final_simulation_plotting_dataset.csv`
- `results/processed/controlled_final_simulation_plotting_dataset.json`

These are generated artifacts and are not committed.

## Decision

The controlled final 10,000-request baseline is complete.

The final larger/deployability experiment is not allowed yet. The next phase
should optimize only after reviewing the failed SLOs. Recommended candidates
are generation-contract JSON repair, prompt/output parsing repair, evidence
selection alignment, context formatting for groundedness, and then a targeted
MM4 bounded-agentic quality repair.
