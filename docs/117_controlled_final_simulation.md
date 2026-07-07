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
| Wall-clock runtime | 1,834.33 seconds |
| A100 GPU cost estimate | `$0.759211` |
| API token cost estimate | `$0.034657` |
| Total measured cost estimate | `$0.793868` |

Quality summary:

| Metric | Value |
| --- | ---: |
| Joined rate | 100.00% |
| JSON-valid rate | 99.92% |
| Generation-contract-valid rate | 81.41% |
| Format-valid rate | 81.41% |
| Evidence-match rate | 61.92% |
| Grounded rate | 60.26% |
| Safety violations | 97 |
| Truncations | 0 |

The repaired baseline completed operationally and the per-config SLO comparison
reported zero failed SLOs across 25 configs. The aggregate raw-output evaluation
still shows residual contract, safety, evidence, and groundedness findings, so
the next phase should optimize against those measured bottlenecks before making
deployability claims.

## Runtime Results

Overall runtime metrics:

| Metric | Value |
| --- | ---: |
| Mean E2E latency | 1,911.71 ms |
| P50 E2E latency | 1,748.03 ms |
| P95 E2E latency | 3,620.76 ms |
| P99 E2E latency | 4,441.53 ms |
| Mean TTFT | 462.48 ms |
| P95 TTFT | 1,155.00 ms |
| P99 TTFT | 1,754.86 ms |
| Mean TPOT | 46.89 ms |
| Mean tokens/sec | 487.61 |

Engine comparison on the self-hosted track:

| Engine | Mean latency | Mean tokens/sec | Result |
| --- | ---: | ---: | --- |
| vLLM | 2,017.25 ms | 452.97 | Latency and throughput winner |
| SGLang | 2,143.01 ms | 429.09 | Slower in this baseline |

Concurrency comparison:

| Track | Concurrency | Mean latency | Mean tokens/sec |
| --- | ---: | ---: | ---: |
| Self-hosted | 16 | 1,758.40 ms | 492.60 |
| Self-hosted | 32 | 2,401.87 ms | 389.45 |
| API provider | 4 | 1,238.02 ms | 673.94 |

Memory-mode latency ranking:

| Rank | Memory mode | Mean latency | Mean tokens/sec |
| ---: | --- | ---: | ---: |
| 1 | `mm0_no_context` | 903.64 ms | 481.15 |
| 2 | `mm2_hybrid_top5` | 2,116.51 ms | 488.62 |
| 3 | `mm3_compressed_hybrid_top5` | 2,145.13 ms | 485.81 |
| 4 | `mm1_dense_top5` | 2,190.33 ms | 481.47 |
| 5 | `mm4_bounded_agentic` | 2,202.94 ms | 501.00 |

Per-config SLO rows did not meaningfully separate memory modes on quality; each
memory-mode aggregate reported 95.25% evidence match, 94.25% groundedness, and
99.0% generation-contract validity. The aggregate raw-output evaluator remains
the stricter bottleneck signal for the next optimization phase.

## GPU Telemetry

GPU telemetry was captured for self-hosted configurations only. API-provider
rows did not report GPU telemetry.

| Metric | Value |
| --- | ---: |
| Telemetry samples | 1,534 |
| Mean GPU utilization | 50.92% |
| Max GPU utilization | 100.00% |
| Mean VRAM used | 73,482.70 MiB |
| Max VRAM used | 74,419 MiB |
| Mean power draw | 225.21 W |
| Mean temperature | 45.19 C |
| Max temperature | 63 C |

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

The repaired controlled final 10,000-request baseline is complete. Artifact sync
and backup verification passed with a 1.0 completeness score, 18 synced
artifacts, and 19/19 backup verification checks passing.

The optimization phase can begin from this baseline. Recommended candidates are
contract normalization at the final-answer boundary, safety-wording cleanup for
final answers, groundedness/evidence selection, and a concurrency-32 self-hosted
latency/throughput pass.
