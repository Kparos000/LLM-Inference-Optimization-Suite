# Controlled Final-Experiment Simulation

## Scope

This block ran the controlled final-experiment simulation safety gate on the
RunPod A100 SXM pod. It did not apply optimizations and did not weaken SLOs,
evaluators, or gold data.

The requested matrix was frozen as:

- Dataset: five verticals, 100 prompts per vertical.
- Self-hosted GPU track: `model3_7b` / `Qwen/Qwen2.5-7B-Instruct` on A100 SXM.
- Self-hosted engines: vLLM and SGLang.
- Self-hosted memory modes: `mm0_no_context`, `mm1_dense_top5`,
  `mm2_hybrid_top5`, `mm3_compressed_hybrid_top5`, `mm4_bounded_agentic`.
- Self-hosted concurrency: 16 and 32.
- API track: `model6_gated` / Llama 3.1 8B API route.
- API memory modes: `mm0_no_context`, `mm1_dense_top5`, `mm2_hybrid_top5`,
  `mm3_compressed_hybrid_top5`, `mm4_bounded_agentic`.
- API concurrency: 4 and 8.

Matrix cardinality:

| Track | Configs | Requests/config | Planned requests |
| --- | ---: | ---: | ---: |
| Self-hosted GPU | 20 | 500 | 10,000 |
| API provider | 10 | 500 | 5,000 |
| Total | 30 | 500 | 15,000 |

## Safety Gate Result

The matrix preflight passed and wrote
`data/generated/phase4/controlled_final_simulation_100_per_vertical_matrix.jsonl`
with 15,000 planned request rows. The full simulation did not run because the
required smoke gates did not pass.

Smoke status:

| Track | Status | Reason |
| --- | --- | --- |
| vLLM `model3_7b` | blocked | No local vLLM server was serving `Qwen/Qwen2.5-7B-Instruct` at `http://localhost:8000/v1`. |
| SGLang `model3_7b` | blocked | `sglang` is not installed/importable, and the runtime registry does not allow SGLang for `model3_7b` on `a100_sxm_80gb`. |
| API `model6_gated` | blocked | `HF_TOKEN` and a provider API key were not present. |
| MM4 | smoke-ready only | The bounded LangGraph mm4 runner is importable, but no full-matrix mm4 request ran because the required track smokes were blocked. |

Requests attempted: 0.

Configs completed: 0.

Configs failed/not run: 30.

## Reports

The safety-gated run wrote the requested report files:

- `results/raw/controlled_final_simulation_results.jsonl`
- `results/raw/controlled_final_simulation_manifest.json`
- `results/raw/controlled_final_simulation_gpu_telemetry.jsonl`
- `results/processed/controlled_final_simulation_eval_report.json`
- `results/processed/controlled_final_simulation_eval_summary.csv`
- `results/processed/controlled_final_simulation_engine_comparison.csv`
- `results/processed/controlled_final_simulation_memory_mode_comparison.csv`
- `results/processed/controlled_final_simulation_concurrency_comparison.csv`
- `results/processed/controlled_final_simulation_api_track_comparison.csv`
- `results/processed/controlled_final_simulation_slo_report.json`
- `results/processed/controlled_final_simulation_slo_summary.csv`
- `results/processed/controlled_final_simulation_cost_report.json`
- `results/processed/controlled_final_simulation_artifact_sync_report.json`

These are generated artifacts and are not committed.

## Findings

vLLM did not run for this block because the required `model3_7b` server was not
available. SGLang did not run and was not silently replaced with vLLM. The API
route did not run and did not report GPU telemetry or GPU hourly cost. MM4 did
not run in the matrix and was not silently replaced by `mm2`.

Because no config completed, engine comparisons, memory-mode comparisons,
concurrency comparisons, API-vs-self-hosted comparisons, and SLO diagnosis are
reported as `NOT_RUN` or `NOT_EVALUATED` rather than inferred from unrelated
data.

## Decision

The final 10,000-prompt experiment is not allowed yet.

Before it can run:

- serve `Qwen/Qwen2.5-7B-Instruct` through vLLM and pass the 10-request smoke;
- install/configure SGLang for A100 SXM, register the compatible route, and pass
  the 10-request smoke;
- provide validated `model6_gated` API credentials and pricing route, then pass
  the 10-request API smoke;
- run the MM4 smoke in the same controlled context if MM4 remains in the matrix.
