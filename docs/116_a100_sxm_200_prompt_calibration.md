# A100 SXM 200-Prompt Calibration

## Scope

The live A100 SXM calibration ran on July 7, 2026 against
`Qwen/Qwen2.5-3B-Instruct` through a local vLLM OpenAI-compatible server.

- Run ID: `a100_sxm_model2_3b_mm2_c1_200`
- GPU: `NVIDIA A100-SXM4-80GB`
- Model alias: `model2_3b`
- Memory mode: `mm2_hybrid_top5`
- Engine: `vllm`
- Concurrency: 1
- Temperature: 0
- Traffic profile: `online_low_latency`
- Prompt count: 200, split 40 each across airline, healthcare admin, retail, finance, and research AI
- Hourly price used for measured-cost estimate: `$1.49/hr`

vLLM was started separately with:

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 8192
```

## Outputs

Generated artifacts are intentionally ignored and must not be committed.

- `data/generated/phase4/a100_sxm_model2_3b_mm2_c1_200_runner_input.jsonl`
- `results/raw/a100_sxm_model2_3b_mm2_c1_200_results.jsonl`
- `results/raw/a100_sxm_model2_3b_mm2_c1_200_manifest.json`
- `results/raw/a100_sxm_model2_3b_mm2_c1_200_gpu_telemetry.jsonl`
- `results/processed/a100_sxm_model2_3b_mm2_c1_200_eval_report.json`
- `results/processed/a100_sxm_model2_3b_mm2_c1_200_eval_summary.csv`
- `results/processed/a100_sxm_model2_3b_mm2_c1_200_artifact_sync_report.json`
- `results/processed/a100_sxm_model2_3b_mm2_c1_200_runtime_projection.json`

## Result

Preflight passed. The runner built a 200-row input with 40 prompts per vertical, verified required evidence in E1-E5, kept canonical IDs hidden from the model, verified checkpoint/resume and manifest controls, and passed the artifact-sync dry run.

The live run completed all 200 prompts with 200 successful requests and 0 failed requests.

Quality:

| Metric | Result |
| --- | ---: |
| JSON valid rate | 99.0% |
| Generation contract valid rate | 98.5% |
| Evidence match rate | 97.5% |
| Grounded rate | 97.0% |
| Safety violations | 0 |
| Truncation rate | 0.5% |

Per-vertical quality:

| Vertical | Evidence match | Grounded | JSON valid | Contract valid |
| --- | ---: | ---: | ---: | ---: |
| airline | 100.0% | 100.0% | 100.0% | 100.0% |
| healthcare_admin | 100.0% | 97.5% | 100.0% | 97.5% |
| retail | 97.5% | 97.5% | 100.0% | 100.0% |
| finance | 97.5% | 97.5% | 97.5% | 97.5% |
| research_ai | 92.5% | 92.5% | 97.5% | 97.5% |

Runtime:

| Metric | Result |
| --- | ---: |
| Wall time | 138.158 s |
| Requests/sec | 1.448 |
| Aggregate tokens/sec | 2,189.29 |
| Aggregate output tokens/sec | 145.59 |
| Mean TTFT | 52.49 ms |
| Mean TPOT | 6.23 ms |
| Mean E2E | 675.71 ms |
| P50 E2E | 584.80 ms |
| P95 E2E | 1,245.48 ms |

GPU telemetry:

| Metric | Result |
| --- | ---: |
| Samples | 128 |
| Mean GPU utilization | 95.80% |
| Max GPU utilization | 100.00% |
| Mean VRAM used | 73,892.92 MB |
| Max VRAM used | 74,247.00 MB |
| Mean power | 266.52 W |
| Max power | 409.48 W |
| Mean temperature | 47.80 C |
| Max temperature | 53.00 C |

Measured cost estimate: `$0.0572` for the 200-prompt run at `$1.49/hr`.

Artifact sync completed successfully with local backup verification passing. The artifact sync report itself was written after final verification and synced in the final `artifact_sync_report_written` event.

## Baseline Decision

The 1,000-prompt A100 baseline is allowed by the calibration rule because the 200-prompt run completed all prompts, passed quality gates, verified artifact sync, and captured GPU telemetry. It remains a separate explicit run decision and must keep concurrency 1 unless a later task authorizes otherwise.
