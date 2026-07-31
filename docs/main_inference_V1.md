# Main Inference V1 Reference

Status: official full pre-optimization inference baseline, completed July 9,
2026.

Artifact root:

```text
experiments/main/main_inference_v1/
```

This document is a repo-grounded handoff for the completed
`Main_Inference_V1` run. It explains what was run, what artifacts are present
locally, what the measured results mean, and which files to use for analysis,
plotting, SLO diagnosis, optimization planning, and portfolio reporting.

## 1. Executive Summary

`Main_Inference_V1` is the official full 250,000-request pre-optimization
inference baseline. It completed end to end with zero request failures:

- 25 configs.
- 10,000 prompts per config.
- 250,000 planned requests.
- 250,000 attempted requests.
- 250,000 completed requests.
- 0 failed requests.

The run validates the benchmark system operationally at full-matrix scale on a
rented A100 environment. It also establishes the measured before-optimization
baseline for `Optimized_Inference_V1`.

The result is intentionally not marked deployable. Runtime and cost SLOs
passed, but quality and safety SLOs failed. The deployability verdict is:

```text
NOT_DEPLOYABLE_SLO_FAILURES
```

## 2. Historical Naming Correction

An earlier artifact was mistakenly named `main_inference_v1` even though it ran
only 10,000 total requests across 25 configs, with 400 requests per config.
That artifact has been reclassified as `Baseline_Inference_V1` and archived
separately under:

```text
experiments/baseline/baseline_inference_v1/
```

The official `Main_Inference_V1` is the later full 250,000-request run archived
under:

```text
experiments/main/main_inference_v1/
```

## 3. Experiment Matrix

The official run executed the corrected full matrix:

| Dimension | Value |
| --- | --- |
| Run ID | `main_inference_v1` |
| Config count | 25 |
| Prompts per config | 10,000 |
| Total requests | 250,000 |
| Verticals | `airline`, `healthcare_admin`, `retail`, `finance`, `research_ai` |
| Prompts per vertical per config | 2,000 |
| Requests per vertical across run | 50,000 |
| Self-hosted model | `model3_7b` / `Qwen/Qwen2.5-7B-Instruct` |
| API model | `model6_gated` / `meta-llama/Llama-3.1-8B-Instruct` |
| Engines | vLLM, SGLang, API provider route |
| Memory modes | `mm0_no_context`, `mm1_dense_top5`, `mm2_hybrid_top5`, `mm3_compressed_hybrid_top5`, `mm4_bounded_agentic` |
| Self-hosted concurrency | 16 and 32 |
| API concurrency | 4 |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU hourly price | `$1.49` |
| Traffic profile | `online_low_latency` |
| Runtime type | mixed self-hosted GPU and API provider route |

The self-hosted portion ran 200,000 requests. The API-provider portion ran
50,000 requests.

## 4. Run Timeline And Completion

The manifest is saved at:

```text
experiments/main/main_inference_v1/raw/main_inference_v1_manifest.json
```

Measured run timeline:

| Field | Value |
| --- | ---: |
| Started at UTC | `2026-07-08T19:14:27.382555+00:00` |
| Completed at UTC | `2026-07-09T06:34:02.325911+00:00` |
| Wall seconds | 42,538.856294736965 |
| Approximate wall time | 11.82 hours |
| Planned requests | 250,000 |
| Attempted requests | 250,000 |
| Completed requests | 250,000 |
| Failed requests | 0 |
| Configs completed | 25 |
| Configs failed | 0 |

The eval report records the vertical distribution:

| Vertical | Requests |
| --- | ---: |
| Airline | 50,000 |
| Finance | 50,000 |
| Healthcare Admin | 50,000 |
| Research AI | 50,000 |
| Retail | 50,000 |

## 5. Artifact Map

The local repo contains the processed and audit artifacts needed for summary
analysis. The full raw 250k response file is not present under the local
experiment folder.

| Category | Repo-relative path | Purpose | Local analysis use | Notes |
| --- | --- | --- | --- | --- |
| Archive readme | `experiments/main/main_inference_v1/main-inference-readme.md` | Short archive overview | Human handoff | Present locally. |
| Checksums | `experiments/main/main_inference_v1/checksums/SHA256SUMS.txt` | Hashes for saved artifacts | Integrity checks | Present locally. |
| Console log | `experiments/main/main_inference_v1/logs/main_inference_v1_console.log` | Run console output | Debug and audit trail | Present locally. |
| Progress log | `experiments/main/main_inference_v1/logs/main_inference_v1_progress.jsonl` | Incremental run progress | Completion timeline | Present locally. |
| Manifest | `experiments/main/main_inference_v1/raw/main_inference_v1_manifest.json` | Run identity, matrix, command, paths | Primary provenance | Present locally. |
| GPU telemetry | `experiments/main/main_inference_v1/raw/main_inference_v1_gpu_telemetry.jsonl` | A100 utilization, memory, power, temperature | GPU reporting | Present locally. |
| Checkpoint | `experiments/main/main_inference_v1/raw/main_inference_v1_checkpoint.json` | Completed request tracking | Resume/completion audit | Present locally. |
| Full raw responses | `experiments/main/main_inference_v1/raw/main_inference_v1_results.jsonl.gz` | Full response-level raw evidence | Row-level raw output analysis | Missing locally. Portable archive contains full raw evidence. |
| Full raw responses, uncompressed | `experiments/main/main_inference_v1/raw/main_inference_v1_results.jsonl` | Full response-level raw evidence | Row-level raw output analysis | Missing locally. Artifact sync recorded this file on the run host. |
| Eval report | `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json` | Aggregate result, latency, telemetry, SLO verdicts | Main analysis source | Present locally. |
| Eval summary | `experiments/main/main_inference_v1/processed/main_inference_v1_eval_summary.csv` | Flat aggregate quality row | Quick quality table | Present locally. |
| SLO report | `experiments/main/main_inference_v1/processed/main_inference_v1_slo_report.json` | SLO comparison and deployability verdict | SLO diagnosis input | Present locally. |
| SLO scorecard | `experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv` | Target, observed, difference, status | Portfolio scorecard | Present locally. |
| SLO summary | `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv` | Per-config SLO summary | Failure-family diagnosis | Present locally. |
| Cost report | `experiments/main/main_inference_v1/processed/main_inference_v1_cost_report.json` | GPU/API/total cost | Cost reporting | Present locally. |
| Post-run automation | `experiments/main/main_inference_v1/processed/main_inference_v1_post_run_automation_report.json` | Generated reports and SLO rows | Report inventory | Present locally. |
| Plotting dataset | `experiments/main/main_inference_v1/processed/main_inference_v1_plotting_dataset.jsonl` | Compact plotting data | Portfolio plots | Present locally. It is CSV-formatted despite the `.jsonl` extension. |
| Engine comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_engine_comparison.csv` | vLLM/SGLang/API rows | Engine comparison | Present locally. |
| Memory comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_memory_comparison.csv` | mm0-mm4 rows | Context/memory analysis | Present locally. |
| Concurrency comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_concurrency_comparison.csv` | concurrency 16/32/API 4 rows | Concurrency analysis | Present locally. |
| API comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_api_comparison.csv` | API provider route rows | API-only analysis | Present locally. |
| API vs self-hosted | `experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv` | API and self-hosted rows | Routing comparison | Present locally. |
| Model comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_model_comparison.csv` | model3 vs model6 rows | Model comparison | Present locally. |
| Context preflight | `experiments/main/main_inference_v1/processed/main_inference_v1_context_preflight_report.json` | Context alignment diagnostics | Retrieval/context audit | Present locally. |
| Context preflight summary | `experiments/main/main_inference_v1/processed/main_inference_v1_context_preflight_summary.csv` | Compact context alignment table | Context summary | Present locally. |
| Contract preflight | `experiments/main/main_inference_v1/processed/main_inference_v1_contract_preflight_report.json` | Contract checks before full run | Prompt/contract audit | Present locally. |
| mm4 safety audit | `experiments/main/main_inference_v1/processed/main_inference_v1_mm4_safety_audit.json` | mm4 safety findings | Safety diagnosis | Present locally. |
| mm4 targeted replay | `experiments/main/main_inference_v1/processed/main_inference_v1_mm4_safety_targeted_report.json` | Targeted mm4 repair replay | Repair evidence | Present locally. |
| mm4 targeted summary | `experiments/main/main_inference_v1/processed/main_inference_v1_mm4_safety_targeted_summary.csv` | Targeted replay summary | Repair summary | Present locally. |
| Repair readiness | `experiments/main/main_inference_v1/processed/main_inference_v1_repair_ready_report.json` | Full-run readiness after repair gates | Run authorization audit | Present locally. |
| Repair comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_repair_vs_broken_comparison_report.json` | Broken vs repaired evidence | Repair narrative | Present locally. |
| 25-row failure audit CSV | `experiments/main/main_inference_v1/processed/main_inference_v1_repaired_25_failure_audit.csv` | Failure classes | Targeted audit | Present locally. |
| 25-row failure audit JSON | `experiments/main/main_inference_v1/processed/main_inference_v1_repaired_25_failure_audit.json` | Failure classes | Targeted audit | Present locally. |
| 25-row replay | `experiments/main/main_inference_v1/processed/main_inference_v1_repaired_25_replay_report.json` | Small repair replay | Repair evidence | Present locally. |
| 500-row validation | `experiments/main/main_inference_v1/processed/main_inference_v1_repaired_500_validation_report.json` | Repair validation | Repair evidence | Present locally. |
| 500-row validation summary | `experiments/main/main_inference_v1/processed/main_inference_v1_repaired_500_validation_summary.csv` | Flat repair validation row | Repair summary | Present locally. |

## 6. Top-Line Result

The run completed operationally:

- Benchmark execution verdict: `COMPLETED`.
- Runtime SLO verdict: `PASS`.
- Cost SLO verdict: `PASS`.
- Quality SLO verdict: `FAIL`.
- Safety SLO verdict: `FAIL`.
- Overall deployability verdict: `NOT_DEPLOYABLE_SLO_FAILURES`.

The measured total cost was `$18.461296726432796`, split into
`$17.606359966432798` GPU cost and `$0.854936759999998` API cost.

The result is a valid before-optimization baseline. It is not a deployment
approval.

## 7. SLO Scorecard

Source:

```text
experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv
```

| Metric | Target | Observed | Difference | Status |
| --- | --- | ---: | ---: | --- |
| JSON validity | >= 0.95 | 0.99822 | 0.04822000000000004 | PASS |
| Contract validity | >= 0.95 | 0.805388 | -0.14461199999999996 | FAIL |
| Format validity | >= 0.95 | 0.805388 | -0.14461199999999996 | FAIL |
| Evidence match | >= 0.95 | 0.589724 | -0.36027599999999993 | FAIL |
| Groundedness | >= 0.98 | 0.567204 | -0.41279599999999994 | FAIL |
| Safety findings | = 0 | 2757 | 2757 | FAIL |
| Runtime | repo configured runtime SLO | 42538.856294736965 | n/a | PASS |
| Cost | repo configured cost SLO | 18.461296726432796 | n/a | PASS |

## 8. Latency / Throughput

Source:

```text
experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json
```

Aggregate latency and throughput:

| Metric | Value |
| --- | ---: |
| Mean E2E latency ms | 1555.083189807053 |
| p50 E2E latency ms | 1357.596322428435 |
| p95 E2E latency ms | 2997.51268675318 |
| p99 E2E latency ms | 3990.831322551244 |
| Mean TTFT ms | 316.182404887296 |
| p50 TTFT ms | 177.4042589822784 |
| p95 TTFT ms | 881.2036044429988 |
| p99 TTFT ms | 1206.8634418747388 |
| Mean TPOT ms | 40.81780315907682 |
| p50 TPOT ms | 37.353379669200095 |
| p95 TPOT ms | 83.32632407512622 |
| p99 TPOT ms | 116.49905651574956 |
| Mean total tokens/sec | 574.1231592602473 |

Runtime passed the configured SLO scorecard. The blocking failures are quality
and safety.

## 9. Cost

Source:

```text
experiments/main/main_inference_v1/processed/main_inference_v1_cost_report.json
```

| Metric | Value |
| --- | ---: |
| Total cost USD | 18.461296726432796 |
| GPU cost USD | 17.606359966432798 |
| API cost USD | 0.854936759999998 |
| Self-hosted request count | 200,000 |
| API request count | 50,000 |
| Self-hosted GPU hourly price USD | 1.49 |
| Wall seconds | 42,538.856294736965 |

Cost passed the configured SLO scorecard.

## 10. GPU Telemetry

Source:

```text
experiments/main/main_inference_v1/raw/main_inference_v1_gpu_telemetry.jsonl
```

Telemetry summary from the eval report:

| Metric | Value |
| --- | ---: |
| GPU | NVIDIA A100-SXM4-80GB |
| Telemetry samples | 33,564 |
| Max GPU utilization percent | 100.0 |
| Mean GPU utilization percent | 45.415445119771185 |
| Total VRAM MB | 81,920 |
| Mean VRAM used MB | 74,000.83610415921 |
| Max VRAM used MB | 74,421 |
| Max temperature C | 68 |
| Mean temperature C | 44.34465498748659 |
| Max power draw W | 476.52 |
| Mean power draw W | 201.71575169824814 |
| Observed process names | `VLLM::EngineCore`, `sglang::scheduler` |

Mean GPU utilization is lower than peak because the full run mixes self-hosted
GPU configs and API-provider configs. VRAM was heavily allocated during
self-hosted serving.

## 11. Quality And Safety Findings

Source:

```text
experiments/main/main_inference_v1/processed/main_inference_v1_eval_summary.csv
```

Aggregate quality and safety:

| Metric | Value |
| --- | ---: |
| JSON valid rate | 0.99822 |
| Generation contract valid rate | 0.805388 |
| Format valid rate | 0.805388 |
| Evidence ID presence rate | 0.702976 |
| Evidence match rate | 0.589724 |
| Grounded rate | 0.567204 |
| Safety violation count | 2,757 |
| Safety violation rate | 0.011028 |
| Joined count | 250,000 |
| Joined rate | 1.0 |
| Retry row count | 0 |
| Retry rate | 0.0 |
| Truncation count | 0 |
| Truncation rate | 0.0 |

JSON validity passed comfortably. The main failures are stricter contract
validity, evidence match, groundedness, and safety. This means the system can
complete the workload and return parseable JSON, but the answer contract and
grounded evidence behavior are not yet reliable enough for deployment.

## 12. Comparison Signals

Comparison files:

- `experiments/main/main_inference_v1/processed/main_inference_v1_engine_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_memory_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_concurrency_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_api_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_model_comparison.csv`

Observed comparison signals from those CSVs:

- API `model6_gated` rows had stronger average evidence and groundedness than
  self-hosted `model3_7b` rows in this run.
- API-provider rows averaged 0.773740 evidence match and 0.710560
  groundedness across five configs.
- Self-hosted GPU rows averaged 0.543720 evidence match and 0.531365
  groundedness across twenty configs.
- vLLM averaged lower E2E latency than SGLang across the self-hosted rows in
  this run.
- Self-hosted concurrency 32 increased latency and reduced average total
  tokens/sec compared with concurrency 16 across the self-hosted rows.
- `mm0_no_context` behaved as a no-context ablation: it was comparatively fast
  and had zero safety findings, but evidence match and groundedness collapsed.
- `mm4_bounded_agentic` had the strongest memory-mode average groundedness and
  contract validity, but still had safety findings.

These are comparison signals for the measured matrix, not final optimization
claims. They identify where failed-SLO diagnosis should focus next.

## 13. Best Files For Local Analysis

| Task | File |
| --- | --- |
| SLO table | `experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv` |
| SLO diagnosis input | `experiments/main/main_inference_v1/processed/main_inference_v1_slo_report.json` |
| Per-config SLO rows | `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv` |
| Eval summary | `experiments/main/main_inference_v1/processed/main_inference_v1_eval_summary.csv` |
| Full eval aggregate | `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json` |
| Cost | `experiments/main/main_inference_v1/processed/main_inference_v1_cost_report.json` |
| Plotting | `experiments/main/main_inference_v1/processed/main_inference_v1_plotting_dataset.jsonl` |
| GPU telemetry | `experiments/main/main_inference_v1/raw/main_inference_v1_gpu_telemetry.jsonl` |
| Engine comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_engine_comparison.csv` |
| Memory comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_memory_comparison.csv` |
| Concurrency comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_concurrency_comparison.csv` |
| API-only comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_api_comparison.csv` |
| API vs self-hosted comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv` |
| Model comparison | `experiments/main/main_inference_v1/processed/main_inference_v1_model_comparison.csv` |
| Optimization report inventory | `experiments/main/main_inference_v1/processed/main_inference_v1_post_run_automation_report.json` |
| Portfolio/demo reporting | `experiments/main/main_inference_v1/main-inference-readme.md` and this document |

## 14. Local Analysis Commands

PowerShell-friendly checks:

```powershell
Test-Path experiments/main/main_inference_v1
Get-ChildItem experiments/main/main_inference_v1 -Recurse -File |
  Select-Object FullName, Length
```

Open the artifact folder:

```powershell
Invoke-Item experiments/main/main_inference_v1
```

Read the SLO scorecard:

```powershell
python -c "import pandas as pd; print(pd.read_csv('experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv').to_string(index=False))"
```

Read comparison CSVs:

```powershell
python -c "import pandas as pd; p='experiments/main/main_inference_v1/processed/main_inference_v1_engine_comparison.csv'; print(pd.read_csv(p).head().to_string(index=False))"
python -c "import pandas as pd; p='experiments/main/main_inference_v1/processed/main_inference_v1_memory_comparison.csv'; print(pd.read_csv(p).head().to_string(index=False))"
python -c "import pandas as pd; p='experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv'; print(pd.read_csv(p).head().to_string(index=False))"
```

Read the plotting dataset. Despite the `.jsonl` extension, this file is
CSV-formatted, so use `pd.read_csv`:

```powershell
python -c "import pandas as pd; p='experiments/main/main_inference_v1/processed/main_inference_v1_plotting_dataset.jsonl'; print(pd.read_csv(p).head().to_string(index=False))"
```

Read the eval report JSON:

```powershell
python -c "import json; p='experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json'; d=json.load(open(p)); print(d['status']); print(d['slo_verdicts'])"
```

Verify checksums for files listed in the checksum manifest. This assumes the
paths in `SHA256SUMS.txt` exist in the current checkout:

```powershell
@'
from hashlib import sha256
from pathlib import Path

manifest = Path('experiments/main/main_inference_v1/checksums/SHA256SUMS.txt')
for line in manifest.read_text().splitlines():
    expected, rel = line.split('  ', 1)
    path = Path(rel)
    if not path.exists():
        print(f'MISSING {rel}')
        continue
    observed = sha256(path.read_bytes()).hexdigest()
    print(('OK' if observed == expected else 'MISMATCH'), rel)
'@ | python -
```

## 15. Optimization Readiness

The run now has a UI-facing failed-SLO diagnosis and optimization option
export. Relevant configuration and code:

- `configs/bottleneck_catalog.yaml`
- `configs/optimization_catalog.yaml`
- `configs/optimization_negative_rules.yaml`
- `configs/slo_targets.yaml`
- `configs/slo_profiles.yaml`
- `src/inference_bench/slo_diagnosis.py`
- `src/inference_bench/optimization_recommender.py`
- `src/inference_bench/optimization_catalog.py`
- `src/inference_bench/main_inference_optimization_ui.py`
- `scripts/phase4/build_main_inference_optimization_ui.py`
- `scripts/phase4/diagnose_phase2_optimization.py`
- `scripts/phase4/generate_b2_slo_diagnosis_reports.py`

The UI-ready artifacts are:

- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_diagnosis.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_optimization_options.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_apply_plan.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_story.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_deployability_repairs.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_repair_gate.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_core_optimization_catalog.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_core_optimization_applicability.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_experiment_stage.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_ui_optimization_story.json`

They are generated with:

```powershell
python scripts/phase4/build_main_inference_optimization_ui.py
```

The UI layer uses the measured failed SLOs from
`main_inference_v1_slo_report.json`, `main_inference_v1_slo_scorecard.csv`,
and `main_inference_v1_slo_summary.csv`. It applies the bottleneck catalog,
optimization catalog, and negative-rule filtering so the product dropdowns
show only compatible options for the selected failed SLO.

Optimization should target quality and safety first while preserving the
runtime and cost SLO passes. The product-facing layer now separates this into
two tracks: mandatory deployability repairs first, then core inference
optimization after measured repair validation. The current repair gate is
`NOT_MEASURED`, so core optimizations are visible for education but locked for
the champion optimized run. See
`docs/128_inference_optimization_two_track_architecture.md`.

## 16. GPU Shutdown / CPU-Only Analysis

Local analysis, plotting, SLO comparison, report writing, and optimization
strategy planning are CPU-only tasks. The A100 is not needed for:

- reading the saved reports;
- building plots from comparison CSVs;
- reviewing SLO failures;
- writing documentation;
- planning optimization experiments;
- preparing portfolio/demo narratives.

The A100 is needed again only for live inference reruns such as
`Optimized_Inference_V1` or targeted GPU validation runs.

## 16A. Interactive Platform Replay

Platform Foundation V1 now exposes this run through a read-only product demo
under:

```text
platform/frontend/
platform/backend/
```

The platform replays saved Main_Inference_V1 progress, telemetry, SLO, cost,
diagnosis, and optimization-planning artifacts without running inference. The
Main run is labeled `measured`. Optimized Inference, before/after comparison,
and conclusions remain `planned` until exact saved optimized artifacts are
created or imported.

The first five product routes are now documented in
`docs/127_platform_ux_storytelling_upgrade.md`. The Main Inference route uses a
time-compressed replay over saved progress events and aggregate reports; it
does not fabricate request-level latency series where the run only saved
per-config or aggregate latency data.

## 17. Caveats

- The full raw 250k response file is not present under
  `experiments/main/main_inference_v1/raw/` in this local repo checkout.
- The artifact sync report records the full raw file on the run host at
  `backups/main_inference_v1/results/raw/main_inference_v1_results.jsonl` with
  size 4,591,606,250 bytes. The downloaded portable archive contains the full
  raw evidence separately from the Git-tracked repo artifacts.
- The repo-saved artifacts are enough for local summary analysis, SLO
  comparison, cost reporting, GPU telemetry reporting, comparison tables, and
  plotting.
- `.env` secrets must not be printed, copied into docs, or committed.
- `main_inference_v1_plotting_dataset.jsonl` has a misleading extension: it is
  CSV-formatted and should be read with `pd.read_csv`, not `pd.read_json`.
- The context preflight artifacts are diagnostic history. The official full
  run status is determined by the manifest, eval report, SLO report, and
  scorecard for `main_inference_v1`.

## 18. Next Steps

1. Finalize this `Main_Inference_V1` reference document.
2. Keep `docs/95_definitive_technical_briefing.md` pointing to this document
   as the canonical detailed reference.
3. Run post-run optimization diagnosis over the failed quality and safety SLOs.
4. Choose one controlled optimization strategy.
5. Prepare `Optimized_Inference_V1`.
6. Rerun optimized inference on GPU.
7. Compare Main_Inference_V1 against Optimized_Inference_V1.
8. Update the portfolio narrative around the measured before/after result.
