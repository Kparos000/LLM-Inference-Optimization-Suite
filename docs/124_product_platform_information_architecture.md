# Product Platform Information Architecture

Status: design blueprint only. No frontend, backend, API, or generated UI
artifact is implemented by this document.

Purpose: define the final interactive AI Inference Engineering Platform. The
platform is not a benchmark dashboard. It is a guided product experience that
lets recruiters and engineers replay the completed inference experiment,
inspect SLO failures, understand bottlenecks, select only applicable inference
optimization strategies, and compare saved before/after artifacts without
running GPUs.

## Product Principle

The deployed demo must run from saved artifacts. Users do not start vLLM,
SGLang, API provider jobs, RunPod pods, or GPU inference. The platform loads
repository artifacts for `Main_Inference_V1` and, when available,
`Optimized_Inference_V1`.

The core interaction is:

```text
Measured result
-> failed SLO
-> deterministic diagnosis
-> bottleneck
-> compatible optimization options
-> explanation and risk
-> apply
-> replay saved optimized result or show controlled rerun plan
-> compare before and after
```

The product must never show an optimization option that cannot plausibly
improve the selected failed SLO under the current model, engine, memory mode,
hardware, backend, and measured telemetry context.

## Source-Of-Truth Inputs

Current repository sources:

- `docs/main_inference_V1.md`
- `experiments/main/main_inference_v1/`
- `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_eval_summary.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_cost_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_engine_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_memory_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_concurrency_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_api_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_model_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_plotting_dataset.jsonl`
- `data/generated/dataset_10000/`
- `data/generated/context_engineering/`
- `configs/models.yaml`
- `configs/memory_modes.yaml`
- `configs/runtime_engines.yaml`
- `configs/backend_matrix.yaml`
- `configs/slo_targets.yaml`
- `configs/slo_profiles.yaml`
- `configs/bottleneck_catalog.yaml`
- `configs/optimization_catalog.yaml`
- `configs/optimization_negative_rules.yaml`
- `src/inference_bench/slo_diagnosis.py`
- `src/inference_bench/optimization_recommender.py`
- `src/inference_bench/optimization_catalog.py`
- `src/inference_bench/optimization_negative_rules.py`

Known missing product sources:

- `experiments/optimized/optimized_inference_v1/`
- `optimized_inference_v1` eval, SLO, cost, telemetry, comparison, and plotting
  artifacts.
- UI-ready JSON bundles such as `main_inference_v1_ui_summary.json`.
- A Main_Inference-specific diagnosis export that applies catalog,
  compatibility, and negative-rule filtering per failed row.
- A saved before/after replay bundle for optimized inference.

## Frontend Information Architecture

Primary navigation:

1. Landing
2. Why Inference Engineering
3. Project Overview
4. Dataset
5. Dataset Exploration
6. Knowledge Bases
7. Context Engineering
8. Retrieval Pipeline
9. Memory Modes
10. Serving Engines
11. Experiment Matrix
12. Main_Inference_V1
13. SLO Dashboard
14. Optimization Intelligence
15. Optimized_Inference_V1
16. Before vs After Comparison
17. Engineering Lessons
18. About the Project

Secondary persistent controls:

- Experiment selector: `Main_Inference_V1`, `Optimized_Inference_V1` when
  available.
- Scope selector: aggregate, per config, per vertical, per model, per engine,
  per memory mode, per concurrency.
- Metric family selector: quality, safety, latency, throughput, resource,
  cost.
- Evidence drawer: source file, row count, timestamp, checksum status.
- "No GPU required" status badge.

## Page Contracts

### Landing Page

Purpose: introduce the product as an AI inference engineering platform and
state that the demo replays saved full-scale artifacts.

User journey: user lands, sees the full 250,000-request result, understands
that runtime/cost passed while quality/safety failed, then enters the guided
replay.

Data source: curated summary generated from Main_Inference artifacts.

Existing artifacts:

- `docs/main_inference_V1.md`
- `experiments/main/main_inference_v1/main-inference-readme.md`
- `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_cost_report.json`

Missing artifacts:

- `main_inference_v1_ui_summary.json`
- optimized run headline summary.

Visualizations:

- Hero metric strip: 250,000/250,000 completed, 0 failed, A100, cost, runtime.
- Verdict cards: runtime pass, cost pass, quality fail, safety fail.
- Journey map from dataset to optimization replay.

Interactions:

- Start replay.
- Jump to failed SLOs.
- Toggle "show engineering evidence".

APIs required:

- `GET /api/experiments`
- `GET /api/experiments/{run_id}/summary`
- `GET /api/experiments/{run_id}/verdicts`

UI components:

- Hero banner.
- Metric cards.
- Verdict badges.
- Timeline preview.
- Call-to-action buttons.

### Why Inference Engineering

Purpose: explain why inference optimization is more than model comparison.

User journey: user learns the engineering concepts behind TTFT, TPOT, E2E,
throughput, KV cache, retrieval, safety, cost, and SLOs.

Data source: static educational copy plus glossary and measured examples.

Existing artifacts:

- `docs/95_definitive_technical_briefing.md`
- `docs/main_inference_V1.md`
- `configs/slo_targets.yaml`

Missing artifacts:

- UI educational copy bundle.

Visualizations:

- Inference pipeline diagram.
- Prefill vs decode split.
- Quality-latency-cost frontier illustration using saved metrics.

Interactions:

- Hover definitions for TTFT, TPOT, groundedness, evidence match, SLO.
- Metric family tabs.

APIs required:

- `GET /api/reference/glossary`
- `GET /api/reference/slo-targets`

UI components:

- Explainer panels.
- Glossary popovers.
- Pipeline stepper.

### Project Overview

Purpose: show the architecture of the suite and how artifacts flow from
dataset to inference to diagnosis.

User journey: user understands the system as an end-to-end inference
engineering stack.

Data source: repository docs and current configuration.

Existing artifacts:

- `docs/95_definitive_technical_briefing.md`
- `configs/models.yaml`
- `configs/memory_modes.yaml`
- `configs/runtime_engines.yaml`
- `configs/backend_matrix.yaml`

Missing artifacts:

- product-ready architecture graph JSON.

Visualizations:

- System architecture graph.
- Artifact dependency graph.
- Baseline to optimized workflow.

Interactions:

- Click subsystem to reveal source files.
- Toggle code/config/report layers.

APIs required:

- `GET /api/reference/architecture`
- `GET /api/reference/config-index`

UI components:

- Architecture graph.
- Source-file cards.
- Dependency drawer.

### Dataset

Purpose: describe the five-vertical 10,000-prompt benchmark.

User journey: user sees what tasks were measured and why the dataset is
representative for inference engineering.

Data source: promoted dataset and EDA reports.

Existing artifacts:

- `data/scaleup_2000_full/`
- `data/generated/dataset_10000/`
- `docs/53_phase2a_10000_dataset_eda.md`
- `docs/95_definitive_technical_briefing.md`

Missing artifacts:

- `dataset_ui_summary.json`
- sampled safe prompt examples for UI display.

Visualizations:

- Vertical distribution.
- Prompt/gold/KB counts.
- Task type distribution.
- Safety-boundary distribution.

Interactions:

- Filter by vertical.
- Filter by task type.
- Inspect prompt/gold schema.

APIs required:

- `GET /api/dataset/summary`
- `GET /api/dataset/verticals`
- `GET /api/dataset/samples?vertical=&task_type=`

UI components:

- Dataset stat cards.
- Distribution charts.
- Schema viewer.
- Sample table.

### Dataset Exploration

Purpose: provide interactive exploration of the benchmark without exposing raw
sensitive or oversized artifacts.

User journey: user filters the dataset, inspects representative examples, and
connects workload shape to inference behavior.

Data source: EDA outputs and curated sample extracts.

Existing artifacts:

- `data/generated/dataset_10000/`
- `data/generated/dataset_10000/README.md`

Missing artifacts:

- small UI-safe sample bundle.
- precomputed search index for prompt examples.

Visualizations:

- Treemaps.
- Term bars.
- Word/term distributions.
- Vertical task heatmap.

Interactions:

- Search prompts.
- Select vertical.
- Compare prompt lengths.
- Open prompt/gold side panel.

APIs required:

- `GET /api/dataset/explore`
- `GET /api/dataset/search?q=`
- `GET /api/dataset/prompt/{prompt_id}`

UI components:

- Search box.
- Faceted filters.
- Charts.
- Prompt detail drawer.

### Knowledge Bases

Purpose: explain the vertical knowledge corpora and provenance boundaries.

User journey: user learns what evidence the model could use and why each
vertical has different retrieval challenges.

Data source: KB files, corpus registry, and corpus build reports.

Existing artifacts:

- `data/scaleup_2000_full/`
- `data/generated/context_engineering/corpus_registry.json`
- `data/generated/context_engineering/corpus_build_report.json`
- `data/generated/context_engineering/corpus_build_summary.csv`

Missing artifacts:

- `knowledge_base_ui_summary.json`
- source-family thumbnails or compact cards.

Visualizations:

- KB count by vertical.
- Source type distribution.
- Chunk strategy comparison.
- Evidence family map.

Interactions:

- Select vertical.
- Inspect chunk examples.
- Compare KB row counts.

APIs required:

- `GET /api/kb/summary`
- `GET /api/kb/vertical/{vertical}`
- `GET /api/kb/chunks/sample?vertical=`

UI components:

- Vertical tabs.
- Corpus cards.
- Chunk preview drawer.

### Context Engineering

Purpose: show how raw KB records become model-facing context.

User journey: user sees context normalization, chunking, top-k selection,
compression, evidence labels, and prompt rendering.

Data source: context engineering reports and memory workload code/config.

Existing artifacts:

- `data/generated/context_engineering/corpus_build_report.json`
- `data/generated/context_engineering/compression_diagnostic_report.json`
- `data/generated/context_engineering/compression_diagnostic_summary.csv`
- `configs/memory_modes.yaml`
- `src/inference_bench/memory_workloads.py`

Missing artifacts:

- rendered context examples for all five memory modes.
- UI-safe before/after compression examples.

Visualizations:

- Context packing diagram.
- Compression before/after token chart.
- Evidence label mapping.

Interactions:

- Toggle mm0-mm4.
- Inspect E1-E5 evidence labels.
- View original vs compressed context.

APIs required:

- `GET /api/context/summary`
- `GET /api/context/memory-mode/{memory_mode}`
- `GET /api/context/example?prompt_id=&memory_mode=`

UI components:

- Memory-mode tabs.
- Evidence cards.
- Context diff viewer.

### Retrieval Pipeline

Purpose: show BM25, Qdrant vector retrieval, hybrid fusion, reranking, and
retrieval SLOs.

User journey: user sees how retrieval quality affects both latency and
groundedness.

Data source: promoted retrieval reports.

Existing artifacts:

- `data/generated/context_engineering/retrieval_source_of_truth_manifest.json`
- `data/generated/context_engineering/retrieval_quality_gate_report.json`
- `data/generated/context_engineering/retrieval_evaluation_report.json`
- `data/generated/context_engineering/qdrant_index_report.json`
- `docs/79_phase4_handoff_and_retrieval_promotion.md`

Missing artifacts:

- UI-ready retrieval pipeline graph.
- per-prompt retrieval trace samples.

Visualizations:

- Candidate recall@20/@50 by vertical.
- Final recall@5 by vertical.
- MRR by vertical.
- Sparse/dense/hybrid pipeline graph.

Interactions:

- Select vertical.
- Compare retrieval ablations.
- Inspect failed retrieval examples where available.

APIs required:

- `GET /api/retrieval/summary`
- `GET /api/retrieval/vertical/{vertical}`
- `GET /api/retrieval/examples?status=failed|passed`

UI components:

- Retrieval metric cards.
- Pipeline graph.
- Evidence rank table.

### Memory Modes

Purpose: explain mm0-mm4 and connect each mode to measured inference results.

User journey: user sees why no-context is an ablation, dense/hybrid are RAG
baselines, compressed hybrid targets prefill cost, and mm4 adds bounded repair.

Data source: memory mode config and Main_Inference comparison rows.

Existing artifacts:

- `configs/memory_modes.yaml`
- `experiments/main/main_inference_v1/processed/main_inference_v1_memory_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv`

Missing artifacts:

- memory-mode UI explanations as structured JSON.

Visualizations:

- Memory mode comparison table.
- Quality vs latency scatter.
- Evidence match and groundedness by memory mode.

Interactions:

- Toggle memory mode.
- Compare modes side by side.
- Open optimization implications.

APIs required:

- `GET /api/memory-modes`
- `GET /api/experiments/{run_id}/memory-comparison`

UI components:

- Segmented memory-mode control.
- Comparison table.
- Scatter chart.

### Serving Engines

Purpose: explain vLLM, SGLang, and API provider route and show measured
differences.

User journey: user understands engine choice as a serving-runtime decision with
latency, throughput, memory, and operational implications.

Data source: runtime registry and comparison CSVs.

Existing artifacts:

- `configs/runtime_engines.yaml`
- `configs/backend_matrix.yaml`
- `experiments/main/main_inference_v1/processed/main_inference_v1_engine_comparison.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv`

Missing artifacts:

- engine capability UI matrix.

Visualizations:

- Engine comparison table.
- TTFT/TPOT/E2E charts by engine.
- API vs self-hosted route diagram.

Interactions:

- Filter by model, memory mode, concurrency.
- Click engine row to view capability and risk.

APIs required:

- `GET /api/engines`
- `GET /api/experiments/{run_id}/engine-comparison`
- `GET /api/experiments/{run_id}/api-vs-self-hosted`

UI components:

- Engine cards.
- Comparison chart.
- Capability table.

### Experiment Matrix

Purpose: present the exact 25-config matrix.

User journey: user sees every measured config and can drill into a row before
viewing SLO outcomes.

Data source: manifest, SLO summary, comparison CSVs.

Existing artifacts:

- `experiments/main/main_inference_v1/raw/main_inference_v1_manifest.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_model_comparison.csv`

Missing artifacts:

- `main_inference_v1_ui_matrix.json`
- optimized matrix artifacts.

Visualizations:

- Matrix grid by model, engine, memory mode, concurrency.
- Completion status row badges.
- Cost/latency/quality sparklines.

Interactions:

- Filter matrix.
- Click config row.
- Pin configs for comparison.

APIs required:

- `GET /api/experiments/{run_id}/matrix`
- `GET /api/experiments/{run_id}/config/{config_id}`

UI components:

- Data grid.
- Filter chips.
- Config detail drawer.

### Main_Inference_V1

Purpose: tell the completed baseline story from execution through failed
deployability.

User journey: user reviews the official full run before entering the SLO and
optimization pages.

Data source: Main_Inference artifacts.

Existing artifacts:

- `docs/main_inference_V1.md`
- `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_cost_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_artifact_sync_report.json`

Missing artifacts:

- `main_inference_v1_ui_summary.json`

Visualizations:

- Run timeline.
- Request completion cards.
- Cost split.
- SLO verdict cards.
- A100 telemetry summary.

Interactions:

- Open artifact evidence.
- Jump to failed SLOs.
- Download report references.

APIs required:

- `GET /api/experiments/main_inference_v1/summary`
- `GET /api/experiments/main_inference_v1/artifacts`

UI components:

- Timeline.
- Metric cards.
- Artifact table.

### SLO Dashboard

Purpose: show pass/fail status across metrics, configs, and metric families.

User journey: user clicks a failed SLO and enters diagnosis.

Data source: SLO scorecard, SLO report, SLO summary.

Existing artifacts:

- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_report.json`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv`
- `configs/slo_targets.yaml`
- `configs/slo_profiles.yaml`

Missing artifacts:

- `main_inference_v1_ui_slo.json`
- per-row diagnosis export.

Visualizations:

- SLO scorecard.
- Failed metric heatmap.
- Config x SLO matrix.
- Severity bars.

Interactions:

- Click failed SLO.
- Filter by metric family.
- Toggle aggregate/config/vertical scope.
- Open target definition.

APIs required:

- `GET /api/experiments/{run_id}/slo`
- `GET /api/experiments/{run_id}/slo/{slo_id}`
- `GET /api/slo-targets`

UI components:

- SLO table.
- Heatmap.
- Metric detail drawer.
- Status badges.

### Optimization Intelligence

Purpose: convert failed SLOs into explainable, compatible optimization options.

User journey:

```text
Click failed SLO
-> show diagnosis
-> show bottleneck
-> show compatible optimizations
-> explain why each applies
-> explain why excluded options are hidden or disabled
-> apply selected option
```

Data source: SLO diagnosis code, bottleneck catalog, optimization catalog,
negative rules, and Main_Inference SLO rows.

Existing artifacts:

- `configs/bottleneck_catalog.yaml`
- `configs/optimization_catalog.yaml`
- `configs/optimization_negative_rules.yaml`
- `src/inference_bench/slo_diagnosis.py`
- `src/inference_bench/optimization_recommender.py`
- `experiments/main/main_inference_v1/processed/main_inference_v1_slo_summary.csv`

Missing artifacts:

- `main_inference_v1_ui_diagnosis.json`
- `main_inference_v1_ui_optimization_options.json`
- Main_Inference-specific diagnosis wrapper.
- negative-rule filtering wired into UI option generation.
- saved optimized result map.

Visualizations:

- SLO -> bottleneck -> optimization graph.
- Recommendation rank list.
- Applicability/exclusion cards.
- Risk badges for quality, cost, latency, and hardware.

Interactions:

- Click failed SLO.
- Select optimization from filtered dropdown.
- View "why shown" and "why not shown".
- Apply one optimization.
- Apply all compatible optimizations as a controlled plan.
- Toggle planned-only, implemented-only, safe-to-replay.

APIs required:

- `GET /api/experiments/{run_id}/diagnosis`
- `GET /api/experiments/{run_id}/slo/{slo_id}/optimization-options`
- `POST /api/replay/apply`
- `POST /api/replay/apply-all`

UI components:

- Diagnosis graph.
- Filtered dropdown.
- Optimization cards.
- Apply button.
- Apply-all planner.
- Exclusion drawer.

### Optimized_Inference_V1

Purpose: present the saved optimized run after optimization is measured.

User journey: user sees what changed, what stayed constant, and whether failed
SLOs improved.

Data source: future optimized artifacts.

Existing artifacts:

- No `Optimized_Inference_V1` artifact folder was found in the repo.

Missing artifacts:

- `experiments/optimized/optimized_inference_v1/`
- optimized manifest.
- optimized eval report.
- optimized SLO report and scorecard.
- optimized cost report.
- optimized comparison artifacts.
- optimized UI summary JSON.

Visualizations:

- Optimized run summary.
- Changed-factor cards.
- Optimization application timeline.
- Quality/safety/runtime/cost verdicts.

Interactions:

- Select optimized run.
- Inspect changed factors.
- Compare against Main_Inference_V1.

APIs required:

- `GET /api/experiments/optimized_inference_v1/summary`
- `GET /api/experiments/optimized_inference_v1/artifacts`
- `GET /api/experiments/optimized_inference_v1/applied-optimizations`

UI components:

- Optimized result cards.
- Changed-factor table.
- Artifact evidence table.

### Before vs After Comparison

Purpose: show measured deltas between Main_Inference_V1 and
Optimized_Inference_V1.

User journey: user sees whether the optimization improved the failed SLOs
without hiding regressions.

Data source: baseline and optimized UI summaries plus comparison artifact.

Existing artifacts:

- Main_Inference comparison CSVs.
- baseline-v1 repair comparison artifacts under
  `experiments/baseline/baseline_v1_quality_repair_v1/evidence/`.

Missing artifacts:

- `main_vs_optimized_inference_v1_ui_comparison.json`
- optimized result artifacts.
- saved before/after row mapping.

Visualizations:

- SLO delta table.
- Before/after bars.
- Quality/safety/runtime/cost radar.
- Regression callouts.

Interactions:

- Compare by aggregate, config, metric, model, engine, memory mode.
- Show only improved.
- Show regressions.
- Open evidence trace.

APIs required:

- `GET /api/comparisons/main-vs-optimized`
- `GET /api/comparisons/main-vs-optimized/config/{config_id}`
- `GET /api/comparisons/main-vs-optimized/slo/{slo_id}`

UI components:

- Delta table.
- Before/after charts.
- Regression badges.
- Evidence drawer.

### Engineering Lessons

Purpose: explain what the experiment teaches about inference engineering.

User journey: user connects measured evidence to professional engineering
judgment.

Data source: docs, SLO failures, comparison signals, repair reports.

Existing artifacts:

- `docs/main_inference_V1.md`
- `docs/95_definitive_technical_briefing.md`
- `docs/121_phase2_optimization_diagnosis.md`
- `experiments/main/main_inference_v1/processed/main_inference_v1_repair_vs_broken_comparison_report.json`

Missing artifacts:

- curated lesson cards.

Visualizations:

- Quality-latency-cost tradeoff map.
- Failure-to-lesson cards.
- "Do not optimize blindly" sequence.

Interactions:

- Click lesson to reveal supporting artifact.
- Compare intuition vs measured result.

APIs required:

- `GET /api/lessons`
- `GET /api/lessons/{lesson_id}`

UI components:

- Lesson cards.
- Evidence links.
- Tradeoff chart.

### About The Project

Purpose: explain project scope, authorial intent, reproducibility, and role
relevance for AI Inference Engineer interviews.

User journey: recruiter or engineer understands what was built, what is
measured, and what remains.

Data source: docs and repository metadata.

Existing artifacts:

- `README.md`
- `PROJECT_STATE.md`
- `docs/00_project_scope.md`
- `docs/95_definitive_technical_briefing.md`

Missing artifacts:

- product about-page copy bundle.

Visualizations:

- Timeline.
- Capability checklist.
- Repo artifact map.

Interactions:

- Open GitHub/repo references.
- Download project summary.

APIs required:

- `GET /api/about`

UI components:

- Timeline.
- Capability grid.
- Links panel.

## Backend API Contract

The backend should be artifact-backed and read-only for saved experiment data.
No endpoint should start inference in the deployed demo.

### Experiment APIs

```http
GET /api/experiments
GET /api/experiments/{run_id}/summary
GET /api/experiments/{run_id}/artifacts
GET /api/experiments/{run_id}/matrix
GET /api/experiments/{run_id}/config/{config_id}
GET /api/experiments/{run_id}/telemetry
GET /api/experiments/{run_id}/cost
GET /api/experiments/{run_id}/quality
```

### SLO And Diagnosis APIs

```http
GET /api/experiments/{run_id}/slo
GET /api/experiments/{run_id}/slo/{slo_id}
GET /api/experiments/{run_id}/diagnosis
GET /api/experiments/{run_id}/slo/{slo_id}/diagnosis
GET /api/experiments/{run_id}/slo/{slo_id}/optimization-options
```

### Optimization Replay APIs

```http
POST /api/replay/apply
POST /api/replay/apply-all
GET /api/replay/{replay_id}
GET /api/comparisons/main-vs-optimized
GET /api/comparisons/main-vs-optimized/config/{config_id}
```

`POST /api/replay/apply` does not run inference. It returns either a saved
optimized artifact mapping or a controlled rerun plan.

### Reference APIs

```http
GET /api/reference/slo-targets
GET /api/reference/optimization-catalog
GET /api/reference/optimization-negative-rules
GET /api/reference/bottleneck-catalog
GET /api/reference/models
GET /api/reference/memory-modes
GET /api/reference/engines
GET /api/reference/glossary
```

## UI JSON Object Designs

These objects are design contracts only. Do not generate them until the UI data
pipeline is implemented.

### `experiment_ui_index.json`

```json
{
  "experiments": [
    {
      "run_id": "main_inference_v1",
      "label": "Main Inference V1",
      "role": "pre_optimization_baseline",
      "status": "available",
      "artifact_root": "experiments/main/main_inference_v1",
      "has_raw_responses_local": false,
      "has_ui_bundles": false
    },
    {
      "run_id": "optimized_inference_v1",
      "label": "Optimized Inference V1",
      "role": "post_optimization_result",
      "status": "missing",
      "artifact_root": "experiments/optimized/optimized_inference_v1",
      "has_raw_responses_local": false,
      "has_ui_bundles": false
    }
  ]
}
```

### `main_inference_v1_ui_summary.json`

```json
{
  "run_id": "main_inference_v1",
  "title": "Main Inference V1",
  "role": "official_full_pre_optimization_baseline",
  "artifact_root": "experiments/main/main_inference_v1",
  "status": "completed",
  "completed_requests": 250000,
  "failed_requests": 0,
  "config_count": 25,
  "gpu": "NVIDIA A100-SXM4-80GB",
  "wall_seconds": 42538.856294736965,
  "cost": {
    "total_usd": 18.461296726432796,
    "gpu_usd": 17.606359966432798,
    "api_usd": 0.854936759999998
  },
  "verdicts": {
    "runtime": "PASS",
    "cost": "PASS",
    "quality": "FAIL",
    "safety": "FAIL",
    "deployability": "NOT_DEPLOYABLE_SLO_FAILURES"
  },
  "source_artifacts": []
}
```

### `main_inference_v1_ui_slo.json`

```json
{
  "run_id": "main_inference_v1",
  "slo_rows": [
    {
      "slo_id": "aggregate.groundedness",
      "metric_family": "quality",
      "metric_name": "groundedness",
      "scope": "aggregate",
      "target": 0.98,
      "operator": ">=",
      "observed": 0.567204,
      "difference": -0.41279599999999994,
      "status": "FAIL",
      "severity": 0.421,
      "source_artifact": "experiments/main/main_inference_v1/processed/main_inference_v1_slo_scorecard.csv"
    }
  ]
}
```

### `main_inference_v1_ui_diagnosis.json`

```json
{
  "run_id": "main_inference_v1",
  "diagnosis_rows": [
    {
      "diagnosis_id": "aggregate.groundedness.low_groundedness",
      "slo_id": "aggregate.groundedness",
      "bottleneck_id": "low_groundedness",
      "category": "quality",
      "description": "Generated answers do not satisfy grounded evidence coverage.",
      "evidence": [
        {
          "metric_name": "groundedness",
          "target": 0.98,
          "observed": 0.567204,
          "gap": -0.41279599999999994
        }
      ],
      "confidence": 1.0,
      "severity": 0.421,
      "source_catalog": "configs/bottleneck_catalog.yaml"
    }
  ]
}
```

### `main_inference_v1_ui_optimization_options.json`

```json
{
  "run_id": "main_inference_v1",
  "options_by_slo": {
    "aggregate.groundedness": [
      {
        "optimization_id": "prompt_contract_repair",
        "label": "Prompt and contract repair",
        "shown": true,
        "enabled": true,
        "why_applicable": "Groundedness and contract validity failed.",
        "expected_effect": ["quality", "latency", "cost"],
        "may_hurt": [],
        "implementation_status": "implemented",
        "application_method": "code_change",
        "requires_gpu_rerun": true,
        "has_saved_optimized_result": false,
        "risk": {
          "quality": "low",
          "cost": "low"
        },
        "negative_rule_checks": [
          {
            "rule_id": "context_compression",
            "triggered": false
          }
        ],
        "source_catalog": "configs/optimization_catalog.yaml"
      }
    ]
  },
  "excluded_options_by_slo": {
    "aggregate.groundedness": [
      {
        "optimization_id": "increase_concurrency",
        "shown": false,
        "reason": "Concurrency does not address groundedness failure."
      }
    ]
  }
}
```

### `main_inference_v1_ui_comparison.json`

```json
{
  "comparison_id": "main_vs_optimized_inference_v1",
  "baseline_run_id": "main_inference_v1",
  "optimized_run_id": "optimized_inference_v1",
  "status": "blocked_until_optimized_artifacts_exist",
  "rows": [
    {
      "metric_name": "groundedness",
      "baseline": 0.567204,
      "optimized": null,
      "delta": null,
      "status": "missing_optimized_result"
    }
  ]
}
```

### `optimization_apply_request.json`

```json
{
  "run_id": "main_inference_v1",
  "scope": "row",
  "slo_id": "aggregate.groundedness",
  "config_id": null,
  "optimization_id": "prompt_contract_repair",
  "mode": "replay_saved_result"
}
```

### `optimization_apply_response.json`

```json
{
  "replay_id": "main_inference_v1.prompt_contract_repair.aggregate.groundedness",
  "status": "plan_only",
  "reason": "No saved Optimized_Inference_V1 artifact is available.",
  "change_exactly_one_factor": "prompt_renderer",
  "hold_constant": [
    "prompt_ids",
    "model_alias",
    "engine",
    "hardware",
    "memory_mode",
    "temperature",
    "max_new_tokens",
    "evaluator"
  ],
  "requires_gpu_rerun": true,
  "source_experiment": "main_inference_v1"
}
```

## Interaction Design

### Failed SLO Drilldown

```text
User clicks failed SLO
-> frontend requests SLO detail
-> frontend requests diagnosis for SLO
-> diagnosis drawer opens
-> bottleneck evidence is shown
-> optimization dropdown loads only compatible options
-> excluded options are available in an explanation drawer
```

### Optimization Dropdown Filtering

Filter sequence:

1. Start from failed SLO metric.
2. Map metric to bottleneck.
3. Load compatible optimizations from `configs/bottleneck_catalog.yaml`.
4. Load definitions from `configs/optimization_catalog.yaml`.
5. Filter by model, engine, memory mode, backend, hardware, and active
   capabilities.
6. Apply `configs/optimization_negative_rules.yaml`.
7. Hide irrelevant options.
8. Disable risky but educational options when they are not safe to apply.
9. Sort by severity, confidence, implementation status, and expected impact.

### Apply One

```text
User selects optimization
-> clicks Apply
-> if saved optimized artifact exists, replay before/after result
-> otherwise show controlled rerun plan
-> record selected optimization in local UI state
```

### Apply All

Apply all should not combine arbitrary optimizations. It should build a
controlled plan:

1. Safety fixes.
2. Contract fixes.
3. Evidence and groundedness fixes.
4. Latency/throughput/cost fixes only after quality and safety are acceptable.
5. One-factor changes unless a saved optimized run measured a bundle.

### Before/After Replay

```text
User opens comparison
-> select baseline run
-> select optimized run
-> compare SLOs and metrics
-> show improvements and regressions
-> link each delta to source artifacts
```

## Already Implemented

- Full `Main_Inference_V1` artifact archive.
- Main_Inference reference document.
- Dataset, context, retrieval, memory-mode, serving, evaluation, telemetry,
  SLO, and cost artifacts.
- SLO targets and profiles.
- Bottleneck catalog.
- Optimization catalog.
- Negative optimization rules catalog.
- Deterministic SLO diagnosis engine.
- Deterministic optimization recommender.
- Main_Inference-specific UI diagnosis wrapper.
- UI-ready Main_Inference diagnosis, optimization option, apply-plan, and
  story JSON artifacts.
- Negative-rule enforcement in UI option generation.
- Main_Inference SLO summary and comparison CSVs.
- Artifact sync report, checkpoint, progress log, manifest, and checksums.
- Baseline repair evidence from earlier phases.

## Missing

- Frontend application.
- Backend read API.
- UI-ready summary, matrix, SLO, and before/after comparison bundles.
- Optimized_Inference_V1 artifacts.
- Saved before/after comparison bundle.
- Row-level raw response access in the local repo checkout.
- Curated UI-safe prompt/context/evidence examples.
- Product copy bundles for educational pages.
- Product visual design system.
- Replay state model for apply/apply-all.

## Implementation Roadmap

### Phase P0: Product Data Contract

1. Freeze this document as the product blueprint.
2. Define JSON schemas for UI bundles.
3. Add schema validation tests.
4. Confirm artifact roots for Main and Optimized runs.

### Phase P1: UI Bundle Generator

1. Generate `main_inference_v1_ui_summary.json`.
2. Generate `main_inference_v1_ui_slo.json`.
3. `main_inference_v1_ui_diagnosis.json` is implemented.
4. `main_inference_v1_ui_optimization_options.json` is implemented.
5. `main_inference_v1_ui_apply_plan.json` is implemented.
6. `main_inference_v1_ui_story.json` is implemented.
7. Generate `experiment_ui_index.json`.
8. Do not require GPUs.

### Phase P2: Diagnosis And Filtering

1. Main_Inference diagnosis wrapper is implemented.
2. Catalog-backed optimization options are implemented for failed SLOs.
3. Compatibility filters are implemented through existing recommender logic.
4. Negative-rule filters are implemented for UI option generation.
5. Excluded-option explanations are preserved for transparency.

### Phase P3: Read-Only Backend

1. Implement artifact-backed API endpoints.
2. Add endpoint tests against saved fixtures.
3. Ensure endpoints cannot start inference.
4. Add checksum/source metadata to API responses.

### Phase P4: Frontend Shell

1. Implement navigation and layout.
2. Build Landing, Project Overview, Dataset, and Main_Inference pages.
3. Build shared chart/table/card components.
4. Keep first release read-only.

### Phase P5: Optimization Intelligence UI

1. Build SLO Dashboard.
2. Build failed-SLO drilldown.
3. Build bottleneck graph.
4. Build filtered optimization dropdown.
5. Build apply/apply-all plan mode.

### Phase P6: Optimized_Inference_V1 Integration

1. Run or import optimized artifacts.
2. Generate optimized UI bundles.
3. Generate before/after comparison bundle.
4. Enable saved-result replay.

### Phase P7: Portfolio Polish

1. Add guided walkthrough.
2. Add recruiter-friendly overview mode.
3. Add engineer detail mode.
4. Add exportable summary cards.
5. Validate all claims against repo artifacts.

## Product Guardrails

- Do not run GPUs in the deployed demo.
- Do not expose `.env` or secrets.
- Do not invent optimized results before artifacts exist.
- Do not show optimization options that do not target the selected failed SLO.
- Do not hide regressions in before/after comparisons.
- Do not treat planned optimizations as measured improvements.
- Do not combine multiple changes unless a saved artifact measured that bundle.
- Always link UI claims back to source artifacts.
