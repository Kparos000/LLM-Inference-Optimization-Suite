# Platform UX Storytelling Upgrade

Status: implemented on July 13, 2026

This document records the first major UX and storytelling upgrade for the
interactive AI Inference Engineering Platform. The upgrade is limited to the
first five routes:

- `/`
- `/slo-metrics`
- `/data`
- `/preparation`
- `/main-inference`

The platform remains a read-only replay product. It does not run GPUs, start
vLLM or SGLang, call API providers, mutate `Main_Inference_V1` artifacts, or
fabricate optimized results.

The later Optimization Lab route is governed by the two-track architecture in
`docs/128_inference_optimization_two_track_architecture.md`: deployability
repairs first, then core inference optimization after measured repair
validation.

## Design Goal

The goal is to make the completed inference engineering project understandable
without reducing it to a static dashboard. The first five routes now tell the
pre-run and measured-run story in order:

1. What inference engineering means.
2. What SLOs were designed before the run.
3. How the dataset, gold contract, knowledge base, and evaluator connect.
4. How retrieval, context, memory modes, models, engines, and the matrix were
   prepared.
5. How the measured `Main_Inference_V1` run executed and why the final verdict
   was not deployable.

Every page separates:

- designed-before-run content;
- measured-during-run content;
- learned-after-run content;
- educational visualizations;
- sampled replay events;
- aggregate metrics.

## Page To Artifact Map

| Route | Product Purpose | Primary Artifacts |
| --- | --- | --- |
| `/` | One-minute project overview and honest matrix snapshot | `docs/main_inference_V1.md`, `experiments/main/main_inference_v1/raw/main_inference_v1_manifest.json`, `experiments/main/main_inference_v1/processed/main_inference_v1_eval_report.json`, `configs/models.yaml`, `configs/memory_modes.yaml` |
| `/slo-metrics` | Pre-run SLO and metric education | `configs/slo_targets.yaml`, `configs/slo_profiles.yaml` |
| `/data` | Dataset and workflow explorer | `data/generated/dataset_10000/*`, `data/scaleup_2000_full/*/*_prompts_2000.jsonl`, `*_gold_2000.jsonl`, `*_kb_2000.jsonl` |
| `/preparation` | Retrieval/context/memory/model/serving/matrix setup | `configs/models.yaml`, `configs/runtime_engines.yaml`, `configs/memory_modes.yaml`, `configs/slo_targets.yaml`, `data/generated/context_engineering/*`, `experiments/main/main_inference_v1/processed/main_inference_v1_api_vs_self_hosted_comparison.csv` |
| `/main-inference` | Time-compressed measured replay of Main_Inference_V1 | `experiments/main/main_inference_v1/raw/main_inference_v1_manifest.json`, `logs/main_inference_v1_progress.jsonl`, `raw/main_inference_v1_gpu_telemetry.jsonl`, processed eval/cost/SLO/comparison CSV and JSON files |

## Frontend Structure

New storytelling components live in:

```text
platform/frontend/src/components/StoryRoutes.tsx
```

`ChapterScreen` dispatches the first five routes into the story components and
keeps the existing later-route behavior for optimization, optimized inference,
comparison, and conclusions.

The guided navigation now includes the new `/slo-metrics` route. The browser
session and sidebar use the same chapter IDs, which fixes route highlighting
for the expanded journey.

## Backend API Contracts

The product backend exposes compact read-only contracts through the existing
Pydantic response envelope in
`src/inference_bench/product_platform_contracts.py`.

New or expanded endpoints:

| Endpoint | Result Type | Purpose |
| --- | --- | --- |
| `GET /api/slo-metrics` | `planned` | Metric families, target sources, applicability rules, and evaluation flow |
| `GET /api/dataset/explorer` | `planned` | Dataset totals, vertical workload summaries, pressure dimensions, and Research AI coverage explanation |
| `GET /api/dataset/cases` | `planned` | Paginated safe prompt/gold/KB/evaluation case viewer |
| `GET /api/preparation/modules` | `planned` | Retrieval, context, memory mode, model registry, serving/hardware, and matrix setup modules |
| `GET /api/matrix` | `planned` | Exact 25-config matrix rows and construction formulas |
| `GET /api/main-inference/replay-detail` | `measured` | Run contract, replay phases/events, telemetry summaries, latency/throughput, cost, verdict gates, comparison tabs, and artifact reliability |
| `GET /api/main-inference/comparisons` | `measured` | Engine, memory, concurrency, API/self-hosted, model, and SLO scorecard datasets |

The backend does not expose arbitrary raw files. Dataset examples are joined
server-side by `prompt_id`, filtered, paginated, and reduced to UI-safe fields.
Knowledge base chunks are exposed only through the curated prompt/gold/KB case
contract and preserve source labels without leaking local absolute paths.

## SLO And Metrics Page

`/slo-metrics` is pre-run educational content. It deliberately avoids showing
`Main_Inference_V1` observed results. It explains:

- what an SLO is;
- why deployability needs targets before experiments;
- metric families for UX, capacity, answer usefulness, safety, retrieval,
  infrastructure, and economics;
- metric definitions, user impact, system influences, optimization levers,
  tradeoffs, and target source;
- applicability rules such as API-only costs, GPU telemetry requirements,
  compression-only targets, and MM4 trace targets;
- evaluation flow: metric -> target -> measured value -> gap -> pass/fail ->
  deployability decision.

## Dataset Explorer

`/data` uses the promoted 10,000-record dataset summaries and linked
2,000-record vertical files. It shows:

- 10,000 prompts;
- 10,000 gold records;
- 4,740 KB rows;
- five verticals;
- 2,000 prompts per vertical;
- evidence coverage by vertical;
- prompt/gold/KB/evaluation as one synchronized case.

Research AI's 98% evidence coverage is explained as deliberate:

- 1,960 prompts require evidence;
- 40 out-of-scope prompts intentionally require no evidence;
- no answerable Research AI prompts are missing required evidence.

The page adds a qualitative workload-pressure model for input pressure, output
pressure, evidence complexity, retrieval difficulty, and contract/safety
complexity. These are education signals, not measured post-run predictions.

## Preparation Page

`/preparation` replaces the oversized generic pipeline with six modules:

1. Retrieval Engineering
2. Context Engineering
3. Memory Modes
4. Model Registry
5. Serving & Hardware
6. SLO Setup & Experiment Matrix

The matrix is shown with the actual construction:

```text
Self-hosted:
1 model x 2 engines x 2 concurrency levels x 5 memory modes = 20 configs

API:
1 model x 1 API route x 1 concurrency level x 5 memory modes = 5 configs

Total:
25 configs x 10,000 prompts = 250,000 requests
```

This avoids implying a larger Cartesian product that did not run.

## Main Inference Replay

`/main-inference` is a measured artifact replay. It uses saved progress,
telemetry, manifest, cost, eval, SLO, and comparison artifacts. No inference is
executed in the browser.

The replay contract now starts at zero and ends at:

- 250,000 completed requests;
- 25 completed configs;
- 2,500 checkpoints;
- 0 failures;
- measured final cost;
- measured wall time of about 11.82 hours.

Replay phases:

1. Preflight
2. Matrix load
3. vLLM execution
4. SGLang execution
5. API execution
6. Artifact finalization
7. Evaluation
8. SLO scoring

The page includes chart-resolution labels. Request progress and checkpoint
accumulation are replay-event based. GPU telemetry uses sampled measured
telemetry. Latency and throughput trends are per-config aggregate updates where
request-level latency series are not available.

The old mixed-unit completion verdict chart is replaced by deployability gates:

- Execution: PASS
- Reliability: PASS
- Latency: PASS
- Throughput: PASS
- Cost: PASS
- JSON validity: PASS
- Contract validity: FAIL
- Format validity: FAIL
- Evidence match: FAIL
- Groundedness: FAIL
- Safety: FAIL

The final verdict remains:

```text
NOT_DEPLOYABLE_SLO_FAILURES
```

## Data Safety

The upgrade follows these safety rules:

- no `.env` or secret exposure;
- no local absolute paths in UI-facing artifact labels;
- no raw 10,000-row browser dropdowns;
- server-side filtering and pagination for cases;
- no mutation of experiment artifacts;
- no GPU/API execution;
- no fabricated optimized results;
- no fabricated request-level latency series.

## Local Startup

From the repository root:

```powershell
cd platform/frontend
npm install
npm run backend
```

In a second terminal:

```powershell
cd platform/frontend
npm run dev
```

Open:

```text
http://127.0.0.1:3001
```

The API runs at:

```text
http://127.0.0.1:8011
```

## Validation

Primary validation commands:

```powershell
pytest tests/test_product_platform_api.py
ruff check .
mypy src tests
cd platform/frontend
npm run typecheck
npm run lint
npm test
npm run build
```

API smoke routes:

```text
/api/slo-metrics
/api/dataset/explorer
/api/dataset/cases
/api/preparation/modules
/api/matrix
/api/main-inference/replay-detail
/api/main-inference/comparisons
```

UI smoke routes:

```text
/
/slo-metrics
/data
/preparation
/main-inference
```

## Known Limitations

- `Optimized_Inference_V1` is still planned until measured optimized artifacts
  exist or are imported.
- The full raw 250,000-response file is not required by this UI and remains
  outside the Git-tracked local artifact set.
- Main replay uses available saved progress and aggregate reports; it does not
  invent request-level TTFT, TPOT, E2E, or throughput series.
- Some deeper retrieval-ranking examples depend on compact UI-safe retrieval
  exports that can be added in a later pass.
- Screenshots should be regenerated after the local browser server is running
  cleanly.
