# Platform Foundation V1

Status: implemented on July 12, 2026

Platform Foundation V1 is the first implementation of the interactive AI
Inference Engineering Platform. It is not a benchmark dashboard and does not
run inference. It is a read-only product surface over saved repository
artifacts, with a guided nine-page story from data construction to
optimization planning.

## Product Scope

The product currently supports:

- artifact-backed replay of `Main_Inference_V1`;
- guided navigation across the nine product chapters;
- a dedicated pre-run SLO and metrics education page;
- a synchronized prompt -> gold contract -> knowledge base -> evaluation case
  explorer;
- interactive preparation modules for retrieval, context, memory modes, model
  registry, serving, and matrix construction;
- browser-persisted experiment session state;
- a measured Main_Inference replay using saved progress and telemetry events;
- a two-lane Optimization Lab separating mandatory system repairs from core
  inference-engineering strategies;
- visible disabled states for negative-rule-blocked strategies;
- plan-only apply/apply-all behavior;
- honest planned placeholders for `Optimized_Inference_V1`, comparison, and
  conclusion artifacts.

It explicitly does not:

- run GPUs;
- start vLLM, SGLang, API provider inference, or RunPod jobs;
- mutate `Main_Inference_V1` artifacts;
- fabricate optimized results;
- label planned or modeled outputs as measured.

## Frontend

Frontend root:

```text
platform/frontend/
```

Stack:

- Next.js App Router;
- TypeScript;
- Tailwind CSS;
- Recharts;
- Framer Motion;
- lucide-react;
- browser-local `ExperimentSession`.

Routes:

| Route | Page |
| --- | --- |
| `/` | About |
| `/slo-metrics` | SLO & Metrics |
| `/data` | Data & Workflow Explorer |
| `/preparation` | Inference Experiment Preparation |
| `/main-inference` | Main Inference Simulation |
| `/optimization` | Inference Optimization Lab |
| `/optimized-inference` | Optimized Inference Simulation target |
| `/comparison` | Before/After Comparison target |
| `/conclusions` | Conclusions & Recommendations target |

The frontend uses the FastAPI backend when available and keeps conservative
repo-grounded fallback facts for local static rendering.

## Backend

Backend root:

```text
platform/backend/
```

Reusable data layer:

```text
src/inference_bench/product_platform.py
src/inference_bench/product_platform_contracts.py
```

The backend is FastAPI with Pydantic response contracts. It exposes compact
read-only UI contracts from existing artifacts and existing repository logic.

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Service and artifact availability |
| `GET /api/project/overview` | Project headline metrics and page artifact map |
| `GET /api/slo-metrics` | Pre-run SLO/metric families, definitions, targets, and applicability |
| `GET /api/dataset/workflow` | Dataset, vertical, and workflow summary |
| `GET /api/dataset/explorer` | Dataset totals, workload pressure, and vertical summaries |
| `GET /api/dataset/cases` | Paginated safe prompt/gold/KB/evaluation case viewer |
| `GET /api/preparation/pipeline` | Preparation pipeline, retrieval, context, and matrix summary |
| `GET /api/preparation/modules` | Retrieval/context/memory/model/serving/matrix modules |
| `GET /api/matrix` | Exact 25-config matrix rows and construction formulas |
| `GET /api/models` | Model registry and aliases |
| `GET /api/engines` | Runtime engine registry |
| `GET /api/memory-modes` | MM0-MM4 configuration |
| `GET /api/slo-targets` | SLO target/profile configuration |
| `GET /api/main-inference/manifest` | Main run manifest |
| `GET /api/main-inference/replay-events` | Time-compressed replay events |
| `GET /api/main-inference/telemetry` | Sampled A100 telemetry |
| `GET /api/main-inference/results` | Eval, cost, and SLO scorecard |
| `GET /api/main-inference/replay-detail` | Main replay contract, phases, charts, gates, lessons, and artifacts |
| `GET /api/main-inference/comparisons` | Engine, memory, concurrency, API/self-hosted, model, and SLO comparisons |
| `GET /api/main-inference/diagnosis` | Existing UI diagnosis/options/apply/story artifacts |
| `GET /api/optimizations/mandatory-repairs` | Mandatory repair plan lane |
| `GET /api/optimizations/core-catalog` | Full optimization catalog for education |
| `GET /api/optimizations/applicability` | Optimization states and negative-rule blocks |
| `POST /api/optimizations/recipe/validate` | Plan-only recipe validation |
| `GET /api/scenarios` | Measured/planned scenario registry |
| `GET /api/comparison/availability` | Before/after availability contract |
| `GET /api/conclusions/availability` | Conclusion/chat availability contract |

## Experiment Session

The browser session tracks:

- baseline run ID;
- current chapter;
- selected mandatory system repairs;
- selected core inference optimizations;
- validated recipe;
- result type: `measured`, `modeled`, or `planned`;
- selected scenario ID;
- selected optimized run ID when available.

Session state is persisted in `localStorage` for demo continuity.

## Local Startup

Backend:

```powershell
cd platform/frontend
npm run backend
```

Frontend:

```powershell
cd platform/frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:3001
```

Optional API base override:

```powershell
$env:NEXT_PUBLIC_PLATFORM_API_BASE = "http://127.0.0.1:8011"
```

## Verification

Implemented tests:

```text
tests/test_product_platform_api.py
platform/frontend/src/lib/facts.test.ts
```

Important checks:

- backend response schemas;
- read-only health/results endpoints;
- replay ends at exactly 250,000 completed and zero failed;
- quantization remains visible but negative-rule-blocked;
- recipe validation rejects blocked strategies;
- frontend has all guided routes including `/slo-metrics`;
- frontend labels optimized/comparison routes as planned.

## Current Limitations

- The first five routes have the richer storytelling upgrade documented in
  `docs/127_platform_ux_storytelling_upgrade.md`.
- `Optimized_Inference_V1` is not yet completed or imported.
- Before/after comparison is therefore blocked by design.
- Conclusion/chat endpoints are contract placeholders.
- The frontend uses conservative fallback facts when the API is not running.
- The full raw 250,000-response file is not required for this product
  foundation and remains outside the local Git-tracked artifact set.
- `npm install` resolves with zero known npm audit vulnerabilities after the
  Vitest update and patched PostCSS override.

## Next Phase

Recommended next phase:

```text
PLATFORM_V2_UI_POLISH_AND_OPTIMIZED_ARTIFACT_INTEGRATION
```

Priorities:

1. Add deeper retrieval-ranking UI-safe exports for the Preparation route.
2. Replace remaining frontend fallback facts with API-only contracts for
   production.
3. Add deeper route/component tests and screenshot regression coverage.
4. Import measured `Optimized_Inference_V1` artifacts when they exist.
5. Generate the measured before/after comparison bundle.
6. Add project-grounded conclusion interpretation artifacts.
