"""FastAPI entrypoint for the read-only product platform backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.product_platform import platform_data  # noqa: E402
from inference_bench.product_platform_contracts import (  # noqa: E402
    PlatformResponse,
    RecipeValidationRequest,
    RecipeValidationResponse,
)

app = FastAPI(
    title="AI Inference Engineering Platform API",
    version="0.1.0",
    description="Read-only artifact API for the saved inference engineering demo.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3001",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _response(data: dict[str, Any], *, result_type: str = "measured") -> PlatformResponse:
    return PlatformResponse(
        status="ok",
        result_type=result_type,  # type: ignore[arg-type]
        source_artifacts=[],
        data=data,
    )


@app.get("/api/health", response_model=PlatformResponse)
def health() -> PlatformResponse:
    return _response(platform_data().health())


@app.get("/api/project/overview", response_model=PlatformResponse)
def project_overview() -> PlatformResponse:
    return _response(platform_data().project_overview())


@app.get("/api/slo-metrics", response_model=PlatformResponse)
def slo_metric_catalog() -> PlatformResponse:
    return _response(platform_data().slo_metric_catalog(), result_type="planned")


@app.get("/api/dataset/workflow", response_model=PlatformResponse)
def dataset_workflow() -> PlatformResponse:
    return _response(platform_data().dataset_workflow_summary())


@app.get("/api/dataset/explorer", response_model=PlatformResponse)
def dataset_explorer() -> PlatformResponse:
    return _response(platform_data().dataset_explorer(), result_type="planned")


@app.get("/api/dataset/cases", response_model=PlatformResponse)
def dataset_cases(
    vertical: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=120),
    expected_status: str | None = Query(default=None),
    min_evidence_count: int | None = Query(default=None, ge=0, le=10),
    max_evidence_count: int | None = Query(default=None, ge=0, le=10),
    sort_by: str = Query(default="prompt_id", pattern="^(prompt_id|prompt_length|evidence_count)$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=12, ge=1, le=50),
) -> PlatformResponse:
    return _response(
        platform_data().dataset_cases(
            vertical=vertical,
            search=search,
            expected_status=expected_status,
            min_evidence_count=min_evidence_count,
            max_evidence_count=max_evidence_count,
            sort_by=sort_by,
            offset=offset,
            limit=limit,
        ),
        result_type="planned",
    )


@app.get("/api/preparation/pipeline", response_model=PlatformResponse)
def preparation_pipeline() -> PlatformResponse:
    return _response(platform_data().preparation_pipeline())


@app.get("/api/preparation/modules", response_model=PlatformResponse)
def preparation_modules() -> PlatformResponse:
    return _response(platform_data().preparation_modules(), result_type="planned")


@app.get("/api/matrix", response_model=PlatformResponse)
def matrix_rows() -> PlatformResponse:
    return _response(platform_data().matrix_rows(), result_type="planned")


@app.get("/api/models", response_model=PlatformResponse)
def models() -> PlatformResponse:
    return _response(platform_data().models())


@app.get("/api/engines", response_model=PlatformResponse)
def engines() -> PlatformResponse:
    return _response(platform_data().engines())


@app.get("/api/memory-modes", response_model=PlatformResponse)
def memory_modes() -> PlatformResponse:
    return _response(platform_data().memory_modes())


@app.get("/api/slo-targets", response_model=PlatformResponse)
def slo_targets() -> PlatformResponse:
    return _response(platform_data().slo_targets())


@app.get("/api/main-inference/manifest", response_model=PlatformResponse)
def main_manifest() -> PlatformResponse:
    return _response(platform_data().main_manifest())


@app.get("/api/main-inference/replay-events", response_model=PlatformResponse)
def replay_events() -> PlatformResponse:
    return _response(platform_data().replay_events())


@app.get("/api/main-inference/telemetry", response_model=PlatformResponse)
def telemetry() -> PlatformResponse:
    return _response(platform_data().telemetry())


@app.get("/api/main-inference/results", response_model=PlatformResponse)
def measured_results() -> PlatformResponse:
    return _response(platform_data().main_results())


@app.get("/api/main-inference/replay-detail", response_model=PlatformResponse)
def main_replay_detail() -> PlatformResponse:
    return _response(platform_data().main_replay_detail())


@app.get("/api/main-inference/comparisons", response_model=PlatformResponse)
def main_comparisons() -> PlatformResponse:
    return _response(platform_data().comparison_datasets())


@app.get("/api/main-inference/diagnosis", response_model=PlatformResponse)
def diagnosis() -> PlatformResponse:
    return _response(platform_data().diagnosis())


@app.get("/api/optimizations/mandatory-repairs", response_model=PlatformResponse)
def mandatory_repairs() -> PlatformResponse:
    return _response(platform_data().mandatory_repairs(), result_type="planned")


@app.get("/api/optimizations/deployability-repairs", response_model=PlatformResponse)
def deployability_repairs() -> PlatformResponse:
    return _response(platform_data().deployability_repairs(), result_type="planned")


@app.get("/api/optimizations/repair-gate", response_model=PlatformResponse)
def repair_gate() -> PlatformResponse:
    return _response(platform_data().repair_gate(), result_type="planned")


@app.get("/api/optimizations/core-catalog", response_model=PlatformResponse)
def core_optimization_catalog() -> PlatformResponse:
    return _response(platform_data().core_optimization_catalog(), result_type="planned")


@app.get("/api/optimizations/core-catalog-v2", response_model=PlatformResponse)
def core_optimization_catalog_v2() -> PlatformResponse:
    return _response(platform_data().core_optimization_catalog_v2(), result_type="planned")


@app.get("/api/optimizations/applicability", response_model=PlatformResponse)
def optimization_applicability() -> PlatformResponse:
    return _response(platform_data().optimization_applicability(), result_type="planned")


@app.get("/api/optimizations/core-applicability", response_model=PlatformResponse)
def core_optimization_applicability() -> PlatformResponse:
    return _response(
        platform_data().core_optimization_applicability_v2(),
        result_type="planned",
    )


@app.get("/api/optimizations/experiment-stage", response_model=PlatformResponse)
def optimization_experiment_stage() -> PlatformResponse:
    return _response(platform_data().experiment_stage(), result_type="planned")


@app.get("/api/optimizations/story", response_model=PlatformResponse)
def optimization_story() -> PlatformResponse:
    return _response(platform_data().optimization_story_v2(), result_type="planned")


@app.get("/api/optimizations/observability/registry", response_model=PlatformResponse)
def optimization_observability_registry() -> PlatformResponse:
    return _response(platform_data().core_observability_registry(), result_type="planned")


@app.get("/api/optimizations/observability/readiness", response_model=PlatformResponse)
def optimization_observability_readiness() -> PlatformResponse:
    return _response(platform_data().core_observability_readiness(), result_type="planned")


@app.get("/api/optimizations/observability/inventory", response_model=PlatformResponse)
def optimization_observability_inventory() -> PlatformResponse:
    return _response(platform_data().main_observability_inventory(), result_type="planned")


@app.get("/api/optimizations/observability/prefix-opportunity", response_model=PlatformResponse)
def optimization_prefix_opportunity() -> PlatformResponse:
    return _response(platform_data().prefix_opportunity_analysis(), result_type="planned")


@app.get("/api/optimizations/observability/event-schema", response_model=PlatformResponse)
def optimization_observability_event_schema() -> PlatformResponse:
    return _response(platform_data().core_observability_event_schema(), result_type="planned")


@app.get(
    "/api/optimizations/observability/missing-instrumentation",
    response_model=PlatformResponse,
)
def optimization_missing_instrumentation() -> PlatformResponse:
    return _response(
        platform_data().core_observability_missing_instrumentation(),
        result_type="planned",
    )


@app.get("/api/optimizations/observability/cards", response_model=PlatformResponse)
def optimization_observability_cards() -> PlatformResponse:
    return _response(platform_data().core_observability_cards(), result_type="planned")


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1/summary",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_summary() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_summary(),
        result_type="planned",
    )


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1/layouts",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_layouts() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_layouts(),
        result_type="planned",
    )


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1/prefix-metrics",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_prefix_metrics() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_prefix_metrics(),
        result_type="planned",
    )


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1/equivalence",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_equivalence() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_equivalence(),
        result_type="planned",
    )


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1/decision",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_decision() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_decision(),
        result_type="planned",
    )


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1/story",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_story() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_story(),
        result_type="planned",
    )


@app.get(
    "/api/optimizations/coreopt-prefix-layout-static-v1",
    response_model=PlatformResponse,
)
def coreopt_prefix_layout_static_experiment() -> PlatformResponse:
    return _response(
        platform_data().coreopt_prefix_layout_static_experiment(),
        result_type="planned",
    )


@app.post("/api/optimizations/recipe/validate", response_model=RecipeValidationResponse)
def validate_recipe(request: RecipeValidationRequest) -> RecipeValidationResponse:
    payload = platform_data().validate_recipe(request)
    return RecipeValidationResponse(status="ok", result_type="planned", **payload)


@app.get("/api/scenarios", response_model=PlatformResponse)
def scenarios() -> PlatformResponse:
    return _response(platform_data().scenario_registry(), result_type="planned")


@app.get("/api/comparison/availability", response_model=PlatformResponse)
def comparison_availability() -> PlatformResponse:
    return _response(platform_data().comparison_availability(), result_type="planned")


@app.get("/api/conclusions/availability", response_model=PlatformResponse)
def conclusion_availability() -> PlatformResponse:
    return _response(platform_data().conclusion_availability(), result_type="planned")
