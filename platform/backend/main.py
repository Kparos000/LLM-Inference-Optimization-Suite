"""FastAPI entrypoint for the read-only product platform backend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
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


@app.get("/api/dataset/workflow", response_model=PlatformResponse)
def dataset_workflow() -> PlatformResponse:
    return _response(platform_data().dataset_workflow_summary())


@app.get("/api/preparation/pipeline", response_model=PlatformResponse)
def preparation_pipeline() -> PlatformResponse:
    return _response(platform_data().preparation_pipeline())


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


@app.get("/api/main-inference/diagnosis", response_model=PlatformResponse)
def diagnosis() -> PlatformResponse:
    return _response(platform_data().diagnosis())


@app.get("/api/optimizations/mandatory-repairs", response_model=PlatformResponse)
def mandatory_repairs() -> PlatformResponse:
    return _response(platform_data().mandatory_repairs(), result_type="planned")


@app.get("/api/optimizations/core-catalog", response_model=PlatformResponse)
def core_optimization_catalog() -> PlatformResponse:
    return _response(platform_data().core_optimization_catalog(), result_type="planned")


@app.get("/api/optimizations/applicability", response_model=PlatformResponse)
def optimization_applicability() -> PlatformResponse:
    return _response(platform_data().optimization_applicability(), result_type="planned")


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
