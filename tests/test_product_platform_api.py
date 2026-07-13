from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from fastapi.testclient import TestClient

from inference_bench.product_platform import ProductPlatformData
from inference_bench.product_platform_contracts import RecipeValidationRequest


def _load_backend_module() -> ModuleType:
    path = Path("platform/backend/main.py")
    spec = importlib.util.spec_from_file_location("product_platform_backend", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platform_data_project_overview_uses_measured_main_artifacts() -> None:
    overview = ProductPlatformData().project_overview()

    assert overview["headline_metrics"]["completed_requests"] == 250000
    assert overview["headline_metrics"]["failed_requests"] == 0
    assert overview["verdicts"]["overall_deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"


def test_replay_events_end_at_exact_measured_totals() -> None:
    replay = ProductPlatformData().replay_events(limit=24)

    assert replay["result_type"] == "measured"
    assert replay["final_completed"] == 250000
    assert replay["final_failed"] == 0
    assert replay["events"][-1]["completed_requests"] == 250000
    assert replay["events"][-1]["failure_count"] == 0


def test_optimization_applicability_preserves_blocked_core_strategies() -> None:
    applicability = ProductPlatformData().optimization_applicability()
    states = {str(item["optimization_id"]): item for item in applicability["states"]}

    assert states["prompt_contract_repair"]["state"] == "applicable_measured"
    assert states["use_quantized_model"]["state"] == "blocked_by_negative_rule"
    assert states["use_quantized_model"]["negative_rule"] == "quantization"


def test_recipe_validation_rejects_blocked_negative_rule_strategy() -> None:
    payload = ProductPlatformData().validate_recipe(
        RecipeValidationRequest(core_optimization_ids=["use_quantized_model"])
    )

    assert payload["valid"] is False
    assert payload["blocked"][0]["negative_rule"] == "quantization"
    assert payload["plan"]["does_not_execute_inference"] is True


def test_fastapi_health_and_results_are_read_only() -> None:
    module = _load_backend_module()
    client = TestClient(module.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["data"]["read_only"] is True

    results = client.get("/api/main-inference/results")
    body: dict[str, Any] = results.json()
    assert results.status_code == 200
    assert body["data"]["result_type"] == "measured"
    assert body["data"]["eval_report"]["total_requests_completed"] == 250000
