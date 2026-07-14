from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
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


def test_platform_data_resolves_artifacts_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    health = ProductPlatformData().health()

    assert health["main_inference_available"] is True


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


def test_fastapi_allows_frontend_origin_for_api_fetches() -> None:
    module = _load_backend_module()
    client = TestClient(module.app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://127.0.0.1:3001",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3001"


def test_slo_metrics_are_pre_run_educational() -> None:
    module = _load_backend_module()
    client = TestClient(module.app)

    response = client.get("/api/slo-metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["result_type"] == "planned"
    data = payload["data"]
    assert data["chronology"] == "Designed before the run"
    assert {family["id"] for family in data["families"]} >= {
        "user_experience",
        "answer_usefulness",
        "economics",
    }
    serialized = str(data)
    assert "250000" not in serialized
    assert "NOT_DEPLOYABLE_SLO_FAILURES" not in serialized


def test_dataset_explorer_research_ai_coverage_explanation() -> None:
    module = _load_backend_module()
    client = TestClient(module.app)

    response = client.get("/api/dataset/explorer")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["totals"]["prompt_count"] == 10000
    assert payload["totals"]["kb_count"] == 4740
    explanation = payload["research_ai_coverage_explanation"]
    assert explanation["coverage_rate"] == 0.98
    assert explanation["prompts_requiring_evidence"] == 1960
    assert explanation["out_of_scope_prompts_requiring_no_evidence"] == 40
    assert explanation["answerable_prompts_missing_evidence"] == 0


def test_dataset_cases_are_joined_and_paginated() -> None:
    module = _load_backend_module()
    client = TestClient(module.app)

    response = client.get("/api/dataset/cases?vertical=airline&limit=2")
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["limit"] == 2
    assert len(payload["cases"]) == 2
    first = payload["cases"][0]
    assert first["prompt"]["prompt_id"] == first["gold_contract"]["prompt_id"]
    assert first["knowledge_base"]["required_evidence"]
    assert payload["public_safety"]["raw_files_exposed"] is False


def test_matrix_and_replay_contracts() -> None:
    module = _load_backend_module()
    client = TestClient(module.app)

    matrix_response = client.get("/api/matrix")
    assert matrix_response.status_code == 200
    matrix = matrix_response.json()["data"]
    assert matrix["totals"]["config_count"] == 25
    assert matrix["totals"]["self_hosted_configs"] == 20
    assert matrix["totals"]["api_configs"] == 5

    replay_response = client.get("/api/main-inference/replay-events")
    assert replay_response.status_code == 200
    replay = replay_response.json()["data"]
    assert replay["events"][0]["completed_requests"] == 0
    assert replay["events"][-1]["completed_requests"] == 250000
    assert replay["events"][-1]["checkpoint_count"] == 2500
    assert replay["final_failed"] == 0
