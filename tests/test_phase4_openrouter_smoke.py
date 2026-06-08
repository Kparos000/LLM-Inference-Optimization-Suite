from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest

from inference_bench.api_pricing import (
    estimate_api_cost_from_pricing,
    resolve_api_pricing,
)
from inference_bench.api_routes import (
    OPENROUTER_BASE_URL,
    api_key_for_route,
    resolve_api_provider_route,
)
from inference_bench.config import load_project_config
from inference_bench.model_smoke_comparison import build_model5_comparison_report
from inference_bench.streaming_metrics import (
    TimedStreamChunk,
    calculate_streaming_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_ROW = run_path(str(REPO_ROOT / "scripts/phase4/run_api_priced_smoke.py"))["_base_row"]


def _evaluation(grounded: float) -> dict[str, object]:
    return {
        "summary": {
            "json_valid_rate": 1.0,
            "generation_contract_valid_rate": 1.0,
            "evidence_id_presence_rate": 1.0,
            "evidence_match_rate": grounded,
            "grounded_rate": grounded,
            "safety_violation_rate": 0.0,
        }
    }


def _result_row(alias: str, *, streaming: bool = True) -> dict[str, object]:
    return {
        "model_alias": alias,
        "model_id": alias,
        "provider": "fixture",
        "success": True,
        "streaming_available": streaming,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "latency_ms": 200.0,
    }


def test_openrouter_config_and_route_load() -> None:
    config = load_project_config()
    model = config.resolve_model_config("model5_gated")
    pricing = resolve_api_pricing("model5_gated")
    route = resolve_api_provider_route(model=model, pricing=pricing)

    assert model.model_id == "mistralai/ministral-3b-2512"
    assert route.provider == "openrouter"
    assert route.base_url == OPENROUTER_BASE_URL
    assert route.api_key_env == "OPENROUTER_API_KEY"
    assert route.supports_streaming is True


def test_missing_openrouter_key_fails_cleanly_without_exposing_a_secret() -> None:
    model = load_project_config().resolve_model_config("model5_gated")
    route = resolve_api_provider_route(
        model=model,
        pricing=resolve_api_pricing("model5_gated"),
    )

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        api_key_for_route(route, {})


def test_runner_base_row_does_not_persist_api_key() -> None:
    pricing = resolve_api_pricing("model5_gated")
    item = SimpleNamespace(
        metadata={
            "workload_id": "w1",
            "vertical": "finance",
            "memory_mode": "mm2_hybrid_top5",
            "ablation_mode": "prompt_plus_metadata",
            "dataset_split": "smoke_500",
            "citation_id_aliases": {},
            "gold_evidence_ids": [],
        },
        workload_name="smoke",
        prompt_id="p1",
        expected_output="generation_contract_json",
        prompt="fixture",
    )

    row = BASE_ROW(
        item=item,
        model_alias="model5_gated",
        model_id=pricing.model_id,
        provider=pricing.provider,
        pricing=pricing,
        stream=True,
        backend="openrouter",
        api_route=f"{OPENROUTER_BASE_URL}/chat/completions",
    )

    assert "api_key" not in row
    assert "OPENROUTER_API_KEY" not in str(row)


def test_openrouter_streaming_fixture_captures_latency_metrics() -> None:
    metrics = calculate_streaming_metrics(
        [
            TimedStreamChunk(
                80.0,
                {"choices": [{"delta": {"content": "one"}}]},
            ),
            TimedStreamChunk(
                100.0,
                {"choices": [{"delta": {"content": " two"}}]},
            ),
            TimedStreamChunk(
                130.0,
                {
                    "choices": [{"delta": {"content": " three"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                },
            ),
        ],
        e2e_latency_ms=140.0,
        prompt="fixture prompt",
    )

    assert metrics.ttft_ms == 80.0
    assert metrics.itl_p50_ms == 25.0
    assert metrics.itl_p95_ms == pytest.approx(29.5)
    assert metrics.itl_p99_ms == pytest.approx(29.9)
    assert metrics.tpot_ms == 30.0


def test_model5_cost_uses_checked_in_openrouter_pricing() -> None:
    cost = estimate_api_cost_from_pricing(
        input_tokens=1_000_000,
        output_tokens=500_000,
        pricing=resolve_api_pricing("model5_gated"),
    )

    assert cost["input_cost_usd"] == pytest.approx(0.10)
    assert cost["output_cost_usd"] == pytest.approx(0.05)
    assert cost["total_api_cost_usd"] == pytest.approx(0.15)


def test_offline_comparison_does_not_trigger_api_or_gpu() -> None:
    model5 = [_result_row("model5_gated") for _ in range(5)]
    model6 = [_result_row("model6_gated") for _ in range(5)]
    local = [_result_row("model1_0_5b", streaming=False) for _ in range(5)]
    cost = {
        "total_cost_usd": 0.001,
        "cost_per_request_usd": 0.0002,
        "cost_per_successful_answer_usd": 0.0002,
        "cost_per_grounded_answer_usd": 0.0004,
    }
    latency = {
        "ttft_ms": {"mean": 50.0},
        "itl_p50_ms": {"mean": 2.0},
        "itl_p95_ms": {"mean": 4.0},
        "itl_p99_ms": {"mean": 5.0},
        "tpot_ms": {"mean": 3.0},
        "e2e_latency_ms": {"mean": 200.0},
    }

    report, rows = build_model5_comparison_report(
        model5_results=model5,
        model5_eval=_evaluation(0.8),
        model5_cost=cost,
        model5_latency=latency,
        model6_results=model6,
        model6_eval=_evaluation(0.6),
        model6_cost=cost,
        model6_latency=latency,
        local_results=local,
        local_eval=_evaluation(0.2),
    )

    assert report["model5_final_benchmark_recommendation"] == "RETAIN"
    assert report["prompt_id_sets_match"] is True
    assert report["models"]["local_qwen_0_5b"]["total_tokens"] == 600
    assert report["no_additional_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert any(row["metric"] == "ttft_ms" for row in rows)
