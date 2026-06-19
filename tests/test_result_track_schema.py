from __future__ import annotations

import pytest

from inference_bench.result_track_schema import (
    result_track_join_key,
    track_description,
    validate_result_track_row,
)


def _base_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-1",
        "config_id": "cfg-1",
        "prompt_id": "prompt-1",
        "vertical": "research_ai",
        "model_alias": "model2_1_5b",
        "memory_mode": "mm2_hybrid_top5",
        "runtime": "vllm",
        "backend_type": "self_hosted_gpu",
        "engine": "vllm",
        "hardware": "remote_rtx3070",
        "provider": "huggingface",
        "concurrency": 1,
        "gpu_cost_usd": None,
        "gpu_hourly_price_usd": None,
        "api_cost_usd": None,
    }
    row.update(overrides)
    return row


def test_result_track_join_key_is_stable_tuple() -> None:
    key = result_track_join_key(_base_row())

    assert key == (
        "run-1",
        "cfg-1",
        "prompt-1",
        "research_ai",
        "model2_1_5b",
        "mm2_hybrid_top5",
        "vllm",
        "self_hosted_gpu",
        "vllm",
        "remote_rtx3070",
        "huggingface",
        "1",
    )


def test_api_provider_requires_provider_and_api_cost_without_gpu_telemetry() -> None:
    errors = validate_result_track_row(
        _base_row(
            backend_type="api_provider",
            api_provider="novita",
            api_cost_usd=0.0001,
            gpu_telemetry_available=False,
            runtime="api_provider_route",
            engine="hf_router",
            hardware="provider_managed",
            provider="novita",
        )
    )

    assert errors == []


def test_api_provider_cannot_claim_gpu_telemetry() -> None:
    errors = validate_result_track_row(
        _base_row(
            backend_type="api_provider",
            api_provider="openrouter",
            api_cost_usd=0.0001,
            gpu_telemetry_available=True,
            runtime="api_provider_route",
            engine="openrouter",
            hardware="provider_managed",
            provider="openrouter",
        )
    )

    assert "api_provider_must_not_claim_gpu_telemetry" in errors


def test_self_hosted_gpu_cost_requires_hourly_price_and_no_api_cost() -> None:
    errors = validate_result_track_row(
        _base_row(gpu_cost_usd=0.01, api_cost_usd=0.0001),
    )

    assert "gpu_cost_requires_hourly_price" in errors
    assert "self_hosted_gpu_must_not_claim_api_cost" in errors


def test_api_provider_requires_provider_managed_hardware() -> None:
    errors = validate_result_track_row(
        _base_row(
            backend_type="api_provider",
            runtime="api_provider_route",
            api_provider="openrouter",
            api_cost_usd=0.0001,
            gpu_telemetry_available=False,
            engine="openrouter",
            hardware="runpod_gpu",
            provider="openrouter",
        )
    )

    assert "api_provider_requires_provider_managed_hardware" in errors


def test_local_compute_track_is_valid_without_api_or_gpu_cost() -> None:
    errors = validate_result_track_row(
        _base_row(
            backend_type="local_compute",
            runtime="huggingface_transformers",
            engine="huggingface",
            hardware="developer_workstation",
            provider="huggingface",
        )
    )

    assert errors == []


def test_track_descriptions_distinguish_execution_tracks() -> None:
    assert "API provider track" in track_description("api_provider")
    assert "Local compute track" in track_description("local_compute")
    assert "Self-hosted GPU track" in track_description("self_hosted_gpu")
    with pytest.raises(ValueError, match="Unknown backend_type"):
        track_description("unknown")
