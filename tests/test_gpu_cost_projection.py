from __future__ import annotations

import pytest

from inference_bench.gpu_price_registry import estimate_gpu_cost
from inference_bench.gpu_telemetry import build_runtime_projections


def test_gpu_cost_projection_adds_token_and_success_efficiency_fields() -> None:
    projection = build_runtime_projections(
        measured_prompt_count=100,
        measured_wall_seconds=600.0,
        mean_latency_ms=1200.0,
        p50_latency_ms=1000.0,
        p95_latency_ms=2400.0,
        target_prompt_counts=(1000,),
        gpu_name="A100 SXM 80GB",
        total_tokens=240_000,
        successful_requests=96,
    )

    assert projection["gpu_hourly_cost"] == pytest.approx(1.49)
    assert projection["estimated_run_cost"] == pytest.approx(0.24833333333333332)
    assert projection["projected_1000_cost"] == pytest.approx(2.4833333333333334)
    assert projection["projected_10000_cost"] == pytest.approx(24.833333333333332)
    assert projection["projected_40000_cost"] == pytest.approx(99.33333333333333)
    assert projection["tokens_per_gpu_dollar"] == pytest.approx(966442.9530201342)
    assert projection["successful_requests_per_gpu_dollar"] == pytest.approx(386.5771812080537)


def test_gpu_cost_projection_never_applies_to_api_provider_track() -> None:
    estimate = estimate_gpu_cost(
        gpu_name="A100 SXM 80GB",
        elapsed_hours=1.0,
        total_tokens=10_000,
        successful_requests=10,
        backend_type="api_provider",
        provider="openrouter",
    )

    assert estimate["cost_applicable"] is False
    assert estimate["cost_blocked_reason"] == "api_provider_track"
    assert estimate["estimated_run_cost"] is None
    assert estimate["tokens_per_gpu_dollar"] is None
    assert estimate["successful_requests_per_gpu_dollar"] is None
