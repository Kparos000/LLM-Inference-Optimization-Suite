from __future__ import annotations

from typing import Any, cast

from inference_bench.b7_controlled_baseline import build_b7_runtime_projection


def test_b7_runtime_projection_includes_required_scale_targets() -> None:
    projection = build_b7_runtime_projection(
        measured_prompt_count=1000,
        measured_wall_seconds=2000.0,
        mean_latency_ms=1800.0,
        p50_latency_ms=1500.0,
        p95_latency_ms=4200.0,
        selected_full_matrix_config_count=4,
    )

    targets = cast(dict[str, Any], projection["requested_projection_targets"])
    assert targets["controlled_2000_prompt_run"] == 2000
    assert targets["final_10000_prompt_single_config"] == 10000
    assert targets["selected_full_matrix_prompt_count"] == 40000
    assert projection["runpod_readiness_claimed"] is False
    assert len(cast(list[dict[str, Any]], projection["projections"])) == 3


def test_b7_runtime_projection_has_hour_estimates() -> None:
    projection = build_b7_runtime_projection(
        measured_prompt_count=1000,
        measured_wall_seconds=3600.0,
        mean_latency_ms=3600.0,
        p50_latency_ms=3000.0,
        p95_latency_ms=6000.0,
    )

    first = cast(list[dict[str, Any]], projection["projections"])[0]
    assert first["prompt_count"] == 2000
    assert first["estimated_hours_from_measured_throughput"] == 2.0
