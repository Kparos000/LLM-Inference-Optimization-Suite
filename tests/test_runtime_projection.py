from __future__ import annotations

from inference_bench.b1_quality import build_b1_runtime_projection


def test_b1_runtime_projection_includes_required_scales_and_matrix() -> None:
    report = build_b1_runtime_projection(
        measured_prompt_count=100,
        measured_wall_seconds=200.0,
        mean_latency_ms=1900.0,
        p50_latency_ms=1800.0,
        p95_latency_ms=2500.0,
        runpod_targets={
            "rtx4090": {
                "hourly_price_usd": None,
                "throughput_multiplier_vs_rtx3070": None,
            }
        },
    )

    projections = report["rtx3070_prompt_projections"]
    assert isinstance(projections, list)
    assert [row["prompt_count"] for row in projections] == [500, 2500, 5000, 10000]
    matrix = report["rtx3070_full_matrix_projection"]
    assert isinstance(matrix, dict)
    assert matrix["config_count"] == 8
    assert matrix["total_prompt_executions"] == 80000


def test_runpod_projection_keeps_unconfigured_values_null() -> None:
    report = build_b1_runtime_projection(
        measured_prompt_count=100,
        measured_wall_seconds=100.0,
        mean_latency_ms=900.0,
        p50_latency_ms=850.0,
        p95_latency_ms=1200.0,
        runpod_targets={
            "h100": {
                "hourly_price_usd": None,
                "throughput_multiplier_vs_rtx3070": None,
            }
        },
    )

    targets = report["runpod_full_matrix_placeholders"]
    assert isinstance(targets, dict)
    h100 = targets["h100"]
    assert h100["estimated_full_matrix_seconds"] is None
    assert h100["estimated_full_matrix_cost_usd"] is None
    assert h100["status"] == "placeholder_requires_price_and_measured_multiplier"


def test_runpod_projection_calculates_only_from_configured_inputs() -> None:
    report = build_b1_runtime_projection(
        measured_prompt_count=100,
        measured_wall_seconds=100.0,
        mean_latency_ms=900.0,
        p50_latency_ms=850.0,
        p95_latency_ms=1200.0,
        runpod_targets={
            "l40s": {
                "hourly_price_usd": 2.0,
                "throughput_multiplier_vs_rtx3070": 4.0,
            }
        },
    )

    targets = report["runpod_full_matrix_placeholders"]
    assert isinstance(targets, dict)
    l40s = targets["l40s"]
    assert l40s["estimated_full_matrix_seconds"] == 20000.0
    assert l40s["estimated_full_matrix_cost_usd"] == 20000.0 / 3600.0 * 2.0
    assert l40s["status"] == "estimated"
