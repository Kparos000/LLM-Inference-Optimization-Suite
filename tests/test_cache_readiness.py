from __future__ import annotations

from inference_bench.cache_readiness import (
    calculate_cache_readiness,
    estimate_kv_cache_pressure,
)


def test_cache_readiness_reports_prefix_context_and_cache_score() -> None:
    metrics = calculate_cache_readiness(
        prompts=[
            "System: answer from evidence only. Question A",
            "System: answer from evidence only. Question B",
            "System: answer from evidence only. Question C",
        ],
        context_blocks=[
            ["policy_a", "shared_context"],
            ["policy_b", "shared_context"],
            ["policy_c", "shared_context"],
        ],
        input_tokens=[100, 100, 100],
        output_tokens=[20, 20, 20],
        concurrency=2,
        context_window_tokens=1000,
    )

    assert metrics.repeated_prefix_tokens > 0
    assert metrics.shared_context_percentage > 0
    assert metrics.prefix_reuse_potential > 0
    assert 0 <= metrics.kv_cache_pressure_estimate <= 1
    assert 0 <= metrics.cacheability_score <= 1
    assert 0 <= metrics.estimated_prefix_cache_benefit <= 1


def test_kv_cache_pressure_increases_with_concurrency() -> None:
    low = estimate_kv_cache_pressure(
        input_tokens=[1000],
        output_tokens=[200],
        concurrency=1,
        context_window_tokens=4000,
    )
    high = estimate_kv_cache_pressure(
        input_tokens=[1000],
        output_tokens=[200],
        concurrency=4,
        context_window_tokens=4000,
    )

    assert high > low
