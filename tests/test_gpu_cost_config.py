from dataclasses import replace

import pytest

from inference_bench.phase4_readiness import (
    calculate_configured_gpu_cost,
    calculate_elapsed_gpu_cost,
    load_gpu_cost_configs,
)


def test_gpu_cost_config_loads_with_runpod_placeholders() -> None:
    config = load_gpu_cost_configs()["runpod_default"]

    assert config.provider == "runpod"
    assert config.gpu_type is None
    assert config.hourly_price_usd is None
    assert config.live_run_values_configured is False
    assert config.cost_formula == "elapsed_hours * hourly_price_usd"


def test_elapsed_gpu_cost_formula_is_usable() -> None:
    cost = calculate_elapsed_gpu_cost(
        hourly_price_usd=2.0,
        measured_start_time="2026-06-05T00:00:00+00:00",
        measured_end_time="2026-06-05T01:30:00+00:00",
    )

    assert cost == pytest.approx(3.0)


def test_configured_gpu_cost_refuses_unfilled_placeholders() -> None:
    config = load_gpu_cost_configs()["runpod_default"]

    with pytest.raises(ValueError, match="must be configured"):
        calculate_configured_gpu_cost(config)

    configured = replace(
        config,
        gpu_type="NVIDIA test fixture",
        hourly_price_usd=1.5,
        region="test-region",
        measured_start_time="2026-06-05T00:00:00Z",
        measured_end_time="2026-06-05T02:00:00Z",
    )
    assert calculate_configured_gpu_cost(configured) == pytest.approx(3.0)
