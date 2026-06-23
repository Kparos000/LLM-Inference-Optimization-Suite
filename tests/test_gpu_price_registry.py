from __future__ import annotations

from pathlib import Path

import pytest

from inference_bench.gpu_price_registry import (
    estimate_gpu_cost,
    get_gpu_metadata,
    get_gpu_price,
    list_supported_gpus,
    load_gpu_price_registry,
)


def test_gpu_price_registry_loads_requested_runpod_gpus() -> None:
    registry = load_gpu_price_registry()
    supported = list_supported_gpus()

    assert len(registry) == 22
    assert "A100 SXM 80GB" in supported
    assert "H100 SXM 80GB" in supported
    assert "L40S" in supported
    assert "RTX Pro 4000" in supported
    assert get_gpu_price("A100 SXM 80GB") is None
    assert get_gpu_metadata("l40s")["vram_gb"] == 48.0


def test_missing_gpu_price_blocks_cost_claim_without_fabricating() -> None:
    estimate = estimate_gpu_cost(
        gpu_name="H100 SXM 80GB",
        elapsed_hours=2.0,
        projected_seconds_by_prompt_count={1000: 600.0, 10000: 6000.0, 40000: 24000.0},
    )

    assert estimate["cost_blocked_reason"] == "gpu_hourly_price_missing"
    assert estimate["gpu_hourly_cost"] is None
    assert estimate["estimated_run_cost"] is None
    assert estimate["projected_10000_cost"] is None


def test_api_provider_track_never_uses_gpu_cost() -> None:
    estimate = estimate_gpu_cost(
        gpu_name="H100 SXM 80GB",
        elapsed_hours=1.0,
        backend_type="api_provider",
        provider="hf_inference_provider",
    )

    assert estimate["cost_applicable"] is False
    assert estimate["cost_blocked_reason"] == "api_provider_track"
    assert estimate["estimated_run_cost"] is None


def test_reviewed_price_fixture_enables_projection_math(tmp_path: Path) -> None:
    path = tmp_path / "gpu_prices.yaml"
    path.write_text(
        """
test_h100:
  gpu_name: "Test H100"
  provider: runpod
  hourly_price: 4.0
  vram_gb: 80
  system_ram_gb: null
  vcpus: null
  generation: "test"
  recommended_use: "test fixture"
""".lstrip(),
        encoding="utf-8",
    )

    estimate = estimate_gpu_cost(
        gpu_name="Test H100",
        elapsed_hours=1.5,
        projected_seconds_by_prompt_count={1000: 900.0, 10000: 9000.0, 40000: 36000.0},
        registry_path=path,
    )

    assert estimate["gpu_hourly_cost"] == pytest.approx(4.0)
    assert estimate["estimated_run_cost"] == pytest.approx(6.0)
    assert estimate["projected_1000_cost"] == pytest.approx(1.0)
    assert estimate["projected_10000_cost"] == pytest.approx(10.0)
    assert estimate["projected_40000_cost"] == pytest.approx(40.0)
