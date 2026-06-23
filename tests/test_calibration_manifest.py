from __future__ import annotations

from pathlib import Path

import pytest

from inference_bench.calibration_manifest import (
    build_calibration_manifest,
    load_runpod_calibration_profiles,
    validate_calibration_profile,
    write_calibration_manifest,
)


def test_runpod_calibration_profiles_load_with_null_prices() -> None:
    profiles = load_runpod_calibration_profiles()

    assert set(profiles) == {
        "A100_SXM_CALIBRATION",
        "H100_SXM_CALIBRATION",
        "L40S_CALIBRATION",
    }
    assert profiles["A100_SXM_CALIBRATION"].gpu_name == "A100 SXM 80GB"
    assert profiles["A100_SXM_CALIBRATION"].hourly_price is None
    assert profiles["H100_SXM_CALIBRATION"].prompt_counts == (100, 200)


def test_calibration_profile_validation_blocks_missing_price_only() -> None:
    profile = load_runpod_calibration_profiles()["L40S_CALIBRATION"]
    validation = validate_calibration_profile(profile)

    assert validation["gpu_registered"] is True
    assert validation["gpu_price_registered"] is False
    assert validation["runtime_profile_valid"] is True
    assert validation["registered_hourly_price"] is None


def test_calibration_manifest_generation_and_write(tmp_path: Path) -> None:
    profile = load_runpod_calibration_profiles()["A100_SXM_CALIBRATION"]
    manifest = build_calibration_manifest(
        profile=profile,
        model_alias="model2_3b",
        memory_mode="mm2_hybrid_top5",
        concurrency=1,
        prompt_count=100,
        artifact_paths={"raw": "results/raw/calibration.jsonl"},
        repo_root=tmp_path,
    )

    assert manifest.profile_id == "A100_SXM_CALIBRATION"
    assert manifest.cost_estimate["cost_blocked_reason"] == "gpu_hourly_price_missing"
    output = write_calibration_manifest(manifest, tmp_path / "manifest.json")
    assert output.exists()


def test_calibration_manifest_rejects_non_calibration_prompt_count() -> None:
    profile = load_runpod_calibration_profiles()["A100_SXM_CALIBRATION"]

    with pytest.raises(ValueError, match="prompt_count"):
        build_calibration_manifest(
            profile=profile,
            model_alias="model2_3b",
            memory_mode="mm2_hybrid_top5",
            concurrency=1,
            prompt_count=500,
            artifact_paths={"raw": "results/raw/calibration.jsonl"},
        )
