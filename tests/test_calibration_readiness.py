from __future__ import annotations

from pathlib import Path
from typing import cast

from inference_bench.calibration_manifest import (
    calibration_readiness_verdict,
    load_runpod_calibration_profiles,
    validate_calibration_profile,
)
from inference_bench.full_run_readiness_audit import build_full_run_readiness_audit


def test_calibration_readiness_is_ready_when_price_and_run_safety_gates_pass() -> None:
    profile = load_runpod_calibration_profiles()["H100_SXM_CALIBRATION"]
    validation = validate_calibration_profile(profile)
    readiness = calibration_readiness_verdict(
        profile=profile,
        artifact_sync_enabled=True,
        checkpoint_resume_enabled=True,
        manifest_enabled=True,
        runtime_profile_valid=bool(validation["runtime_profile_valid"]),
        gpu_price_registered=bool(validation["gpu_price_registered"]),
        backup_verification_dry_run_passed=True,
    )

    assert validation["registered_hourly_price"] == 3.29
    assert readiness["verdict"] == "READY_FOR_H100_CALIBRATION"
    assert readiness["ready"] is True
    assert cast(list[str], readiness["failed_checks"]) == []


def test_calibration_readiness_still_blocks_without_backup_dry_run() -> None:
    profile = load_runpod_calibration_profiles()["A100_SXM_CALIBRATION"]
    validation = validate_calibration_profile(profile)
    readiness = calibration_readiness_verdict(
        profile=profile,
        artifact_sync_enabled=True,
        checkpoint_resume_enabled=True,
        manifest_enabled=True,
        runtime_profile_valid=bool(validation["runtime_profile_valid"]),
        gpu_price_registered=bool(validation["gpu_price_registered"]),
        backup_verification_dry_run_passed=False,
    )

    assert readiness["verdict"] == "CALIBRATION_NOT_READY"
    assert readiness["ready"] is False
    assert "backup_verification_dry_run_passed" in cast(list[str], readiness["failed_checks"])


def test_calibration_readiness_can_emit_ready_verdict_with_all_gates() -> None:
    profile = load_runpod_calibration_profiles()["L40S_CALIBRATION"]
    readiness = calibration_readiness_verdict(
        profile=profile,
        artifact_sync_enabled=True,
        checkpoint_resume_enabled=True,
        manifest_enabled=True,
        runtime_profile_valid=True,
        gpu_price_registered=True,
        backup_verification_dry_run_passed=True,
    )

    assert readiness["ready"] is True
    assert readiness["verdict"] == "READY_FOR_L40S_CALIBRATION"


def test_full_run_readiness_reports_calibration_not_ready(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    for name in (
        "models.yaml",
        "runtime_engines.yaml",
        "gpu_prices.yaml",
        "runpod_calibration_profiles.yaml",
    ):
        source = Path("configs") / name
        target = tmp_path / "configs" / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    for relative in (
        "src/inference_bench/artifact_sync.py",
        "src/inference_bench/checkpoint_resume.py",
        "src/inference_bench/run_manifest.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("present\n", encoding="utf-8")

    report = build_full_run_readiness_audit(repo_root=tmp_path)

    assert report["runpod_calibration_readiness"]["status"] == "CALIBRATION_NOT_READY"
    assert report["runpod_calibration_readiness"]["ready_profiles"] == []
