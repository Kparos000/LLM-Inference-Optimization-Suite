from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, cast


def _load_builder() -> Callable[..., dict[str, Any]]:
    script = Path("scripts/phase4/prepare_runpod_a100_calibration.py")
    spec = spec_from_file_location("prepare_runpod_a100_calibration", script)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(Callable[..., dict[str, Any]], module.build_a100_calibration_package)


def _copy_required_files(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    for name in (
        "models.yaml",
        "runtime_engines.yaml",
        "gpu_prices.yaml",
        "runpod_calibration_profiles.yaml",
    ):
        shutil.copyfile(Path("configs") / name, root / "configs" / name)
    for relative in (
        "src/inference_bench/artifact_sync.py",
        "src/inference_bench/checkpoint_resume.py",
        "src/inference_bench/run_manifest.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("present\n", encoding="utf-8")
    report_path = root / "results/processed/long_run_recovery_dry_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"backup_verification": {"passed": True}}),
        encoding="utf-8",
    )


def test_a100_calibration_package_writes_manifests_and_blocks_live_without_ssh(
    tmp_path: Path,
) -> None:
    _copy_required_files(tmp_path)
    build_a100_calibration_package = _load_builder()

    report = build_a100_calibration_package(repo_root=tmp_path, env_path=tmp_path / ".env")

    manifest_100 = tmp_path / "results/processed/a100_sxm_calibration_manifest_100.json"
    manifest_200 = tmp_path / "results/processed/a100_sxm_calibration_manifest_200.json"
    readiness = tmp_path / "results/processed/a100_sxm_calibration_readiness_report.json"
    payload = json.loads(readiness.read_text(encoding="utf-8"))

    assert manifest_100.exists()
    assert manifest_200.exists()
    assert payload["live_runpod_calibration_allowed"] is False
    assert payload["live_runpod_blocked_reason"] == "RUNPOD_SSH_HOST_missing"
    assert payload["profile"]["gpu_name"] == "A100 SXM 80GB"
    assert payload["profile"]["hourly_price"] == 1.49
    assert payload["readiness"]["verdict"] == "READY_FOR_A100_CALIBRATION"
    assert payload["cost_examples_usd"]["one_hour"] == 1.49
    assert report["manifest_paths"]["100"] == (
        "results\\processed\\a100_sxm_calibration_manifest_100.json"
        if "\\" in str(readiness)
        else "results/processed/a100_sxm_calibration_manifest_100.json"
    )


def test_a100_calibration_package_allows_live_only_with_runpod_ssh(tmp_path: Path) -> None:
    _copy_required_files(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("RUNPOD_SSH_HOST=example-runpod\n", encoding="utf-8")
    build_a100_calibration_package = _load_builder()

    report = build_a100_calibration_package(repo_root=tmp_path, env_path=env_file)

    assert report["live_runpod_calibration_allowed"] is True
    assert report["live_runpod_blocked_reason"] is None
