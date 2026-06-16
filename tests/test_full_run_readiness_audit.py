from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from inference_bench.full_run_readiness_audit import (
    build_full_run_readiness_audit,
    partial_run_completion_check,
    telemetry_availability_check,
)


def _write(root: Path, relative_path: str, text: str = "") -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _check_by_name(report: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [check for check in report["checks"] if check["name"] == name]
    assert matches
    return cast(dict[str, Any], matches[-1])


def test_readiness_audit_detects_checkpoint_resume_support(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/inference_bench/runners/openai_load_runner.py",
        "checkpoint_path = 'x'\ncompleted_prompt_ids = set()\n",
    )

    report = build_full_run_readiness_audit(repo_root=tmp_path)

    assert _check_by_name(report, "checkpoint_resume_supported")["status"] == "PASS"
    assert _check_by_name(report, "completed_prompt_ids_tracked")["status"] == "PASS"


def test_readiness_audit_reports_missing_checkpoint_resume_as_gap(tmp_path: Path) -> None:
    _write(tmp_path, "src/inference_bench/runners/openai_load_runner.py", "pass\n")

    report = build_full_run_readiness_audit(repo_root=tmp_path)

    checkpoint = _check_by_name(report, "checkpoint_resume_supported")
    assert checkpoint["status"] == "GAP"
    assert checkpoint["blocking"] is False


def test_partial_completed_run_is_blocking_failure() -> None:
    check = partial_run_completion_check(
        expected_count=500,
        observed_count=499,
        manifest_status="completed",
    )

    assert check["status"] == "FAIL"
    assert check["blocking"] is True


def test_partial_running_run_is_not_marked_failed() -> None:
    check = partial_run_completion_check(
        expected_count=500,
        observed_count=499,
        manifest_status="running",
    )

    assert check["status"] == "PASS"
    assert check["blocking"] is False


def test_missing_telemetry_is_unavailable_not_failure() -> None:
    check = telemetry_availability_check(None)

    assert check["status"] == "UNAVAILABLE"
    assert check["blocking"] is False


def test_gpu_cost_requires_hourly_price_gap_when_missing(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "results/processed/b6_runtime_projection_report.json",
        '{"runpod_gpu_projections":{"h100":{"hourly_price_usd":null}}}',
    )

    report = build_full_run_readiness_audit(repo_root=tmp_path)
    price = _check_by_name(report, "runpod_hourly_price_configured")

    assert price["status"] == "GAP"
    assert price["blocking"] is False
