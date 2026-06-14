import json
from pathlib import Path
from typing import Any

from inference_bench.controlled_run_readiness import (
    ControlledReadinessCheck,
    inspect_controlled_inference_readiness,
    write_controlled_readiness_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _checks_by_category() -> tuple[
    dict[str, Any],
    dict[str, ControlledReadinessCheck],
]:
    report, checks = inspect_controlled_inference_readiness(REPO_ROOT)
    return report, {check.category: check for check in checks}


def test_audit_detects_retrieval_promotion_and_workload_splits() -> None:
    report, checks = _checks_by_category()

    assert checks["dataset_readiness"].status == "PASS"
    assert checks["workload_control"].status == "PASS"
    assert set(report["workload_materialization"]) == {
        "smoke_500",
        "controlled_2000",
        "final_10000",
    }


def test_audit_detects_telemetry_cost_and_active_bounded_mm4() -> None:
    report, checks = _checks_by_category()

    assert checks["logging_observability"].status == "PASS"
    assert report["cost_schema_ready"] is True
    assert report["live_gpu_cost_ready"] is False
    assert report["mm4_status"] == "active_bounded"
    assert report["mm4_benchmark_ready"] is True


def test_audit_is_not_ready_until_gpu_inputs_are_frozen() -> None:
    report, checks = _checks_by_category()

    assert report["readiness_status"] == "NOT_READY"
    assert checks["gpu_execution_inputs"].status == "FAIL"
    assert checks["cost_controls"].status == "FAIL"
    assert report["no_gpu_call_triggered"] is True
    assert report["no_vllm_call_triggered"] is True
    assert report["no_sglang_call_triggered"] is True


def test_audit_artifacts_are_written_without_inference(tmp_path: Path) -> None:
    report, checks = inspect_controlled_inference_readiness(REPO_ROOT)
    report_path, summary_path = write_controlled_readiness_artifacts(
        output_root=tmp_path,
        report=report,
        checks=checks,
    )

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["no_model_inference_triggered"] is True
    assert "gpu_execution_inputs" in summary_path.read_text(encoding="utf-8")
