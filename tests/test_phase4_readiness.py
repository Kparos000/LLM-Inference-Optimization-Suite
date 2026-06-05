import csv
import json
from pathlib import Path

from inference_bench.phase4_readiness import (
    build_phase4_readiness_report,
    inspect_phase4_readiness,
)


def test_readiness_checker_reports_promoted_retrieval() -> None:
    report, rows = inspect_phase4_readiness(repo_root=Path.cwd())
    by_area = {row.area: row for row in rows}

    assert report["promoted_retrieval"]["retrieval_promotion_status"] == "PROMOTED"
    assert report["promoted_retrieval"]["retrieval_slo_status"] == "PASS"
    assert by_area["promoted_retrieval"].status == "PASS"
    assert by_area["retrieval_slo"].status == "PASS"
    assert by_area["latency_cost_resource_metrics"].status == "NOT_AVAILABLE"
    assert report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert report["no_paid_api_call_triggered"] is True


def test_readiness_report_writes_json_and_csv(tmp_path: Path) -> None:
    report = build_phase4_readiness_report(
        repo_root=Path.cwd(),
        output_root=tmp_path,
    )
    report_path = tmp_path / "phase4_readiness_report.json"
    summary_path = tmp_path / "phase4_readiness_summary.csv"

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    with summary_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert report["pre_gpu_plumbing_ready"] is True
    assert loaded["overall_status"] == "PRE_GPU_PLUMBING_READY"
    assert any(row["area"] == "gpu_cost_values" for row in rows)
    assert all(row["status"] != "FAIL" for row in rows)
