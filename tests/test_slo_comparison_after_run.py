from __future__ import annotations

import csv
import json
from pathlib import Path


def test_slo_report_records_safety_gate_status_and_verdicts() -> None:
    report = json.loads(
        Path("results/processed/controlled_final_simulation_slo_report.json").read_text(
            encoding="utf-8"
        )
    )

    if report["status"] == "SLO_COMPARISON_NOT_RUN_SAFETY_GATED":
        assert report["deployability_verdict"] == "NOT_DEPLOYABLE_SIMULATION_BLOCKED"
        assert report["benchmark_execution_verdict"] == "NOT_READY"
        assert report["optimization_needed_verdict"] == "NOT_EVALUATED"
        assert len(report["config_slo_results"]) == 25
        return

    assert report["status"] == "SLO_COMPARISON_COMPLETE"
    assert report["benchmark_execution_verdict"] == "COMPLETED"
    assert len(report["config_slo_results"]) == 25
    failed_slo_total = sum(int(result["failed_slos"]) for result in report["config_slo_results"])
    if failed_slo_total:
        assert report["deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"
        assert report["optimization_needed_verdict"] == "OPTIMIZATION_NEEDED"
    else:
        assert report["deployability_verdict"] == "DEPLOYABLE_BASELINE"
        assert report["optimization_needed_verdict"] == "NO_OPTIMIZATION_REQUIRED"


def test_slo_summary_has_one_row_per_config_with_no_applied_optimizations() -> None:
    with Path("results/processed/controlled_final_simulation_slo_summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 25
    if {row["status"] for row in rows} == {"NOT_RUN"}:
        assert all(row["requests_attempted"] == "0" for row in rows)
        assert all(row["requests_completed"] == "0" for row in rows)
        assert all(row["recommended_optimization_candidates"] == "" for row in rows)
        return

    assert {row["status"] for row in rows} == {"COMPLETED"}
    failed_slo_counts = [int(row["failed_slos"]) for row in rows]
    if any(failed_slo_counts):
        assert all(count > 0 for count in failed_slo_counts)
        assert "generation_contract_prompt_repair" in {
            item
            for row in rows
            for item in row["recommended_optimization_candidates"].split(";")
            if item
        }
        return

    assert all(count == 0 for count in failed_slo_counts)
    assert not {
        item
        for row in rows
        for item in row["recommended_optimization_candidates"].split(";")
        if item
    }
