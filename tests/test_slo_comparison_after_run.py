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

    assert report["status"] == "SLO_COMPARISON_COMPLETE"
    assert report["deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"
    assert report["benchmark_execution_verdict"] == "COMPLETED"
    assert report["optimization_needed_verdict"] == "OPTIMIZATION_NEEDED"
    assert len(report["config_slo_results"]) == 25


def test_slo_summary_has_one_row_per_config_with_no_applied_optimizations() -> None:
    with Path("results/processed/controlled_final_simulation_slo_summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 25
    assert {row["status"] for row in rows} == {"COMPLETED"}
    assert all(int(row["failed_slos"]) > 0 for row in rows)
    assert "generation_contract_prompt_repair" in {
        item
        for row in rows
        for item in row["recommended_optimization_candidates"].split(";")
        if item
    }
