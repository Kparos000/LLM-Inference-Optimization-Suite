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

    assert report["status"] == "SLO_COMPARISON_NOT_RUN_SAFETY_GATED"
    assert report["deployability_verdict"] == "NOT_DEPLOYABLE_SIMULATION_BLOCKED"
    assert report["benchmark_execution_verdict"] == "NOT_READY"
    assert report["optimization_needed_verdict"] == "NOT_EVALUATED"
    assert len(report["config_slo_results"]) == 25


def test_slo_summary_has_one_row_per_config_with_no_applied_optimizations() -> None:
    with Path("results/processed/controlled_final_simulation_slo_summary.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 25
    assert {row["status"] for row in rows} == {"NOT_RUN"}
    assert {row["bottleneck_category"] for row in rows} == {"safety_gate"}
    assert {row["recommended_optimization_candidates"] for row in rows} == {""}
