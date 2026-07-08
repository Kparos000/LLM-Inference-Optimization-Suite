from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType


def _load_runner_module() -> ModuleType:
    path = Path("scripts/phase4/run_controlled_final_simulation.py")
    spec = importlib.util.spec_from_file_location("controlled_final_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


def test_slo_report_records_safety_gate_status_and_verdicts() -> None:
    report = json.loads(
        Path("results/processed/controlled_final_simulation_slo_report_fixed.json").read_text(
            encoding="utf-8"
        )
    )

    if report["status"] == "SLO_COMPARISON_NOT_RUN_SAFETY_GATED":
        assert report["overall_deployability_verdict"] == "NOT_DEPLOYABLE_SIMULATION_BLOCKED"
        assert report["deployability_verdict"] == "NOT_DEPLOYABLE_SIMULATION_BLOCKED"
        assert report["benchmark_execution_verdict"] == "NOT_READY"
        assert report["optimization_needed_verdict"] == "NOT_EVALUATED"
        assert len(report["config_slo_results"]) == 25
        return

    assert report["status"] == "SLO_COMPARISON_COMPLETE"
    assert report["benchmark_execution_verdict"] == "COMPLETED"
    assert report["runtime_slo_verdict"] == "PASS"
    assert report["quality_slo_verdict"] == "FAIL"
    assert report["safety_slo_verdict"] == "FAIL"
    assert report["overall_deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"
    assert report["deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"
    assert report["optimization_needed_verdict"] == "OPTIMIZATION_NEEDED"
    assert len(report["config_slo_results"]) == 25
    assert any(
        row["scope"] == "mm0_no_context" and row["mode_type"] == "no_context_ablation"
        for row in report["aggregate_slo_results"]
    )
    assert any(
        row["scope"] == "mm4_bounded_agentic" and row["mode_type"] == "agentic_workflow"
        for row in report["aggregate_slo_results"]
    )


def test_slo_summary_has_one_row_per_config_with_no_applied_optimizations() -> None:
    with Path("results/processed/controlled_final_simulation_slo_summary_fixed.csv").open(
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
    assert any(count > 0 for count in failed_slo_counts)
    assert "final_answer_safety_boundary_repair" in {
        item
        for row in rows
        for item in row["recommended_optimization_candidates"].split(";")
        if item
    }
    assert {row["slo_scope"] for row in rows} == {
        "deployability_contextual",
        "no_context_ablation",
    }


def test_fixed_slo_re_score_verdict_exists() -> None:
    verdict = json.loads(
        Path("results/processed/controlled_final_simulation_verdict_fixed.json").read_text(
            encoding="utf-8"
        )
    )
    assert verdict["benchmark_execution_verdict"] == "COMPLETED"
    assert verdict["overall_deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"
    assert verdict["final_main_10k_rerun_needed"] is False
    assert verdict["optimization_can_begin"] is True


def test_slo_fails_when_quality_or_safety_fails() -> None:
    report = runner._build_slo_report(
        run_id="test",
        gates={},
        config_slo_rows=[
            {
                "failed_metric_family": "",
                "failed_slos": 0,
            }
        ],
        quality_group_rows=[
            {
                "scope": "contextual_all",
                "failed_metric_family": "evidence_match;groundedness;safety",
            }
        ],
        cost_report={},
        total_requests_planned=100,
        total_requests_completed=100,
        total_requests_failed=0,
    )
    assert report["runtime_slo_verdict"] == "PASS"
    assert report["quality_slo_verdict"] == "FAIL"
    assert report["safety_slo_verdict"] == "FAIL"
    assert report["benchmark_execution_verdict"] == "COMPLETED"
    assert report["overall_deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"


def test_progress_logging_emits_every_100_requests(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    state = runner.ProgressState(
        total_requests=200,
        completed_requests=99,
        success_count=99,
        failure_count=0,
        started_at_monotonic=time.monotonic() - 10,
        hourly_price=1.49,
        checkpoint_path="checkpoint.json",
        progress_log_path=str(progress_path),
    )
    config = runner.ConfigSpec(
        config_id="cfg",
        model_alias="model3_7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        backend_type="self_hosted_gpu",
        engine="vllm",
        runtime="vllm",
        memory_mode="mm2_hybrid_top5",
        concurrency=16,
    )
    runner._record_progress(progress=state, config=config, result={"success": True}, force=False)
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["completed_requests"] == 100
    assert rows[0]["remaining_requests"] == 100
    assert rows[0]["current_config_id"] == "cfg"
