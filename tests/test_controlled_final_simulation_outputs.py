from __future__ import annotations

import json
from pathlib import Path


def _report() -> dict[str, object]:
    path = Path("results/processed/controlled_final_simulation_eval_report.json")
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_controlled_final_simulation_outputs_exist_after_safety_gate_run() -> None:
    expected = [
        "data/generated/phase4/controlled_final_simulation_100_per_vertical_matrix.jsonl",
        "results/raw/controlled_final_simulation_results.jsonl",
        "results/raw/controlled_final_simulation_manifest.json",
        "results/raw/controlled_final_simulation_gpu_telemetry.jsonl",
        "results/processed/controlled_final_simulation_eval_report.json",
        "results/processed/controlled_final_simulation_eval_summary.csv",
        "results/processed/controlled_final_simulation_engine_comparison.csv",
        "results/processed/controlled_final_simulation_memory_mode_comparison.csv",
        "results/processed/controlled_final_simulation_concurrency_comparison.csv",
        "results/processed/controlled_final_simulation_api_track_comparison.csv",
        "results/processed/controlled_final_simulation_slo_report.json",
        "results/processed/controlled_final_simulation_slo_summary.csv",
        "results/processed/controlled_final_simulation_cost_report.json",
        "results/processed/controlled_final_simulation_artifact_sync_report.json",
    ]

    for path in expected:
        assert Path(path).exists(), path


def test_controlled_final_simulation_report_records_blocked_smoke_without_fake_runs() -> None:
    report = _report()

    assert report["status"] == "CONTROLLED_FINAL_SIMULATION_BLOCKED_BY_SAFETY_GATES"
    assert report["total_requests_planned"] == 15_000
    assert report["total_requests_attempted"] == 0
    assert report["configs_completed"] == 0
    assert report["configs_failed"] == 30
    assert report["vllm_ran"] is False
    assert report["sglang_ran"] is False
    assert report["api_route_ran"] is False
    assert report["mm4_ran"] is False
    assert report["final_10000_prompt_experiment_allowed"] is False


def test_controlled_final_simulation_cost_report_separates_api_and_gpu_costs() -> None:
    payload = json.loads(
        Path("results/processed/controlled_final_simulation_cost_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["status"] == "COST_NOT_MEASURED_SAFETY_GATED"
    assert payload["api_cost_usd"] == 0.0
    assert payload["gpu_cost_usd"] == 0.0
    assert payload["self_hosted_gpu_hourly_price_usd"] == 1.49
