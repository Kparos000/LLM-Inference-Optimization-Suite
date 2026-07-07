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
        "data/generated/phase4/controlled_final_simulation_80_per_vertical_matrix.jsonl",
        "results/raw/controlled_final_simulation_results.jsonl",
        "results/raw/controlled_final_simulation_manifest.json",
        "results/raw/controlled_final_simulation_gpu_telemetry.jsonl",
        "results/processed/controlled_final_simulation_eval_report.json",
        "results/processed/controlled_final_simulation_eval_summary.csv",
        "results/processed/controlled_final_simulation_engine_comparison.csv",
        "results/processed/controlled_final_simulation_memory_mode_comparison.csv",
        "results/processed/controlled_final_simulation_concurrency_comparison.csv",
        "results/processed/controlled_final_simulation_api_track_comparison.csv",
        "results/processed/controlled_final_simulation_api_vs_self_hosted_comparison.csv",
        "results/processed/controlled_final_simulation_model_comparison.csv",
        "results/processed/controlled_final_simulation_slo_report.json",
        "results/processed/controlled_final_simulation_slo_summary.csv",
        "results/processed/controlled_final_simulation_cost_report.json",
        "results/processed/controlled_final_simulation_artifact_sync_report.json",
    ]

    for path in expected:
        assert Path(path).exists(), path


def test_controlled_final_simulation_report_records_completed_baseline() -> None:
    report = _report()

    assert report["status"] == "CONTROLLED_FINAL_SIMULATION_COMPLETED"
    assert report["total_requests_planned"] == 10_000
    assert report["total_requests_attempted"] == 10_000
    assert report["total_requests_completed"] == 10_000
    assert report["total_requests_failed"] == 0
    assert report["configs_completed"] == 25
    assert report["configs_failed"] == 0
    assert report["vllm_ran"] is True
    assert report["sglang_ran"] is True
    assert report["api_route_ran"] is True
    assert report["mm4_ran"] is True
    assert report["final_10000_prompt_experiment_allowed"] is True


def test_controlled_final_simulation_report_records_sglang_health_check() -> None:
    report = _report()
    gate_report = report["gate_report"]
    assert isinstance(gate_report, dict)
    checks = gate_report["checks"]
    assert isinstance(checks, dict)
    sglang = checks["sglang_model3_7b"]

    assert sglang["runtime_registry_allows_sglang"] is True
    assert sglang["startup_command"].startswith("python -m sglang.launch_server")
    assert sglang["health_check_url"] == "http://localhost:30000/v1/models"
    health_check = sglang["health_check"]
    assert isinstance(health_check, dict)
    assert health_check["endpoint"] == "http://localhost:30000/v1/models"


def test_controlled_final_simulation_cost_report_separates_api_and_gpu_costs() -> None:
    payload = json.loads(
        Path("results/processed/controlled_final_simulation_cost_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["status"] == "COST_MEASURED"
    assert payload["api_cost_usd"] > 0.0
    assert payload["gpu_cost_usd"] > 0.0
    assert payload["total_cost_usd"] == payload["api_cost_usd"] + payload["gpu_cost_usd"]
    assert payload["self_hosted_gpu_hourly_price_usd"] == 1.49
