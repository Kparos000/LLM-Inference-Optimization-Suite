from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

RUN_ID = "a100_sxm_model2_3b_mm2_c1_200"


def _minimal_report() -> dict[str, Any]:
    return {
        "block": "A100_SXM_CALIBRATION",
        "model_alias": "model2_3b",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "engine": "vllm",
        "hardware": "a100_sxm_80gb",
        "memory_mode": "mm2_hybrid_top5",
        "traffic_profile": "online_low_latency",
        "concurrency": 1,
        "prompt_count": 200,
        "summary": {
            "completed_prompts": 200,
            "request_success_count": 200,
            "request_failure_count": 0,
            "json_valid_rate": 0.99,
            "generation_contract_valid_rate": 0.985,
            "evidence_match_rate": 0.975,
            "grounded_rate": 0.97,
            "requests_per_second": 1.44,
            "aggregate_tokens_per_second": 2189.0,
            "mean_ttft_ms": 52.0,
            "mean_tpot_ms": 6.2,
            "mean_e2e_latency_ms": 675.0,
        },
        "per_vertical_quality": [
            {
                "vertical": vertical,
                "row_count": 40,
                "evidence_match_rate": 0.925,
                "grounded_rate": 0.925,
            }
            for vertical in ("airline", "healthcare_admin", "retail", "finance", "research_ai")
        ],
        "gpu_telemetry_summary": {
            "sample_count": 128,
            "gpu_names": ["NVIDIA A100-SXM4-80GB"],
            "mean_utilization_gpu_percent": 95.8,
            "max_memory_used_mb": 74247.0,
            "mean_power_draw_w": 266.5,
            "max_temperature_c": 53.0,
        },
        "quality_gate": {
            "status": "A100_SXM_200_PROMPT_CALIBRATION_READY",
            "passed": True,
        },
        "cost_estimate": {
            "hourly_price_usd": 1.49,
            "estimated_cost_usd": 0.057,
        },
        "a100_1000_prompt_baseline_allowed": True,
    }


def _report_payload() -> dict[str, Any]:
    path = Path(f"results/processed/{RUN_ID}_eval_report.json")
    if path.exists():
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return _minimal_report()


def test_a100_calibration_report_contract() -> None:
    report = _report_payload()

    assert report["block"] == "A100_SXM_CALIBRATION"
    assert report["model_alias"] == "model2_3b"
    assert report["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert report["engine"] == "vllm"
    assert report["hardware"] == "a100_sxm_80gb"
    assert report["memory_mode"] == "mm2_hybrid_top5"
    assert report["traffic_profile"] == "online_low_latency"
    assert report["concurrency"] == 1
    assert report["prompt_count"] == 200
    assert report["summary"]["completed_prompts"] == 200
    assert report["summary"]["request_success_count"] == 200
    assert report["summary"]["request_failure_count"] == 0
    assert report["quality_gate"]["passed"] is True
    assert report["a100_1000_prompt_baseline_allowed"] is True


def test_a100_calibration_report_has_quality_runtime_gpu_and_cost_fields() -> None:
    report = _report_payload()
    summary = report["summary"]
    telemetry = report["gpu_telemetry_summary"]

    for metric in (
        "json_valid_rate",
        "generation_contract_valid_rate",
        "evidence_match_rate",
        "grounded_rate",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
        "requests_per_second",
        "aggregate_tokens_per_second",
    ):
        assert metric in summary
    assert {row["vertical"] for row in report["per_vertical_quality"]} == {
        "airline",
        "healthcare_admin",
        "retail",
        "finance",
        "research_ai",
    }
    assert telemetry["sample_count"] >= 1
    assert "NVIDIA A100-SXM4-80GB" in telemetry["gpu_names"]
    assert telemetry["max_memory_used_mb"] > 0
    assert report["cost_estimate"]["hourly_price_usd"] == 1.49
    assert report["cost_estimate"]["estimated_cost_usd"] > 0


def test_a100_artifact_paths_remain_generated_outputs() -> None:
    expected = [
        Path(f"data/generated/phase4/{RUN_ID}_runner_input.jsonl"),
        Path(f"results/raw/{RUN_ID}_results.jsonl"),
        Path(f"results/raw/{RUN_ID}_manifest.json"),
        Path(f"results/raw/{RUN_ID}_gpu_telemetry.jsonl"),
        Path(f"results/processed/{RUN_ID}_eval_report.json"),
        Path(f"results/processed/{RUN_ID}_eval_summary.csv"),
        Path(f"results/processed/{RUN_ID}_artifact_sync_report.json"),
        Path(f"results/processed/{RUN_ID}_runtime_projection.json"),
    ]

    for path in expected:
        assert path.as_posix().startswith(("data/generated/", "results/"))
