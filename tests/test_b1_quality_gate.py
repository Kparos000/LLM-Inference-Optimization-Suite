from __future__ import annotations

from pathlib import Path

from inference_bench.b1_quality import (
    build_per_vertical_quality,
    build_quality_gate,
)
from inference_bench.config import load_yaml_file


def test_b1_experiment_config_is_frozen_and_bounded() -> None:
    config = load_yaml_file(
        Path("configs/experiments/b1_remote_rtx3070_vllm_1_5b_quality_smoke.yaml")
    )

    assert config["hardware"] == "remote_rtx3070"
    assert config["engine"] == "vllm"
    assert config["model_alias"] == "model2_1_5b"
    assert config["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["memory_mode"] == "mm2_hybrid_top5"
    assert config["prompts_per_vertical"] == 20
    assert config["total_prompts"] == 100
    assert config["concurrency"] == 1
    assert config["optional_concurrency_sweep"] == [2, 4]
    assert config["streaming"] is True
    assert config["temperature"] == 0.0
    assert config["max_new_tokens"] == 128
    assert config["retrieval_source"] == "promoted_retrieval_baseline"
    assert config["output_contract"] == "enabled"


def test_b1_quality_gate_passes_only_when_every_threshold_passes() -> None:
    gate = build_quality_gate(
        {
            "json_valid_rate": 0.95,
            "generation_contract_valid_rate": 0.85,
            "evidence_match_rate": 0.60,
            "grounded_rate": 0.60,
            "safety_violation_count": 0,
        }
    )

    assert gate["status"] == "PASSED"
    assert gate["passed"] is True
    assert gate["failed_metrics"] == []


def test_b1_quality_gate_reports_missing_and_failed_metrics_honestly() -> None:
    gate = build_quality_gate(
        {
            "json_valid_rate": 0.99,
            "generation_contract_valid_rate": 0.84,
            "evidence_match_rate": 0.61,
            "safety_violation_count": 1,
        }
    )

    assert gate["status"] == "QUALITY_BLOCKED"
    assert gate["failed_metrics"] == [
        "generation_contract_valid_rate",
        "grounded_rate",
        "safety_violation_count",
    ]
    checks = gate["checks"]
    assert isinstance(checks, dict)
    assert checks["grounded_rate"]["reason"] == "metric_missing"


def test_per_vertical_quality_uses_unchanged_evaluator_fields() -> None:
    evaluation_rows = [
        {
            "prompt_id": "finance-1",
            "vertical": "finance",
            "json_validity": True,
            "generation_contract_valid": True,
            "evidence_id_presence": True,
            "evidence_match": False,
            "groundedness": False,
            "safety_violation": False,
        },
        {
            "prompt_id": "finance-2",
            "vertical": "finance",
            "json_validity": True,
            "generation_contract_valid": False,
            "evidence_id_presence": False,
            "evidence_match": False,
            "groundedness": False,
            "safety_violation": True,
        },
    ]
    result_rows = [
        {"prompt_id": "finance-1", "truncation_detected": False},
        {"prompt_id": "finance-2", "truncation_detected": True},
    ]

    rows = build_per_vertical_quality(
        evaluation_rows,
        result_rows,
        verticals=["finance"],
    )

    assert rows[0]["json_valid_rate"] == 1.0
    assert rows[0]["generation_contract_valid_rate"] == 0.5
    assert rows[0]["evidence_match_rate"] == 0.0
    assert rows[0]["safety_violation_count"] == 1
    assert rows[0]["truncation_rate"] == 0.5
