from __future__ import annotations

from pathlib import Path

from inference_bench.config import load_yaml_file

EXPERIMENT_PATH = Path("configs/experiments/a2_remote_rtx3070_sglang_smoke.yaml")


def test_a2_sglang_experiment_matrix_is_frozen_and_bounded() -> None:
    config = load_yaml_file(EXPERIMENT_PATH)

    assert config["experiment_id"] == "a2_remote_rtx3070_sglang_smoke"
    assert config["hardware"] == "remote_rtx3070"
    assert config["engine"] == "sglang"
    assert config["model_alias"] == "model1_0_5b"
    assert config["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config["memory_mode"] == "mm2_hybrid_top5"
    assert config["ablation_mode"] == "prompt_plus_metadata"
    assert config["prompts_per_vertical"] == 10
    assert config["total_prompts"] == 50
    assert config["concurrency"] == 1
    assert config["streaming"] is True
    assert config["temperature"] == 0.0
    assert config["max_new_tokens"] == 128
    assert config["retrieval_source"] == "promoted_retrieval_baseline"
    assert config["output_contract"] == "enabled"
    assert config["base_url"] == "http://localhost:30000/v1"
    assert config["verticals"] == [
        "airline",
        "healthcare_admin",
        "retail",
        "finance",
        "research_ai",
    ]


def test_a2_sglang_config_requires_no_live_gpu() -> None:
    assert EXPERIMENT_PATH.exists()
