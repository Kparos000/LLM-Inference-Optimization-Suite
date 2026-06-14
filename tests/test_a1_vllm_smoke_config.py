from __future__ import annotations

from pathlib import Path

from inference_bench.config import load_yaml_file

HARDWARE_PATH = Path("configs/hardware/remote_rtx3070.yaml")
EXPERIMENT_PATH = Path("configs/experiments/a1_remote_rtx3070_vllm_smoke.yaml")


def test_remote_rtx3070_hardware_config_loads() -> None:
    config = load_yaml_file(HARDWARE_PATH)

    assert config["hardware_alias"] == "remote_rtx3070"
    assert config["gpu_name"] == "NVIDIA GeForce RTX 3070"
    assert config["vram_gb"] == 8
    assert config["docker_gpu_runtime_verified"] is True
    assert config["access_method"] == "ssh_over_tailscale"


def test_a1_experiment_matrix_is_frozen_and_bounded() -> None:
    config = load_yaml_file(EXPERIMENT_PATH)

    assert config["hardware"] == "remote_rtx3070"
    assert config["engine"] == "vllm"
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
    assert config["verticals"] == [
        "airline",
        "healthcare_admin",
        "retail",
        "finance",
        "research_ai",
    ]
