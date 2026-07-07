from __future__ import annotations

import pytest

from inference_bench.serving_profiles import load_serving_profiles, select_serving_profile


def test_b7r1_safe_serving_profile_validates() -> None:
    profiles = load_serving_profiles()
    safe = profiles["remote_rtx3070_qwen3b_safe_v1"]

    assert safe.status == "ready"
    assert safe.model_alias == "model2_3b"
    assert safe.engine == "vllm"
    assert safe.hardware == "remote_rtx3070"
    assert safe.gpu_memory_utilization <= 0.82
    assert safe.max_model_len == 3584
    assert safe.max_num_seqs == 1
    assert safe.max_num_batched_tokens <= safe.max_model_len
    assert "--enforce-eager" in safe.vllm_server_args()
    assert "--disable-custom-all-reduce" in safe.vllm_server_args()


def test_b7_baseline_profile_is_documented_as_unstable_and_not_live_selectable() -> None:
    profiles = load_serving_profiles()
    baseline = profiles["remote_rtx3070_qwen3b_baseline_b7"]

    assert baseline.status == "unstable_observed"
    assert baseline.live_run_allowed is False
    with pytest.raises(ValueError, match="not live-run ready"):
        select_serving_profile("remote_rtx3070_qwen3b_baseline_b7", live_run=True)


def test_select_safe_profile_for_live_run() -> None:
    selected = select_serving_profile("remote_rtx3070_qwen3b_safe_v1", live_run=True)

    assert selected.profile_id == "remote_rtx3070_qwen3b_safe_v1"


def test_a100_sxm_sglang_qwen7b_profile_validates() -> None:
    profiles = load_serving_profiles()
    profile = profiles["a100_sxm_qwen7b_sglang_final_sim_v1"]

    assert profile.status == "ready"
    assert profile.engine == "sglang"
    assert profile.model_alias == "model3_7b"
    assert profile.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert profile.hardware == "a100_sxm_80gb"
    assert profile.backend_type == "self_hosted_gpu"
    assert profile.gpu_memory_utilization == 0.90
    assert profile.max_model_len == 4096
    assert profile.max_num_seqs == 32
    assert profile.max_num_batched_tokens == 8192
    assert profile.live_run_allowed is True
    assert "python -m sglang.launch_server" in profile.notes
    with pytest.raises(ValueError, match="only vLLM"):
        profile.vllm_server_args()
