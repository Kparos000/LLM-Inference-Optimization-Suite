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
