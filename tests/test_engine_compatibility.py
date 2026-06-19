from __future__ import annotations

import pytest

from inference_bench.runtime_registry import select_runtime_for_model


def test_api_models_select_api_provider_route_not_self_hosted_gpu() -> None:
    model5 = select_runtime_for_model(
        model_alias="model5_gated",
        runtime="api_provider_route",
        live_run=True,
    )
    model6 = select_runtime_for_model(
        model_alias="model6_gated",
        runtime="api_provider_route",
        live_run=True,
    )

    assert model5.backend_type == "api_provider"
    assert model5.backend_route == "openrouter"
    assert model5.hardware_type == "provider_managed"
    assert model6.backend_route == "hf_inference_provider"
    assert model6.provider == "hf_inference_provider"
    with pytest.raises(ValueError, match="not compatible"):
        select_runtime_for_model(
            model_alias="model6_gated",
            runtime="vllm",
            hardware_type="runpod_gpu",
            live_run=True,
        )


def test_open_weight_models_can_use_hf_vllm_and_sglang_when_compatible() -> None:
    hf = select_runtime_for_model(
        model_alias="model2_3b",
        runtime="huggingface_transformers",
        hardware_type="developer_workstation",
        live_run=True,
    )
    vllm = select_runtime_for_model(
        model_alias="model2_3b",
        runtime="vllm",
        hardware_type="remote_rtx3070",
        live_run=True,
    )
    sglang = select_runtime_for_model(
        model_alias="model2_3b",
        runtime="sglang",
        hardware_type="remote_rtx3070",
        live_run=True,
    )

    assert hf.backend_type == "local_compute"
    assert vllm.backend_type == "self_hosted_gpu"
    assert sglang.backend_type == "self_hosted_gpu"


def test_open_weight_model_cannot_use_api_provider_runtime() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        select_runtime_for_model(
            model_alias="model2_3b",
            runtime="api_provider_route",
            live_run=True,
        )


def test_api_model_cannot_use_tensorrt_even_for_non_live_inspection() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        select_runtime_for_model(
            model_alias="model7_gated",
            runtime="tensorrt_llm",
            hardware_type="runpod_gpu",
            live_run=False,
        )
