from __future__ import annotations

import pytest

from inference_bench.runtime_registry import (
    build_engine_compatibility_rows,
    load_runtime_registry,
    select_runtime_for_model,
)


def test_runtime_registry_loads_production_engines() -> None:
    registry = load_runtime_registry()

    assert set(registry) == {
        "huggingface_transformers",
        "vllm",
        "sglang",
        "api_provider_route",
        "tensorrt_llm",
    }
    assert registry["huggingface_transformers"].status == "ready"
    assert registry["vllm"].status == "ready"
    assert registry["sglang"].status == "ready"
    assert registry["api_provider_route"].backend_type == "api_provider"


def test_tensorrt_llm_is_planned_and_not_live_selectable() -> None:
    registry = load_runtime_registry()
    tensorrt = registry["tensorrt_llm"]

    assert tensorrt.status == "planned"
    assert tensorrt.planned_engine is True
    assert tensorrt.smoke_tested is False
    assert tensorrt.live_run_supported is False
    with pytest.raises(ValueError, match="not selectable for live runs"):
        select_runtime_for_model(
            model_alias="model2_3b",
            runtime="tensorrt_llm",
            hardware_type="runpod_gpu",
            live_run=True,
        )


def test_tensorrt_can_only_be_inspected_as_non_live_planned_config() -> None:
    selection = select_runtime_for_model(
        model_alias="model2_3b",
        runtime="tensorrt_llm",
        hardware_type="runpod_gpu",
        live_run=False,
    )

    assert selection.runtime == "tensorrt_llm"
    assert selection.status == "planned"
    assert selection.live_run_allowed is False


def test_engine_compatibility_rows_include_runtime_metadata() -> None:
    rows = build_engine_compatibility_rows()
    model2_vllm = next(
        row for row in rows if row["model_alias"] == "model2_3b" and row["runtime"] == "vllm"
    )
    model7_api = next(
        row
        for row in rows
        if row["model_alias"] == "model7_gated" and row["runtime"] == "api_provider_route"
    )

    assert model2_vllm["compatible"] is True
    assert model2_vllm["backend_type"] == "self_hosted_gpu"
    assert model7_api["compatible"] is True
    assert model7_api["backend_type"] == "api_provider"
    assert model7_api["backend_route"] == "hf_inference_provider"
