from __future__ import annotations

from inference_bench.config import load_project_config


def test_final_active_production_aliases_resolve() -> None:
    config = load_project_config()

    expected = {
        "model1_0_5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "model2_3b": "Qwen/Qwen2.5-3B-Instruct",
        "model3_7b": "Qwen/Qwen2.5-7B-Instruct",
        "model4_32b": "Qwen/Qwen2.5-32B-Instruct",
        "model5_gated": "mistralai/ministral-3b-2512",
        "model6_gated": "meta-llama/Llama-3.1-8B-Instruct",
        "model7_gated": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    }

    active_aliases = [
        alias
        for alias in config.model_aliases
        if alias.startswith("model") and alias not in {"model2_1_5b", "model7_large_placeholder"}
    ][:7]
    assert active_aliases == list(expected)
    for alias, model_id in expected.items():
        assert config.resolve_model_config(alias).model_id == model_id


def test_model6_gated_remains_llama_3_1_8b() -> None:
    config = load_project_config()

    assert config.resolve_model_key("model6_gated") == "llama_3_1_8b_instruct_api"
    assert config.resolve_model_config("model6_gated").model_id == (
        "meta-llama/Llama-3.1-8B-Instruct"
    )


def test_deprecated_aliases_preserve_historical_compatibility() -> None:
    config = load_project_config()

    assert config.resolve_model_config("model2_1_5b").model_id == ("Qwen/Qwen2.5-1.5B-Instruct")
    assert config.resolve_model_config("model7_large_placeholder").model_id == (
        "placeholder/large-model"
    )
    assert config.resolve_model_config("old_model5_llama_3_2_3b").model_id == (
        "meta-llama/Llama-3.2-3B-Instruct"
    )


def test_model_compatibility_metadata_includes_expected_tracks() -> None:
    config = load_project_config()

    qwen3b = config.resolve_model_config("model2_3b")
    assert qwen3b.execution_target == "local_or_self_hosted"
    assert {"huggingface", "vllm_optional", "sglang_optional", "tensorrt_llm_optional"}.issubset(
        set(qwen3b.allowed_backends)
    )

    model7 = config.resolve_model_config("model7_gated")
    assert model7.execution_target == "hf_inference_provider_api"
    assert {
        "hf_inference_provider",
        "vllm_optional",
        "sglang_optional",
        "tensorrt_llm_optional",
    }.issubset(set(model7.allowed_backends))
