from inference_bench.phase4_readiness import load_backend_matrix


def test_backend_matrix_loads_required_backends() -> None:
    backends = load_backend_matrix()

    assert set(backends) == {
        "hf_local",
        "hf_inference_provider_api",
        "openai_compatible_vllm",
        "sglang_openai_compatible_future",
        "tensorrt_llm_future",
    }
    assert backends["hf_local"].status == "ready"
    assert backends["sglang_openai_compatible_future"].status == "future"
    assert backends["tensorrt_llm_future"].status == "future"


def test_vllm_backend_requires_server_and_gpu() -> None:
    backend = load_backend_matrix()["openai_compatible_vllm"]

    assert backend.requires_server is True
    assert backend.requires_gpu is True
    assert backend.supports_concurrency is True
    assert backend.cost_model == "gpu_infra"
    assert backend.status == "dry_run_ready"


def test_hf_provider_backend_uses_api_token_cost_model() -> None:
    backend = load_backend_matrix()["hf_inference_provider_api"]

    assert backend.requires_server is False
    assert backend.requires_gpu is False
    assert backend.supports_streaming is True
    assert backend.cost_model == "api_token"
