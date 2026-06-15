from inference_bench.optimization_catalog import (
    APPLICATION_METHODS,
    IMPLEMENTATION_STATUSES,
    load_optimization_catalog,
)

EXPECTED_OPTIMIZATIONS = {
    "reduce_context_tokens",
    "enable_context_compression",
    "reduce_top_k",
    "improve_context_ordering",
    "improve_evidence_formatting",
    "reduce_max_new_tokens",
    "prompt_contract_repair",
    "repair_retrieval",
    "switch_to_hybrid_retrieval",
    "improve_reranking",
    "improve_chunking",
    "improve_embedding_model",
    "use_qdrant_vector_retrieval",
    "tune_retrieval_top_k",
    "use_stronger_model",
    "use_smaller_model",
    "use_distilled_model",
    "use_quantized_model",
    "enable_int8_quantization",
    "enable_awq_int4",
    "enable_gptq_int4",
    "enable_fp8_where_supported",
    "switch_engine_to_vllm",
    "switch_engine_to_sglang",
    "switch_engine_to_tensorrt_llm",
    "use_pagedattention_capable_engine",
    "enable_continuous_batching",
    "tune_scheduler",
    "tune_max_num_seqs",
    "tune_max_model_len",
    "tune_gpu_memory_utilization",
    "enable_prefix_cache",
    "tune_kv_cache",
    "enable_cuda_graphs",
    "enable_speculative_decoding",
    "use_flashattention_where_available",
    "use_flashinfer_where_available",
    "increase_concurrency",
    "decrease_concurrency",
    "concurrency_sweep",
    "admission_control",
    "request_queue_tuning",
    "route_long_and_short_requests_separately",
    "increase_gpu_size",
    "move_to_runpod_rtx4090",
    "move_to_runpod_l40s",
    "move_to_runpod_a100",
    "move_to_runpod_h100",
    "use_tensor_parallelism",
    "use_pipeline_parallelism",
    "use_data_parallelism",
    "prefill_decode_disaggregation",
    "use_mm4_agentic_repair",
    "enable_bounded_citation_repair",
    "enable_escalation_path",
    "reduce_agent_retrieval_rounds",
    "cap_agent_tool_calls",
}


def test_optimization_catalog_has_required_coverage() -> None:
    catalog = load_optimization_catalog()

    assert EXPECTED_OPTIMIZATIONS <= set(catalog)
    assert all(item.implementation_status in IMPLEMENTATION_STATUSES for item in catalog.values())
    assert all(item.application_method in APPLICATION_METHODS for item in catalog.values())


def test_pagedattention_is_vllm_engine_builtin() -> None:
    optimization = load_optimization_catalog()["use_pagedattention_capable_engine"]

    assert optimization.implementation_status == "engine_builtin"
    assert optimization.required_engines == ("vllm",)
    assert optimization.current_project_support == "active_when_vllm_is_selected"
