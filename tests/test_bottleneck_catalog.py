from inference_bench.bottleneck_catalog import BOTTLENECK_FIELDS, load_bottleneck_catalog

EXPECTED_BOTTLENECKS = {
    "low_groundedness",
    "low_evidence_match",
    "low_citation_accuracy",
    "low_contract_validity",
    "low_json_validity",
    "safety_violations",
    "high_truncation_rate",
    "insufficient_evidence_misuse",
    "low_candidate_recall",
    "low_final_recall",
    "low_mrr",
    "high_retrieval_latency",
    "high_input_tokens",
    "excessive_context_tokens",
    "compression_recall_loss",
    "poor_context_ordering",
    "weak_context_formatting",
    "high_ttft_p50",
    "high_ttft_p95",
    "high_ttft_p99",
    "high_tpot",
    "high_itl_p95",
    "high_itl_p99",
    "high_e2e_p95",
    "high_e2e_p99",
    "high_queue_delay",
    "high_prefill_time",
    "high_decode_time",
    "low_requests_per_second",
    "low_tokens_per_second",
    "low_successful_requests_per_second",
    "low_gpu_utilization",
    "high_gpu_utilization_with_bad_latency",
    "high_gpu_memory_utilization",
    "oom",
    "low_vram_headroom",
    "high_cpu_utilization",
    "high_ram_usage",
    "high_power_draw",
    "thermal_pressure",
    "high_api_cost_per_request",
    "high_gpu_cost_per_request",
    "high_cost_per_successful_answer",
    "high_cost_per_grounded_answer",
    "low_tokens_per_gpu_dollar",
    "engine_not_serving_optimized",
    "no_streaming_metrics",
    "no_gpu_telemetry",
    "no_checkpoint_resume",
    "backend_unavailable",
    "provider_pricing_unavailable",
}


def test_bottleneck_catalog_has_required_coverage() -> None:
    catalog = load_bottleneck_catalog()

    assert EXPECTED_BOTTLENECKS <= set(catalog)
    assert len(BOTTLENECK_FIELDS) == 10


def test_every_bottleneck_has_diagnostic_evidence_and_actions() -> None:
    catalog = load_bottleneck_catalog()

    for definition in catalog.values():
        assert definition.description
        assert definition.required_metrics
        assert definition.trigger_conditions
        assert definition.compatible_optimizations
        assert definition.evidence_fields
