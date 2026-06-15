from inference_bench.slo_profiles import resolve_slo_profile, select_slos


def test_users_can_enable_and_disable_slo_groups() -> None:
    profile = resolve_slo_profile(
        enabled_groups=["quality", "latency", "resource"],
        disabled_groups=["latency"],
        priority_mode="quality_first",
    )

    assert profile.enabled_groups == ("quality", "resource")
    assert profile.disabled_groups == ("latency",)
    assert profile.group_weights["quality"] > profile.group_weights["resource"]


def test_mm0_marks_retrieval_not_applicable() -> None:
    profile = resolve_slo_profile(enabled_groups=["retrieval"])
    selected = select_slos(
        profile,
        vertical="airline",
        memory_mode="mm0_no_context",
        engine="huggingface",
    )

    assert selected
    assert all(not item.applicable for item in selected)
    assert all("mm0_no_context" in str(item.applicability_reason) for item in selected)


def test_mm3_applies_compression_slos() -> None:
    profile = resolve_slo_profile(enabled_groups=["compression"])
    selected = select_slos(
        profile,
        vertical="retail",
        memory_mode="mm3_compressed_hybrid_top5",
        engine="vllm",
    )

    assert {item.metric_name for item in selected} == {
        "compression_token_reduction_pct_min",
        "compression_recall_loss_max",
    }
    assert all(item.applicable for item in selected)


def test_mm4_applies_agentic_trace_slos() -> None:
    profile = resolve_slo_profile(enabled_groups=["agentic_trace"])
    selected = select_slos(
        profile,
        vertical="finance",
        memory_mode="mm4_bounded_agentic",
        engine="vllm",
    )

    assert selected
    assert all(item.applicable for item in selected)
    assert "tool_calls_max" in {item.metric_name for item in selected}


def test_cost_and_resource_applicability_are_conditional() -> None:
    profile = resolve_slo_profile(enabled_groups=["api_cost", "gpu_cost", "resource"])
    selected = select_slos(
        profile,
        vertical="airline",
        memory_mode="mm2_hybrid_top5",
        engine="vllm",
        telemetry_available=False,
        gpu_hourly_price=None,
    )

    assert all(not item.applicable for item in selected)
