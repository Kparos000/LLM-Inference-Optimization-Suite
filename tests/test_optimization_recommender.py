from typing import Any

from inference_bench.slo_diagnosis import diagnose_slos
from inference_bench.slo_profiles import resolve_slo_profile


def _diagnose(
    metrics: dict[str, object],
    *,
    groups: list[str],
    engine: str = "vllm",
    memory_mode: str = "mm2_hybrid_top5",
    concurrency: int = 1,
    telemetry_available: bool = True,
) -> dict[str, Any]:
    return diagnose_slos(
        run_metrics=metrics,
        profile=resolve_slo_profile(enabled_groups=groups),
        experiment_config={"experiment_id": "synthetic", "concurrency": concurrency},
        model_metadata={"model_alias": "model1_0_5b"},
        hardware_profile={
            "hardware_alias": "remote_rtx3070",
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vram_gb": 8,
        },
        engine=engine,
        memory_mode=memory_mode,
        vertical="airline",
        telemetry_available=telemetry_available,
    )


def _recommendation_ids(diagnosis: dict[str, Any]) -> list[str]:
    return [str(item["optimization_id"]) for item in diagnosis["recommended_optimizations"]]


def test_low_gpu_utilization_recommends_sweep_before_hardware() -> None:
    diagnosis = _diagnose(
        {"mean_gpu_utilization_percent": 20.0},
        groups=["resource"],
    )
    ids = _recommendation_ids(diagnosis)

    assert ids[0] == "concurrency_sweep"
    assert ids.index("concurrency_sweep") < ids.index("increase_gpu_size")


def test_high_ttft_with_high_input_tokens_recommends_context_optimization() -> None:
    diagnosis = _diagnose(
        {
            "p95_ttft_ms": 2000.0,
            "high_input_tokens": True,
        },
        groups=["latency"],
    )

    assert diagnosis["primary_recommendation"]["optimization_id"] == ("reduce_context_tokens")


def test_low_quality_with_passed_retrieval_recommends_model_context_mm4_path() -> None:
    diagnosis = _diagnose(
        {
            "candidate_recall_at_20_min": 1.0,
            "candidate_recall_at_50_min": 1.0,
            "final_recall_at_5_min": 1.0,
            "mrr_min": 1.0,
            "grounded_rate": 0.2,
            "evidence_match_rate": 0.2,
        },
        groups=["retrieval", "quality"],
    )
    ids = _recommendation_ids(diagnosis)

    assert ids[0] == "use_stronger_model"
    assert "improve_evidence_formatting" in ids
    assert "use_mm4_agentic_repair" in ids


def test_low_quality_with_failed_retrieval_repairs_retrieval_first() -> None:
    diagnosis = _diagnose(
        {
            "candidate_recall_at_20_min": 0.4,
            "candidate_recall_at_50_min": 0.5,
            "final_recall_at_5_min": 0.3,
            "mrr_min": 0.2,
            "grounded_rate": 0.2,
        },
        groups=["retrieval", "quality"],
    )

    assert diagnosis["primary_recommendation"]["optimization_id"] == ("repair_retrieval")


def test_vllm_marks_pagedattention_already_active() -> None:
    diagnosis = _diagnose(
        {"mean_gpu_utilization_percent": 20.0},
        groups=["resource"],
    )
    active = diagnosis["already_active_capabilities"]

    assert any(
        item["optimization_id"] == "use_pagedattention_capable_engine"
        and item["status"] == "already_active"
        for item in active
    )
    assert "use_pagedattention_capable_engine" not in _recommendation_ids(diagnosis)


def test_hf_can_recommend_switching_to_vllm() -> None:
    diagnosis = _diagnose(
        {"p95_ttft_ms": 2000.0, "high_input_tokens": True},
        groups=["latency"],
        engine="huggingface",
    )

    assert "switch_engine_to_vllm" in _recommendation_ids(diagnosis)


def test_incompatible_optimizations_are_filtered() -> None:
    diagnosis = _diagnose(
        {"grounded_rate": 0.2},
        groups=["quality"],
        memory_mode="mm0_no_context",
    )
    incompatible = diagnosis["incompatible_optimizations"]

    assert any(item["optimization_id"] == "use_mm4_agentic_repair" for item in incompatible)
    assert "use_mm4_agentic_repair" not in _recommendation_ids(diagnosis)


def test_recommendation_engine_never_uses_an_llm() -> None:
    diagnosis = _diagnose({"grounded_rate": 0.2}, groups=["quality"])

    assert diagnosis["llm_used"] is False
    assert diagnosis["primary_recommendation"]["catalog_definition"]
