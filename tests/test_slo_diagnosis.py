from typing import Any

from inference_bench.slo_diagnosis import diagnose_slos
from inference_bench.slo_profiles import resolve_slo_profile


def _context() -> dict[str, Any]:
    return {
        "experiment_config": {"experiment_id": "test", "concurrency": 1},
        "model_metadata": {"model_alias": "model1_0_5b"},
        "hardware_profile": {
            "hardware_alias": "remote_rtx3070",
            "gpu_name": "NVIDIA GeForce RTX 3070",
            "vram_gb": 8,
        },
        "engine": "vllm",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "airline",
    }


def test_diagnosis_only_runs_on_failed_slos() -> None:
    profile = resolve_slo_profile(enabled_groups=["retrieval", "quality"])
    metrics = {
        "candidate_recall_at_20_min": 1.0,
        "candidate_recall_at_50_min": 1.0,
        "final_recall_at_5_min": 1.0,
        "mrr_min": 1.0,
        "grounded_rate": 0.3,
        "evidence_match_rate": 0.3,
        "generation_contract_valid_rate": 1.0,
        "safety_violation_count": 0,
    }

    diagnosis = diagnose_slos(
        run_metrics=metrics,
        profile=profile,
        telemetry_available=False,
        **_context(),
    )

    assert diagnosis["diagnosed_failed_slo_count"] == len(diagnosis["failed_slos"])
    assert {item["id"] for item in diagnosis["bottlenecks"]} == {
        "low_evidence_match",
        "low_groundedness",
    }
    assert all(item["status"] == "PASS" for item in diagnosis["passed_slos"])


def test_missing_telemetry_is_unavailable_not_a_fake_failure() -> None:
    profile = resolve_slo_profile(enabled_groups=["resource"])

    diagnosis = diagnose_slos(
        run_metrics={},
        profile=profile,
        telemetry_available=True,
        **_context(),
    )

    assert not diagnosis["failed_slos"]
    assert diagnosis["unavailable_metrics"]
    assert all(item["status"] == "UNAVAILABLE" for item in diagnosis["unavailable_metrics"])
    assert not diagnosis["bottlenecks"]


def test_not_applicable_slos_are_separate_from_unavailable_metrics() -> None:
    profile = resolve_slo_profile(enabled_groups=["retrieval"])
    context = _context()
    context["memory_mode"] = "mm0_no_context"

    diagnosis = diagnose_slos(
        run_metrics={},
        profile=profile,
        telemetry_available=False,
        **context,
    )

    assert diagnosis["not_applicable_slos"]
    assert not diagnosis["unavailable_metrics"]
    assert not diagnosis["failed_slos"]


def test_recommendations_include_reasons_and_evidence() -> None:
    profile = resolve_slo_profile(enabled_groups=["quality"])

    diagnosis = diagnose_slos(
        run_metrics={"grounded_rate": 0.2},
        profile=profile,
        telemetry_available=False,
        **_context(),
    )

    primary = diagnosis["primary_recommendation"]
    assert primary["reason"]
    assert primary["evidence"]
    assert diagnosis["llm_used"] is False
