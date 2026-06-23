from __future__ import annotations

from inference_bench.vllm_stability_audit import (
    build_vllm_stability_audit,
    classify_b7r1_stability_gate,
    is_backend_connection_failure,
    is_fatal_engine_error,
)


def test_b7_vllm_audit_detects_fatal_cascading_engine_failure() -> None:
    rows = [
        {"prompt_id": "p1", "vertical": "airline", "success": True, "input_tokens": 900},
        {
            "prompt_id": "p2",
            "vertical": "finance",
            "success": False,
            "error_message": "HTTP 500: EngineCore encountered an issue; CUBLAS failed",
            "input_tokens": 0,
            "output_tokens": 0,
        },
        *[
            {
                "prompt_id": f"p{index}",
                "vertical": "finance",
                "success": False,
                "error_message": "Remote end closed connection without response",
                "input_tokens": 0,
                "output_tokens": 0,
            }
            for index in range(3, 15)
        ],
    ]

    report = build_vllm_stability_audit(result_rows=rows, expected_count=14)

    assert report["serving_diagnosis"] == "vllm_engine_core_cuda_cublas_collapse"
    assert report["likely_primary_cause"] == "serving_stability_failure"
    assert report["fatal_engine_error_count"] == 1
    assert report["first_failure"]["prompt_id"] == "p2"
    assert report["cascading_failure"]["cascading_failure_observed"] is True
    assert report["failure_by_vertical"] == {"finance": 13}


def test_engine_error_and_connection_failure_patterns_are_explicit() -> None:
    assert is_fatal_engine_error("EngineCore encountered an issue")
    assert is_fatal_engine_error("CUDA error: CUBLAS_STATUS_ALLOC_FAILED")
    assert is_backend_connection_failure("Connection reset by peer")


def test_b7r1_readiness_blocks_if_engine_collapses() -> None:
    gate = classify_b7r1_stability_gate(
        completed_count=617,
        expected_count=1000,
        success_count=616,
        fatal_engine_errors=1,
        cascading_backend_failure=True,
        safety_violation_count=0,
        artifact_sync_complete=True,
        manifest_valid=True,
        checkpoint_valid=True,
        peak_vram_mb=7770,
        peak_vram_threshold_mb=7600,
        quality_passed=False,
    )

    assert gate["status"] == "B7R1_STABILITY_BLOCKED"
    assert gate["passed"] is False
    assert gate["rtx3070_qwen3b_suitability"] == "unstable"


def test_b7r1_stable_with_quality_caveat_when_serving_passes() -> None:
    gate = classify_b7r1_stability_gate(
        completed_count=1000,
        expected_count=1000,
        success_count=1000,
        fatal_engine_errors=0,
        cascading_backend_failure=False,
        safety_violation_count=0,
        artifact_sync_complete=True,
        manifest_valid=True,
        checkpoint_valid=True,
        peak_vram_mb=7300,
        peak_vram_threshold_mb=7600,
        quality_passed=False,
    )

    assert gate["status"] == "B7R1_STABLE_WITH_QUALITY_CAVEAT"
    assert gate["next_api_load_probe_allowed"] is True
