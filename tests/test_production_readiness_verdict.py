from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_bench.production_readiness import (
    ProductionRunReadinessInput,
    build_production_readiness_verdict,
)
from inference_bench.profiling_hooks import build_profiling_config, disabled_profiling_metadata
from inference_bench.run_manifest import RunManifest, utc_now, write_run_manifest


def _result_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-1",
        "config_id": "cfg-1",
        "prompt_id": "prompt-1",
        "vertical": "finance",
        "model_alias": "model2_3b",
        "memory_mode": "mm2_hybrid_top5",
        "runtime": "vllm",
        "backend_type": "self_hosted_gpu",
        "engine": "vllm",
        "hardware": "runpod_gpu",
        "provider": "huggingface",
        "concurrency": 1,
        "gpu_cost_usd": None,
        "gpu_hourly_price_usd": None,
        "api_cost_usd": None,
    }
    row.update(overrides)
    return row


def test_long_run_guardrails_block_without_sync_checkpoint_and_price() -> None:
    report = build_production_readiness_verdict(
        ProductionRunReadinessInput(
            planned_prompt_count=1000,
            expected_prompt_count=1000,
            observed_prompt_count=1000,
            manifest_status="planned",
            backend_type="self_hosted_gpu",
            traffic_profile="offline_throughput",
            concurrency=1,
            request_arrival_mode="closed_loop",
            artifact_sync_configured=False,
            making_gpu_cost_claim=True,
            checkpoint_resume_supported=False,
            result_track_rows=[_result_row()],
        )
    )

    failed = {check["name"] for check in report["checks"] if check["blocking"]}
    assert report["status"] == "NOT_READY"
    assert "artifact_sync_before_long_run" in failed
    assert "gpu_hourly_price_required_before_gpu_cost_claim" in failed
    assert "checkpoint_resume_required_for_1000_plus" in failed


def test_partial_completed_run_is_blocked() -> None:
    report = build_production_readiness_verdict(
        ProductionRunReadinessInput(
            planned_prompt_count=500,
            expected_prompt_count=500,
            observed_prompt_count=499,
            manifest_status="completed",
            backend_type="self_hosted_gpu",
            traffic_profile="online_low_latency",
            concurrency=1,
            request_arrival_mode="jittered_poisson",
            result_track_rows=[_result_row()],
        )
    )

    assert any(
        check["name"] == "partial_runs_cannot_be_marked_complete" and check["blocking"]
        for check in report["checks"]
    )


def test_large_api_run_requires_load_probe_and_unified_result_schema() -> None:
    report = build_production_readiness_verdict(
        ProductionRunReadinessInput(
            planned_prompt_count=1000,
            expected_prompt_count=1000,
            observed_prompt_count=0,
            manifest_status="planned",
            backend_type="api_provider",
            traffic_profile="offline_throughput",
            concurrency=4,
            request_arrival_mode="closed_loop",
            checkpoint_resume_supported=True,
            api_provider_load_probe_completed=False,
            result_track_rows=[
                _result_row(
                    model_alias="model6_gated",
                    runtime="api_provider_route",
                    backend_type="api_provider",
                    engine="api_provider",
                    hardware="provider_managed",
                    provider="hf_inference_provider",
                    api_provider="hf_inference_provider",
                    api_cost_usd=0.001,
                    gpu_telemetry_available=False,
                )
            ],
        )
    )

    assert any(
        check["name"] == "api_provider_load_probe_required_before_large_api_runs"
        and check["blocking"]
        for check in report["checks"]
    )
    assert any(
        check["name"] == "api_and_gpu_tracks_join_through_unified_result_schema"
        and check["status"] == "PASS"
        for check in report["checks"]
    )


def test_ready_verdict_when_required_guardrails_are_satisfied() -> None:
    report = build_production_readiness_verdict(
        ProductionRunReadinessInput(
            planned_prompt_count=1000,
            expected_prompt_count=1000,
            observed_prompt_count=0,
            manifest_status="planned",
            backend_type="self_hosted_gpu",
            traffic_profile="offline_throughput",
            concurrency=1,
            request_arrival_mode="closed_loop",
            artifact_sync_configured=True,
            gpu_hourly_price_usd=0.5,
            making_gpu_cost_claim=True,
            checkpoint_resume_supported=True,
            result_track_rows=[_result_row(gpu_hourly_price_usd=0.5)],
        )
    )

    assert report["status"] == "READY"
    assert report["blocking_count"] == 0


def test_profiling_disabled_by_default_and_optional_manifest_metadata(tmp_path: Path) -> None:
    assert disabled_profiling_metadata()["mode"] == "disabled"
    profiling = build_profiling_config(mode="pytorch", output_path="profiles/run-1")
    manifest = RunManifest(
        run_id="run-1",
        timestamp_utc=utc_now(),
        backend="openai_compatible_vllm",
        model_alias="model2_3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path="input.jsonl",
        output_path="output.csv",
        max_records=5,
        git_commit="abc",
        command="cmd",
        status="planned",
        start_time=utc_now(),
        end_time=None,
        error_count=0,
        profiling_enabled=profiling.enabled,
        profiling_mode=profiling.mode,
        profiler_output_path=profiling.output_path,
        profiling_metadata=profiling.to_manifest_metadata(),
    )

    path = write_run_manifest(manifest, tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["profiling_enabled"] is True
    assert payload["profiling_mode"] == "pytorch"
    assert payload["profiling_metadata"]["output_path"] == "profiles/run-1"


def test_enabled_profiling_requires_output_path() -> None:
    with pytest.raises(ValueError, match="enabled profiling requires output_path"):
        build_profiling_config(mode="nsys")
