from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_bench.run_manifest import (
    RunManifest,
    file_sha256,
    hash_existing_paths,
    read_run_manifest,
    utc_now,
    write_run_manifest,
)


def test_production_run_manifest_writes_required_long_run_fields(tmp_path: Path) -> None:
    workload = tmp_path / "workload.jsonl"
    config = tmp_path / "config.yaml"
    workload.write_text('{"prompt_id":"p1"}\n', encoding="utf-8")
    config.write_text("runtime: vllm\n", encoding="utf-8")
    now = utc_now()
    manifest = RunManifest(
        run_id="run-1",
        timestamp_utc=now,
        backend="openai_compatible_vllm",
        model_alias="model2_3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        memory_mode="mm2_hybrid_top5",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(workload),
        output_path="results/raw/run-1.jsonl",
        max_records=20,
        git_commit="abc123",
        command="dry-run",
        status="initialized",
        start_time=now,
        end_time=None,
        error_count=0,
        config_id="phase1e",
        vertical="finance",
        runtime="vllm",
        engine="vllm",
        backend_type="self_hosted_gpu",
        hardware="runpod_gpu",
        provider="huggingface",
        concurrency=1,
        traffic_profile="offline_throughput",
        prompt_count=20,
        dataset_workload_hash=hash_existing_paths([workload]),
        config_hash=hash_existing_paths([config]),
        started_at=now,
        updated_at=now,
        completed_count=0,
        failed_count=0,
        expected_count=20,
        artifact_paths={"raw_jsonl": "results/raw/run-1.jsonl"},
    )

    path = write_run_manifest(manifest, tmp_path / "manifest.json")
    payload = read_run_manifest(path)

    assert payload["config_id"] == "phase1e"
    assert payload["runtime"] == "vllm"
    assert payload["backend_type"] == "self_hosted_gpu"
    assert payload["traffic_profile"] == "offline_throughput"
    assert payload["expected_count"] == 20
    assert payload["artifact_paths"] == {"raw_jsonl": "results/raw/run-1.jsonl"}
    assert payload["dataset_workload_hash"] == hash_existing_paths([workload])
    assert file_sha256(workload)


def test_completed_manifest_cannot_hide_partial_run() -> None:
    now = utc_now()
    with pytest.raises(ValueError, match="completed manifests cannot be partial"):
        RunManifest(
            run_id="run-1",
            timestamp_utc=now,
            backend="dry",
            model_alias="model2_3b",
            model_id="Qwen/Qwen2.5-3B-Instruct",
            memory_mode="mm2_hybrid_top5",
            split="smoke",
            ablation_mode="prompt_plus_metadata",
            input_workload_path="input.jsonl",
            output_path="output.jsonl",
            max_records=20,
            git_commit="abc",
            command="cmd",
            status="completed",
            start_time=now,
            end_time=now,
            error_count=0,
            completed_count=10,
            failed_count=0,
            expected_count=20,
        )


def test_manifest_statuses_include_partial_and_initialized(tmp_path: Path) -> None:
    now = utc_now()
    for status in ("initialized", "running", "partial", "completed", "failed"):
        manifest = RunManifest(
            run_id=f"run-{status}",
            timestamp_utc=now,
            backend="dry",
            model_alias="model2_3b",
            model_id="Qwen/Qwen2.5-3B-Instruct",
            memory_mode="mm2_hybrid_top5",
            split="smoke",
            ablation_mode="prompt_plus_metadata",
            input_workload_path="input.jsonl",
            output_path="output.jsonl",
            max_records=1,
            git_commit="abc",
            command="cmd",
            status=status,
            start_time=now,
            end_time=now if status in {"completed", "failed"} else None,
            error_count=0,
            completed_count=1 if status == "completed" else 0,
            failed_count=0,
            expected_count=1,
        )
        path = write_run_manifest(manifest, tmp_path / f"{status}.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == status
