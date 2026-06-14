from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_bench.gpu_telemetry import (
    build_runtime_projections,
    parse_nvidia_smi_csv,
    summarize_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)


def test_parse_nvidia_smi_fixture_output() -> None:
    output = "2026/06/14 15:52:56.862, NVIDIA GeForce RTX 3070, 87, 6144, 8192, 191.50, 72\n"

    samples = parse_nvidia_smi_csv(
        output,
        process_info="1234, VLLM::EngineCore, 5900 MiB",
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.gpu_name == "NVIDIA GeForce RTX 3070"
    assert sample.utilization_gpu_percent == 87.0
    assert sample.memory_used_mb == 6144.0
    assert sample.memory_total_mb == 8192.0
    assert sample.power_draw_w == 191.5
    assert sample.temperature_c == 72.0
    assert "VLLM::EngineCore" in sample.process_info


def test_parse_nvidia_smi_rejects_wrong_column_count() -> None:
    with pytest.raises(ValueError, match="Expected seven"):
        parse_nvidia_smi_csv("timestamp, gpu, 10\n")


def test_gpu_summary_and_writers(tmp_path: Path) -> None:
    samples = parse_nvidia_smi_csv(
        "\n".join(
            [
                "2026/06/14 15:00:00, NVIDIA GeForce RTX 3070, 10, 1000, 8192, 50, 60",
                "2026/06/14 15:00:01, NVIDIA GeForce RTX 3070, 90, 6000, 8192, 180, 75",
            ]
        )
    )

    summary = summarize_gpu_telemetry(samples)
    assert summary["sample_count"] == 2
    assert summary["utilization_gpu_percent"]["mean"] == 50.0  # type: ignore[index]

    csv_path = write_gpu_telemetry_csv(tmp_path / "telemetry.csv", samples)
    json_path = write_gpu_telemetry_summary(tmp_path / "telemetry.json", samples)
    assert csv_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["sample_count"] == 2


def test_runtime_projection_uses_measured_throughput_and_latency() -> None:
    report = build_runtime_projections(
        measured_prompt_count=50,
        measured_wall_seconds=100.0,
        mean_latency_ms=2000.0,
        p50_latency_ms=1800.0,
        p95_latency_ms=3000.0,
        target_prompt_counts=(500,),
    )

    assert report["is_guarantee"] is False
    assert report["measured_requests_per_second"] == 0.5
    projection = report["projections"][0]  # type: ignore[index]
    assert projection["estimated_seconds_from_measured_throughput"] == 1000.0
    assert projection["estimated_seconds_from_mean_latency"] == 1000.0
