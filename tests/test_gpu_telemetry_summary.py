from __future__ import annotations

from inference_bench.gpu_telemetry import (
    parse_nvidia_smi_csv,
    summarize_gpu_telemetry,
)


def test_gpu_telemetry_summary_exposes_operational_fields() -> None:
    samples = parse_nvidia_smi_csv(
        "\n".join(
            [
                "2026/06/14 15:00:00, NVIDIA GeForce RTX 3070, 20, 1000, 8192, 50, 45",
                "2026/06/14 15:00:01, NVIDIA GeForce RTX 3070, 80, 6200, 8192, 100, 55",
            ]
        ),
        process_info="1234, SGLang::EngineCore, 6000",
    )

    summary = summarize_gpu_telemetry(
        samples,
        interval_seconds=1.0,
        requested_duration_seconds=60.0,
    )

    assert summary["sampling_interval_seconds"] == 1.0
    assert summary["requested_duration_seconds"] == 60.0
    assert summary["sample_start_timestamp"] == "2026/06/14 15:00:00"
    assert summary["sample_end_timestamp"] == "2026/06/14 15:00:01"
    assert summary["process_names"] == ["SGLang::EngineCore"]
    assert summary["max_memory_used_mb"] == 6200.0
    assert summary["mean_utilization_gpu_percent"] == 50.0
    assert summary["max_utilization_gpu_percent"] == 80.0
    assert summary["mean_power_draw_w"] == 75.0
    assert summary["max_temperature_c"] == 55.0


def test_empty_gpu_telemetry_summary_keeps_missing_values_null() -> None:
    summary = summarize_gpu_telemetry([])

    assert summary["sample_count"] == 0
    assert summary["sample_start_timestamp"] is None
    assert summary["max_memory_used_mb"] is None
    assert summary["process_names"] == []
