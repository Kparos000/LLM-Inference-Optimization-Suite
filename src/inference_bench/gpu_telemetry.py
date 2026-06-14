"""GPU telemetry parsing, sampling, summaries, and runtime projections."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

GPU_QUERY_FIELDS = (
    "timestamp",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)
GPU_QUERY = ",".join(GPU_QUERY_FIELDS)
PROCESS_QUERY = "pid,process_name,used_gpu_memory"
TELEMETRY_CSV_FIELDS = [
    "timestamp",
    "gpu_name",
    "utilization_gpu_percent",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "temperature_c",
    "process_info",
]

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _float_value(value: str) -> float:
    cleaned = value.strip()
    if cleaned in {"", "N/A", "[Not Supported]"}:
        msg = f"Expected numeric nvidia-smi value, received {value!r}"
        raise ValueError(msg)
    return float(cleaned)


@dataclass(frozen=True)
class GpuTelemetrySample:
    """One nvidia-smi polling sample."""

    timestamp: str
    gpu_name: str
    utilization_gpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    power_draw_w: float
    temperature_c: float
    process_info: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return a stable dictionary representation."""

        return asdict(self)


def parse_nvidia_smi_csv(
    output: str,
    *,
    process_info: str = "",
) -> list[GpuTelemetrySample]:
    """Parse ``nvidia-smi --format=csv,noheader,nounits`` output."""

    samples: list[GpuTelemetrySample] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != len(GPU_QUERY_FIELDS):
            msg = (
                "Expected seven nvidia-smi columns "
                f"({', '.join(GPU_QUERY_FIELDS)}), received {len(row)}"
            )
            raise ValueError(msg)
        samples.append(
            GpuTelemetrySample(
                timestamp=row[0].strip(),
                gpu_name=row[1].strip(),
                utilization_gpu_percent=_float_value(row[2]),
                memory_used_mb=_float_value(row[3]),
                memory_total_mb=_float_value(row[4]),
                power_draw_w=_float_value(row[5]),
                temperature_c=_float_value(row[6]),
                process_info=process_info.strip(),
            )
        )
    return samples


def default_command_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a telemetry command and capture text output."""

    return subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )


def nvidia_smi_command(*, ssh_host: str | None = None) -> list[str]:
    """Build the GPU query command for local or SSH execution."""

    query = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    return ["ssh", ssh_host, *query] if ssh_host else query


def nvidia_smi_process_command(*, ssh_host: str | None = None) -> list[str]:
    """Build the process query command for local or SSH execution."""

    query = [
        "nvidia-smi",
        f"--query-compute-apps={PROCESS_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    return ["ssh", ssh_host, *query] if ssh_host else query


def collect_gpu_sample(
    *,
    ssh_host: str | None = None,
    command_runner: CommandRunner = default_command_runner,
) -> list[GpuTelemetrySample]:
    """Collect one GPU sample and best-effort compute process information."""

    gpu_result = command_runner(nvidia_smi_command(ssh_host=ssh_host))
    process_info = ""
    try:
        process_result = command_runner(nvidia_smi_process_command(ssh_host=ssh_host))
        process_info = " | ".join(
            line.strip() for line in process_result.stdout.splitlines() if line.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        process_info = ""
    return parse_nvidia_smi_csv(gpu_result.stdout, process_info=process_info)


def sample_gpu_telemetry(
    *,
    duration_seconds: float,
    interval_seconds: float,
    ssh_host: str | None = None,
    command_runner: CommandRunner = default_command_runner,
    stop_requested: Callable[[], bool] | None = None,
) -> list[GpuTelemetrySample]:
    """Poll nvidia-smi until duration expires or a stop callback returns true."""

    if duration_seconds <= 0:
        msg = "duration_seconds must be > 0"
        raise ValueError(msg)
    if interval_seconds <= 0:
        msg = "interval_seconds must be > 0"
        raise ValueError(msg)

    samples: list[GpuTelemetrySample] = []
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        if stop_requested is not None and stop_requested():
            break
        samples.extend(
            collect_gpu_sample(
                ssh_host=ssh_host,
                command_runner=command_runner,
            )
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval_seconds, remaining))
    return samples


def write_gpu_telemetry_csv(
    path: str | Path,
    samples: list[GpuTelemetrySample],
) -> Path:
    """Write raw GPU samples to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TELEMETRY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sample.to_dict() for sample in samples)
    return output_path


def _metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize_gpu_telemetry(samples: list[GpuTelemetrySample]) -> dict[str, object]:
    """Build aggregate GPU telemetry statistics."""

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "gpu_names": sorted({sample.gpu_name for sample in samples}),
        "utilization_gpu_percent": _metric_summary(
            [sample.utilization_gpu_percent for sample in samples]
        ),
        "memory_used_mb": _metric_summary([sample.memory_used_mb for sample in samples]),
        "memory_total_mb": _metric_summary([sample.memory_total_mb for sample in samples]),
        "power_draw_w": _metric_summary([sample.power_draw_w for sample in samples]),
        "temperature_c": _metric_summary([sample.temperature_c for sample in samples]),
        "process_info_observed": sorted(
            {sample.process_info for sample in samples if sample.process_info}
        ),
    }


def write_gpu_telemetry_summary(
    path: str | Path,
    samples: list[GpuTelemetrySample],
) -> Path:
    """Write aggregate GPU telemetry statistics as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            summarize_gpu_telemetry(samples),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def build_runtime_projections(
    *,
    measured_prompt_count: int,
    measured_wall_seconds: float,
    mean_latency_ms: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    target_prompt_counts: Sequence[int] = (500, 2500, 5000, 10000),
) -> dict[str, object]:
    """Project concurrency-one runtimes from measured latency and throughput."""

    if measured_prompt_count <= 0:
        msg = "measured_prompt_count must be > 0"
        raise ValueError(msg)
    if measured_wall_seconds <= 0:
        msg = "measured_wall_seconds must be > 0"
        raise ValueError(msg)
    for field_name, value in (
        ("mean_latency_ms", mean_latency_ms),
        ("p50_latency_ms", p50_latency_ms),
        ("p95_latency_ms", p95_latency_ms),
    ):
        if value < 0:
            msg = f"{field_name} must be >= 0"
            raise ValueError(msg)

    measured_requests_per_second = measured_prompt_count / measured_wall_seconds
    projections: list[dict[str, object]] = []
    for prompt_count in target_prompt_counts:
        if prompt_count <= 0:
            msg = "target prompt counts must be > 0"
            raise ValueError(msg)
        throughput_seconds = prompt_count / measured_requests_per_second
        projections.append(
            {
                "prompt_count": prompt_count,
                "estimated_seconds_from_measured_throughput": throughput_seconds,
                "estimated_seconds_from_mean_latency": prompt_count * mean_latency_ms / 1000.0,
                "estimated_seconds_from_p50_latency": prompt_count * p50_latency_ms / 1000.0,
                "estimated_seconds_from_p95_latency": prompt_count * p95_latency_ms / 1000.0,
            }
        )
    return {
        "projection_type": "measured_concurrency_one_linear_estimate",
        "is_guarantee": False,
        "measured_prompt_count": measured_prompt_count,
        "measured_wall_seconds": measured_wall_seconds,
        "measured_requests_per_second": measured_requests_per_second,
        "assumptions": [
            "concurrency remains 1",
            "prompt and output length distributions remain comparable",
            "server remains warm and free of competing GPU workloads",
            "network and queue conditions remain comparable",
        ],
        "projections": projections,
    }


def write_runtime_projection(path: str | Path, payload: dict[str, object]) -> Path:
    """Write a runtime projection report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
