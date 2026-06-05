"""Telemetry schemas for Phase 4 serving validation and future GPU runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

BackendName = Literal["huggingface_local", "vllm", "sglang"]

TELEMETRY_FIELDS = [
    "timestamp",
    "backend",
    "model",
    "memory_mode",
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "throughput_tokens_per_second",
    "requests_per_second",
    "success",
    "error_type",
    "gpu_utilization",
    "gpu_memory",
    "gpu_cost",
    "runpod_cost",
]

BACKEND_COMPARISON_FIELDS = [
    "backend",
    "status",
    "model",
    "memory_mode",
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "throughput_tokens_per_second",
    "requests_per_second",
    "quality",
    "groundedness",
    "cost",
    "gpu_cost",
    "notes",
]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _validate_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_optional_non_negative(value: float | None, field_name: str) -> None:
    if value is not None and value < 0:
        msg = f"{field_name} must be >= 0 when available"
        raise ValueError(msg)


@dataclass(frozen=True)
class TelemetryRecord:
    """One request-level telemetry record.

    GPU and RunPod fields are intentionally nullable in Phase 4 local validation.
    Phase 5 GPU experiments can populate them without changing the schema.
    """

    timestamp: str
    backend: str
    model: str
    memory_mode: str
    latency_ms: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    throughput_tokens_per_second: float | None
    requests_per_second: float | None
    success: bool
    error_type: str | None = None
    gpu_utilization: float | None = None
    gpu_memory: float | None = None
    gpu_cost: float | None = None
    runpod_cost: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("timestamp", "backend", "model", "memory_mode"):
            _validate_non_empty(str(getattr(self, field_name)), field_name)
        for field_name in (
            "latency_ms",
            "ttft_ms",
            "tpot_ms",
            "throughput_tokens_per_second",
            "requests_per_second",
            "gpu_utilization",
            "gpu_memory",
            "gpu_cost",
            "runpod_cost",
        ):
            _validate_optional_non_negative(
                getattr(self, field_name),
                field_name,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable dictionary representation."""

        return asdict(self)


@dataclass(frozen=True)
class BackendComparisonRow:
    """Backend comparison placeholder row for HF, vLLM, and future SGLang."""

    backend: BackendName
    status: str
    model: str
    memory_mode: str
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tokens_per_second: float | None = None
    requests_per_second: float | None = None
    quality: float | None = None
    groundedness: float | None = None
    cost: float | None = None
    gpu_cost: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"available", "validated", "not_run", "future"}:
            msg = "status must be one of: available, validated, not_run, future"
            raise ValueError(msg)
        for field_name in ("backend", "status", "model", "memory_mode"):
            _validate_non_empty(str(getattr(self, field_name)), field_name)
        for field_name in (
            "latency_ms",
            "ttft_ms",
            "tpot_ms",
            "throughput_tokens_per_second",
            "requests_per_second",
            "quality",
            "groundedness",
            "cost",
            "gpu_cost",
        ):
            _validate_optional_non_negative(
                getattr(self, field_name),
                field_name,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a stable dictionary representation."""

        return asdict(self)


def telemetry_record_from_result_row(
    row: dict[str, Any],
    *,
    backend: str,
    model: str,
    memory_mode: str,
) -> TelemetryRecord:
    """Build telemetry from a runner output row."""

    latency_ms = _optional_float(row.get("latency_ms") or row.get("end_to_end_latency_ms"))
    requests_per_second = 1000.0 / latency_ms if latency_ms and latency_ms > 0 else None
    return TelemetryRecord(
        timestamp=str(row.get("timestamp_utc") or row.get("timestamp") or utc_now()),
        backend=backend,
        model=model,
        memory_mode=memory_mode,
        latency_ms=latency_ms,
        ttft_ms=_optional_float(row.get("ttft_ms")),
        tpot_ms=_optional_float(row.get("tpot_ms")),
        throughput_tokens_per_second=_optional_float(row.get("throughput_tokens_per_second")),
        requests_per_second=requests_per_second,
        success=_truthy(row.get("success")),
        error_type=str(row.get("error_type") or "") or None,
    )


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_backend_comparison_framework(
    *,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    memory_mode: str = "mm2_hybrid_top5",
) -> list[BackendComparisonRow]:
    """Return comparison rows for current and future serving backends."""

    return [
        BackendComparisonRow(
            backend="huggingface_local",
            status="available",
            model=model,
            memory_mode=memory_mode,
            notes="Local HF path validates correctness plumbing, not serving throughput.",
        ),
        BackendComparisonRow(
            backend="vllm",
            status="not_run",
            model=model,
            memory_mode=memory_mode,
            notes="OpenAI-compatible vLLM path is validated when localhost:8000 is reachable.",
        ),
        BackendComparisonRow(
            backend="sglang",
            status="future",
            model=model,
            memory_mode=memory_mode,
            notes="SGLang integration is reserved for a later Phase 4/5 block.",
        ),
    ]


def write_telemetry_json(path: str | Path, records: list[TelemetryRecord]) -> Path:
    """Write telemetry records as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": TELEMETRY_FIELDS,
        "record_count": len(records),
        "records": [record.to_dict() for record in records],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_backend_comparison_csv(
    path: str | Path,
    rows: list[BackendComparisonRow],
) -> Path:
    """Write backend comparison framework rows as CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BACKEND_COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    return output_path
