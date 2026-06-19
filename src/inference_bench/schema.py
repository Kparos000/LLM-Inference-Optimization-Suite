"""Benchmark data structures and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TypedDict


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


def _validate_non_negative_float(value: float | None, field_name: str) -> None:
    if value is not None and value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


def _validate_optional_string(value: str | None, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        msg = f"{field_name} must be a string when provided"
        raise ValueError(msg)


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None:
        _validate_non_negative_int(value, field_name)


def _optional_int_from_metadata(
    metadata: dict[str, str],
    field_name: str,
) -> int | None:
    raw_value = metadata.get(field_name)
    if raw_value is None or raw_value == "":
        return None
    return int(raw_value)


class BenchmarkMetadata(TypedDict):
    """Optional metadata fields carried into benchmark result rows."""

    workload_id: str | None
    vertical: str | None
    memory_mode: str | None
    ablation_mode: str | None
    context_token_estimate: int | None
    gold_evidence_ids: str | None


def benchmark_metadata_from_workload_item(item: WorkloadItem) -> BenchmarkMetadata:
    """Return optional benchmark metadata carried by an adapted workload item."""

    metadata = item.metadata
    return {
        "workload_id": metadata.get("workload_id"),
        "vertical": metadata.get("vertical"),
        "memory_mode": metadata.get("memory_mode"),
        "ablation_mode": metadata.get("ablation_mode"),
        "context_token_estimate": _optional_int_from_metadata(
            metadata,
            "context_token_estimate",
        ),
        "gold_evidence_ids": metadata.get("gold_evidence_ids"),
    }


def empty_benchmark_metadata() -> BenchmarkMetadata:
    """Return an empty optional benchmark metadata payload."""

    return {
        "workload_id": None,
        "vertical": None,
        "memory_mode": None,
        "ablation_mode": None,
        "context_token_estimate": None,
        "gold_evidence_ids": None,
    }


@dataclass(frozen=True)
class WorkloadItem:
    """A single prompt record in a benchmark workload."""

    prompt_id: str
    workload_name: str
    prompt: str
    expected_output: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.prompt_id, "prompt_id")
        _validate_non_empty_string(self.workload_name, "workload_name")
        _validate_non_empty_string(self.prompt, "prompt")

        for key, value in self.metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                msg = "metadata must contain only string keys and string values"
                raise ValueError(msg)


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics and metadata produced by running one workload item."""

    run_id: str
    timestamp_utc: str
    backend: str
    model_name: str
    optimization: str
    workload_name: str
    prompt_id: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None
    end_to_end_latency_ms: float
    throughput_tokens_per_second: float | None
    peak_memory_mb: float | None
    estimated_cost_usd: float | None
    success: bool
    error_message: str | None = None
    workload_id: str | None = None
    vertical: str | None = None
    memory_mode: str | None = None
    ablation_mode: str | None = None
    context_token_estimate: int | None = None
    gold_evidence_ids: str | None = None
    runtime: str | None = None
    engine: str | None = None
    backend_type: str | None = None
    hardware: str | None = None
    provider: str | None = None
    traffic_profile: str | None = None
    request_arrival_mode: str | None = None
    concurrency: int | None = None
    input_token_bucket: str | None = None
    output_token_bucket: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "timestamp_utc",
            "backend",
            "model_name",
            "optimization",
            "workload_name",
            "prompt_id",
        ):
            _validate_non_empty_string(getattr(self, field_name), field_name)

        _validate_optional_string(self.workload_id, "workload_id")
        _validate_optional_string(self.vertical, "vertical")
        _validate_optional_string(self.memory_mode, "memory_mode")
        _validate_optional_string(self.ablation_mode, "ablation_mode")
        _validate_optional_string(self.runtime, "runtime")
        _validate_optional_string(self.engine, "engine")
        _validate_optional_string(self.backend_type, "backend_type")
        _validate_optional_string(self.hardware, "hardware")
        _validate_optional_string(self.provider, "provider")
        _validate_optional_string(self.traffic_profile, "traffic_profile")
        _validate_optional_string(self.request_arrival_mode, "request_arrival_mode")
        _validate_optional_non_negative_int(self.concurrency, "concurrency")
        _validate_optional_string(self.input_token_bucket, "input_token_bucket")
        _validate_optional_string(self.output_token_bucket, "output_token_bucket")
        _validate_optional_non_negative_int(
            self.context_token_estimate,
            "context_token_estimate",
        )
        _validate_optional_string(self.gold_evidence_ids, "gold_evidence_ids")
        _validate_non_negative_int(self.input_tokens, "input_tokens")
        _validate_non_negative_int(self.output_tokens, "output_tokens")
        _validate_non_negative_float(self.ttft_ms, "ttft_ms")
        _validate_non_negative_float(self.tpot_ms, "tpot_ms")
        _validate_non_negative_float(
            self.end_to_end_latency_ms,
            "end_to_end_latency_ms",
        )
        _validate_non_negative_float(
            self.throughput_tokens_per_second,
            "throughput_tokens_per_second",
        )
        _validate_non_negative_float(self.peak_memory_mb, "peak_memory_mb")
        _validate_non_negative_float(self.estimated_cost_usd, "estimated_cost_usd")

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation suitable for CSV or JSON output."""

        return asdict(self)

    @staticmethod
    def csv_fieldnames() -> list[str]:
        """Return the stable CSV column order for benchmark results."""

        return [
            "run_id",
            "timestamp_utc",
            "backend",
            "model_name",
            "optimization",
            "workload_name",
            "prompt_id",
            "workload_id",
            "vertical",
            "memory_mode",
            "ablation_mode",
            "context_token_estimate",
            "gold_evidence_ids",
            "runtime",
            "engine",
            "backend_type",
            "hardware",
            "provider",
            "traffic_profile",
            "request_arrival_mode",
            "concurrency",
            "input_token_bucket",
            "output_token_bucket",
            "input_tokens",
            "output_tokens",
            "ttft_ms",
            "tpot_ms",
            "end_to_end_latency_ms",
            "throughput_tokens_per_second",
            "peak_memory_mb",
            "estimated_cost_usd",
            "success",
            "error_message",
        ]
