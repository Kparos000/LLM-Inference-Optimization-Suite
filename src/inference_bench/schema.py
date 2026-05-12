"""Benchmark data structures and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
