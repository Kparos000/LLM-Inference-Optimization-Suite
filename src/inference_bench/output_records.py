"""Generated text output records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
class GenerationRecord:
    """Generated text and prompt trace for one benchmark item."""

    run_id: str
    timestamp_utc: str
    prompt_id: str
    workload_name: str
    backend: str
    model_name: str
    optimization: str
    prompt: str
    generated_text: str | None
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
    expected_output_format: str | None = None
    citation_id_aliases: str | None = None
    generation_contract_valid: bool = False
    generation_contract_error: str | None = None
    generation_contract_missing_fields: list[str] = field(default_factory=list)
    parse_error_type: str | None = None
    parse_repair_applied: bool = False
    truncation_detected: bool = False
    answer: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    insufficient_evidence: bool | None = None
    citation_notes: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "timestamp_utc",
            "prompt_id",
            "workload_name",
            "backend",
            "model_name",
            "optimization",
            "prompt",
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

        if self.success and self.generated_text is None:
            msg = "generated_text must not be None when success is True"
            raise ValueError(msg)
        if not self.success and self.error_message is None:
            msg = "error_message must not be None when success is False"
            raise ValueError(msg)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must be between 0 and 1 when provided"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


def write_generation_records_jsonl(
    records: Sequence[GenerationRecord],
    output_path: str | Path,
) -> Path:
    """Write generation records to a JSONL file."""

    jsonl_path = Path(output_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    return jsonl_path
