"""Generated text output records."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


@dataclass(frozen=True)
class GenerationRecord:
    """Generated text and prompt trace for one benchmark item."""

    run_id: str
    prompt_id: str
    workload_name: str
    backend: str
    model_name: str
    optimization: str
    prompt: str
    generated_text: str | None
    input_tokens: int
    output_tokens: int
    success: bool
    error_message: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
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

        if self.success and self.generated_text is None:
            msg = "generated_text must not be None when success is True"
            raise ValueError(msg)
        if not self.success and self.error_message is None:
            msg = "error_message must not be None when success is False"
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
