"""Phase 3 context and workload schemas.

These schemas define generated context/workload records for future context
engineering and RAG experiments. They do not implement retrieval or inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference_bench.config import resolve_memory_mode

VALID_VERTICALS = {
    "airline",
    "healthcare_admin",
    "retail",
    "finance",
    "research_ai",
}

VALID_DATASET_SPLITS = {
    "smoke_500",
    "controlled_2000",
    "final_10000",
    "test_fixture",
}


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be an integer >= 0"
        raise ValueError(msg)


def _validate_dict(value: dict[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        msg = f"{field_name} must be an object/dict"
        raise ValueError(msg)


@dataclass(frozen=True)
class ContextRecord:
    """A normalized context chunk available to a generated workload record."""

    context_id: str
    vertical: str
    source_id: str
    parent_id: str
    chunk_id: str
    chunk_strategy: str
    source_type: str
    title: str
    text: str
    metadata: dict[str, Any]
    token_estimate: int
    provenance: str
    is_gold_linked: bool

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.context_id, "context_id")
        _validate_non_empty_string(self.vertical, "vertical")
        if self.vertical not in VALID_VERTICALS:
            msg = f"vertical must be one of: {', '.join(sorted(VALID_VERTICALS))}"
            raise ValueError(msg)
        _validate_non_empty_string(self.text, "text")
        _validate_dict(self.metadata, "metadata")
        _validate_non_negative_int(self.token_estimate, "token_estimate")
        if not isinstance(self.is_gold_linked, bool):
            msg = "is_gold_linked must be boolean"
            raise ValueError(msg)


def _coerce_context_record(record: ContextRecord | dict[str, Any]) -> ContextRecord:
    if isinstance(record, ContextRecord):
        return record
    if isinstance(record, dict):
        return ContextRecord(**record)
    msg = "context_records must contain ContextRecord instances or dictionaries"
    raise ValueError(msg)


@dataclass(frozen=True)
class WorkloadRecord:
    """A Phase 3 generated workload record for context/memory experiments."""

    workload_id: str
    prompt_id: str
    vertical: str
    memory_mode: str
    messages: list[dict[str, str]]
    context_records: list[ContextRecord | dict[str, Any]]
    context_token_estimate: int
    retrieval_metadata: dict[str, Any]
    expected_output_format: str
    gold_evidence_ids: list[str]
    dataset_split: str
    source_prompt_record: dict[str, Any]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.workload_id, "workload_id")
        _validate_non_empty_string(self.prompt_id, "prompt_id")
        _validate_non_empty_string(self.vertical, "vertical")
        if self.vertical not in VALID_VERTICALS:
            msg = f"vertical must be one of: {', '.join(sorted(VALID_VERTICALS))}"
            raise ValueError(msg)
        resolve_memory_mode(self.memory_mode)
        if not isinstance(self.messages, list) or not self.messages:
            msg = "messages must be non-empty"
            raise ValueError(msg)
        for message in self.messages:
            _validate_dict(message, "messages")
        if not isinstance(self.context_records, list):
            msg = "context_records must be a list"
            raise ValueError(msg)
        coerced_context_records = [
            _coerce_context_record(record) for record in self.context_records
        ]
        for context_record in coerced_context_records:
            if context_record.vertical != self.vertical:
                msg = "context_records must use the same vertical as the workload"
                raise ValueError(msg)
        _validate_non_negative_int(self.context_token_estimate, "context_token_estimate")
        _validate_dict(self.retrieval_metadata, "retrieval_metadata")
        _validate_non_empty_string(self.expected_output_format, "expected_output_format")
        if not isinstance(self.gold_evidence_ids, list):
            msg = "gold_evidence_ids must be a list"
            raise ValueError(msg)
        if not all(isinstance(evidence_id, str) for evidence_id in self.gold_evidence_ids):
            msg = "gold_evidence_ids must contain only strings"
            raise ValueError(msg)
        if self.dataset_split not in VALID_DATASET_SPLITS:
            msg = f"dataset_split must be one of: {', '.join(sorted(VALID_DATASET_SPLITS))}"
            raise ValueError(msg)
        _validate_dict(self.source_prompt_record, "source_prompt_record")
