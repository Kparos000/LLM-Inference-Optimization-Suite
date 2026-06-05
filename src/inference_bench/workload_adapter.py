"""Adapters from Phase 3 memory workloads to runner workload items."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.generation_contract import (
    GENERATION_CONTRACT_FORMAT,
    citation_aliases,
    citation_label,
    render_generation_contract_prompt,
)
from inference_bench.schema import WorkloadItem


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {jsonl_path} at line {line_number}: {exc.msg}"
                raise ValueError(msg) from exc
            if not isinstance(row, dict):
                msg = f"Invalid JSONL row in {jsonl_path} at line {line_number}: expected object"
                raise ValueError(msg)
            rows.append(row)
    return rows


def load_phase3_workload_records(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[WorkloadRecord]:
    """Load Phase 3 WorkloadRecord JSONL records."""

    if limit is not None and limit <= 0:
        msg = "limit must be > 0 when provided"
        raise ValueError(msg)

    records: list[WorkloadRecord] = []
    for row in read_jsonl(path):
        records.append(WorkloadRecord(**row))
        if limit is not None and len(records) >= limit:
            break
    return records


def phase3_workload_path(
    workload_root: str | Path,
    *,
    split: str,
    memory_mode: str,
    ablation_mode: str | None = None,
) -> Path:
    """Resolve a generated Phase 3 workload path."""

    root = Path(workload_root)
    if ablation_mode:
        candidate = root / split / ablation_mode / f"{memory_mode}.jsonl"
        if candidate.exists():
            return candidate
    return root / split / f"{memory_mode}.jsonl"


def render_messages_for_runner(messages: list[dict[str, str]]) -> str:
    """Render chat messages into a single prompt string for existing runners."""

    rendered_parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().upper()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        rendered_parts.append(f"{role}:\n{content}")

    rendered = "\n\n".join(rendered_parts).strip()
    if not rendered:
        msg = "Rendered prompt must not be empty"
        raise ValueError(msg)
    return rendered


def _json_metadata(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _context_record(value: ContextRecord | dict[str, Any]) -> ContextRecord:
    if isinstance(value, ContextRecord):
        return value
    return ContextRecord(**value)


def _source_question(record: WorkloadRecord) -> str:
    """Return the user-visible question without answer-side evidence IDs."""

    for field_name in ("question", "prompt", "request", "task"):
        value = record.source_prompt_record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for message in reversed(record.messages):
        if str(message.get("role") or "").lower() == "user":
            content = str(message.get("content") or "").strip()
            if content:
                return content
    return render_messages_for_runner(record.messages)


def workload_record_metadata(record: WorkloadRecord) -> dict[str, str]:
    """Return string metadata preserved in the runner workload item."""

    retrieval_metadata = dict(record.retrieval_metadata)
    ablation_mode = str(retrieval_metadata.get("ablation_mode") or "none")
    selected_context_ids = retrieval_metadata.get("selected_context_ids", [])
    context_ids = [
        context.context_id if hasattr(context, "context_id") else str(context.get("context_id"))
        for context in record.context_records
    ]
    if not selected_context_ids:
        selected_context_ids = context_ids
    contexts = [_context_record(context) for context in record.context_records]
    citation_id_aliases = {
        citation_label(index): citation_aliases(context)
        for index, context in enumerate(contexts, start=1)
    }

    return {
        "workload_id": record.workload_id,
        "phase3_prompt_id": record.prompt_id,
        "vertical": record.vertical,
        "memory_mode": record.memory_mode,
        "ablation_mode": ablation_mode,
        "dataset_split": record.dataset_split,
        "expected_output_format": GENERATION_CONTRACT_FORMAT,
        "source_expected_output_format": record.expected_output_format,
        "context_token_estimate": str(record.context_token_estimate),
        "context_record_count": str(len(record.context_records)),
        "gold_evidence_ids": _json_metadata(record.gold_evidence_ids),
        "selected_context_ids": _json_metadata(selected_context_ids),
        "citation_id_aliases": _json_metadata(citation_id_aliases),
        "retrieval_metadata": _json_metadata(record.retrieval_metadata),
        "source_prompt_record": _json_metadata(record.source_prompt_record),
    }


def workload_record_to_runner_item(record: WorkloadRecord) -> WorkloadItem:
    """Convert one Phase 3 WorkloadRecord to the existing runner WorkloadItem shape."""

    contexts = [_context_record(context) for context in record.context_records]
    return WorkloadItem(
        prompt_id=record.prompt_id,
        workload_name=f"{record.dataset_split}_{record.memory_mode}",
        prompt=render_generation_contract_prompt(
            question=_source_question(record),
            context_records=contexts,
            memory_mode=record.memory_mode,
        ),
        expected_output=GENERATION_CONTRACT_FORMAT,
        metadata=workload_record_metadata(record),
    )


def convert_phase3_workload_to_runner_items(
    workload_path: str | Path,
    *,
    limit: int | None = None,
) -> list[WorkloadItem]:
    """Convert Phase 3 workload JSONL records to runner WorkloadItem objects."""

    return [
        workload_record_to_runner_item(record)
        for record in load_phase3_workload_records(workload_path, limit=limit)
    ]


def write_runner_workload_jsonl(
    items: list[WorkloadItem],
    output_path: str | Path,
) -> Path:
    """Write runner-compatible workload items as JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(asdict(item), ensure_ascii=True, sort_keys=True) + "\n")
    return path


def export_runner_workload(
    *,
    workload_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    report_path: str | Path = "data/generated/phase4/smoke_workload_export_report.json",
    summary_path: str | Path = "data/generated/phase4/smoke_workload_export_summary.csv",
) -> dict[str, Any]:
    """Convert and write a Phase 3 workload for runner consumption."""

    records = load_phase3_workload_records(workload_path, limit=limit)
    items = [workload_record_to_runner_item(record) for record in records]
    written_output_path = write_runner_workload_jsonl(items, output_path)

    by_vertical: dict[str, int] = {}
    memory_modes = sorted({record.memory_mode for record in records})
    ablation_modes = sorted(
        {str(record.retrieval_metadata.get("ablation_mode") or "none") for record in records}
    )
    for record in records:
        by_vertical[record.vertical] = by_vertical.get(record.vertical, 0) + 1

    report = {
        "input_workload_path": str(workload_path),
        "output_path": str(written_output_path),
        "limit": limit,
        "record_count": len(items),
        "memory_modes": memory_modes,
        "ablation_modes": ablation_modes,
        "by_vertical": by_vertical,
        "metadata_fields_preserved": [
            "prompt_id",
            "workload_id",
            "vertical",
            "memory_mode",
            "ablation_mode",
            "retrieval_metadata",
            "context_token_estimate",
            "gold_evidence_ids",
        ],
        "no_model_inference_triggered": True,
    }

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_workload_path",
        "output_path",
        "record_count",
        "memory_modes",
        "ablation_modes",
        "vertical",
        "vertical_record_count",
    ]
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        if by_vertical:
            for vertical, count in sorted(by_vertical.items()):
                writer.writerow(
                    {
                        "input_workload_path": str(workload_path),
                        "output_path": str(written_output_path),
                        "record_count": len(items),
                        "memory_modes": ";".join(memory_modes),
                        "ablation_modes": ";".join(ablation_modes),
                        "vertical": vertical,
                        "vertical_record_count": count,
                    }
                )
        else:
            writer.writerow(
                {
                    "input_workload_path": str(workload_path),
                    "output_path": str(written_output_path),
                    "record_count": 0,
                    "memory_modes": "",
                    "ablation_modes": "",
                    "vertical": "",
                    "vertical_record_count": 0,
                }
            )
    return report
