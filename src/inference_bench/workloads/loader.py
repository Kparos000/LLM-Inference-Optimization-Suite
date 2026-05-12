"""Workload loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

from inference_bench.schema import WorkloadItem


def load_jsonl_workload(path: str | Path) -> list[WorkloadItem]:
    """Load a JSONL workload file into workload items."""

    workload_path = Path(path)
    if not workload_path.exists():
        raise FileNotFoundError(workload_path)

    items: list[WorkloadItem] = []
    with workload_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {workload_path} at line {line_number}: {exc.msg}"
                raise ValueError(msg) from exc

            if not isinstance(record, dict):
                msg = (
                    f"Invalid workload record in {workload_path} "
                    f"at line {line_number}: expected object"
                )
                raise ValueError(msg)

            try:
                items.append(WorkloadItem(**record))
            except (TypeError, ValueError) as exc:
                msg = f"Invalid workload record in {workload_path} at line {line_number}: {exc}"
                raise ValueError(msg) from exc

    return items
