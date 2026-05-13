"""Utilities for comparing multiple benchmark result CSV files."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from inference_bench.reporting.summary import summarize_results

STANDARD_COMPARISON_FIELDS = [
    "source_file",
    "row_count",
    "success_count",
    "failure_count",
    "backends",
    "models",
    "optimizations",
    "workloads",
    "avg_end_to_end_latency_ms",
    "avg_ttft_ms",
    "avg_tpot_ms",
    "avg_throughput_tokens_per_second",
    "total_estimated_cost_usd",
]


def compare_result_files(paths: Sequence[str | Path]) -> list[dict[str, object]]:
    """Return one benchmark summary row per input CSV file."""

    comparison_rows: list[dict[str, object]] = []
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        summary = summarize_results(csv_path)
        comparison_rows.append({"source_file": str(path), **summary})

    return comparison_rows


def _csv_value(value: object) -> object:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def _fieldnames_for_rows(rows: Sequence[dict[str, object]]) -> list[str]:
    fieldnames = list(STANDARD_COMPARISON_FIELDS)
    for row in rows:
        for field_name in row:
            if field_name not in fieldnames:
                fieldnames.append(field_name)
    return fieldnames


def write_comparison_csv(
    rows: Sequence[dict[str, object]],
    output_path: str | Path,
) -> Path:
    """Write comparison rows to a CSV file."""

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames_for_rows(rows)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field_name: _csv_value(row.get(field_name)) for field_name in fieldnames}
            )

    return csv_path
