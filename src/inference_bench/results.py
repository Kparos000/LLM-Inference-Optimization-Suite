"""Benchmark result output utilities."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from inference_bench.schema import BenchmarkResult


def write_results_csv(results: Sequence[BenchmarkResult], output_path: str | Path) -> Path:
    """Write benchmark results to a CSV file and return the output path."""

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = BenchmarkResult.csv_fieldnames()
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return csv_path
