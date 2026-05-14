"""CSV result summary utilities."""

from __future__ import annotations

import csv
from pathlib import Path


def read_results_csv(path: str | Path) -> list[dict[str, str]]:
    """Read benchmark result rows from a CSV file."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _non_blank_values(rows: list[dict[str, str]], field_name: str) -> list[str]:
    return [row[field_name] for row in rows if row.get(field_name, "").strip()]


def _sorted_unique_values(rows: list[dict[str, str]], field_name: str) -> list[str]:
    return sorted(set(_non_blank_values(rows, field_name)))


def _float_values(rows: list[dict[str, str]], field_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(field_name, "").strip()
        if raw_value:
            values.append(float(raw_value))
    return values


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile <= 100:
        msg = "percentile must be between 0 and 100"
        raise ValueError(msg)

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * weight
    )


def _percentile_summary(rows: list[dict[str, str]], field_name: str) -> dict[str, float | None]:
    values = _float_values(rows, field_name)
    return {
        f"p50_{field_name}": _percentile(values, 50),
        f"p90_{field_name}": _percentile(values, 90),
        f"p95_{field_name}": _percentile(values, 95),
        f"p99_{field_name}": _percentile(values, 99),
    }


def summarize_results(path: str | Path) -> dict[str, object]:
    """Summarize a benchmark result CSV."""

    rows = read_results_csv(path)
    success_count = sum(1 for row in rows if row.get("success", "").strip().lower() == "true")
    cost_values = _float_values(rows, "estimated_cost_usd")
    latency_values = _float_values(rows, "end_to_end_latency_ms")
    ttft_values = _float_values(rows, "ttft_ms")
    tpot_values = _float_values(rows, "tpot_ms")
    throughput_values = _float_values(rows, "throughput_tokens_per_second")

    return {
        "row_count": len(rows),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "backends": _sorted_unique_values(rows, "backend"),
        "models": _sorted_unique_values(rows, "model_name"),
        "optimizations": _sorted_unique_values(rows, "optimization"),
        "workloads": _sorted_unique_values(rows, "workload_name"),
        "avg_end_to_end_latency_ms": _average(latency_values),
        **_percentile_summary(rows, "end_to_end_latency_ms"),
        "avg_ttft_ms": _average(ttft_values),
        **_percentile_summary(rows, "ttft_ms"),
        "avg_tpot_ms": _average(tpot_values),
        **_percentile_summary(rows, "tpot_ms"),
        "avg_throughput_tokens_per_second": _average(throughput_values),
        **_percentile_summary(rows, "throughput_tokens_per_second"),
        "total_estimated_cost_usd": sum(cost_values) if cost_values else None,
    }
