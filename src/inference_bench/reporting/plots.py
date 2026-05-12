"""Report-ready plot generation utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from inference_bench.reporting.summary import read_results_csv


def _float_or_none(value: str) -> float | None:
    stripped_value = value.strip()
    if not stripped_value:
        return None
    return float(stripped_value)


def _group_numeric_values(
    input_csv: str | Path,
    metric_field: str,
) -> dict[str, list[float]]:
    grouped_values: dict[str, list[float]] = {}
    for row in read_results_csv(input_csv):
        optimization = row.get("optimization", "").strip()
        if not optimization:
            continue

        value = _float_or_none(row.get(metric_field, ""))
        if value is None:
            continue

        grouped_values.setdefault(optimization, []).append(value)
    return grouped_values


def _write_bar_plot(
    values_by_optimization: dict[str, float],
    output_path: str | Path,
    title: str,
    y_label: str,
) -> Path:
    if not values_by_optimization:
        msg = f"No usable data available for {title.lower()}"
        raise ValueError(msg)

    png_path = Path(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    optimizations = sorted(values_by_optimization)
    values = [values_by_optimization[optimization] for optimization in optimizations]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(optimizations, values, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel("Optimization")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png_path, format="png", dpi=150)
    plt.close(fig)
    return png_path


def _plot_by_optimization(
    input_csv: str | Path,
    output_path: str | Path,
    metric_field: str,
    reducer: Callable[[list[float]], float],
    title: str,
    y_label: str,
) -> Path:
    grouped_values = _group_numeric_values(input_csv, metric_field)
    reduced_values = {
        optimization: reducer(values) for optimization, values in grouped_values.items() if values
    }
    return _write_bar_plot(reduced_values, output_path, title, y_label)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def plot_latency_by_optimization(input_csv: str | Path, output_path: str | Path) -> Path:
    """Plot average end-to-end latency by optimization."""

    return _plot_by_optimization(
        input_csv=input_csv,
        output_path=output_path,
        metric_field="end_to_end_latency_ms",
        reducer=_mean,
        title="Average Latency by Optimization",
        y_label="End-to-end latency (ms)",
    )


def plot_throughput_by_optimization(input_csv: str | Path, output_path: str | Path) -> Path:
    """Plot average token throughput by optimization."""

    return _plot_by_optimization(
        input_csv=input_csv,
        output_path=output_path,
        metric_field="throughput_tokens_per_second",
        reducer=_mean,
        title="Average Throughput by Optimization",
        y_label="Tokens per second",
    )


def plot_cost_by_optimization(input_csv: str | Path, output_path: str | Path) -> Path:
    """Plot total estimated cost by optimization."""

    return _plot_by_optimization(
        input_csv=input_csv,
        output_path=output_path,
        metric_field="estimated_cost_usd",
        reducer=sum,
        title="Total Cost by Optimization",
        y_label="Estimated cost (USD)",
    )
