"""Phase 1 report plot generation from curated comparison artifacts."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402,I001


DEFAULT_INPUT_CSV = (
    "results/samples/processed/vllm_qwen0_5b_all_workloads_5000_concurrency_comparison_sample.csv"
)
DEFAULT_OUTPUT_DIR = "results/samples/figures/phase1"
DEFAULT_TITLE_PREFIX = "Qwen 0.5B vLLM 5,000-Prompt Benchmark"


class Phase1PlotManifest(TypedDict):
    input_csv: str
    output_dir: str
    generated_plots: list[str]
    skipped_plots: list[str]
    created_at_utc: str


_PlotRow = dict[str, str | int | float]


class _PlotSpec(TypedDict):
    file_name: str
    x_field: str
    y_field: str
    title: str
    x_label: str
    y_label: str
    kind: str


_CONCURRENCY_RE = re.compile(r"conc(\d+)")


def _read_comparison_rows(input_csv: str | Path) -> list[dict[str, str]]:
    csv_path = Path(input_csv)
    with csv_path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _extract_workload(row: dict[str, str]) -> str | None:
    workload = row.get("workloads", "").strip()
    if workload:
        return workload
    source_file = row.get("source_file", "")
    for candidate in (
        "short_chat",
        "code_helpdesk",
        "long_context",
        "shared_prefix",
        "structured_output",
    ):
        if candidate in source_file:
            return candidate
    return None


def _extract_concurrency(row: dict[str, str]) -> int | None:
    source_file = row.get("source_file", "")
    match = _CONCURRENCY_RE.search(source_file)
    if match is None:
        return None
    return int(match.group(1))


def _metadata_candidates(input_csv: Path, source_file: str) -> list[Path]:
    source_path = Path(source_file)
    metadata_name = source_path.name.replace("_results.csv", "_metadata.json")

    candidates = [
        source_path.with_name(metadata_name),
        input_csv.parent.parent / "raw" / metadata_name,
        Path("results/samples/raw") / metadata_name,
    ]
    return list(dict.fromkeys(candidates))


def _read_metadata(input_csv: Path, source_file: str) -> dict[str, object]:
    for candidate in _metadata_candidates(input_csv, source_file):
        if candidate.exists():
            with candidate.open(encoding="utf-8") as file:
                loaded = json.load(file)
            if isinstance(loaded, dict):
                return loaded
    return {}


def _enrich_rows(input_csv: str | Path) -> list[_PlotRow]:
    csv_path = Path(input_csv)
    enriched_rows: list[_PlotRow] = []
    numeric_fields = (
        "success_count",
        "failure_count",
        "avg_end_to_end_latency_ms",
        "p95_end_to_end_latency_ms",
        "p99_end_to_end_latency_ms",
        "avg_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "avg_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
    )

    for row in _read_comparison_rows(csv_path):
        workload = _extract_workload(row)
        concurrency = _extract_concurrency(row)
        if workload is None or concurrency is None:
            continue

        plot_row: _PlotRow = {
            "source_file": row.get("source_file", ""),
            "workload": workload,
            "concurrency": concurrency,
        }
        for field_name in numeric_fields:
            value = _float_or_none(row.get(field_name))
            if value is not None:
                plot_row[field_name] = value

        metadata = _read_metadata(csv_path, row.get("source_file", ""))
        for field_name in (
            "aggregate_requests_per_second",
            "aggregate_output_tokens_per_second",
        ):
            metadata_value = metadata.get(field_name)
            if isinstance(metadata_value, int | float):
                plot_row[field_name] = float(metadata_value)

        enriched_rows.append(plot_row)

    return enriched_rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _group_mean(rows: list[_PlotRow], group_field: str, value_field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(value_field)
        group_value = row.get(group_field)
        if isinstance(value, int | float) and group_value is not None:
            grouped.setdefault(str(group_value), []).append(float(value))
    return {key: _mean(values) for key, values in grouped.items() if values}


def _sort_group_keys(keys: list[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, str]:
        if value.isdigit():
            return (0, f"{int(value):08d}")
        return (1, value)

    return sorted(keys, key=sort_key)


def _write_bar_plot(
    values_by_group: dict[str, float],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = _sort_group_keys(list(values_by_group))
    values = [values_by_group[label] for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=150)
    plt.close(fig)
    return output_path


def _write_scatter_plot(
    rows: list[_PlotRow],
    output_path: Path,
    x_field: str,
    y_field: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values: list[float] = []
    y_values: list[float] = []
    labels: list[str] = []
    for row in rows:
        x_value = row.get(x_field)
        y_value = row.get(y_field)
        workload = row.get("workload")
        concurrency = row.get("concurrency")
        if (
            isinstance(x_value, int | float)
            and isinstance(y_value, int | float)
            and workload is not None
            and concurrency is not None
        ):
            x_values.append(float(x_value))
            y_values.append(float(y_value))
            labels.append(f"{workload} c{concurrency}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x_values, y_values)
    for x_value, y_value, label in zip(x_values, y_values, labels, strict=True):
        ax.annotate(label, (x_value, y_value), fontsize=7, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=150)
    plt.close(fig)
    return output_path


def _plot_has_required_data(rows: list[_PlotRow], fields: tuple[str, ...]) -> bool:
    for row in rows:
        if all(isinstance(row.get(field_name), int | float) for field_name in fields):
            return True
    return False


def _required_bar_specs(title_prefix: str) -> list[_PlotSpec]:
    return [
        {
            "file_name": "aggregate_requests_per_second_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "aggregate_requests_per_second",
            "title": f"{title_prefix}: Aggregate Requests/sec by Concurrency",
            "x_label": "Concurrency",
            "y_label": "Aggregate requests/sec",
            "kind": "bar",
        },
        {
            "file_name": "aggregate_output_tokens_per_second_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "aggregate_output_tokens_per_second",
            "title": f"{title_prefix}: Aggregate Output Tokens/sec by Concurrency",
            "x_label": "Concurrency",
            "y_label": "Aggregate output tokens/sec",
            "kind": "bar",
        },
        {
            "file_name": "avg_latency_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "avg_end_to_end_latency_ms",
            "title": f"{title_prefix}: Average Latency by Concurrency",
            "x_label": "Concurrency",
            "y_label": "Average latency (ms)",
            "kind": "bar",
        },
        {
            "file_name": "p95_latency_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "p95_end_to_end_latency_ms",
            "title": f"{title_prefix}: p95 Latency by Concurrency",
            "x_label": "Concurrency",
            "y_label": "p95 latency (ms)",
            "kind": "bar",
        },
        {
            "file_name": "p99_latency_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "p99_end_to_end_latency_ms",
            "title": f"{title_prefix}: p99 Latency by Concurrency",
            "x_label": "Concurrency",
            "y_label": "p99 latency (ms)",
            "kind": "bar",
        },
        {
            "file_name": "avg_ttft_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "avg_ttft_ms",
            "title": f"{title_prefix}: Average TTFT by Concurrency",
            "x_label": "Concurrency",
            "y_label": "Average TTFT (ms)",
            "kind": "bar",
        },
        {
            "file_name": "p95_ttft_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "p95_ttft_ms",
            "title": f"{title_prefix}: p95 TTFT by Concurrency",
            "x_label": "Concurrency",
            "y_label": "p95 TTFT (ms)",
            "kind": "bar",
        },
        {
            "file_name": "p99_ttft_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "p99_ttft_ms",
            "title": f"{title_prefix}: p99 TTFT by Concurrency",
            "x_label": "Concurrency",
            "y_label": "p99 TTFT (ms)",
            "kind": "bar",
        },
        {
            "file_name": "avg_tpot_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "avg_tpot_ms",
            "title": f"{title_prefix}: Average TPOT by Concurrency",
            "x_label": "Concurrency",
            "y_label": "Average TPOT (ms)",
            "kind": "bar",
        },
        {
            "file_name": "p95_tpot_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "p95_tpot_ms",
            "title": f"{title_prefix}: p95 TPOT by Concurrency",
            "x_label": "Concurrency",
            "y_label": "p95 TPOT (ms)",
            "kind": "bar",
        },
        {
            "file_name": "p99_tpot_by_concurrency.png",
            "x_field": "concurrency",
            "y_field": "p99_tpot_ms",
            "title": f"{title_prefix}: p99 TPOT by Concurrency",
            "x_label": "Concurrency",
            "y_label": "p99 TPOT (ms)",
            "kind": "bar",
        },
    ]


def _workload_specs(title_prefix: str) -> list[_PlotSpec]:
    return [
        {
            "file_name": "workload_avg_latency_at_conc32.png",
            "x_field": "workload",
            "y_field": "avg_end_to_end_latency_ms",
            "title": f"{title_prefix}: Workload Average Latency at Concurrency 32",
            "x_label": "Workload",
            "y_label": "Average latency (ms)",
            "kind": "bar",
        },
        {
            "file_name": "workload_aggregate_requests_at_conc32.png",
            "x_field": "workload",
            "y_field": "aggregate_requests_per_second",
            "title": f"{title_prefix}: Workload Aggregate Requests/sec at Concurrency 32",
            "x_label": "Workload",
            "y_label": "Aggregate requests/sec",
            "kind": "bar",
        },
        {
            "file_name": "workload_p99_latency_at_conc32.png",
            "x_field": "workload",
            "y_field": "p99_end_to_end_latency_ms",
            "title": f"{title_prefix}: Workload p99 Latency at Concurrency 32",
            "x_label": "Workload",
            "y_label": "p99 latency (ms)",
            "kind": "bar",
        },
        {
            "file_name": "workload_p99_ttft_at_conc32.png",
            "x_field": "workload",
            "y_field": "p99_ttft_ms",
            "title": f"{title_prefix}: Workload p99 TTFT at Concurrency 32",
            "x_label": "Workload",
            "y_label": "p99 TTFT (ms)",
            "kind": "bar",
        },
    ]


def _tradeoff_specs(title_prefix: str) -> list[_PlotSpec]:
    return [
        {
            "file_name": "throughput_vs_avg_latency.png",
            "x_field": "aggregate_requests_per_second",
            "y_field": "avg_end_to_end_latency_ms",
            "title": f"{title_prefix}: Throughput vs Average Latency",
            "x_label": "Aggregate requests/sec",
            "y_label": "Average latency (ms)",
            "kind": "scatter",
        },
        {
            "file_name": "throughput_vs_p99_latency.png",
            "x_field": "aggregate_requests_per_second",
            "y_field": "p99_end_to_end_latency_ms",
            "title": f"{title_prefix}: Throughput vs p99 Latency",
            "x_label": "Aggregate requests/sec",
            "y_label": "p99 latency (ms)",
            "kind": "scatter",
        },
        {
            "file_name": "aggregate_requests_vs_p99_ttft.png",
            "x_field": "aggregate_requests_per_second",
            "y_field": "p99_ttft_ms",
            "title": f"{title_prefix}: Aggregate Requests/sec vs p99 TTFT",
            "x_label": "Aggregate requests/sec",
            "y_label": "p99 TTFT (ms)",
            "kind": "scatter",
        },
    ]


def _success_failure_specs(title_prefix: str) -> list[_PlotSpec]:
    return [
        {
            "file_name": "failure_count_by_workload_concurrency.png",
            "x_field": "workload",
            "y_field": "failure_count",
            "title": f"{title_prefix}: Failure Count by Workload and Concurrency",
            "x_label": "Workload/concurrency",
            "y_label": "Failure count",
            "kind": "bar_with_concurrency",
        },
        {
            "file_name": "success_count_by_workload_concurrency.png",
            "x_field": "workload",
            "y_field": "success_count",
            "title": f"{title_prefix}: Success Count by Workload and Concurrency",
            "x_label": "Workload/concurrency",
            "y_label": "Success count",
            "kind": "bar_with_concurrency",
        },
    ]


def _write_workload_concurrency_bar(
    rows: list[_PlotRow],
    output_path: Path,
    value_field: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    values: dict[str, float] = {}
    for row in rows:
        workload = row.get("workload")
        concurrency = row.get("concurrency")
        value = row.get(value_field)
        if (
            isinstance(workload, str)
            and isinstance(concurrency, int)
            and isinstance(value, int | float)
        ):
            values[f"{workload}\nc{concurrency}"] = float(value)
    return _write_bar_plot(values, output_path, title, x_label, y_label)


def _generate_plot_for_spec(
    rows: list[_PlotRow],
    output_dir: Path,
    spec: _PlotSpec,
) -> Path | str:
    file_name = spec["file_name"]
    x_field = spec["x_field"]
    y_field = spec["y_field"]
    kind = spec["kind"]
    output_path = output_dir / file_name

    if kind == "bar":
        if not _plot_has_required_data(rows, (y_field,)):
            return f"{file_name}: missing required column {y_field}"
        values = _group_mean(rows, x_field, y_field)
        if not values:
            return f"{file_name}: no usable rows"
        return _write_bar_plot(values, output_path, spec["title"], spec["x_label"], spec["y_label"])

    if kind == "scatter":
        if not _plot_has_required_data(rows, (x_field, y_field)):
            return f"{file_name}: missing required columns {x_field}, {y_field}"
        return _write_scatter_plot(
            rows, output_path, x_field, y_field, spec["title"], spec["x_label"], spec["y_label"]
        )

    if kind == "bar_with_concurrency":
        if not _plot_has_required_data(rows, (y_field,)):
            return f"{file_name}: missing required column {y_field}"
        return _write_workload_concurrency_bar(
            rows, output_path, y_field, spec["title"], spec["x_label"], spec["y_label"]
        )

    return f"{file_name}: unsupported plot kind {kind}"


def generate_phase1_plots(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    title_prefix: str = DEFAULT_TITLE_PREFIX,
) -> Phase1PlotManifest:
    """Generate Phase 1 report plots and a JSON manifest."""

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    plot_rows = _enrich_rows(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_plots: list[str] = []
    skipped_plots: list[str] = []
    rows_at_concurrency_32 = [row for row in plot_rows if row.get("concurrency") == 32]

    specs = (
        _required_bar_specs(title_prefix)
        + _workload_specs(title_prefix)
        + _tradeoff_specs(title_prefix)
        + _success_failure_specs(title_prefix)
    )
    for spec in specs:
        rows_for_plot = (
            rows_at_concurrency_32 if spec["file_name"].startswith("workload_") else plot_rows
        )
        result = _generate_plot_for_spec(rows_for_plot, output_path, spec)
        if isinstance(result, Path):
            generated_plots.append(result.as_posix())
        else:
            skipped_plots.append(result)
            print(f"Warning: skipped plot: {result}")

    manifest: Phase1PlotManifest = {
        "input_csv": input_path.as_posix(),
        "output_dir": output_path.as_posix(),
        "generated_plots": generated_plots,
        "skipped_plots": skipped_plots,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    manifest_path = output_path / "plot_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    return manifest
