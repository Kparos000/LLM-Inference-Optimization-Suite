"""Helpers for stronger-model generation-contract validation."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from inference_bench.context_corpora import VERTICALS
from inference_bench.context_schema import WorkloadRecord
from inference_bench.workload_adapter import (
    load_phase3_workload_records,
    workload_record_to_runner_item,
    write_runner_workload_jsonl,
)
from inference_bench.workloads.loader import load_jsonl_workload

COMPARISON_METRICS = (
    "json_valid_rate",
    "generation_contract_valid_rate",
    "evidence_id_presence_rate",
    "evidence_match_rate",
    "grounded_rate",
    "mean_latency_ms",
    "median_latency_ms",
    "total_input_tokens",
    "total_output_tokens",
)


def _json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return cast(dict[str, Any], loaded) if isinstance(loaded, dict) else {}


def validate_promoted_runner_metadata(metadata: dict[str, object]) -> None:
    """Require the promoted Qdrant prompt-plus-metadata retrieval baseline."""

    if str(metadata.get("memory_mode") or "") != "mm2_hybrid_top5":
        msg = "Stronger-model validation requires memory_mode mm2_hybrid_top5"
        raise ValueError(msg)
    if str(metadata.get("ablation_mode") or "") != "prompt_plus_metadata":
        msg = "Stronger-model validation requires ablation_mode prompt_plus_metadata"
        raise ValueError(msg)
    retrieval = _json_mapping(metadata.get("retrieval_metadata"))
    dense_backend = str(
        retrieval.get("dense_backend") or retrieval.get("retrieval_backend_label") or ""
    )
    if dense_backend != "qdrant_vector":
        msg = "Stronger-model validation requires the promoted qdrant_vector retrieval path"
        raise ValueError(msg)
    if str(retrieval.get("vector_store") or "") != "qdrant_local":
        msg = "Stronger-model validation requires vector_store qdrant_local"
        raise ValueError(msg)
    if retrieval.get("source_hints_used") is True:
        msg = "Stronger-model validation must not use source-hint-assisted retrieval"
        raise ValueError(msg)


def select_one_per_vertical(records: list[WorkloadRecord]) -> list[WorkloadRecord]:
    """Select the first promoted workload record for each vertical."""

    selected: dict[str, WorkloadRecord] = {}
    for record in records:
        if record.vertical in selected:
            continue
        metadata = {
            "memory_mode": record.memory_mode,
            "ablation_mode": record.retrieval_metadata.get("ablation_mode"),
            "retrieval_metadata": record.retrieval_metadata,
        }
        validate_promoted_runner_metadata(metadata)
        selected[record.vertical] = record
    missing = [vertical for vertical in VERTICALS if vertical not in selected]
    if missing:
        msg = f"Could not select one promoted workload record for verticals: {missing}"
        raise ValueError(msg)
    return [selected[vertical] for vertical in VERTICALS]


def build_promoted_runner_input(
    *,
    workload_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export one promoted mm2 workload item per vertical."""

    records = load_phase3_workload_records(workload_path)
    selected = select_one_per_vertical(records)
    items = [workload_record_to_runner_item(record) for record in selected]
    for item in items:
        validate_promoted_runner_metadata(cast(dict[str, object], item.metadata))
    return write_runner_workload_jsonl(items, output_path)


def load_and_validate_runner_input(path: str | Path) -> list[dict[str, Any]]:
    """Load runner items and validate promoted retrieval metadata."""

    items = load_jsonl_workload(path)
    if len(items) != len(VERTICALS):
        msg = f"Expected exactly {len(VERTICALS)} runner items, found {len(items)}"
        raise ValueError(msg)
    verticals: set[str] = set()
    rows: list[dict[str, Any]] = []
    for item in items:
        validate_promoted_runner_metadata(cast(dict[str, object], item.metadata))
        verticals.add(str(item.metadata.get("vertical") or ""))
        rows.append(asdict(item))
    if verticals != set(VERTICALS):
        msg = f"Runner input must contain one record per vertical; found {sorted(verticals)}"
        raise ValueError(msg)
    return rows


def huggingface_cache_roots(explicit_root: str | Path | None = None) -> list[Path]:
    """Return candidate Hugging Face Hub cache roots."""

    if explicit_root is not None:
        return [Path(explicit_root)]
    roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        roots.append(Path(local_app_data) / "huggingface" / "hub")
    return list(dict.fromkeys(roots))


def model_cache_directory(model_id: str, cache_root: Path) -> Path:
    """Return the Hugging Face cache directory for one model ID."""

    return cache_root / f"models--{model_id.replace('/', '--')}"


def is_model_cached(
    model_id: str,
    *,
    cache_root: str | Path | None = None,
) -> bool:
    """Return whether model weights and config exist in a local Hub snapshot."""

    for root in huggingface_cache_roots(cache_root):
        snapshots = model_cache_directory(model_id, root) / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
                continue
            if any(snapshot.glob("*.safetensors")) or any(snapshot.glob("*.bin")):
                return True
    return False


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL result rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    msg = "Expected JSON object result row"
                    raise ValueError(msg)
                rows.append(cast(dict[str, Any], loaded))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write JSONL result rows."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def runtime_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Calculate latency and token totals from generation result rows."""

    latencies = sorted(_numeric_values(rows, "latency_ms"))
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    median_latency: float | None = None
    if latencies:
        midpoint = len(latencies) // 2
        if len(latencies) % 2:
            median_latency = latencies[midpoint]
        else:
            median_latency = (latencies[midpoint - 1] + latencies[midpoint]) / 2
    return {
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
    }


def load_baseline_metrics(
    *,
    baseline_report_path: str | Path,
) -> dict[str, float | int | None]:
    """Load Block 24 quality and runtime metrics when local artifacts exist."""

    report_path = Path(baseline_report_path)
    if not report_path.is_file():
        return {metric: None for metric in COMPARISON_METRICS}
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {metric: None for metric in COMPARISON_METRICS}
    summary = loaded.get("summary")
    metrics: dict[str, float | int | None] = {metric: None for metric in COMPARISON_METRICS}
    if isinstance(summary, dict):
        for metric in COMPARISON_METRICS[:5]:
            value = summary.get(metric)
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics[metric] = float(value)
    results_path = loaded.get("results_path")
    if isinstance(results_path, str) and Path(results_path).is_file():
        metrics.update(runtime_metrics(read_jsonl(results_path)))
    return metrics


def write_comparison_artifacts(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    validation_status: str,
    model_alias: str,
    model_id: str,
    execution_path: str,
    baseline_metrics: dict[str, float | int | None],
    stronger_metrics: dict[str, float | int | None],
    evaluation_rows: list[dict[str, Any]],
    reason: str | None,
) -> tuple[Path, Path]:
    """Write Block 24 versus stronger-model comparison artifacts."""

    comparison: dict[str, dict[str, float | int | None]] = {}
    for metric in COMPARISON_METRICS:
        baseline_value = baseline_metrics.get(metric)
        stronger_value = stronger_metrics.get(metric)
        delta = (
            float(stronger_value) - float(baseline_value)
            if stronger_value is not None and baseline_value is not None
            else None
        )
        comparison[metric] = {
            "block24_model1_0_5b": baseline_value,
            "stronger_model": stronger_value,
            "delta": delta,
        }
    report = {
        "validation_status": validation_status,
        "model_alias": model_alias,
        "model_id": model_id,
        "execution_path": execution_path,
        "reason": reason,
        "comparison": comparison,
        "evaluation_rows": evaluation_rows,
        "no_gpu_work_triggered": True,
        "no_vllm_triggered": True,
        "paid_api_call_triggered": execution_path == "hf_inference_provider_paid",
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
        "validation_status",
        "model_alias",
        "model_id",
        "execution_path",
        "metric",
        "block24_model1_0_5b",
        "stronger_model",
        "delta",
        "reason",
    ]
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for metric, values in comparison.items():
            writer.writerow(
                {
                    "validation_status": validation_status,
                    "model_alias": model_alias,
                    "model_id": model_id,
                    "execution_path": execution_path,
                    "metric": metric,
                    **values,
                    "reason": reason or "",
                }
            )
    return report_output, summary_output
