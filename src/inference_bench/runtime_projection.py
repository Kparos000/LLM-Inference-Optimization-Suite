"""Runtime and RunPod projection helpers for measured GPU gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _seconds_hours(seconds: float) -> dict[str, float]:
    return {"seconds": seconds, "hours": seconds / 3600.0}


def _projection_status(
    *,
    hourly_price: float | None,
    multiplier: float | None,
) -> str:
    missing: list[str] = []
    if hourly_price is None:
        missing.append("price_missing")
    if multiplier is None:
        missing.append("throughput_multiplier_missing")
    return "projected" if not missing else "_and_".join(missing)


def build_runtime_projection_report(
    *,
    measured_prompt_count: int,
    measured_wall_seconds: float,
    runpod_profiles: dict[str, dict[str, object]],
    target_prompt_counts: tuple[int, ...] = (500, 2500, 5000, 10000),
    matrix_config_counts: tuple[int, ...] = (8, 16, 32),
) -> dict[str, Any]:
    """Build measured RTX 3070 and projected RunPod runtime/cost estimates."""

    if measured_prompt_count <= 0:
        raise ValueError("measured_prompt_count must be > 0")
    if measured_wall_seconds <= 0:
        raise ValueError("measured_wall_seconds must be > 0")
    measured_requests_per_second = measured_prompt_count / measured_wall_seconds
    seconds_per_prompt = measured_wall_seconds / measured_prompt_count
    prompt_projections = [
        {
            "prompt_count": prompt_count,
            **_seconds_hours(prompt_count * seconds_per_prompt),
        }
        for prompt_count in target_prompt_counts
    ]
    matrix_projections = [
        {
            "config_count": config_count,
            "prompt_count_per_config": 10000,
            "total_prompt_executions": config_count * 10000,
            **_seconds_hours(config_count * 10000 * seconds_per_prompt),
        }
        for config_count in matrix_config_counts
    ]

    runpod: dict[str, Any] = {}
    for gpu_name, profile in runpod_profiles.items():
        hourly_price = _float_or_none(profile.get("hourly_price_usd"))
        multiplier = _float_or_none(profile.get("throughput_multiplier_vs_rtx3070"))
        status = _projection_status(hourly_price=hourly_price, multiplier=multiplier)
        projected_cost_by_prompt_count: dict[int, float | None] = {}
        for prompt_count in (1000, 10000, 40000):
            projected_seconds = (
                prompt_count * seconds_per_prompt / multiplier if multiplier is not None else None
            )
            projected_cost_by_prompt_count[prompt_count] = (
                projected_seconds / 3600.0 * hourly_price
                if projected_seconds is not None and hourly_price is not None
                else None
            )
        gpu_matrices: list[dict[str, object]] = []
        for matrix in matrix_projections:
            baseline_seconds = float(matrix["seconds"])
            projected_seconds = baseline_seconds / multiplier if multiplier is not None else None
            projected_hours = projected_seconds / 3600.0 if projected_seconds is not None else None
            projected_cost = (
                projected_hours * hourly_price
                if projected_hours is not None and hourly_price is not None
                else None
            )
            gpu_matrices.append(
                {
                    "config_count": matrix["config_count"],
                    "total_prompt_executions": matrix["total_prompt_executions"],
                    "projected_seconds": projected_seconds,
                    "projected_hours": projected_hours,
                    "hourly_price_usd": hourly_price,
                    "projected_cost_usd": projected_cost,
                    "status": status,
                }
            )
        runpod[gpu_name] = {
            "hourly_price_usd": hourly_price,
            "estimated_run_cost": None,
            "projected_1000_cost": projected_cost_by_prompt_count[1000],
            "projected_10000_cost": projected_cost_by_prompt_count[10000],
            "projected_40000_cost": projected_cost_by_prompt_count[40000],
            "tokens_per_gpu_dollar": None,
            "successful_requests_per_gpu_dollar": None,
            "throughput_multiplier_vs_rtx3070": multiplier,
            "status": status,
            "is_measured": False,
            "projection_basis": "B6 measured RTX 3070 concurrency-one throughput",
            "matrix_projections": gpu_matrices,
        }

    return {
        "projection_type": "b6_measured_rtx3070_to_runpod_projection",
        "is_projection": True,
        "measured_prompt_count": measured_prompt_count,
        "measured_wall_seconds": measured_wall_seconds,
        "measured_requests_per_second": measured_requests_per_second,
        "measured_seconds_per_prompt": seconds_per_prompt,
        "rtx3070_prompt_projections": prompt_projections,
        "rtx3070_matrix_projections": matrix_projections,
        "runpod_gpu_projections": runpod,
        "assumptions": [
            "concurrency remains one",
            "prompt and output distributions remain comparable",
            "RunPod numbers are projections until measured on that GPU",
            "cost requires configured hourly price",
            "runtime requires configured throughput multiplier",
        ],
    }


def runtime_projection_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten projection output for CSV."""

    rows: list[dict[str, Any]] = []
    for row in report["rtx3070_prompt_projections"]:
        rows.append({"scope": "rtx3070_prompt", "gpu": "rtx3070", **row})
    for row in report["rtx3070_matrix_projections"]:
        rows.append({"scope": "rtx3070_matrix", "gpu": "rtx3070", **row})
    for gpu_name, payload in report["runpod_gpu_projections"].items():
        for row in payload["matrix_projections"]:
            rows.append({"scope": "runpod_matrix", "gpu": gpu_name, **row})
    return rows


def write_runtime_projection_artifacts(
    *,
    report: dict[str, Any],
    report_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and CSV runtime projection artifacts."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = runtime_projection_summary_rows(report)
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_output, summary_output
