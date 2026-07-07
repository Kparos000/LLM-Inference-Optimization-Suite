"""Post-run automation for SLO, comparison, and plotting artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from inference_bench.run_manifest import read_run_manifest, utc_now
from inference_bench.slo import load_slo_config

AUTOMATION_METRICS = (
    "ttft_ms",
    "tpot_ms",
    "latency_ms",
    "throughput",
    "gpu_utilization",
    "vram",
    "power",
    "cost",
    "json_validity",
    "groundedness",
    "evidence_match",
)
SLO_TARGET_MAP = {
    "ttft_ms": ("latency_slo", "ttft_p95_ms_max", "max", 1.0),
    "tpot_ms": ("latency_slo", "tpot_p95_ms_max", "max", 1.0),
    "latency_ms": ("latency_slo", "e2e_p95_ms_max", "max", 1.0),
    "throughput": ("throughput_slo", "tokens_per_second_min", "min", 1.0),
    "gpu_utilization": ("resource_slo", "gpu_utilization_min_pct", "min", 1.0),
    "vram": ("resource_slo", "gpu_memory_peak_gb_max", "max", 1024.0),
    "json_validity": ("quality_slo", "format_validity_min", "min", 1.0),
    "groundedness": ("quality_slo", "groundedness_min", "min", 1.0),
    "evidence_match": ("quality_slo", "evidence_match_min", "min", 1.0),
}


@dataclass(frozen=True)
class PostRunAutomationInputs:
    """Artifact inputs produced by a benchmark or smoke runner."""

    run_id: str
    manifest_path: str
    eval_summary_path: str | None = None
    latency_summary_path: str | None = None
    telemetry_summary_path: str | None = None
    cost_report_path: str | None = None
    comparison_paths: tuple[str, ...] = ()


def _read_json(path: str | Path) -> dict[str, Any]:
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, Any], parsed)


def _read_csv_first(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return dict(next(csv.DictReader(file), {}))


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _status_for_metric(
    value: float | None,
    *,
    target: float | None,
    direction: str | None,
) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    if target is None or direction is None:
        return "NOT_AVAILABLE"
    if direction == "min":
        if value >= target:
            return "PASS"
        if value >= target * 0.9:
            return "WARNING"
        return "FAIL"
    if value <= target:
        return "PASS"
    if value <= target * 1.1:
        return "WARNING"
    return "FAIL"


def _load_default_slo_config() -> dict[str, Any]:
    slo_path = Path("configs/slo_targets.yaml")
    return load_slo_config(slo_path) if slo_path.exists() else {}


def _strictest_slo_targets(slo_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verticals = slo_config.get("verticals")
    if not isinstance(verticals, dict):
        return {}
    targets: dict[str, dict[str, Any]] = {}
    for metric, (family, slo_key, direction, multiplier) in SLO_TARGET_MAP.items():
        values: list[float] = []
        for raw_vertical in verticals.values():
            if not isinstance(raw_vertical, dict):
                continue
            raw_family = raw_vertical.get(family)
            if not isinstance(raw_family, dict):
                continue
            raw_value = raw_family.get(slo_key)
            if isinstance(raw_value, bool):
                continue
            if isinstance(raw_value, int | float):
                values.append(float(raw_value) * multiplier)
        if values:
            targets[metric] = {
                "target": max(values) if direction == "min" else min(values),
                "direction": direction,
                "source": f"configs/slo_targets.yaml:{family}.{slo_key}",
            }
    return targets


def _metric_rows(
    *,
    eval_summary: dict[str, Any],
    latency_summary: dict[str, Any],
    telemetry_summary: dict[str, Any],
    cost_report: dict[str, Any],
    slo_targets: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    observed = {
        "ttft_ms": _optional_float(latency_summary.get("mean_ttft_ms")),
        "tpot_ms": _optional_float(latency_summary.get("mean_tpot_ms")),
        "latency_ms": _optional_float(latency_summary.get("mean_e2e_latency_ms")),
        "throughput": _optional_float(latency_summary.get("mean_total_tokens_per_second")),
        "gpu_utilization": _optional_float(
            telemetry_summary.get("mean_gpu_utilization_percent")
            or telemetry_summary.get("gpu_utilization_percent")
        ),
        "vram": _optional_float(
            telemetry_summary.get("max_gpu_memory_used_mb")
            or telemetry_summary.get("gpu_memory_peak_mb")
        ),
        "power": _optional_float(
            telemetry_summary.get("mean_power_draw_watts") or telemetry_summary.get("power_watts")
        ),
        "cost": _optional_float(
            cost_report.get("total_cost_usd")
            or cost_report.get("gpu_cost_usd")
            or cost_report.get("api_cost_usd")
        ),
        "json_validity": _optional_float(eval_summary.get("json_valid_rate")),
        "groundedness": _optional_float(eval_summary.get("grounded_rate")),
        "evidence_match": _optional_float(eval_summary.get("evidence_match_rate")),
    }
    rows = []
    for metric in AUTOMATION_METRICS:
        target = slo_targets.get(metric, {})
        rows.append(
            {
                "metric_name": metric,
                "observed": observed[metric],
                "target": target.get("target"),
                "direction": target.get("direction"),
                "target_source": target.get("source"),
                "status": _status_for_metric(
                    observed[metric],
                    target=_optional_float(target.get("target")),
                    direction=str(target.get("direction")) if target.get("direction") else None,
                ),
            }
        )
    return rows


def build_post_run_automation_report(inputs: PostRunAutomationInputs) -> dict[str, Any]:
    """Build automatic post-run SLO/comparison/plotting metadata."""

    manifest = read_run_manifest(inputs.manifest_path)
    eval_summary = _read_csv_first(inputs.eval_summary_path) if inputs.eval_summary_path else {}
    latency_summary = (
        _read_csv_first(inputs.latency_summary_path) if inputs.latency_summary_path else {}
    )
    telemetry_summary = (
        _read_json(inputs.telemetry_summary_path) if inputs.telemetry_summary_path else {}
    )
    cost_report = _read_json(inputs.cost_report_path) if inputs.cost_report_path else {}
    slo_targets = _strictest_slo_targets(_load_default_slo_config())
    metric_rows = _metric_rows(
        eval_summary=eval_summary,
        latency_summary=latency_summary,
        telemetry_summary=telemetry_summary,
        cost_report=cost_report,
        slo_targets=slo_targets,
    )
    return {
        "run_id": inputs.run_id,
        "generated_at_utc": utc_now(),
        "manifest": {
            key: manifest.get(key)
            for key in (
                "run_id",
                "baseline_or_optimized",
                "engine",
                "model_alias",
                "memory_mode",
                "concurrency",
                "traffic_profile",
                "hardware",
                "prompt_count",
                "vertical",
                "optimization_flags",
                "git_commit",
                "dataset_version",
            )
        },
        "slo_metric_rows": metric_rows,
        "slo_status_counts": {
            status: sum(row["status"] == status for row in metric_rows)
            for status in ("PASS", "WARNING", "FAIL", "NOT_AVAILABLE")
        },
        "comparison_reports": [
            {"path": path, "available": Path(path).exists()} for path in inputs.comparison_paths
        ],
        "optimization_recommendations_status": "available_after_failed_slo_diagnosis",
        "plotting_datasets": [
            "baseline_vs_optimized",
            "engine_comparison",
            "memory_mode_comparison",
            "concurrency_scaling",
            "gpu_utilization",
            "vram",
            "power",
            "cost",
            "latency",
            "throughput",
        ],
    }


def build_plotting_dataset_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return long-form plotting rows for automated chart generation."""

    manifest = report.get("manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    rows: list[dict[str, Any]] = []
    for metric in report.get("slo_metric_rows", []):
        if not isinstance(metric, dict):
            continue
        rows.append(
            {
                "run_id": report.get("run_id"),
                "baseline_or_optimized": manifest.get("baseline_or_optimized"),
                "engine": manifest.get("engine"),
                "model_alias": manifest.get("model_alias"),
                "memory_mode": manifest.get("memory_mode"),
                "concurrency": manifest.get("concurrency"),
                "traffic_profile": manifest.get("traffic_profile"),
                "hardware": manifest.get("hardware"),
                "metric_name": metric.get("metric_name"),
                "metric_value": metric.get("observed"),
                "metric_status": metric.get("status"),
            }
        )
    return rows


def write_post_run_automation_artifacts(
    *,
    report: dict[str, Any],
    report_path: str | Path,
    plotting_dataset_path: str | Path,
) -> tuple[Path, Path]:
    """Write post-run automation JSON and plotting CSV artifacts."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = build_plotting_dataset_rows(report)
    plotting_output = Path(plotting_dataset_path)
    plotting_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row}) if rows else ["run_id"]
    with plotting_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_output, plotting_output
