"""Production SLO loading and readiness evaluation.

The SLO framework centralizes pass/fail checks for retrieval, quality, latency,
throughput, resource usage, and cost. It does not run inference, GPU work, or
external API calls.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

SLO_VERTICALS = ("airline", "retail", "healthcare_admin", "finance", "research_ai")
SLO_METRIC_FAMILIES = (
    "retrieval_slo",
    "quality_slo",
    "latency_slo",
    "throughput_slo",
    "resource_slo",
    "api_cost_slo",
    "gpu_cost_slo",
)
REQUIRED_METRICS_BY_FAMILY = {
    "retrieval_slo": (
        "candidate_recall_at_20_min",
        "candidate_recall_at_50_min",
        "final_recall_at_5_min",
        "mrr_min",
    ),
    "quality_slo": (
        "groundedness_min",
        "citation_accuracy_min",
        "evidence_match_min",
        "task_success_min",
        "format_validity_min",
        "safety_violations_max",
    ),
    "latency_slo": (
        "ttft_p50_ms_max",
        "ttft_p95_ms_max",
        "ttft_p99_ms_max",
        "itl_p50_ms_max",
        "itl_p95_ms_max",
        "itl_p99_ms_max",
        "tpot_p50_ms_max",
        "tpot_p95_ms_max",
        "tpot_p99_ms_max",
        "e2e_p50_ms_max",
        "e2e_p95_ms_max",
        "e2e_p99_ms_max",
    ),
    "throughput_slo": (
        "requests_per_second_min",
        "tokens_per_second_min",
        "successful_requests_per_second_min",
    ),
    "resource_slo": (
        "gpu_utilization_min_pct",
        "gpu_memory_utilization_max_pct",
        "gpu_memory_peak_gb_max",
        "cpu_utilization_max_pct",
        "ram_usage_gb_max",
    ),
    "api_cost_slo": (
        "input_tokens_per_request_max",
        "output_tokens_per_request_max",
        "total_tokens_per_request_max",
        "api_cost_per_request_usd_max",
        "api_cost_per_1000_requests_usd_max",
        "api_cost_per_successful_answer_usd_max",
        "api_cost_per_grounded_successful_answer_usd_max",
    ),
    "gpu_cost_slo": (
        "gpu_hourly_price_usd_required",
        "gpu_cost_per_request_usd_max",
        "gpu_cost_per_1000_requests_usd_max",
        "gpu_cost_per_successful_answer_usd_max",
        "gpu_cost_per_grounded_successful_answer_usd_max",
        "tokens_per_gpu_dollar_min",
    ),
}
RETRIEVAL_REPORT_METRIC_MAP = {
    "candidate_recall_at_20_min": "candidate_recall_at_20",
    "candidate_recall_at_50_min": "candidate_recall_at_50",
    "final_recall_at_5_min": "recall_at_5",
    "mrr_min": "mrr",
}
SUMMARY_FIELDS = [
    "vertical",
    "metric_family",
    "metric_name",
    "target",
    "observed",
    "status",
    "gap",
    "recommended_action",
]
SloStatus = Literal["PASS", "WARN", "BLOCKED", "NOT_AVAILABLE"]


@dataclass(frozen=True)
class SloResult:
    """One metric-level SLO evaluation result."""

    vertical: str
    metric_family: str
    metric_name: str
    target: float | bool
    observed: float | bool | None
    status: SloStatus
    gap: float | None
    recommended_action: str


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def load_slo_config(path: str | Path = "configs/slo_targets.yaml") -> dict[str, Any]:
    """Load and validate a production SLO config."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = "SLO config must be a mapping"
        raise ValueError(msg)
    validate_slo_config(cast(dict[str, Any], payload))
    return cast(dict[str, Any], payload)


def validate_slo_config(config: dict[str, Any]) -> None:
    """Validate vertical and metric-family completeness."""

    verticals = config.get("verticals")
    if not isinstance(verticals, dict):
        msg = "SLO config must define a verticals mapping"
        raise ValueError(msg)

    missing_verticals = [vertical for vertical in SLO_VERTICALS if vertical not in verticals]
    if missing_verticals:
        msg = f"SLO config missing verticals: {', '.join(missing_verticals)}"
        raise ValueError(msg)

    reference_families: set[str] | None = None
    for vertical in SLO_VERTICALS:
        raw_vertical_config = verticals.get(vertical)
        if not isinstance(raw_vertical_config, dict):
            msg = f"SLO config for {vertical} must be a mapping"
            raise ValueError(msg)
        vertical_config = cast(dict[str, Any], raw_vertical_config)
        families = set(vertical_config)
        expected_families = set(SLO_METRIC_FAMILIES)
        if families != expected_families:
            missing = sorted(expected_families - families)
            extra = sorted(families - expected_families)
            msg = f"{vertical} SLO metric families mismatch; missing={missing}, extra={extra}"
            raise ValueError(msg)
        if reference_families is None:
            reference_families = families
        elif families != reference_families:
            msg = "Every vertical must define the same metric families"
            raise ValueError(msg)

        for family, required_metrics in REQUIRED_METRICS_BY_FAMILY.items():
            raw_family_config = vertical_config.get(family)
            if not isinstance(raw_family_config, dict):
                msg = f"{vertical}.{family} must be a mapping"
                raise ValueError(msg)
            family_config = cast(dict[str, Any], raw_family_config)
            missing_metrics = [
                metric_name for metric_name in required_metrics if metric_name not in family_config
            ]
            if missing_metrics:
                msg = f"{vertical}.{family} missing metrics: {', '.join(missing_metrics)}"
                raise ValueError(msg)


def metric_direction(metric_name: str) -> str:
    """Return comparison direction from the metric suffix."""

    if metric_name.endswith("_min"):
        return "min"
    if metric_name.endswith("_max"):
        return "max"
    if metric_name.endswith("_required"):
        return "required"
    msg = f"Cannot infer SLO direction for metric '{metric_name}'"
    raise ValueError(msg)


def evaluate_metric(
    *,
    vertical: str,
    metric_family: str,
    metric_name: str,
    target: float | bool,
    observed: float | bool | None,
) -> SloResult:
    """Evaluate one metric against its target."""

    if observed is None:
        return SloResult(
            vertical=vertical,
            metric_family=metric_family,
            metric_name=metric_name,
            target=target,
            observed=None,
            status="NOT_AVAILABLE",
            gap=None,
            recommended_action=recommended_action(metric_family, "NOT_AVAILABLE"),
        )

    direction = metric_direction(metric_name)
    if direction == "required":
        passed = bool(observed) is bool(target)
        status: SloStatus = "PASS" if passed else "BLOCKED"
        return SloResult(
            vertical=vertical,
            metric_family=metric_family,
            metric_name=metric_name,
            target=target,
            observed=observed,
            status=status,
            gap=0.0 if passed else -1.0,
            recommended_action=recommended_action(metric_family, status),
        )

    target_float = float(target)
    observed_float = float(observed)
    if direction == "min":
        gap = round(observed_float - target_float, 6)
        if observed_float >= target_float:
            status = "PASS"
        else:
            status = warn_or_blocked(abs(target_float - observed_float), abs(target_float))
    else:
        gap = round(target_float - observed_float, 6)
        if observed_float <= target_float:
            status = "PASS"
        else:
            status = warn_or_blocked(abs(observed_float - target_float), abs(target_float))

    return SloResult(
        vertical=vertical,
        metric_family=metric_family,
        metric_name=metric_name,
        target=target_float,
        observed=observed_float,
        status=status,
        gap=gap,
        recommended_action=recommended_action(metric_family, status),
    )


def warn_or_blocked(miss_amount: float, target_amount: float) -> SloStatus:
    """Return WARN if the miss is within 10%, otherwise BLOCKED."""

    denominator = target_amount if target_amount > 0 else 1.0
    return "WARN" if miss_amount / denominator <= 0.10 else "BLOCKED"


def recommended_action(metric_family: str, status: SloStatus) -> str:
    """Return a concise action for an SLO status."""

    if status == "PASS":
        return "No action required."
    if status == "NOT_AVAILABLE":
        return f"Run an experiment/report that measures {metric_family}."
    if metric_family == "retrieval_slo":
        return "Repair retrieval before inference scaling or final benchmark claims."
    if metric_family == "quality_slo":
        return "Improve evaluator score, grounding, citations, or output formatting."
    if metric_family == "latency_slo":
        return "Profile TTFT/ITL/TPOT/E2E latency and optimize serving configuration."
    if metric_family == "throughput_slo":
        return "Tune batching, concurrency, and backend scheduling."
    if metric_family == "resource_slo":
        return "Inspect GPU/CPU/RAM telemetry and memory pressure."
    if metric_family == "api_cost_slo":
        return "Reduce token usage or change API-priced model/provider."
    if metric_family == "gpu_cost_slo":
        return "Improve tokens per GPU dollar or choose a lower-cost serving plan."
    return "Inspect the metric and rerun the benchmark."


def evaluate_metric_family(
    *,
    config: dict[str, Any],
    vertical: str,
    metric_family: str,
    observations: dict[str, float | bool] | None,
) -> list[SloResult]:
    """Evaluate all metrics in one family for one vertical."""

    vertical_config = cast(dict[str, Any], cast(dict[str, Any], config["verticals"])[vertical])
    family_config = cast(dict[str, Any], vertical_config[metric_family])
    active_observations = observations or {}
    return [
        evaluate_metric(
            vertical=vertical,
            metric_family=metric_family,
            metric_name=metric_name,
            target=cast(float | bool, family_config[metric_name]),
            observed=active_observations.get(metric_name),
        )
        for metric_name in REQUIRED_METRICS_BY_FAMILY[metric_family]
    ]


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def retrieval_observations_from_report(
    report_path: str | Path | None,
    *,
    split: str = "final_10000",
    ablation_mode: str = "prompt_plus_metadata",
    memory_mode: str = "mm2_hybrid_top5",
) -> dict[str, dict[str, float]]:
    """Extract per-vertical retrieval observations from a retrieval report."""

    if report_path is None or not Path(report_path).exists():
        return {}
    report = read_json(report_path)
    by_split = report.get("by_split")
    if not isinstance(by_split, dict):
        return {}
    split_payload = by_split.get(split)
    if not isinstance(split_payload, dict):
        return {}
    ablation_payload = split_payload.get(ablation_mode)
    if not isinstance(ablation_payload, dict):
        return {}
    mode_payload = ablation_payload.get(memory_mode)
    if not isinstance(mode_payload, dict):
        return {}

    observations: dict[str, dict[str, float]] = {}
    for vertical in SLO_VERTICALS:
        raw_metrics = mode_payload.get(vertical)
        if not isinstance(raw_metrics, dict):
            continue
        metric_payload = cast(dict[str, Any], raw_metrics)
        observations[vertical] = {
            slo_metric: float(metric_payload[report_metric])
            for slo_metric, report_metric in RETRIEVAL_REPORT_METRIC_MAP.items()
            if metric_payload.get(report_metric) is not None
        }
    return observations


def build_slo_readiness_report(
    *,
    slo_config: dict[str, Any],
    retrieval_report_path: str | Path | None = None,
    quality_gate_report_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build SLO readiness report rows from current available reports."""

    retrieval_observations = retrieval_observations_from_report(retrieval_report_path)
    results: list[SloResult] = []
    for vertical in SLO_VERTICALS:
        for family in SLO_METRIC_FAMILIES:
            observations = (
                retrieval_observations.get(vertical, {}) if family == "retrieval_slo" else None
            )
            results.extend(
                evaluate_metric_family(
                    config=slo_config,
                    vertical=vertical,
                    metric_family=family,
                    observations=observations,
                )
            )

    rows = [slo_result_to_row(result) for result in results]
    status_counts = CounterLike.from_rows(rows, key="status")
    blocked_retrieval_rows = [
        row
        for row in rows
        if row["metric_family"] == "retrieval_slo" and row["status"] == "BLOCKED"
    ]
    unavailable_rows = [row for row in rows if row["status"] == "NOT_AVAILABLE"]
    report = {
        "generated_at_utc": utc_now(),
        "scope": "production_slo_readiness_no_inference_no_gpu_no_api",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "verticals": list(SLO_VERTICALS),
        "metric_families": list(SLO_METRIC_FAMILIES),
        "retrieval_report_path": str(retrieval_report_path) if retrieval_report_path else None,
        "quality_gate_report_path": str(quality_gate_report_path)
        if quality_gate_report_path
        else None,
        "status_counts": status_counts,
        "retrieval_slo_blocked_count": len(blocked_retrieval_rows),
        "inference_scaling_blocked_by_retrieval_slos": bool(blocked_retrieval_rows),
        "blocked_retrieval_metrics": blocked_retrieval_rows,
        "not_available_metric_count": len(unavailable_rows),
        "not_available_metric_families": sorted(
            {str(row["metric_family"]) for row in unavailable_rows}
        ),
        "summary": {
            "overall_status": "BLOCKED" if blocked_retrieval_rows else "READY_WITH_GAPS",
            "retrieval_slos_currently_measured": bool(retrieval_observations),
            "future_inference_cost_resource_metrics_available": False,
        },
        "results": rows,
    }
    return report, rows


def slo_result_to_row(result: SloResult) -> dict[str, Any]:
    """Convert a result to a CSV-safe row."""

    payload = asdict(result)
    payload["target"] = result.target
    payload["observed"] = "" if result.observed is None else result.observed
    payload["gap"] = "" if result.gap is None else result.gap
    return payload


class CounterLike:
    """Tiny local counter helper that keeps this module dependency-light."""

    @staticmethod
    def from_rows(rows: list[dict[str, Any]], *, key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write SLO readiness CSV rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
