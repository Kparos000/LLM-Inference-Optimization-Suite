"""Comparison helpers for mm4 agentic and non-agentic memory modes."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

COMPARISON_FIELDS = [
    "memory_mode",
    "measurement_status",
    "prompt_count",
    "success_count",
    "grounded_rate",
    "evidence_match_rate",
    "generation_contract_valid_rate",
    "safety_violation_rate",
    "mean_ttft_ms",
    "mean_e2e_latency_ms",
    "total_input_tokens",
    "total_output_tokens",
    "total_tokens",
    "token_count_source",
    "total_cost_usd",
    "cost_status",
    "mean_retrieval_rounds",
    "repair_rate",
    "escalation_rate",
    "mean_node_latencies_json",
    "missing_metrics",
]


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(float(str(value)))


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _int_values(rows: list[dict[str, Any]], field: str) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = _optional_int(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _float_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _optional_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _mean_node_latencies(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for node, latency in _json_object(row.get("node_latencies")).items():
            if isinstance(latency, int | float):
                values.setdefault(str(node), []).append(float(latency))
    return {
        node: statistics.fmean(latencies) for node, latencies in sorted(values.items()) if latencies
    }


def build_memory_mode_row(
    *,
    memory_mode: str,
    result_rows: list[dict[str, Any]],
    evaluation_summary: dict[str, Any] | None,
    latency_summary: dict[str, Any] | None,
    measurement_status: str = "measured",
) -> dict[str, Any]:
    """Normalize one memory mode without estimating unavailable values."""

    success_count = sum(bool(row.get("success")) for row in result_rows)
    normalized_input_tokens = _int_values(result_rows, "comparison_input_tokens")
    normalized_output_tokens = _int_values(result_rows, "comparison_output_tokens")
    input_tokens = normalized_input_tokens or _int_values(result_rows, "input_tokens")
    output_tokens = normalized_output_tokens or _int_values(result_rows, "output_tokens")
    retrieval_rounds = _float_values(result_rows, "retrieval_rounds")
    repair_count = sum(int(row.get("repair_attempts") or 0) > 0 for row in result_rows)
    escalation_count = sum(
        str(row.get("final_status") or "") in {"escalate", "insufficient_evidence"}
        for row in result_rows
    )
    explicit_costs = _float_values(result_rows, "total_cost_usd")
    evaluation = evaluation_summary or {}
    latency = latency_summary or {}
    row: dict[str, Any] = {
        "memory_mode": memory_mode,
        "measurement_status": measurement_status,
        "prompt_count": len(result_rows) if result_rows else None,
        "success_count": success_count if result_rows else None,
        "grounded_rate": _optional_float(evaluation.get("grounded_rate")),
        "evidence_match_rate": _optional_float(evaluation.get("evidence_match_rate")),
        "generation_contract_valid_rate": _optional_float(
            evaluation.get("generation_contract_valid_rate")
        ),
        "safety_violation_rate": _optional_float(evaluation.get("safety_violation_rate")),
        "mean_ttft_ms": _optional_float(latency.get("mean_ttft_ms")),
        "mean_e2e_latency_ms": _optional_float(latency.get("mean_e2e_latency_ms")),
        "total_input_tokens": sum(input_tokens) if input_tokens else None,
        "total_output_tokens": sum(output_tokens) if output_tokens else None,
        "total_tokens": (
            sum(input_tokens) + sum(output_tokens) if input_tokens or output_tokens else None
        ),
        "token_count_source": "whitespace_normalized",
        "total_cost_usd": sum(explicit_costs) if explicit_costs else None,
        "cost_status": (
            "measured_api_token_cost" if explicit_costs else "unavailable_no_gpu_hourly_price"
        ),
        "mean_retrieval_rounds": _mean(retrieval_rounds),
        "repair_rate": repair_count / len(result_rows) if result_rows else None,
        "escalation_rate": (escalation_count / len(result_rows) if result_rows else None),
        "mean_node_latencies_json": (
            json.dumps(_mean_node_latencies(result_rows), sort_keys=True)
            if result_rows and _mean_node_latencies(result_rows)
            else None
        ),
    }
    missing = [
        field
        for field in COMPARISON_FIELDS
        if field
        not in {
            "memory_mode",
            "measurement_status",
            "cost_status",
            "token_count_source",
            "missing_metrics",
        }
        and row.get(field) is None
    ]
    row["missing_metrics"] = ";".join(missing)
    return row


def _delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    fields = (
        "grounded_rate",
        "evidence_match_rate",
        "generation_contract_valid_rate",
        "safety_violation_rate",
        "mean_ttft_ms",
        "mean_e2e_latency_ms",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_cost_usd",
        "mean_retrieval_rounds",
        "repair_rate",
        "escalation_rate",
    )
    return {
        field: (
            candidate_value - baseline_value
            if (baseline_value := _optional_float(baseline.get(field))) is not None
            and (candidate_value := _optional_float(candidate.get(field))) is not None
            else None
        )
        for field in fields
    }


def build_mm4_comparison_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the mm4-versus-mm2/mm3 comparison report."""

    by_mode = {str(row["memory_mode"]): row for row in rows}
    mm4 = by_mode.get("mm4_bounded_agentic")
    return {
        "comparison_status": (
            "COMPLETE"
            if mm4 and by_mode.get("mm2_hybrid_top5") and by_mode.get("mm3_compressed_hybrid_top5")
            else "PARTIAL"
        ),
        "rows": rows,
        "mm4_minus_mm2": (
            _delta(by_mode["mm2_hybrid_top5"], mm4) if mm4 and "mm2_hybrid_top5" in by_mode else {}
        ),
        "mm4_minus_mm3": (
            _delta(by_mode["mm3_compressed_hybrid_top5"], mm4)
            if mm4 and "mm3_compressed_hybrid_top5" in by_mode
            else {}
        ),
        "limitations": [
            "Gold evidence is used only by the unchanged offline evaluator, not graph routing.",
            "GPU infrastructure cost is unavailable because no hourly price is registered.",
            "Cross-mode token totals use whitespace-normalized counts; "
            "raw mm4 rows also retain provider tokenizer usage.",
            "Node latency fields exist only for mm4; baseline modes are single-pass runners.",
        ],
    }


def write_mm4_comparison(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write JSON and CSV comparison artifacts."""

    report = build_mm4_comparison_report(rows)
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return report_output, summary_output
