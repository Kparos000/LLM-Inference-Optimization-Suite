"""Honest cross-engine comparison helpers for Phase 4 smoke artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

COMPARISON_FIELDS = [
    "backend",
    "comparison_scope",
    "row_count",
    "success_count",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "mean_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "mean_total_tokens_per_second",
    "json_valid_rate",
    "generation_contract_valid_rate",
    "evidence_match_rate",
    "grounded_rate",
    "safety_violation_rate",
    "mean_gpu_utilization_percent",
    "max_gpu_utilization_percent",
    "max_gpu_memory_used_mb",
    "mean_power_draw_w",
    "max_temperature_c",
    "failure_count",
    "missing_metrics",
]


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return payload


def read_csv_first(path: str | Path) -> dict[str, str]:
    """Read the first CSV row."""

    with Path(path).open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file), None)
    if row is None:
        msg = f"Expected at least one CSV row in {path}"
        raise ValueError(msg)
    return dict(row)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL objects."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"Expected JSON object row in {path}"
                raise ValueError(msg)
            rows.append(payload)
    return rows


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _nested_float(payload: dict[str, Any] | None, *keys: str) -> float | None:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _optional_float(current)


def build_engine_row(
    *,
    backend: str,
    comparison_scope: str,
    result_rows: list[dict[str, Any]],
    evaluation_summary: dict[str, Any],
    latency_summary: dict[str, Any] | None = None,
    telemetry_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one backend into the common comparison schema."""

    success_count = sum(bool(row.get("success")) for row in result_rows)
    row: dict[str, Any] = {
        "backend": backend,
        "comparison_scope": comparison_scope,
        "row_count": len(result_rows),
        "success_count": success_count,
        "mean_ttft_ms": _optional_float(
            latency_summary.get("mean_ttft_ms") if latency_summary else None
        ),
        "p95_ttft_ms": _optional_float(
            latency_summary.get("p95_ttft_ms") if latency_summary else None
        ),
        "mean_tpot_ms": _optional_float(
            latency_summary.get("mean_tpot_ms") if latency_summary else None
        ),
        "p95_tpot_ms": _optional_float(
            latency_summary.get("p95_tpot_ms") if latency_summary else None
        ),
        "mean_e2e_latency_ms": _optional_float(
            latency_summary.get("mean_e2e_latency_ms") if latency_summary else None
        ),
        "p95_e2e_latency_ms": _optional_float(
            latency_summary.get("p95_e2e_latency_ms") if latency_summary else None
        ),
        "mean_total_tokens_per_second": _optional_float(
            latency_summary.get("mean_total_tokens_per_second") if latency_summary else None
        ),
        "json_valid_rate": _optional_float(evaluation_summary.get("json_valid_rate")),
        "generation_contract_valid_rate": _optional_float(
            evaluation_summary.get("generation_contract_valid_rate")
        ),
        "evidence_match_rate": _optional_float(evaluation_summary.get("evidence_match_rate")),
        "grounded_rate": _optional_float(evaluation_summary.get("grounded_rate")),
        "safety_violation_rate": _optional_float(evaluation_summary.get("safety_violation_rate")),
        "mean_gpu_utilization_percent": _nested_float(
            telemetry_summary, "utilization_gpu_percent", "mean"
        ),
        "max_gpu_utilization_percent": _nested_float(
            telemetry_summary, "utilization_gpu_percent", "max"
        ),
        "max_gpu_memory_used_mb": _nested_float(telemetry_summary, "memory_used_mb", "max"),
        "mean_power_draw_w": _nested_float(telemetry_summary, "power_draw_w", "mean"),
        "max_temperature_c": _nested_float(telemetry_summary, "temperature_c", "max"),
        "failure_count": len(result_rows) - success_count,
    }
    missing = [
        field_name
        for field_name in COMPARISON_FIELDS
        if field_name not in {"backend", "comparison_scope", "missing_metrics"}
        and row.get(field_name) is None
    ]
    row["missing_metrics"] = ";".join(missing)
    return row


def build_pairwise_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float | None]:
    """Return candidate-minus-baseline deltas without inventing missing values."""

    deltas: dict[str, float | None] = {}
    for field_name in COMPARISON_FIELDS:
        if field_name in {
            "backend",
            "comparison_scope",
            "missing_metrics",
            "row_count",
            "success_count",
            "failure_count",
        }:
            continue
        baseline_value = _optional_float(baseline.get(field_name))
        candidate_value = _optional_float(candidate.get(field_name))
        deltas[field_name] = (
            candidate_value - baseline_value
            if baseline_value is not None and candidate_value is not None
            else None
        )
    return deltas


def build_comparison_report(
    *,
    rows: list[dict[str, Any]],
    vllm_backend: str = "vllm",
    sglang_backend: str = "sglang",
    prompt_ids_by_backend: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Build the four-backend context and strict vLLM/SGLang comparison."""

    by_backend = {str(row["backend"]): row for row in rows}
    vllm = by_backend.get(vllm_backend)
    sglang = by_backend.get(sglang_backend)
    prompt_ids_match = (
        prompt_ids_by_backend.get(vllm_backend) == prompt_ids_by_backend.get(sglang_backend)
        if prompt_ids_by_backend is not None
        else None
    )
    pairwise_comparable = bool(
        vllm
        and sglang
        and vllm.get("row_count") == sglang.get("row_count") == 50
        and vllm.get("comparison_scope") == sglang.get("comparison_scope")
        and prompt_ids_match is not False
    )
    return {
        "comparison_status": ("COMPARABLE" if pairwise_comparable else "NOT_FULLY_COMPARABLE"),
        "pairwise_scope": ("same_50_prompts_same_model_same_gpu_same_generation_settings"),
        "prompt_id_sets_match": prompt_ids_match,
        "vllm_vs_sglang_comparable": pairwise_comparable,
        "backend_rows": rows,
        "sglang_minus_vllm": (build_pairwise_delta(vllm, sglang) if vllm and sglang else {}),
        "limitations": [
            (
                "HF and API rows are contextual baselines from five-prompt runs and are "
                "not hardware- or workload-equal to the 50-prompt GPU rows."
            ),
            (
                "Only vLLM and SGLang use the same 50 prompts, model, RTX 3070, memory "
                "mode, streaming setting, temperature, and output cap."
            ),
            "Missing metrics remain null and are listed explicitly; no values are estimated.",
        ],
    }


def write_comparison_artifacts(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write comparison JSON and CSV artifacts."""

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
