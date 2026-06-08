"""Offline comparison of measured Phase 4 model smoke artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

QUALITY_FIELDS = (
    "json_valid_rate",
    "generation_contract_valid_rate",
    "evidence_id_presence_rate",
    "evidence_match_rate",
    "grounded_rate",
    "safety_violation_rate",
)
LATENCY_FIELDS = (
    "ttft_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "tpot_ms",
    "e2e_latency_ms",
)
COST_FIELDS = (
    "cost_per_request_usd",
    "cost_per_successful_answer_usd",
    "cost_per_grounded_answer_usd",
)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON object."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return loaded


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL object rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise ValueError(msg)
            rows.append(loaded)
    return rows


def _numeric(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _summary_quality(report: dict[str, Any]) -> dict[str, float | None]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    return {field: _numeric(summary.get(field)) for field in QUALITY_FIELDS}


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    latencies = [
        float(value)
        for row in rows
        if (value := _numeric(row.get("latency_ms") or row.get("e2e_latency_ms"))) is not None
    ]
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    total_tokens = sum(
        int(row.get("total_tokens") or 0)
        or int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)
        for row in rows
    )
    return {
        "request_count": len(rows),
        "success_count": sum(bool(row.get("success")) for row in rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }


def _latency_values(report: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for field in LATENCY_FIELDS:
        aggregate = report.get(field)
        values[field] = _numeric(aggregate.get("mean")) if isinstance(aggregate, dict) else None
    return values


def build_model_metrics(
    *,
    label: str,
    results: list[dict[str, Any]],
    evaluation_report: dict[str, Any],
    cost_report: dict[str, Any] | None = None,
    latency_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one normalized measured model row."""

    metrics: dict[str, Any] = {
        "label": label,
        "model_alias": results[0].get("model_alias") if results else None,
        "model_id": results[0].get("model_id") if results else None,
        "provider": (results[0].get("provider") or "local") if results else "local",
        "streaming_success_count": sum(bool(row.get("streaming_available")) for row in results),
        **_summary_quality(evaluation_report),
        **_aggregate_rows(results),
    }
    metrics.update(
        _latency_values(latency_report or {})
        if latency_report
        else {field: None for field in LATENCY_FIELDS}
    )
    for field in COST_FIELDS:
        metrics[field] = _numeric((cost_report or {}).get(field))
    metrics["total_cost_usd"] = _numeric((cost_report or {}).get("total_cost_usd"))
    return metrics


def build_model5_comparison_report(
    *,
    model5_results: list[dict[str, Any]],
    model5_eval: dict[str, Any],
    model5_cost: dict[str, Any],
    model5_latency: dict[str, Any],
    model6_results: list[dict[str, Any]],
    model6_eval: dict[str, Any],
    model6_cost: dict[str, Any],
    model6_latency: dict[str, Any],
    local_results: list[dict[str, Any]],
    local_eval: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare model5 with model6 and the local Qwen baseline."""

    models = {
        "model5_openrouter": build_model_metrics(
            label="model5_openrouter",
            results=model5_results,
            evaluation_report=model5_eval,
            cost_report=model5_cost,
            latency_report=model5_latency,
        ),
        "model6_hf_novita": build_model_metrics(
            label="model6_hf_novita",
            results=model6_results,
            evaluation_report=model6_eval,
            cost_report=model6_cost,
            latency_report=model6_latency,
        ),
        "local_qwen_0_5b": build_model_metrics(
            label="local_qwen_0_5b",
            results=local_results,
            evaluation_report=local_eval,
        ),
    }
    model5 = models["model5_openrouter"]
    model6 = models["model6_hf_novita"]
    prompt_sets_match = (
        {str(row.get("prompt_id") or "") for row in model5_results}
        == {str(row.get("prompt_id") or "") for row in model6_results}
        == {str(row.get("prompt_id") or "") for row in local_results}
    )
    keep_model5 = bool(
        model5["request_count"] == 5
        and model5["success_count"] == 5
        and model5["streaming_success_count"] == 5
        and model5["total_cost_usd"] is not None
        and (model5["generation_contract_valid_rate"] or 0) > 0
    )
    rows: list[dict[str, Any]] = []
    fields = (
        *QUALITY_FIELDS,
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "mean_latency_ms",
        "median_latency_ms",
        *LATENCY_FIELDS,
        "total_cost_usd",
        *COST_FIELDS,
    )
    for field in fields:
        rows.append(
            {
                "metric": field,
                "model5_openrouter": model5.get(field),
                "model6_hf_novita": models["model6_hf_novita"].get(field),
                "local_qwen_0_5b": models["local_qwen_0_5b"].get(field),
            }
        )
    report = {
        "comparison_scope": (
            "prompt_id_aligned_five_prompt_generation_contract_smokes"
            if prompt_sets_match
            else "non_aligned_smoke_artifacts"
        ),
        "prompt_id_sets_match": prompt_sets_match,
        "models": models,
        "model5_final_benchmark_recommendation": ("RETAIN" if keep_model5 else "DO_NOT_RETAIN_YET"),
        "recommendation_reason": (
            (
                "Retain model5 as a distinct 3B OpenRouter comparison route. In this tiny "
                "sample it completed streaming and cost accounting, but model6 had higher "
                "contract validity, evidence match, and groundedness at lower token cost."
            )
            if keep_model5
            else "Model5 did not complete every guarded streaming and cost requirement."
        ),
        "model5_vs_model6": {
            "grounded_rate_delta": (
                float(model5["grounded_rate"]) - float(model6["grounded_rate"])
                if model5["grounded_rate"] is not None and model6["grounded_rate"] is not None
                else None
            ),
            "evidence_match_rate_delta": (
                float(model5["evidence_match_rate"]) - float(model6["evidence_match_rate"])
                if model5["evidence_match_rate"] is not None
                and model6["evidence_match_rate"] is not None
                else None
            ),
            "cost_per_request_ratio": (
                float(model5["cost_per_request_usd"]) / float(model6["cost_per_request_usd"])
                if model5["cost_per_request_usd"] is not None
                and model6["cost_per_request_usd"] not in (None, 0)
                else None
            ),
            "mean_ttft_delta_ms": (
                float(model5["ttft_ms"]) - float(model6["ttft_ms"])
                if model5["ttft_ms"] is not None and model6["ttft_ms"] is not None
                else None
            ),
            "mean_e2e_delta_ms": (
                float(model5["e2e_latency_ms"]) - float(model6["e2e_latency_ms"])
                if model5["e2e_latency_ms"] is not None and model6["e2e_latency_ms"] is not None
                else None
            ),
        },
        "limitations": [
            "Five prompts measure plumbing and directional quality, not benchmark significance.",
            "Local Qwen has no streaming TTFT/ITL or infrastructure cost measurement.",
        ],
        "no_additional_inference_triggered": True,
        "no_gpu_work_triggered": True,
    }
    return report, rows


def write_model5_comparison_artifacts(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write comparison JSON and CSV files."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "model5_openrouter",
                "model6_hf_novita",
                "local_qwen_0_5b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return report_output, summary_output
