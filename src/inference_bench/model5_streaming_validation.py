"""Blocked and completed artifact helpers for the model5 streaming smoke."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MODEL5_RAW_NAME = "phase4_model5_streaming_smoke_results.jsonl"
MODEL5_EVAL_REPORT_NAME = "phase4_model5_streaming_eval_report.json"
MODEL5_EVAL_SUMMARY_NAME = "phase4_model5_streaming_eval_summary.csv"
MODEL5_COST_REPORT_NAME = "phase4_model5_streaming_cost_report.json"
MODEL5_LATENCY_REPORT_NAME = "phase4_model5_streaming_latency_report.json"


def fallback_allowed_after_primary_execution(
    *,
    primary_execution_attempted: bool,
    primary_success_count: int,
) -> bool:
    """Allow fallback only when primary execution ran and produced no success."""

    return primary_execution_attempted and primary_success_count == 0


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, row: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return path


def write_blocked_model5_artifacts(
    *,
    raw_path: str | Path,
    processed_root: str | Path,
    model_id: str,
    provider: str | None,
    planned_prompt_count: int,
    reason: str,
    pricing_status: str,
    manual_override_configured: bool,
) -> dict[str, Path]:
    """Write explicit non-execution artifacts without fabricating metrics."""

    raw_output = Path(raw_path)
    processed = Path(processed_root)
    status_row = {
        "record_type": "preflight_status",
        "execution_status": "BLOCKED_PRE_EXECUTION",
        "model_alias": "model5_gated",
        "model_id": model_id,
        "provider": provider,
        "planned_prompt_count": planned_prompt_count,
        "executed_prompt_count": 0,
        "memory_mode": "mm2_hybrid_top5",
        "streaming_required": True,
        "pricing_status": pricing_status,
        "manual_override_configured": manual_override_configured,
        "execution_attempted": False,
        "fallback_used": False,
        "paid_api_call_triggered": False,
        "success": False,
        "error_type": "PricingUnavailable",
        "error_message": reason,
        "input_tokens": None,
        "output_tokens": None,
        "total_cost_usd": None,
        "ttft_ms": None,
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "itl_p99_ms": None,
        "tpot_ms": None,
        "e2e_latency_ms": None,
    }
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.write_text(
        json.dumps(status_row, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    common = {
        "execution_status": "BLOCKED_PRE_EXECUTION",
        "model_alias": "model5_gated",
        "model_id": model_id,
        "provider": provider,
        "planned_prompt_count": planned_prompt_count,
        "executed_prompt_count": 0,
        "fallback_used": False,
        "paid_api_call_triggered": False,
        "blocking_reason": reason,
        "pricing_status": pricing_status,
        "manual_override_configured": manual_override_configured,
        "metrics_available": False,
    }
    eval_report: dict[str, Any] = {
        **common,
        "evaluation_rows": [],
        "json_valid_rate": None,
        "generation_contract_valid_rate": None,
        "evidence_match_rate": None,
        "grounded_rate": None,
        "note": "No generation output exists, so quality metrics were not evaluated.",
    }
    eval_summary = {
        **common,
        "json_valid_rate": None,
        "generation_contract_valid_rate": None,
        "evidence_match_rate": None,
        "grounded_rate": None,
    }
    cost_report = {
        **common,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "total_cost_usd": None,
        "cost_per_request_usd": None,
        "cost_per_grounded_answer_usd": None,
        "note": "Cost remains unavailable; zero was not substituted for missing pricing.",
    }
    latency_report = {
        **common,
        "streaming_success_count": 0,
        "ttft_ms": None,
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "itl_p99_ms": None,
        "tpot_ms": None,
        "e2e_latency_ms": None,
        "note": "No request was sent, so no streaming latency was measured.",
    }
    outputs = {
        "raw": raw_output,
        "eval_report": _write_json(processed / MODEL5_EVAL_REPORT_NAME, eval_report),
        "eval_summary": _write_csv(processed / MODEL5_EVAL_SUMMARY_NAME, eval_summary),
        "cost_report": _write_json(processed / MODEL5_COST_REPORT_NAME, cost_report),
        "latency_report": _write_json(
            processed / MODEL5_LATENCY_REPORT_NAME,
            latency_report,
        ),
    }
    return outputs
