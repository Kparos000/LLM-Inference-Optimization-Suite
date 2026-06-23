"""Deterministic vLLM stability audit helpers for B7/B7R1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

FATAL_ENGINE_PATTERNS = (
    "enginecore encountered an issue",
    "engine core encountered an issue",
    "cublas",
    "cuda error",
    "illegal memory access",
    "out of memory",
    "device-side assert",
)
CONNECTION_AFTER_FATAL_PATTERNS = (
    "connection refused",
    "connection reset",
    "remote end closed connection",
    "server disconnected",
    "read timed out",
    "connect timeout",
)
SAFE_B7R1_STATUSES = {"B7R1_STABILITY_READY", "B7R1_STABLE_WITH_QUALITY_CAVEAT"}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    input_path = Path(path)
    if not input_path.exists():
        return None
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write small audit summary rows without importing pandas."""

    import csv

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"metric": "row_count", "value": 0}]
    fields = sorted({field for row in rows for field in row})
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def is_fatal_engine_error(message: object) -> bool:
    """Return whether an error message indicates vLLM engine collapse."""

    text = str(message or "").lower()
    return any(pattern in text for pattern in FATAL_ENGINE_PATTERNS)


def is_backend_connection_failure(message: object) -> bool:
    """Return whether a row likely failed because the backend became unreachable."""

    text = str(message or "").lower()
    return any(pattern in text for pattern in CONNECTION_AFTER_FATAL_PATTERNS)


def _failure_kind(row: dict[str, Any]) -> str:
    error = row.get("error_message")
    if is_fatal_engine_error(error):
        return "fatal_engine_error"
    if is_backend_connection_failure(error):
        return "backend_unreachable_after_engine_failure"
    if str(error or "").strip():
        return "request_error"
    return "unknown_failure"


def _counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))


def _numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = [_float(row.get(key)) for row in rows if row.get(key) not in (None, "")]
    if not values:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    return {"count": len(values), "mean": round(mean(values), 6), "max": max(values)}


def _first_failure(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, row in enumerate(rows, start=1):
        if not _as_bool(row.get("success")):
            return {
                "row_index": index,
                "prompt_id": row.get("prompt_id"),
                "vertical": row.get("vertical"),
                "failure_kind": _failure_kind(row),
                "error_message": row.get("error_message"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
            }
    return None


def _cascading_failure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = _first_failure(rows)
    if first is None:
        return {
            "cascading_failure_observed": False,
            "failure_streak_after_first_failure": 0,
            "post_first_failure_success_count": 0,
        }
    start = int(first["row_index"]) - 1
    streak = 0
    post_first_success = 0
    for row in rows[start:]:
        if _as_bool(row.get("success")):
            post_first_success += 1
            break
        streak += 1
    post_first_failures = [row for row in rows[start:] if not _as_bool(row.get("success"))]
    connection_failures = sum(
        is_backend_connection_failure(row.get("error_message")) for row in post_first_failures
    )
    fatal_failures = sum(
        is_fatal_engine_error(row.get("error_message")) for row in post_first_failures
    )
    return {
        "cascading_failure_observed": streak >= 10 and post_first_success == 0,
        "failure_streak_after_first_failure": streak,
        "post_first_failure_success_count": post_first_success,
        "post_first_failure_count": len(post_first_failures),
        "fatal_engine_error_count_after_first_failure": fatal_failures,
        "connection_failure_count_after_first_failure": connection_failures,
    }


def _gpu_summary(telemetry_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not telemetry_rows:
        return {"sample_count": 0}
    memory_used = [_float(row.get("memory_used_mb")) for row in telemetry_rows]
    memory_total = [_float(row.get("memory_total_mb")) for row in telemetry_rows]
    utilization = [_float(row.get("utilization_gpu_percent")) for row in telemetry_rows]
    peak_used = max(memory_used) if memory_used else 0.0
    total = max(memory_total) if memory_total else 0.0
    return {
        "sample_count": len(telemetry_rows),
        "peak_memory_used_mb": peak_used,
        "peak_memory_total_mb": total,
        "peak_memory_utilization_pct": round((peak_used / total) * 100.0, 6) if total else 0.0,
        "mean_gpu_utilization_percent": round(mean(utilization), 6) if utilization else 0.0,
        "process_info_examples": sorted(
            {
                str(row.get("process_info") or "")
                for row in telemetry_rows
                if row.get("process_info")
            }
        )[:5],
    }


def build_vllm_stability_audit(
    *,
    result_rows: list[dict[str, Any]],
    telemetry_rows: list[dict[str, Any]] | None = None,
    eval_report: dict[str, Any] | None = None,
    expected_count: int = 1000,
) -> dict[str, Any]:
    """Audit B7-style result rows for fatal vLLM serving collapse."""

    failures = [row for row in result_rows if not _as_bool(row.get("success"))]
    successes = [row for row in result_rows if _as_bool(row.get("success"))]
    first_failure = _first_failure(result_rows)
    failure_kinds = dict(sorted(Counter(_failure_kind(row) for row in failures).items()))
    fatal_engine_errors = sum(is_fatal_engine_error(row.get("error_message")) for row in failures)
    gpu = _gpu_summary(telemetry_rows or [])
    cascading = _cascading_failure(result_rows)
    manifest_summary = {
        "expected_count": expected_count,
        "observed_count": len(result_rows),
        "success_count": len(successes),
        "failure_count": len(failures),
        "unique_prompt_count": len({str(row.get("prompt_id") or "") for row in result_rows}),
        "partial": len(result_rows) < expected_count,
    }
    failure_rate = (len(failures) / len(result_rows)) if result_rows else 0.0
    quality_summary = (eval_report or {}).get("summary") if eval_report else None
    if not isinstance(quality_summary, dict):
        quality_summary = {}
    serving_diagnosis = (
        "vllm_engine_core_cuda_cublas_collapse"
        if fatal_engine_errors
        else "no_fatal_vllm_engine_collapse_detected"
        if not failures
        else "nonfatal_request_failures_observed"
    )
    likely_primary_cause = (
        "serving_stability_failure"
        if fatal_engine_errors or cascading["cascading_failure_observed"]
        else "request_or_quality_failure"
    )
    return {
        "block": "B7R1",
        "source_run": "B7",
        "audit_scope": "vllm_cuda_failure",
        "expected_count": expected_count,
        "manifest_summary": manifest_summary,
        "first_failure": first_failure,
        "failure_kinds": failure_kinds,
        "failure_by_vertical": _counter_by(failures, "vertical"),
        "success_by_vertical": _counter_by(successes, "vertical"),
        "failure_rate": round(failure_rate, 6),
        "fatal_engine_error_count": fatal_engine_errors,
        "cascading_failure": cascading,
        "token_risk": {
            "success_input_tokens": _numeric_summary(successes, "input_tokens"),
            "failed_input_tokens": _numeric_summary(failures, "input_tokens"),
            "success_output_tokens": _numeric_summary(successes, "output_tokens"),
            "failed_output_tokens": _numeric_summary(failures, "output_tokens"),
        },
        "gpu_telemetry_summary": gpu,
        "quality_summary": {
            "json_valid_rate": quality_summary.get("json_valid_rate"),
            "generation_contract_valid_rate": quality_summary.get("generation_contract_valid_rate"),
            "evidence_match_rate": quality_summary.get("evidence_match_rate"),
            "grounded_rate": quality_summary.get("grounded_rate"),
            "safety_violation_count": quality_summary.get("safety_violation_count"),
        },
        "serving_diagnosis": serving_diagnosis,
        "likely_primary_cause": likely_primary_cause,
        "retrieval_gold_evaluator_modified": False,
        "recommended_repair_block": "B7R1_VLLM_CUDA_STABILITY_REPAIR",
        "recommended_actions": [
            "freeze the original B7 input and B6R6 repairs",
            "run vLLM with the remote_rtx3070_qwen3b_safe_v1 serving profile",
            "lower GPU memory utilization and bound model length/batched tokens",
            "stop immediately on fatal EngineCore/CUDA/CUBLAS collapse",
            "do not scale to 2,000 or 10,000 prompts until B7R1 completes cleanly",
        ],
    }


def audit_b7_artifacts(
    *,
    raw_results_path: str | Path,
    telemetry_path: str | Path,
    eval_report_path: str | Path | None = None,
    expected_count: int = 1000,
) -> dict[str, Any]:
    """Load B7 artifacts and return the stability audit report."""

    return build_vllm_stability_audit(
        result_rows=_read_jsonl(raw_results_path),
        telemetry_rows=_read_jsonl(telemetry_path),
        eval_report=_read_json(eval_report_path),
        expected_count=expected_count,
    )


def stability_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return flat summary rows for the stability audit CSV."""

    manifest = report.get("manifest_summary") if isinstance(report, dict) else {}
    cascading = report.get("cascading_failure") if isinstance(report, dict) else {}
    gpu = report.get("gpu_telemetry_summary") if isinstance(report, dict) else {}
    rows = [
        {
            "metric": "observed_count",
            "value": (manifest or {}).get("observed_count", 0),
            "category": "manifest",
        },
        {
            "metric": "success_count",
            "value": (manifest or {}).get("success_count", 0),
            "category": "manifest",
        },
        {
            "metric": "failure_count",
            "value": (manifest or {}).get("failure_count", 0),
            "category": "manifest",
        },
        {
            "metric": "fatal_engine_error_count",
            "value": report.get("fatal_engine_error_count", 0),
            "category": "serving",
        },
        {
            "metric": "cascading_failure_observed",
            "value": (cascading or {}).get("cascading_failure_observed", False),
            "category": "serving",
        },
        {
            "metric": "peak_memory_used_mb",
            "value": (gpu or {}).get("peak_memory_used_mb", 0),
            "category": "gpu",
        },
    ]
    for vertical, count in dict(report.get("failure_by_vertical") or {}).items():
        rows.append({"metric": f"failure_count_{vertical}", "value": count, "category": "vertical"})
    return rows


def write_stability_audit_artifacts(
    *,
    report: dict[str, Any],
    report_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write JSON and CSV stability audit artifacts."""

    return _write_json(report_path, report), write_summary_csv(
        summary_path,
        stability_summary_rows(report),
    )


def classify_b7r1_stability_gate(
    *,
    completed_count: int,
    expected_count: int,
    success_count: int,
    fatal_engine_errors: int,
    cascading_backend_failure: bool,
    safety_violation_count: float,
    artifact_sync_complete: bool,
    manifest_valid: bool,
    checkpoint_valid: bool,
    peak_vram_mb: float,
    peak_vram_threshold_mb: float,
    quality_passed: bool,
) -> dict[str, Any]:
    """Classify whether the B7R1 repair made the run stable enough to proceed."""

    checks = {
        "completed_all_expected_prompts": completed_count == expected_count,
        "fatal_engine_errors_zero": fatal_engine_errors == 0,
        "no_cascading_backend_failure": not cascading_backend_failure,
        "successful_requests_at_least_995": success_count >= min(expected_count, 995),
        "real_outputs_only": completed_count == expected_count,
        "safety_violations_zero": safety_violation_count == 0,
        "artifact_sync_complete": artifact_sync_complete,
        "manifest_valid": manifest_valid,
        "checkpoint_report_valid": checkpoint_valid,
        "peak_vram_below_threshold": peak_vram_mb <= peak_vram_threshold_mb,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if fatal_engine_errors or cascading_backend_failure or completed_count < expected_count:
        status = "B7R1_STABILITY_BLOCKED"
    elif failed:
        status = "B7R1_STABILITY_BLOCKED"
    elif quality_passed:
        status = "B7R1_STABILITY_READY"
    else:
        status = "B7R1_STABLE_WITH_QUALITY_CAVEAT"
    return {
        "status": status,
        "passed": status in SAFE_B7R1_STATUSES,
        "failed_checks": failed,
        "checks": checks,
        "benchmark_execution_readiness": (
            "READY"
            if status == "B7R1_STABILITY_READY"
            else "READY_WITH_QUALITY_CAVEAT"
            if status == "B7R1_STABLE_WITH_QUALITY_CAVEAT"
            else "NOT_READY"
        ),
        "next_api_load_probe_allowed": status in SAFE_B7R1_STATUSES,
        "rtx3070_qwen3b_suitability": (
            "stable"
            if status == "B7R1_STABILITY_READY"
            else "stable_but_memory_tight"
            if status == "B7R1_STABLE_WITH_QUALITY_CAVEAT"
            else "unstable"
        ),
    }
