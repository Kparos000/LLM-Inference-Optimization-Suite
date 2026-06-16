"""Phase B6 quality gate and context-preflight helpers."""

from __future__ import annotations

from typing import Any

B6_QUALITY_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.90,
    "grounded_rate": 0.90,
    "safety_violation_count": 0,
    "truncation_rate": 0.02,
    "vertical_evidence_match_rate_min": 0.85,
    "vertical_grounded_rate_min": 0.85,
}


def _float_metric(payload: dict[str, Any], metric: str) -> float:
    value = payload.get(metric)
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _check(
    *,
    observed: float,
    threshold: float,
    operator: str,
) -> dict[str, Any]:
    passed = observed == threshold if operator == "==" else observed >= threshold
    if operator == "<=":
        passed = observed <= threshold
    return {
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "passed": passed,
    }


def classify_b6_quality_gate(
    *,
    summary: dict[str, Any],
    per_vertical_quality: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify the B6 500-prompt scale gate from unchanged evaluator metrics."""

    checks = {
        "json_valid_rate": _check(
            observed=_float_metric(summary, "json_valid_rate"),
            threshold=B6_QUALITY_THRESHOLDS["json_valid_rate"],
            operator=">=",
        ),
        "generation_contract_valid_rate": _check(
            observed=_float_metric(summary, "generation_contract_valid_rate"),
            threshold=B6_QUALITY_THRESHOLDS["generation_contract_valid_rate"],
            operator=">=",
        ),
        "evidence_match_rate": _check(
            observed=_float_metric(summary, "evidence_match_rate"),
            threshold=B6_QUALITY_THRESHOLDS["evidence_match_rate"],
            operator=">=",
        ),
        "grounded_rate": _check(
            observed=_float_metric(summary, "grounded_rate"),
            threshold=B6_QUALITY_THRESHOLDS["grounded_rate"],
            operator=">=",
        ),
        "safety_violation_count": _check(
            observed=_float_metric(summary, "safety_violation_count"),
            threshold=float(B6_QUALITY_THRESHOLDS["safety_violation_count"]),
            operator="==",
        ),
        "truncation_rate": _check(
            observed=_float_metric(summary, "truncation_rate"),
            threshold=B6_QUALITY_THRESHOLDS["truncation_rate"],
            operator="<=",
        ),
    }
    vertical_evidence = [_float_metric(row, "evidence_match_rate") for row in per_vertical_quality]
    vertical_grounded = [_float_metric(row, "grounded_rate") for row in per_vertical_quality]
    min_vertical_evidence = min(vertical_evidence) if vertical_evidence else 0.0
    min_vertical_grounded = min(vertical_grounded) if vertical_grounded else 0.0
    checks["vertical_evidence_match_rate_min"] = _check(
        observed=min_vertical_evidence,
        threshold=B6_QUALITY_THRESHOLDS["vertical_evidence_match_rate_min"],
        operator=">=",
    )
    checks["vertical_grounded_rate_min"] = _check(
        observed=min_vertical_grounded,
        threshold=B6_QUALITY_THRESHOLDS["vertical_grounded_rate_min"],
        operator=">=",
    )
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    if not failed:
        status = "B6_QUALITY_READY"
    elif (
        _float_metric(summary, "evidence_match_rate") >= 0.80
        and _float_metric(summary, "grounded_rate") >= 0.80
    ):
        status = "B6_QUALITY_IMPROVED_BUT_BLOCKED"
    else:
        status = "B6_QUALITY_BLOCKED"
    return {
        "status": status,
        "passed": status == "B6_QUALITY_READY",
        "failed_metrics": failed,
        "checks": checks,
    }


def preflight_inference_allowed(report: dict[str, Any]) -> bool:
    """Return whether context preflight permits live inference."""

    return (
        int(report.get("all_required_evidence_present_count") or 0)
        == int(report.get("row_count") or 0)
        and int(report.get("partial_present_count") or 0) == 0
        and int(report.get("absent_count") or 0) == 0
        and int(report.get("unrecoverable_row_count") or 0) == 0
        and bool(report.get("leakage_guard_passed"))
        and not bool(report.get("canonical_ids_exposed_to_model"))
    )


def preflight_status(report: dict[str, Any]) -> str:
    """Return a stable B6 preflight status string."""

    return (
        "PREFLIGHT_PASSED_B6_CONTEXT_ALIGNMENT"
        if preflight_inference_allowed(report)
        else "PREFLIGHT_BLOCKED_B6_CONTEXT_ALIGNMENT"
    )
