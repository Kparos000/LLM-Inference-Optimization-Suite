"""B6R4 Qwen2.5-3B Research AI validation gates and comparison helpers."""

from __future__ import annotations

from typing import Any

from inference_bench.config import load_project_config

B6R4_MODEL_ALIAS = "model2_3b"
B6R4_DEPRECATED_MODEL_ALIAS = "model2_1_5b"
B6R4_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
B6R4_FROZEN_REPLAY_INPUT = "data/generated/phase4/b6r1_research_ai_failed_replay_input.jsonl"
B6R4_FULL_500_INPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
B6R4_SELECTED_CONTRACT = "research_ai_limitations_v1"
B6R4_FALLBACK_CONTRACT = "research_ai_adaptive_v1"
B6R4_MAX_NEW_TOKENS = 320

B6R4_TARGETED_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.85,
    "grounded_rate": 0.85,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
}

B6R4_FULL_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.90,
    "grounded_rate": 0.90,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
    "vertical_evidence_match_rate_min": 0.85,
    "vertical_grounded_rate_min": 0.85,
}

COMPARISON_BLOCKS = ("B6", "B6R2", "B6R3", "B6R4")


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _check(observed: float, threshold: float, operator: str) -> dict[str, Any]:
    if operator == "==":
        passed = observed == threshold
    elif operator == "<=":
        passed = observed <= threshold
    else:
        passed = observed >= threshold
    return {
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "passed": passed,
    }


def validate_b6r4_model_selection(
    *,
    model_alias: str = B6R4_MODEL_ALIAS,
    models_path: str = "configs/models.yaml",
) -> dict[str, str]:
    """Validate that B6R4 uses model2_3b and not the deprecated 1.5B alias."""

    if model_alias == B6R4_DEPRECATED_MODEL_ALIAS:
        msg = "B6R4 must use model2_3b, not deprecated model2_1_5b"
        raise ValueError(msg)
    if model_alias != B6R4_MODEL_ALIAS:
        msg = f"B6R4 model alias must be {B6R4_MODEL_ALIAS}"
        raise ValueError(msg)
    config = load_project_config(models_path=models_path)
    model = config.resolve_model_config(model_alias)
    if model.model_id != B6R4_MODEL_ID:
        msg = f"{model_alias} resolved to unexpected model {model.model_id}"
        raise ValueError(msg)
    return {"model_alias": model_alias, "model_id": model.model_id}


def validate_b6r4_replay_input(path: str) -> None:
    """Require the frozen B6R1 Research AI failed-row replay input."""

    normalized = path.replace("\\", "/")
    if normalized != B6R4_FROZEN_REPLAY_INPUT:
        msg = (
            "B6R4 targeted replay must use the frozen B6R1 Research AI failed-row "
            f"replay input: {B6R4_FROZEN_REPLAY_INPUT}"
        )
        raise ValueError(msg)


def classify_b6r4_targeted_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Classify the targeted 26-row Qwen2.5-3B Research AI replay."""

    checks: dict[str, dict[str, Any]] = {}
    for metric, threshold in B6R4_TARGETED_THRESHOLDS.items():
        operator = (
            "==" if metric == "safety_violation_count" else "<=" if "truncation" in metric else ">="
        )
        checks[metric] = _check(_float(summary.get(metric)), threshold, operator)
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    return {
        "status": (
            "B6R4_TARGETED_MODEL2_3B_PASSED" if not failed else "B6R4_TARGETED_MODEL2_3B_BLOCKED"
        ),
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def targeted_replay_allows_full_500(targeted_gate: dict[str, Any]) -> bool:
    """Return whether the full frozen 500-row model2_3b run is allowed."""

    return bool(targeted_gate.get("passed")) and (
        targeted_gate.get("status") == "B6R4_TARGETED_MODEL2_3B_PASSED"
    )


def classify_b6r4_full_500_gate(
    *,
    summary: dict[str, Any],
    per_vertical_quality: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify the full frozen 500-row Qwen2.5-3B quality gate."""

    vertical_evidence = [_float(row.get("evidence_match_rate")) for row in per_vertical_quality]
    vertical_grounded = [_float(row.get("grounded_rate")) for row in per_vertical_quality]
    observed = {
        "json_valid_rate": _float(summary.get("json_valid_rate")),
        "generation_contract_valid_rate": _float(summary.get("generation_contract_valid_rate")),
        "evidence_match_rate": _float(summary.get("evidence_match_rate")),
        "grounded_rate": _float(summary.get("grounded_rate")),
        "safety_violation_count": _float(summary.get("safety_violation_count")),
        "truncation_rate": _float(summary.get("truncation_rate")),
        "vertical_evidence_match_rate_min": min(vertical_evidence) if vertical_evidence else 0.0,
        "vertical_grounded_rate_min": min(vertical_grounded) if vertical_grounded else 0.0,
    }
    checks: dict[str, dict[str, Any]] = {}
    for metric, threshold in B6R4_FULL_THRESHOLDS.items():
        operator = (
            "==" if metric == "safety_violation_count" else "<=" if "truncation" in metric else ">="
        )
        checks[metric] = _check(observed[metric], threshold, operator)
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    return {
        "status": "B6R4_MODEL2_3B_500_READY" if not failed else "B6R4_MODEL2_3B_500_BLOCKED",
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def build_model_capacity_comparison(
    *,
    b6_research_ai: dict[str, Any] | None,
    b6r2_best: dict[str, Any] | None,
    b6r3_model6: dict[str, Any] | None,
    b6r4_summary: dict[str, Any] | None,
    b6r4_gate: dict[str, Any],
    full_500_triggered: bool,
) -> dict[str, Any]:
    """Build the B6/B6R2/B6R3/B6R4 Research AI capacity comparison."""

    b6r4_grounded = _float((b6r4_summary or {}).get("grounded_rate"))
    b6r2_grounded = _float((b6r2_best or {}).get("grounded_rate"))
    b6r3_grounded = _float((b6r3_model6 or {}).get("grounded_rate"))
    materially_improved = b6r4_summary is not None and b6r4_grounded > b6r2_grounded
    larger_models_needed = not bool(b6r4_gate.get("passed")) and b6r3_grounded >= 0.85
    return {
        "comparison_blocks": list(COMPARISON_BLOCKS),
        "b6_qwen_1_5b_research_ai": b6_research_ai,
        "b6r2_best_qwen_1_5b_contract": b6r2_best,
        "b6r3_model6_llama_3_1_8b_api": b6r3_model6,
        "b6r4_model2_3b_qwen_targeted": b6r4_summary,
        "b6r4_targeted_gate": b6r4_gate,
        "qwen3b_materially_improves_research_ai": materially_improved,
        "qwen3b_targeted_gate_passed": bool(b6r4_gate.get("passed")),
        "full_500_can_proceed_on_model2_3b": full_500_triggered,
        "larger_models_remain_necessary_for_research_ai_quality": larger_models_needed,
        "workload_specific_routing_introduced": False,
    }


def build_no_live_replay_report(*, reason: str) -> dict[str, Any]:
    """Return an explicit non-equivalent blocked report when vLLM cannot run."""

    gate = classify_b6r4_targeted_gate({})
    return {
        "block": "B6R4",
        "status": "B6R4_TARGETED_MODEL2_3B_BLOCKED",
        "targeted_replay_ran": False,
        "blocked_reason": reason,
        "quality_gate": gate,
        "model_alias": B6R4_MODEL_ALIAS,
        "model_id": B6R4_MODEL_ID,
        "runtime": "vllm",
        "hf_local_dry_run_equivalent": False,
        "full_500_rerun_triggered": False,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "workload_specific_routing_introduced": False,
    }
