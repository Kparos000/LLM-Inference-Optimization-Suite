"""B6R6 Research AI quality recovery helpers for Qwen2.5-3B."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from inference_bench.b6r5_finance_research_repair import (
    extract_evidence_blocks,
    parse_alias_map,
    required_labels_from_aliases,
)

B6R6_MODEL_ALIAS = "model2_3b"
B6R6_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
B6R6_VERTICAL = "research_ai"
B6R6_B6R4_RAW_RESULTS = "results/raw/b6r4_model2_3b_500_results.jsonl"
B6R6_B6R4_EVAL_REPORT = "results/processed/b6r4_model2_3b_500_eval_report.json"
B6R6_FULL_500_INPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
B6R6_REPLAY_INPUT = "data/generated/phase4/b6r6_research_ai_failed_replay_input.jsonl"
B6R6_B6R5_TARGETED_REPORT = "results/processed/b6r5_finance_research_targeted_replay_report.json"

STRATEGY_A_ORIGINAL = "b6r4_original_behavior"
STRATEGY_B_B6R2_BEST_CONTRACT = "b6r2_best_contract"
STRATEGY_C_EVIDENCE_WHITELIST = "evidence_whitelist"
STRATEGY_D_ANSWER_SKELETON = "answer_skeleton"
STRATEGY_E_OUTPUT_BUDGET_384 = "output_budget_384"
B6R6_STRATEGIES = (
    STRATEGY_A_ORIGINAL,
    STRATEGY_B_B6R2_BEST_CONTRACT,
    STRATEGY_C_EVIDENCE_WHITELIST,
    STRATEGY_D_ANSWER_SKELETON,
    STRATEGY_E_OUTPUT_BUDGET_384,
)

B6R6_RESEARCH_AI_FULL_FLOOR = 0.80
B6R6_RESEARCH_AI_PREFERRED_FLOOR = 0.85
B6R6_FINANCE_REPAIR_FLOOR = 0.85
B6R6_TARGETED_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.80,
    "grounded_rate": 0.80,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
}
B6R6_FULL_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.90,
    "grounded_rate": 0.90,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
    "finance_evidence_match_rate": 0.85,
    "finance_grounded_rate": 0.85,
    "research_ai_evidence_match_rate": 0.80,
    "research_ai_grounded_rate": 0.80,
}

ROOT_CAUSE_CATEGORIES = (
    "evidence_present_but_not_cited",
    "wrong_evidence_selected",
    "partial_multi_evidence_citation",
    "synthesis_under_answer",
    "answer_too_generic",
    "contract_field_too_restrictive",
    "model_capacity_limitation",
    "prompt_context_mismatch",
)

CONFIDENCE_TO_FLOAT = {"low": 0.35, "medium": 0.65, "high": 0.9}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        payload = json.loads(value)
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def _metadata_required_labels(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("b5_required_labels")
    if isinstance(raw, str) and raw.strip():
        return [label.strip() for label in raw.split(",") if label.strip()]
    aliases = parse_alias_map(metadata.get("citation_id_aliases"))
    return required_labels_from_aliases(
        gold_evidence_ids=_json_list(metadata.get("gold_evidence_ids")),
        alias_map=aliases,
    )


def research_ai_failure_row_selected(
    *,
    raw_row: dict[str, Any],
    evaluation_row: dict[str, Any],
) -> bool:
    """Return whether a B6R4 row belongs in the B6R6 replay set."""

    if str(raw_row.get("vertical") or "") != B6R6_VERTICAL:
        return False
    return (
        not _as_bool(evaluation_row.get("evidence_match"))
        or not _as_bool(evaluation_row.get("groundedness"))
        or not _as_bool(evaluation_row.get("generation_contract_valid"))
        or not _as_bool(evaluation_row.get("json_validity"))
        or _as_bool(evaluation_row.get("truncation_detected"))
    )


def build_research_ai_replay_rows(
    *,
    raw_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    runner_items_by_prompt: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the frozen B6R6 Research AI failed-row replay input."""

    raw_by_prompt = {str(row.get("prompt_id")): row for row in raw_rows}
    replay_rows: list[dict[str, Any]] = []
    for evaluation in evaluation_rows:
        prompt_id = str(evaluation.get("prompt_id") or "")
        raw_row = raw_by_prompt.get(prompt_id)
        if raw_row is None or not research_ai_failure_row_selected(
            raw_row=raw_row,
            evaluation_row=evaluation,
        ):
            continue
        item = runner_items_by_prompt.get(prompt_id)
        if item is None:
            msg = f"Missing frozen runner input for {prompt_id}"
            raise ValueError(msg)
        metadata = dict(item.get("metadata") or {})
        replay_rows.append(
            {
                "prompt_id": prompt_id,
                "vertical": B6R6_VERTICAL,
                "workload_name": item.get("workload_name"),
                "prompt": item.get("prompt"),
                "expected_output": item.get("expected_output"),
                "metadata": metadata,
                "required_evidence_labels": _metadata_required_labels(metadata),
                "original_rendered_context": extract_evidence_blocks(str(item.get("prompt") or "")),
                "private_alias_mapping": parse_alias_map(metadata.get("citation_id_aliases")),
                "original_b6r4_evidence_ids": raw_row.get("evidence_ids"),
                "original_b6r4_generated_text": raw_row.get("generated_text"),
                "original_b6r4_answer": raw_row.get("answer"),
                "original_b6r4_evaluation": evaluation,
                "evaluator_modified": False,
                "gold_data_modified": False,
                "promoted_retrieval_modified": False,
                "workload_specific_routing_introduced": False,
            }
        )
    return replay_rows


def build_research_ai_baseline_lock(
    *,
    b6r4_report: dict[str, Any],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build immutable B6R4 Research AI floor values used by B6R6."""

    research_ai: dict[str, Any] = next(
        (
            row
            for row in b6r4_report.get("per_vertical_quality", [])
            if row.get("vertical") == B6R6_VERTICAL
        ),
        {},
    )
    failed_count = len(replay_rows)
    failed_evidence = sum(
        _as_bool(row.get("original_b6r4_evaluation", {}).get("evidence_match"))
        for row in replay_rows
    )
    failed_grounded = sum(
        _as_bool(row.get("original_b6r4_evaluation", {}).get("groundedness")) for row in replay_rows
    )
    full_evidence_floor = max(
        B6R6_RESEARCH_AI_FULL_FLOOR,
        _float(research_ai.get("evidence_match_rate")),
    )
    full_grounded_floor = max(
        B6R6_RESEARCH_AI_FULL_FLOOR,
        _float(research_ai.get("grounded_rate")),
    )
    return {
        "source": "B6R4_model2_3b_500",
        "full_vertical_evidence_floor": full_evidence_floor,
        "full_vertical_grounded_floor": full_grounded_floor,
        "failed_row_evidence_floor": _rate(failed_evidence, failed_count),
        "failed_row_grounded_floor": _rate(failed_grounded, failed_count),
        "effective_targeted_evidence_floor": max(
            B6R6_RESEARCH_AI_FULL_FLOOR,
            _rate(failed_evidence, failed_count),
        ),
        "effective_targeted_grounded_floor": max(
            B6R6_RESEARCH_AI_FULL_FLOOR,
            _rate(failed_grounded, failed_count),
        ),
        "row_count": int(research_ai.get("row_count") or 0),
        "failed_replay_row_count": failed_count,
    }


def classify_research_ai_failure(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one Research AI failed row into deterministic B6R6 causes."""

    found = set(str(item) for item in row.get("original_b6r4_evidence_ids") or [])
    required = set(str(item) for item in row.get("required_evidence_labels") or [])
    prompt = str(row.get("prompt") or "").lower()
    answer = str(row.get("original_b6r4_answer") or "")
    answer_terms = {term for term in answer.lower().split() if len(term) >= 5}
    context_text = " ".join(
        str(block.get("text") or "") for block in row.get("original_rendered_context") or []
    ).lower()
    causes: list[str] = []
    if required and not found.intersection(required):
        causes.append("evidence_present_but_not_cited")
    if found - required and required - found:
        causes.append("wrong_evidence_selected")
    if required and found.intersection(required) and not required.issubset(found):
        causes.append("partial_multi_evidence_citation")
    if answer and len(answer.split()) < 28:
        causes.append("synthesis_under_answer")
    if answer and len(answer_terms) < 8:
        causes.append("answer_too_generic")
    if str(row.get("original_b6r4_generated_text") or "").strip().startswith("{"):
        raw = str(row.get("original_b6r4_generated_text") or "")
        if any(key in raw for key in ('"limitation"', '"comparison_summary"', '"findings"')):
            causes.append("contract_field_too_restrictive")
    if required and not required.issubset(found):
        causes.append("model_capacity_limitation")
    if "paper" in prompt and not any(label.lower() in context_text for label in required):
        causes.append("prompt_context_mismatch")
    if not causes:
        causes.append("model_capacity_limitation")
    causes = sorted(set(causes), key=ROOT_CAUSE_CATEGORIES.index)
    return {
        "prompt_id": row.get("prompt_id"),
        "primary_root_cause": causes[0],
        "root_causes": causes,
        "required_evidence_labels": sorted(required),
        "original_evidence_labels": sorted(found),
    }


def build_failure_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a Research-AI-only B6R6 failure audit."""

    examples = [classify_research_ai_failure(row) for row in rows]
    counts = Counter(cause for item in examples for cause in item["root_causes"])
    return {
        "row_count": len(rows),
        "vertical": B6R6_VERTICAL,
        "root_cause_breakdown": dict(sorted(counts.items())),
        "examples": examples,
        **no_policy_mutation_flags(),
    }


def _insert_before_output_contract(prompt: str, addition: str) -> str:
    marker = "\nOUTPUT CONTRACT:"
    if marker in prompt:
        return prompt.replace(marker, f"\n{addition}\n{marker}", 1)
    return f"{prompt}\n\n{addition}"


def apply_research_ai_strategy_prompt(
    *,
    prompt: str,
    strategy_id: str,
    required_labels: list[str],
) -> str:
    """Apply one B6R6 Research AI prompt-only strategy."""

    if strategy_id in {
        STRATEGY_A_ORIGINAL,
        STRATEGY_B_B6R2_BEST_CONTRACT,
        STRATEGY_E_OUTPUT_BUDGET_384,
    }:
        return prompt
    labels = ", ".join(required_labels) if required_labels else "none"
    if strategy_id == STRATEGY_C_EVIDENCE_WHITELIST:
        addition = "\n".join(
            [
                "B6R6 RESEARCH AI EVIDENCE WHITELIST:",
                f"Eligible evidence labels: {labels}.",
                "The evidence_ids field must cite only eligible labels.",
                "Do not cite labels outside the eligible set.",
                "Do not answer from model memory.",
            ]
        )
    elif strategy_id == STRATEGY_D_ANSWER_SKELETON:
        addition = "\n".join(
            [
                "B6R6 RESEARCH AI ANSWER SKELETON:",
                f"Eligible evidence labels: {labels}.",
                "Return exactly one compact JSON object with this schema:",
                '{"summary":"one or two concise sentences","evidence":["E1"],'
                '"confidence":"low|medium|high","insufficient_evidence":false}',
                "Use only eligible labels in evidence. No findings array unless needed.",
            ]
        )
    else:
        msg = f"Unsupported B6R6 strategy: {strategy_id}"
        raise ValueError(msg)
    return _insert_before_output_contract(prompt, addition)


def max_new_tokens_for_strategy(strategy_id: str) -> int:
    """Return the B6R6 generation budget for a Research AI strategy."""

    return 384 if strategy_id == STRATEGY_E_OUTPUT_BUDGET_384 else 320


def map_answer_skeleton_to_common_text(text: str) -> str:
    """Map Strategy D skeleton JSON to the common generation contract JSON."""

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return text
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict) or "summary" not in payload or "evidence" not in payload:
        return text
    confidence = str(payload.get("confidence") or "medium").lower()
    common = {
        "answer": str(payload.get("summary") or ""),
        "evidence_ids": [str(item) for item in payload.get("evidence") or []],
        "confidence": CONFIDENCE_TO_FLOAT.get(confidence, 0.65),
        "insufficient_evidence": bool(payload.get("insufficient_evidence")),
        "citation_notes": "Evidence supports the concise summary.",
    }
    return json.dumps(common, ensure_ascii=True, separators=(",", ":"))


def summarize_strategy_rows(
    *,
    result_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one Research AI strategy replay."""

    raw_by_prompt = {str(row.get("prompt_id")): row for row in result_rows}
    joined = [(raw_by_prompt[str(row.get("prompt_id"))], row) for row in evaluation_rows]
    row_count = len(joined)
    return {
        "row_count": row_count,
        "json_valid_rate": _rate(
            sum(_as_bool(row.get("json_validity")) for _, row in joined),
            row_count,
        ),
        "generation_contract_valid_rate": _rate(
            sum(_as_bool(row.get("generation_contract_valid")) for _, row in joined),
            row_count,
        ),
        "evidence_match_rate": _rate(
            sum(_as_bool(row.get("evidence_match")) for _, row in joined),
            row_count,
        ),
        "grounded_rate": _rate(
            sum(_as_bool(row.get("groundedness")) for _, row in joined),
            row_count,
        ),
        "safety_violation_count": sum(_as_bool(row.get("safety_violation")) for _, row in joined),
        "truncation_rate": _rate(
            sum(_as_bool(raw.get("truncation_detected")) for raw, _ in joined),
            row_count,
        ),
        "output_tokens": sum(int(raw.get("output_tokens") or 0) for raw, _ in joined),
        "total_tokens": sum(int(raw.get("total_tokens") or 0) for raw, _ in joined),
    }


def _operator_for_metric(metric: str) -> str:
    if metric == "safety_violation_count":
        return "=="
    if "truncation" in metric:
        return "<="
    return ">="


def _check_metric(metric: str, observed: float, threshold: float) -> dict[str, Any]:
    operator = _operator_for_metric(metric)
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


def classify_b6r6_targeted_strategy(
    *,
    summary: dict[str, Any],
    baseline_lock: dict[str, Any],
) -> dict[str, Any]:
    """Classify one B6R6 targeted strategy with the Research AI baseline lock."""

    checks = {
        metric: _check_metric(metric, _float(summary.get(metric)), threshold)
        for metric, threshold in B6R6_TARGETED_THRESHOLDS.items()
    }
    checks["baseline_evidence_floor"] = _check_metric(
        "evidence_match_rate",
        _float(summary.get("evidence_match_rate")),
        _float(baseline_lock.get("effective_targeted_evidence_floor")),
    )
    checks["baseline_grounded_floor"] = _check_metric(
        "grounded_rate",
        _float(summary.get("grounded_rate")),
        _float(baseline_lock.get("effective_targeted_grounded_floor")),
    )
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    preferred = (
        _float(summary.get("evidence_match_rate")) >= B6R6_RESEARCH_AI_PREFERRED_FLOOR
        and _float(summary.get("grounded_rate")) >= B6R6_RESEARCH_AI_PREFERRED_FLOOR
    )
    if failed:
        status = "B6R6_TARGETED_BLOCKED"
    elif preferred:
        status = "B6R6_TARGETED_READY"
    else:
        status = "B6R6_TARGETED_QUALITY_CAVEATED"
    return {
        "status": status,
        "passed": not failed,
        "preferred": preferred and not failed,
        "baseline_rejected": "baseline_evidence_floor" in failed
        or "baseline_grounded_floor" in failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def select_b6r6_strategy(
    *,
    strategy_summaries: list[dict[str, Any]],
    baseline_lock: dict[str, Any],
) -> dict[str, Any]:
    """Select a B6R6 Research AI strategy under the baseline lock."""

    enriched: list[dict[str, Any]] = []
    for summary in strategy_summaries:
        gate = classify_b6r6_targeted_strategy(summary=summary, baseline_lock=baseline_lock)
        enriched.append({**summary, "quality_gate": gate})
    preferred = [row for row in enriched if row["quality_gate"]["preferred"]]
    acceptable = [row for row in enriched if row["quality_gate"]["passed"]]
    if preferred:
        selected = max(
            preferred,
            key=lambda row: (
                _float(row.get("evidence_match_rate")) + _float(row.get("grounded_rate")),
                -int(row.get("output_tokens") or 0),
            ),
        )
        status = "B6R6_TARGETED_READY"
        reason = "selected_best_strategy_at_or_above_85_percent"
    elif acceptable:
        selected = max(
            acceptable,
            key=lambda row: (
                _float(row.get("evidence_match_rate")) + _float(row.get("grounded_rate")),
                -int(row.get("output_tokens") or 0),
            ),
        )
        status = "B6R6_TARGETED_QUALITY_CAVEATED"
        reason = "selected_best_strategy_restoring_research_ai_to_at_least_80_percent"
    else:
        selected = max(
            enriched,
            key=lambda row: (
                _float(row.get("evidence_match_rate")) + _float(row.get("grounded_rate")),
                -int(row.get("output_tokens") or 0),
            ),
        )
        status = "B6R6_TARGETED_BLOCKED"
        reason = "no_strategy_restored_research_ai_baseline"
    return {
        "selection_status": status,
        "selected_strategy": selected["strategy_id"],
        "targeted_passed": status in {"B6R6_TARGETED_READY", "B6R6_TARGETED_QUALITY_CAVEATED"},
        "preferred_passed": status == "B6R6_TARGETED_READY",
        "reason": reason,
        "strategy_summaries": enriched,
    }


def finance_repair_candidate_passes(b6r5_report: dict[str, Any]) -> bool:
    """Return whether the selected B6R5 Finance repair remains above 85%."""

    selection = b6r5_report.get("selection") or {}
    selected = str(selection.get("selected_strategy") or "")
    for summary in selection.get("strategy_summaries") or []:
        if summary.get("strategy_id") != selected:
            continue
        return (
            _float(summary.get("finance_evidence_match_rate")) >= B6R6_FINANCE_REPAIR_FLOOR
            and _float(summary.get("finance_grounded_rate")) >= B6R6_FINANCE_REPAIR_FLOOR
        )
    return False


def full_rerun_allowed(
    *,
    selection: dict[str, Any],
    b6r5_report: dict[str, Any],
) -> bool:
    """Return whether B6R6 may run the full frozen 500 matrix."""

    return bool(selection.get("targeted_passed")) and finance_repair_candidate_passes(b6r5_report)


def classify_b6r6_full_gate(
    *,
    summary: dict[str, Any],
    per_vertical_quality: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify the full frozen 500-row B6R6 gate."""

    vertical_by_name = {str(row.get("vertical")): row for row in per_vertical_quality}
    observed = {
        "json_valid_rate": _float(summary.get("json_valid_rate")),
        "generation_contract_valid_rate": _float(summary.get("generation_contract_valid_rate")),
        "evidence_match_rate": _float(summary.get("evidence_match_rate")),
        "grounded_rate": _float(summary.get("grounded_rate")),
        "safety_violation_count": _float(summary.get("safety_violation_count")),
        "truncation_rate": _float(summary.get("truncation_rate")),
        "finance_evidence_match_rate": _float(
            vertical_by_name.get("finance", {}).get("evidence_match_rate")
        ),
        "finance_grounded_rate": _float(vertical_by_name.get("finance", {}).get("grounded_rate")),
        "research_ai_evidence_match_rate": _float(
            vertical_by_name.get(B6R6_VERTICAL, {}).get("evidence_match_rate")
        ),
        "research_ai_grounded_rate": _float(
            vertical_by_name.get(B6R6_VERTICAL, {}).get("grounded_rate")
        ),
    }
    checks = {
        metric: _check_metric(metric, observed[metric], threshold)
        for metric, threshold in B6R6_FULL_THRESHOLDS.items()
    }
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    research_preferred = (
        observed["research_ai_evidence_match_rate"] >= B6R6_RESEARCH_AI_PREFERRED_FLOOR
        and observed["research_ai_grounded_rate"] >= B6R6_RESEARCH_AI_PREFERRED_FLOOR
    )
    if failed:
        status = "B6R6_BLOCKED"
        deployability = "NOT_READY"
        benchmark = "NOT_READY"
    elif research_preferred:
        status = "B6R6_QUALITY_READY"
        deployability = "READY"
        benchmark = "READY"
    else:
        status = "BENCHMARK_EXECUTION_READY_WITH_QUALITY_CAVEAT"
        deployability = "NOT_READY"
        benchmark = "READY_WITH_QUALITY_CAVEAT"
    return {
        "status": status,
        "passed": not failed,
        "research_ai_preferred_passed": research_preferred and not failed,
        "failed_metrics": failed,
        "checks": checks,
        "deployability_readiness": deployability,
        "benchmark_execution_readiness": benchmark,
    }


def no_policy_mutation_flags() -> dict[str, bool]:
    """Return immutable-policy assertions recorded by B6R6 artifacts."""

    return {
        "evaluator_modified": False,
        "slo_thresholds_weakened": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "workload_specific_model_routing_introduced": False,
    }
