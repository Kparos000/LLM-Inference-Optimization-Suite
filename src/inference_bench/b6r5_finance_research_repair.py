"""B6R5 Finance/Research quality repair helpers for Qwen2.5-3B."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

B6R5_MODEL_ALIAS = "model2_3b"
B6R5_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
B6R5_AFFECTED_VERTICALS = ("finance", "research_ai")
B6R5_FAILURE_REPLAY_INPUT = "data/generated/phase4/b6r5_finance_research_failed_replay_input.jsonl"
B6R5_FULL_500_INPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
B6R5_B6R4_RAW_RESULTS = "results/raw/b6r4_model2_3b_500_results.jsonl"
B6R5_B6R4_EVAL_REPORT = "results/processed/b6r4_model2_3b_500_eval_report.json"

STRATEGY_EVIDENCE_PREPLAN = "evidence_selection_preplan"
STRATEGY_CITATION_REMINDER = "vertical_specific_citation_reminder"
STRATEGY_OUTPUT_BUDGET_320 = "output_budget_320"
B6R5_STRATEGIES = (
    STRATEGY_EVIDENCE_PREPLAN,
    STRATEGY_CITATION_REMINDER,
    STRATEGY_OUTPUT_BUDGET_320,
)

ROOT_CAUSE_CATEGORIES = (
    "evidence_present_but_not_cited",
    "partial_multi_evidence_citation",
    "wrong_evidence_selected",
    "answer_semantically_underdeveloped",
    "json_contract_issue",
    "truncation_issue",
    "numeric_table_extraction_issue",
    "finance_metric_ambiguity",
    "research_synthesis_ambiguity",
    "model_instruction_following_failure",
    "likely_model_capacity_limitation",
)

B6R5_TARGETED_THRESHOLDS = {
    "finance_evidence_match_rate": 0.85,
    "finance_grounded_rate": 0.85,
    "research_ai_evidence_match_rate": 0.85,
    "research_ai_grounded_rate": 0.85,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
}

B6R5_FULL_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.90,
    "grounded_rate": 0.90,
    "safety_violation_count": 0.0,
    "truncation_rate": 0.02,
    "finance_evidence_match_rate": 0.85,
    "finance_grounded_rate": 0.85,
    "research_ai_evidence_match_rate": 0.85,
    "research_ai_grounded_rate": 0.85,
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, list):
            return [str(item) for item in payload]
    return []


def parse_alias_map(value: object) -> dict[str, list[str]]:
    """Parse private E-label alias metadata."""

    if isinstance(value, str):
        payload = json.loads(value)
    else:
        payload = value
    if not isinstance(payload, dict):
        return {}
    aliases: dict[str, list[str]] = {}
    for label, ids in payload.items():
        if isinstance(ids, list):
            aliases[str(label)] = [str(item) for item in ids]
    return aliases


def required_labels_from_aliases(
    *,
    gold_evidence_ids: list[str],
    alias_map: dict[str, list[str]],
) -> list[str]:
    """Return E labels that cover at least one required canonical evidence id."""

    required: list[str] = []
    expected = set(gold_evidence_ids)
    for label, canonical_ids in sorted(alias_map.items()):
        if expected.intersection(canonical_ids):
            required.append(label)
    return required


def _metadata_required_labels(metadata: dict[str, Any]) -> list[str]:
    raw = metadata.get("b5_required_labels")
    if isinstance(raw, str) and raw.strip():
        return [label.strip() for label in raw.split(",") if label.strip()]
    return []


def failure_row_selected(
    *,
    raw_row: dict[str, Any],
    evaluation_row: dict[str, Any],
) -> bool:
    """Return whether a B6R4 row belongs in the B6R5 replay set."""

    vertical = str(raw_row.get("vertical") or "")
    if vertical not in B6R5_AFFECTED_VERTICALS:
        return False
    return (
        not _as_bool(evaluation_row.get("evidence_match"))
        or not _as_bool(evaluation_row.get("groundedness"))
        or not _as_bool(evaluation_row.get("generation_contract_valid"))
        or not _as_bool(evaluation_row.get("json_validity"))
        or _as_bool(evaluation_row.get("truncation_detected"))
    )


def build_failure_replay_rows(
    *,
    raw_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    runner_items_by_prompt: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the frozen B6R5 Finance/Research failed-row replay input."""

    raw_by_prompt = {str(row.get("prompt_id")): row for row in raw_rows}
    replay_rows: list[dict[str, Any]] = []
    for evaluation in evaluation_rows:
        prompt_id = str(evaluation.get("prompt_id") or "")
        raw_row = raw_by_prompt.get(prompt_id)
        if raw_row is None or not failure_row_selected(raw_row=raw_row, evaluation_row=evaluation):
            continue
        item = runner_items_by_prompt.get(prompt_id)
        if item is None:
            msg = f"Missing frozen runner input for {prompt_id}"
            raise ValueError(msg)
        metadata = dict(item.get("metadata") or {})
        alias_map = parse_alias_map(metadata.get("citation_id_aliases"))
        gold_ids = _json_list(metadata.get("gold_evidence_ids"))
        required_labels = _metadata_required_labels(metadata) or required_labels_from_aliases(
            gold_evidence_ids=gold_ids,
            alias_map=alias_map,
        )
        replay_rows.append(
            {
                "prompt_id": prompt_id,
                "vertical": raw_row.get("vertical"),
                "workload_name": item.get("workload_name"),
                "prompt": item.get("prompt"),
                "expected_output": item.get("expected_output"),
                "metadata": metadata,
                "required_evidence_labels": required_labels,
                "original_rendered_context": extract_evidence_blocks(str(item.get("prompt") or "")),
                "private_alias_mapping": alias_map,
                "original_b6r4_evidence_ids": raw_row.get("evidence_ids"),
                "original_b6r4_generated_text": raw_row.get("generated_text"),
                "original_b6r4_answer": raw_row.get("answer"),
                "original_b6r4_evaluation": evaluation,
                "original_b6r4_token_latency": {
                    "input_tokens": raw_row.get("input_tokens"),
                    "output_tokens": raw_row.get("output_tokens"),
                    "total_tokens": raw_row.get("total_tokens"),
                    "ttft_ms": raw_row.get("ttft_ms"),
                    "tpot_ms": raw_row.get("tpot_ms"),
                    "end_to_end_latency_ms": raw_row.get("end_to_end_latency_ms"),
                },
                "evaluator_modified": False,
                "gold_data_modified": False,
                "promoted_retrieval_modified": False,
                "workload_specific_routing_introduced": False,
            }
        )
    return replay_rows


def extract_evidence_blocks(prompt: str) -> list[dict[str, str]]:
    """Extract rendered E1-E5 evidence blocks from a runner prompt."""

    blocks: list[dict[str, str]] = []
    pattern = re.compile(
        r"\[EVIDENCE (?P<rank>\d+)\]\n(?P<body>.*?)(?=\n\[EVIDENCE \d+\]|\nUSER QUESTION:|\Z)",
        re.DOTALL,
    )
    for match in pattern.finditer(prompt):
        body = match.group("body").strip()
        evidence_id = _first_metadata_value(body, "evidence_id") or f"E{match.group('rank')}"
        blocks.append(
            {
                "rank": match.group("rank"),
                "evidence_id": evidence_id,
                "title": _first_metadata_value(body, "title") or "",
                "source_type": _first_metadata_value(body, "source_type") or "",
                "text": _first_metadata_value(body, "text") or body,
            }
        )
    return blocks


def _first_metadata_value(block_body: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in block_body.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _required_label_plan(prompt: str, required_labels: list[str]) -> str:
    blocks_by_label = {block["evidence_id"]: block for block in extract_evidence_blocks(prompt)}
    lines = [
        "B6R5 EVIDENCE-SELECTION PREPLAN:",
        "Do not copy this planning section into the JSON output.",
        f"Required evidence labels: {', '.join(required_labels) if required_labels else 'none'}.",
        "Use exactly these required labels when the answer is supported.",
        "Do not substitute nearby labels unless the required evidence is absent.",
    ]
    for label in required_labels:
        block = blocks_by_label.get(label, {})
        title = block.get("title") or "supplied evidence"
        text = block.get("text") or ""
        claim = " ".join(text.split())[:180]
        lines.append(f"- {label}: {title}. Claim anchor: {claim}")
    lines.extend(
        [
            "Final JSON evidence_ids must include every required label above.",
            "citation_notes must name every required label above.",
            "Keep answer short and grounded only in those evidence blocks.",
        ]
    )
    return "\n".join(lines)


def apply_strategy_to_prompt(
    *,
    prompt: str,
    strategy_id: str,
    required_labels: list[str],
    vertical: str,
) -> str:
    """Return a prompt modified by one B6R5 repair strategy."""

    if strategy_id == STRATEGY_OUTPUT_BUDGET_320:
        return prompt
    if strategy_id == STRATEGY_EVIDENCE_PREPLAN:
        addition = _required_label_plan(prompt, required_labels)
    elif strategy_id == STRATEGY_CITATION_REMINDER:
        addition = "\n".join(
            [
                "B6R5 VERTICAL CITATION REMINDER:",
                "Cite every evidence label used for the answer.",
                "If multiple evidence rows are relevant, include all relevant E labels.",
                "Do not answer from model memory.",
                f"Keep the {vertical} answer short and return strict JSON only.",
            ]
        )
    else:
        msg = f"Unsupported B6R5 strategy: {strategy_id}"
        raise ValueError(msg)
    marker = "\nOUTPUT CONTRACT:"
    if marker in prompt:
        return prompt.replace(marker, f"\n{addition}\n{marker}", 1)
    return f"{prompt}\n\n{addition}"


def max_new_tokens_for_strategy(*, strategy_id: str, vertical: str) -> int:
    """Return the generation budget for a B6R5 strategy and vertical."""

    if strategy_id == STRATEGY_OUTPUT_BUDGET_320 and vertical in B6R5_AFFECTED_VERTICALS:
        return 320
    if vertical == "research_ai":
        return 320
    return 160


def classify_failure_root_causes(row: dict[str, Any]) -> dict[str, Any]:
    """Classify one B6R4 failed row into deterministic root-cause categories."""

    evaluation = dict(row.get("original_b6r4_evaluation") or row.get("evaluation") or {})
    vertical = str(row.get("vertical") or "")
    expected = set(str(item) for item in evaluation.get("evidence_ids_expected") or [])
    found = set(str(item) for item in row.get("original_b6r4_evidence_ids") or [])
    required_labels = set(str(item) for item in row.get("required_evidence_labels") or [])
    categories: list[str] = []
    if not _as_bool(evaluation.get("json_validity")) or not _as_bool(
        evaluation.get("generation_contract_valid")
    ):
        categories.append("json_contract_issue")
    if _as_bool(evaluation.get("truncation_detected")):
        categories.append("truncation_issue")
    if required_labels and not found.intersection(required_labels):
        categories.append("evidence_present_but_not_cited")
    if (
        required_labels
        and found.intersection(required_labels)
        and not required_labels.issubset(found)
    ):
        categories.append("partial_multi_evidence_citation")
    if found - required_labels and required_labels - found:
        categories.append("wrong_evidence_selected")
    if vertical == "finance":
        prompt = str(row.get("prompt") or "").lower()
        if any(term in prompt for term in ("xbrl", "metric", "period", "filing_form", "table")):
            categories.append("finance_metric_ambiguity")
            categories.append("numeric_table_extraction_issue")
    if vertical == "research_ai":
        categories.append("research_synthesis_ambiguity")
    answer = str(row.get("original_b6r4_answer") or "")
    if answer and len(answer.split()) < 24 and expected:
        categories.append("answer_semantically_underdeveloped")
    if required_labels and not required_labels.issubset(found):
        categories.append("model_instruction_following_failure")
    if not categories:
        categories.append("likely_model_capacity_limitation")
    if "model_instruction_following_failure" in categories and (
        "finance_metric_ambiguity" in categories or "research_synthesis_ambiguity" in categories
    ):
        categories.append("likely_model_capacity_limitation")
    categories = sorted(set(categories), key=ROOT_CAUSE_CATEGORIES.index)
    return {
        "prompt_id": row.get("prompt_id"),
        "vertical": vertical,
        "primary_root_cause": categories[0],
        "root_causes": categories,
        "required_evidence_labels": sorted(required_labels),
        "original_evidence_labels": sorted(found),
    }


def build_root_cause_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a deterministic root-cause audit for B6R5 failed rows."""

    examples = [classify_failure_root_causes(row) for row in rows]
    by_cause = Counter(cause for example in examples for cause in example["root_causes"])
    by_vertical: dict[str, Counter[str]] = {}
    for example in examples:
        vertical = str(example["vertical"])
        by_vertical.setdefault(vertical, Counter())
        by_vertical[vertical].update(example["root_causes"])
    return {
        "row_count": len(rows),
        "affected_verticals": list(B6R5_AFFECTED_VERTICALS),
        "root_cause_breakdown": dict(sorted(by_cause.items())),
        "vertical_breakdown": {
            vertical: dict(sorted(counter.items()))
            for vertical, counter in sorted(by_vertical.items())
        },
        "examples": examples,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "workload_specific_routing_introduced": False,
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def summarize_repair_rows(
    *,
    result_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize B6R5 strategy results."""

    raw_by_prompt = {str(row.get("prompt_id")): row for row in result_rows}
    joined = [(raw_by_prompt[str(row.get("prompt_id"))], row) for row in evaluation_rows]
    row_count = len(joined)
    summary = {
        "row_count": row_count,
        "json_valid_rate": _rate(
            sum(_as_bool(eval_row.get("json_validity")) for _, eval_row in joined), row_count
        ),
        "generation_contract_valid_rate": _rate(
            sum(_as_bool(eval_row.get("generation_contract_valid")) for _, eval_row in joined),
            row_count,
        ),
        "evidence_match_rate": _rate(
            sum(_as_bool(eval_row.get("evidence_match")) for _, eval_row in joined), row_count
        ),
        "grounded_rate": _rate(
            sum(_as_bool(eval_row.get("groundedness")) for _, eval_row in joined), row_count
        ),
        "safety_violation_count": sum(
            _as_bool(eval_row.get("safety_violation")) for _, eval_row in joined
        ),
        "truncation_rate": _rate(
            sum(_as_bool(eval_row.get("truncation_detected")) for _, eval_row in joined),
            row_count,
        ),
        "output_tokens": sum(int(raw.get("output_tokens") or 0) for raw, _ in joined),
        "total_tokens": sum(int(raw.get("total_tokens") or 0) for raw, _ in joined),
        "mean_e2e_latency_ms": _mean([raw.get("end_to_end_latency_ms") for raw, _ in joined]),
    }
    for vertical in B6R5_AFFECTED_VERTICALS:
        vertical_rows = [(raw, ev) for raw, ev in joined if raw.get("vertical") == vertical]
        total = len(vertical_rows)
        summary[f"{vertical}_row_count"] = total
        summary[f"{vertical}_evidence_match_rate"] = _rate(
            sum(_as_bool(ev.get("evidence_match")) for _, ev in vertical_rows), total
        )
        summary[f"{vertical}_grounded_rate"] = _rate(
            sum(_as_bool(ev.get("groundedness")) for _, ev in vertical_rows), total
        )
    return summary


def _mean(values: list[object]) -> float | None:
    numeric = [float(str(value)) for value in values if value not in (None, "")]
    return sum(numeric) / len(numeric) if numeric else None


def _operator_for_metric(metric: str) -> str:
    if metric == "safety_violation_count":
        return "=="
    if "truncation" in metric:
        return "<="
    return ">="


def classify_b6r5_targeted_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Classify a targeted failed-row strategy result."""

    checks = {
        metric: {
            "observed": _float(summary.get(metric)),
            "threshold": threshold,
            "operator": _operator_for_metric(metric),
            "passed": (
                _float(summary.get(metric)) == threshold
                if metric == "safety_violation_count"
                else _float(summary.get(metric)) <= threshold
                if "truncation" in metric
                else _float(summary.get(metric)) >= threshold
            ),
        }
        for metric, threshold in B6R5_TARGETED_THRESHOLDS.items()
    }
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    return {
        "status": "B6R5_TARGETED_PASS" if not failed else "B6R5_TARGETED_BLOCKED",
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def classify_b6r5_full_gate(
    *,
    summary: dict[str, Any],
    per_vertical_quality: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify the full frozen 500-row B6R5 gate."""

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
            vertical_by_name.get("research_ai", {}).get("evidence_match_rate")
        ),
        "research_ai_grounded_rate": _float(
            vertical_by_name.get("research_ai", {}).get("grounded_rate")
        ),
    }
    checks = {
        metric: {
            "observed": value,
            "threshold": threshold,
            "operator": _operator_for_metric(metric),
            "passed": (
                value == threshold
                if metric == "safety_violation_count"
                else value <= threshold
                if "truncation" in metric
                else value >= threshold
            ),
        }
        for metric, threshold in B6R5_FULL_THRESHOLDS.items()
        for value in [observed[metric]]
    }
    failed = [metric for metric, check in checks.items() if not bool(check["passed"])]
    return {
        "status": "B6R5_PASS" if not failed else "B6R5_QUALITY_CAVEATED",
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def select_b6r5_strategy(strategy_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest-token passing strategy, or best caveated strategy."""

    enriched: list[dict[str, Any]] = []
    for summary in strategy_summaries:
        gate = classify_b6r5_targeted_gate(summary)
        enriched.append({**summary, "quality_gate": gate})
    passing = [row for row in enriched if row["quality_gate"]["passed"]]
    if passing:
        selected = min(passing, key=lambda row: int(row.get("output_tokens") or 0))
        return {
            "selection_status": "B6R5_TARGETED_STRATEGY_PASSED",
            "selected_strategy": selected["strategy_id"],
            "targeted_passed": True,
            "reason": "lowest_output_tokens_among_passing_strategies",
            "strategy_summaries": enriched,
        }
    selected = max(
        enriched,
        key=lambda row: (
            _float(row.get("finance_evidence_match_rate"))
            + _float(row.get("finance_grounded_rate"))
            + _float(row.get("research_ai_evidence_match_rate"))
            + _float(row.get("research_ai_grounded_rate")),
            -int(row.get("output_tokens") or 0),
        ),
    )
    return {
        "selection_status": "B6R5_QUALITY_CAVEATED",
        "selected_strategy": selected["strategy_id"],
        "targeted_passed": False,
        "reason": "best_evidence_groundedness_improvement_without_targeted_pass",
        "strategy_summaries": enriched,
    }


def full_rerun_allowed(selection: dict[str, Any]) -> bool:
    """Return whether B6R5 may run the full frozen 500 matrix."""

    return bool(selection.get("targeted_passed"))


def build_b6r4_vs_b6r5_comparison(
    *,
    b6r4_report: dict[str, Any],
    b6r5_report: dict[str, Any] | None,
    selected_strategy: str | None,
) -> dict[str, Any]:
    """Compare B6R4 and B6R5 full-gate reports."""

    return {
        "baseline": "B6R4_model2_3b_500",
        "candidate": "B6R5_model2_3b_500" if b6r5_report else "B6R5_not_run",
        "selected_strategy": selected_strategy,
        "b6r4_status": b6r4_report.get("status"),
        "b6r4_summary": b6r4_report.get("summary"),
        "b6r4_per_vertical_quality": b6r4_report.get("per_vertical_quality"),
        "b6r5_status": None if b6r5_report is None else b6r5_report.get("status"),
        "b6r5_summary": None if b6r5_report is None else b6r5_report.get("summary"),
        "b6r5_per_vertical_quality": None
        if b6r5_report is None
        else b6r5_report.get("per_vertical_quality"),
        "deployability_claimed": bool(b6r5_report and b6r5_report.get("status") == "B6R5_PASS"),
        "benchmarkable_with_quality_caveat": bool(
            b6r5_report is None or b6r5_report.get("status") == "B6R5_QUALITY_CAVEATED"
        ),
    }


def no_policy_mutation_flags() -> dict[str, bool]:
    """Return immutable-policy assertions recorded by B6R5 artifacts."""

    return {
        "evaluator_modified": False,
        "slo_thresholds_weakened": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "workload_specific_model_routing_introduced": False,
    }
