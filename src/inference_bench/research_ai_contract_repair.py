"""B6R1 Research AI truncation and contract-repair helpers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from inference_bench.schema import WorkloadItem

RESEARCH_AI_CONCISE_STRATEGY = "concise_research_ai_renderer"
RESEARCH_AI_BUDGET_STRATEGY = "research_ai_output_budget_224"
RESEARCH_AI_BASE_MAX_NEW_TOKENS = 160
RESEARCH_AI_BUDGET_MAX_NEW_TOKENS = 224

B6R1_TARGET_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.85,
    "grounded_rate": 0.85,
    "truncation_rate": 0.02,
    "safety_violation_count": 0,
}

B6R1_FULL_THRESHOLDS = {
    "json_valid_rate": 0.97,
    "generation_contract_valid_rate": 0.97,
    "evidence_match_rate": 0.90,
    "grounded_rate": 0.90,
    "truncation_rate": 0.02,
    "safety_violation_count": 0,
    "vertical_evidence_match_rate_min": 0.85,
    "vertical_grounded_rate_min": 0.85,
    "research_ai_json_valid_rate": 0.97,
    "research_ai_generation_contract_valid_rate": 0.97,
    "research_ai_truncation_rate": 0.02,
}


def _json_obj(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return []


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL object rows."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write heterogeneous rows to CSV."""

    if not rows:
        raise ValueError("at least one CSV row is required")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL object rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row in {path}")
            rows.append(payload)
    return rows


def _failed_research_ai_row(
    *,
    result_row: dict[str, Any],
    evaluation_row: dict[str, Any],
) -> bool:
    return str(result_row.get("vertical")) == "research_ai" and (
        _bool(result_row.get("truncation_detected"))
        or not _bool(evaluation_row.get("json_validity"))
        or not _bool(evaluation_row.get("generation_contract_valid"))
        or not _bool(evaluation_row.get("evidence_match"))
        or not _bool(evaluation_row.get("groundedness"))
    )


def build_research_ai_replay_rows(
    *,
    runner_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select frozen B6 Research AI rows requiring B6R1 replay."""

    runner_by_prompt = {str(row["prompt_id"]): row for row in runner_rows}
    result_by_prompt = {str(row["prompt_id"]): row for row in result_rows}
    replay_rows: list[dict[str, Any]] = []
    for evaluation in evaluation_rows:
        prompt_id = str(evaluation.get("prompt_id") or "")
        result = result_by_prompt.get(prompt_id)
        runner = runner_by_prompt.get(prompt_id)
        if result is None or runner is None:
            continue
        if not _failed_research_ai_row(result_row=result, evaluation_row=evaluation):
            continue
        flags = {
            "truncated": _bool(result.get("truncation_detected")),
            "invalid_json": not _bool(evaluation.get("json_validity")),
            "invalid_contract": not _bool(evaluation.get("generation_contract_valid")),
            "evidence_match_failed": not _bool(evaluation.get("evidence_match")),
            "groundedness_failed": not _bool(evaluation.get("groundedness")),
        }
        replay_rows.append(
            {
                "prompt_id": prompt_id,
                "vertical": "research_ai",
                "runner_input": runner,
                "b6_result": result,
                "b6_evaluation": evaluation,
                "b6_failure_flags": flags,
                "b6_token_latency": {
                    key: result.get(key)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "ttft_ms",
                        "tpot_ms",
                        "end_to_end_latency_ms",
                        "throughput_tokens_per_second",
                    )
                },
                "citation_id_aliases": _json_obj(
                    runner.get("metadata", {}).get("citation_id_aliases")
                ),
            }
        )
    return replay_rows


def classify_research_ai_failure(row: dict[str, Any]) -> list[str]:
    """Classify one frozen B6 Research AI failed row deterministically."""

    result = _json_obj(row.get("b6_result"))
    evaluation = _json_obj(row.get("b6_evaluation"))
    flags = _json_obj(row.get("b6_failure_flags"))
    causes: list[str] = []
    output_tokens = int(result.get("output_tokens") or 0)
    generated_text = str(result.get("generated_text") or "")
    answer = str(result.get("answer") or "")
    missing_fields = _list(
        result.get("generation_contract_missing_fields")
        or evaluation.get("generation_contract_missing_fields")
    )
    parse_error = str(result.get("parse_error_type") or evaluation.get("parse_error_type") or "")
    if _bool(flags.get("truncated")) or output_tokens >= RESEARCH_AI_BASE_MAX_NEW_TOKENS - 4:
        causes.append("output_budget_too_small")
    if len(answer.split()) > 55 or output_tokens >= 140:
        causes.append("answer_too_verbose")
    if parse_error == "truncated_json" or ("{" in generated_text and "}" not in generated_text):
        causes.append("json_closing_missing")
    if _bool(flags.get("truncated")) and _bool(flags.get("evidence_match_failed")):
        causes.append("evidence_ids_missing_due_to_truncation")
    if missing_fields:
        causes.append("contract_field_missing")
    if any(
        phrase in generated_text.lower()
        for phrase in (
            "the answer is",
            "based on the evidence",
            "here is",
            "in conclusion",
            "to answer",
        )
    ):
        causes.append("answer_contains_unneeded_explanation")
    if _bool(flags.get("evidence_match_failed")) and _bool(
        evaluation.get("generation_contract_valid")
    ):
        causes.append("citation_selection_failure")
    if (
        _bool(flags.get("invalid_json"))
        or _bool(flags.get("invalid_contract"))
        or _bool(flags.get("groundedness_failed"))
    ):
        causes.append("model_instruction_following_failure")
    return list(dict.fromkeys(causes))


def build_failure_audit_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build B6R1 Research AI failure audit payload."""

    audited_rows = []
    counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for row in rows:
        causes = classify_research_ai_failure(row)
        counts.update(causes)
        flags = _json_obj(row.get("b6_failure_flags"))
        flag_counts.update(key for key, value in flags.items() if _bool(value))
        audited_rows.append({**row, "root_causes": causes})
    return {
        "block": "B6R1",
        "scope": "research_ai_failed_truncated_invalid_b6_rows",
        "row_count": len(rows),
        "root_cause_counts": dict(sorted(counts.items())),
        "failure_flag_counts": dict(sorted(flag_counts.items())),
        "rows": audited_rows,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
    }


def failure_audit_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return root-cause count rows."""

    rows = [
        {"category": "root_cause", "name": name, "count": count}
        for name, count in report["root_cause_counts"].items()
    ]
    rows.extend(
        {"category": "failure_flag", "name": name, "count": count}
        for name, count in report["failure_flag_counts"].items()
    )
    return rows or [{"category": "none", "name": "none", "count": 0}]


def write_failure_audit_artifacts(
    *,
    report: dict[str, Any],
    report_path: str | Path,
    summary_path: str | Path,
) -> None:
    """Write B6R1 failure audit JSON and CSV."""

    _write_json(report_path, report)
    write_csv(summary_path, failure_audit_summary_rows(report))


def apply_research_ai_strategy(
    item: WorkloadItem,
    *,
    strategy: str,
) -> tuple[WorkloadItem, int]:
    """Apply a Research AI repair strategy without changing aliases or gold data."""

    if strategy == RESEARCH_AI_CONCISE_STRATEGY:
        instruction = "\n".join(
            [
                "RESEARCH AI COMPACT ANSWER RULES:",
                "Write the answer field as 1-3 compact sentences.",
                "Use only supplied E-label evidence IDs in evidence_ids.",
                "Do not add setup prose, explanations, markdown, or text outside JSON.",
                "Keep citation_notes short and only map labels to claims.",
            ]
        )
        marker = "\n\nOUTPUT CONTRACT:\n"
        prompt = (
            item.prompt.replace(marker, f"\n\n{instruction}{marker}", 1)
            if marker in item.prompt
            else f"{item.prompt.rstrip()}\n\n{instruction}\n"
        )
        metadata = {
            **item.metadata,
            "b6r1_strategy": strategy,
            "b6r1_research_ai_concise_renderer": "true",
            "b6r1_max_new_tokens": str(RESEARCH_AI_BASE_MAX_NEW_TOKENS),
        }
        return (
            WorkloadItem(
                prompt_id=item.prompt_id,
                workload_name=item.workload_name,
                prompt=prompt,
                expected_output=item.expected_output,
                metadata=metadata,
            ),
            RESEARCH_AI_BASE_MAX_NEW_TOKENS,
        )
    if strategy == RESEARCH_AI_BUDGET_STRATEGY:
        metadata = {
            **item.metadata,
            "b6r1_strategy": strategy,
            "b6r1_research_ai_budget_increase": "true",
            "b6r1_max_new_tokens": str(RESEARCH_AI_BUDGET_MAX_NEW_TOKENS),
        }
        return (
            WorkloadItem(
                prompt_id=item.prompt_id,
                workload_name=item.workload_name,
                prompt=item.prompt,
                expected_output=item.expected_output,
                metadata=metadata,
            ),
            RESEARCH_AI_BUDGET_MAX_NEW_TOKENS,
        )
    raise ValueError(f"Unknown Research AI repair strategy: {strategy}")


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return sum(_bool(row.get(field)) for row in rows) / len(rows) if rows else 0.0


def summarize_research_ai_strategy(
    *,
    strategy: str,
    evaluation_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one targeted Research AI strategy replay."""

    retry_count = sum(int(row.get("retry_attempt_count") or 0) for row in result_rows)
    latency_values = [
        _float(row.get("end_to_end_latency_ms"))
        for row in result_rows
        if row.get("end_to_end_latency_ms") not in (None, "")
    ]
    ttft_values = [
        _float(row.get("ttft_ms")) for row in result_rows if row.get("ttft_ms") not in (None, "")
    ]
    tpot_values = [
        _float(row.get("tpot_ms")) for row in result_rows if row.get("tpot_ms") not in (None, "")
    ]
    output_tokens = [int(row.get("output_tokens") or 0) for row in result_rows]
    summary = {
        "strategy": strategy,
        "row_count": len(evaluation_rows),
        "json_valid_rate": _rate(evaluation_rows, "json_validity"),
        "generation_contract_valid_rate": _rate(evaluation_rows, "generation_contract_valid"),
        "evidence_match_rate": _rate(evaluation_rows, "evidence_match"),
        "grounded_rate": _rate(evaluation_rows, "groundedness"),
        "safety_violation_count": sum(
            _bool(row.get("safety_violation")) for row in evaluation_rows
        ),
        "truncation_rate": _rate(result_rows, "truncation_detected"),
        "mean_output_tokens": fmean(output_tokens) if output_tokens else None,
        "mean_e2e_latency_ms": fmean(latency_values) if latency_values else None,
        "mean_ttft_ms": fmean(ttft_values) if ttft_values else None,
        "mean_tpot_ms": fmean(tpot_values) if tpot_values else None,
        "retry_count": retry_count,
    }
    summary["passed"] = targeted_strategy_passes(summary)
    return summary


def targeted_strategy_passes(summary: dict[str, Any]) -> bool:
    """Return whether a targeted strategy clears B6R1 Research AI thresholds."""

    return (
        _float(summary.get("json_valid_rate")) >= B6R1_TARGET_THRESHOLDS["json_valid_rate"]
        and _float(summary.get("generation_contract_valid_rate"))
        >= B6R1_TARGET_THRESHOLDS["generation_contract_valid_rate"]
        and _float(summary.get("evidence_match_rate"))
        >= B6R1_TARGET_THRESHOLDS["evidence_match_rate"]
        and _float(summary.get("grounded_rate")) >= B6R1_TARGET_THRESHOLDS["grounded_rate"]
        and _float(summary.get("truncation_rate")) <= B6R1_TARGET_THRESHOLDS["truncation_rate"]
        and int(summary.get("safety_violation_count") or 0)
        == B6R1_TARGET_THRESHOLDS["safety_violation_count"]
    )


def select_research_ai_strategy(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the passing strategy with lower token use then latency."""

    passing = [summary for summary in summaries if targeted_strategy_passes(summary)]
    if not passing:
        return {
            "selected_strategy": None,
            "selection_status": "NO_STRATEGY_PASSED",
            "reason": "Neither targeted Research AI repair strategy passed all thresholds.",
        }
    selected = sorted(
        passing,
        key=lambda row: (
            _float(row.get("mean_output_tokens")),
            _float(row.get("mean_e2e_latency_ms")),
            str(row.get("strategy")),
        ),
    )[0]
    return {
        "selected_strategy": selected["strategy"],
        "selection_status": "SELECTED_PASSING_STRATEGY",
        "reason": "Selected the passing strategy with lower output tokens, then lower E2E latency.",
    }


def classify_b6r1_full_gate(
    *,
    summary: dict[str, Any],
    per_vertical_quality: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify full frozen B6R1 500-row rerun quality."""

    research_ai = next(
        (row for row in per_vertical_quality if row.get("vertical") == "research_ai"),
        {},
    )
    vertical_evidence = [_float(row.get("evidence_match_rate")) for row in per_vertical_quality]
    vertical_grounded = [_float(row.get("grounded_rate")) for row in per_vertical_quality]
    observed = {
        "json_valid_rate": _float(summary.get("json_valid_rate")),
        "generation_contract_valid_rate": _float(summary.get("generation_contract_valid_rate")),
        "evidence_match_rate": _float(summary.get("evidence_match_rate")),
        "grounded_rate": _float(summary.get("grounded_rate")),
        "truncation_rate": _float(summary.get("truncation_rate")),
        "safety_violation_count": int(summary.get("safety_violation_count") or 0),
        "vertical_evidence_match_rate_min": min(vertical_evidence) if vertical_evidence else 0.0,
        "vertical_grounded_rate_min": min(vertical_grounded) if vertical_grounded else 0.0,
        "research_ai_json_valid_rate": _float(research_ai.get("json_valid_rate")),
        "research_ai_generation_contract_valid_rate": _float(
            research_ai.get("generation_contract_valid_rate")
        ),
        "research_ai_truncation_rate": _float(research_ai.get("truncation_rate")),
    }
    checks: dict[str, dict[str, Any]] = {}
    for metric, target in B6R1_FULL_THRESHOLDS.items():
        operator = "<=" if metric.endswith("_rate") and "truncation" in metric else ">="
        if metric == "safety_violation_count":
            passed = observed[metric] == target
            operator = "=="
        elif operator == "<=":
            passed = float(observed[metric]) <= float(target)
        else:
            passed = float(observed[metric]) >= float(target)
        checks[metric] = {
            "observed": observed[metric],
            "threshold": target,
            "operator": operator,
            "passed": passed,
        }
    failed = [metric for metric, check in checks.items() if not check["passed"]]
    return {
        "status": "B6R1_READY" if not failed else "B6R1_BLOCKED",
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }
