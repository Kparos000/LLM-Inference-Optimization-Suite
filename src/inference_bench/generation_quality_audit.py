"""Deterministic root-cause audit for failed grounded-generation rows."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

FAILURE_CLASSES = (
    "retrieved_gold_absent_from_context",
    "evidence_present_but_not_cited",
    "partial_multi_evidence_citation",
    "invalid_json",
    "invalid_contract",
    "safety_violation",
    "truncation",
    "insufficient_evidence_wrongly_used",
    "answer_semantically_underdeveloped",
    "finance_metric_period_missing",
    "context_ordering_issue",
    "model_instruction_following_failure",
)
VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")
FINANCE_SAFETY_PATTERNS = (
    "investment advice",
    "investment recommendation",
    "recommend buying",
    "recommend selling",
    "buy the stock",
    "sell the stock",
    "price target",
    "guaranteed return",
    "guaranteed outcome",
    "projection",
    "projected",
    "forecast",
)
FINANCE_METADATA_KEYS = {
    "ticker": ("ticker",),
    "company": ("company_name", "company"),
    "period": (
        "period",
        "fiscal_period",
        "fiscal_year",
        "report_date",
        "filing_date",
    ),
    "metric": (
        "metric",
        "metrics",
        "financial_metric",
        "concept",
        "concepts",
        "xbrl_concept",
    ),
    "form": ("filing_form", "form"),
}


def _json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def _string_list(value: Any) -> list[str]:
    parsed = _json_value(value, [])
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return [str(item) for item in parsed if str(item)]


def _mapping(value: Any) -> dict[str, Any]:
    parsed = _json_value(value, {})
    if not isinstance(parsed, dict):
        return {}
    return cast(dict[str, Any], parsed)


def _aliases(value: Any) -> dict[str, list[str]]:
    parsed = _mapping(value)
    return {str(label): _string_list(identifiers) for label, identifiers in parsed.items()}


def _runner_metadata(runner_input: dict[str, Any]) -> dict[str, Any]:
    return _mapping(runner_input.get("metadata"))


def _source_prompt(runner_input: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_runner_metadata(runner_input).get("source_prompt_record"))


def _retrieval_metadata(runner_input: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_runner_metadata(runner_input).get("retrieval_metadata"))


def _context_text(result_row: dict[str, Any], runner_input: dict[str, Any]) -> str:
    prompt = str(result_row.get("prompt") or runner_input.get("prompt") or "")
    return prompt.split("\nUSER QUESTION:", maxsplit=1)[0]


def _label_rank(label: str) -> int | None:
    match = re.fullmatch(r"E(\d+)", label.upper())
    return int(match.group(1)) if match else None


def _normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _finance_values(source_prompt: dict[str, Any], group: str) -> list[str]:
    values: list[str] = []
    for key in FINANCE_METADATA_KEYS[group]:
        raw = source_prompt.get(key)
        if isinstance(raw, (list, tuple, set)):
            values.extend(str(item) for item in raw if str(item).strip())
        elif raw not in (None, ""):
            values.append(str(raw))
    if group == "company" and values:
        company = re.sub(r"\s*\([^)]*\).*$", "", values[0]).strip()
        values = [company] if company else []
    if group == "period" and not values:
        question = str(source_prompt.get("question") or "")
        values.extend(re.findall(r"\b(?:19|20)\d{2}\b|\bQ[1-4]\b|\bFY\d{2,4}\b", question))
    return list(dict.fromkeys(values))


def finance_metadata_presence(
    source_prompt: dict[str, Any],
    rendered_context: str,
) -> dict[str, dict[str, Any]]:
    """Report whether prompt-side Finance metadata is represented in E1-E5."""

    normalized_context = _normalized_text(rendered_context)
    result: dict[str, dict[str, Any]] = {}
    for group in FINANCE_METADATA_KEYS:
        values = _finance_values(source_prompt, group)
        missing = [value for value in values if _normalized_text(value) not in normalized_context]
        result[group] = {
            "values": values,
            "status": ("not_specified" if not values else "missing" if missing else "present"),
            "missing_values": missing,
        }
    return result


def _semantic_terms_missing(evaluation_row: dict[str, Any]) -> list[str]:
    expected = set(_string_list(evaluation_row.get("evidence_ids_expected")))
    return [
        term
        for term in _string_list(evaluation_row.get("must_include_missing"))
        if term not in expected and "_" not in term
    ]


def _finance_metric_period_issue(
    *,
    vertical: str,
    gold_absent_ids: list[str],
    finance_presence: dict[str, dict[str, Any]],
) -> bool:
    if vertical != "finance":
        return False
    metadata_missing = any(
        finance_presence[group]["status"] == "missing" for group in ("metric", "period")
    )
    exact_finance_source_missing = any(
        "_xbrl_" in evidence_id.lower()
        or "_10k_" in evidence_id.lower()
        or "_10q_" in evidence_id.lower()
        or "_8k_" in evidence_id.lower()
        for evidence_id in gold_absent_ids
    )
    return metadata_missing or exact_finance_source_missing


def is_failed_quality_row(evaluation_row: dict[str, Any]) -> bool:
    """Return whether a row failed any B1 grounded-output quality condition."""

    return not (
        bool(evaluation_row.get("json_validity"))
        and bool(evaluation_row.get("generation_contract_valid"))
        and bool(evaluation_row.get("evidence_match"))
        and bool(evaluation_row.get("groundedness"))
        and not bool(evaluation_row.get("safety_violation"))
        and not bool(evaluation_row.get("truncation_detected"))
    )


def classify_quality_failure(
    evaluation_row: dict[str, Any],
    result_row: dict[str, Any],
    runner_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one failed row using frozen evaluator and runner artifacts."""

    runner = runner_input or {}
    metadata = _runner_metadata(runner)
    alias_map = _aliases(
        result_row.get("citation_id_aliases") or metadata.get("citation_id_aliases")
    )
    expected = set(_string_list(evaluation_row.get("evidence_ids_expected")))
    found = set(_string_list(evaluation_row.get("evidence_ids_found")))
    emitted_labels = _string_list(result_row.get("evidence_ids") or result_row.get("citations"))
    labels_by_expected: dict[str, list[str]] = {
        evidence_id: sorted(label for label, aliases in alias_map.items() if evidence_id in aliases)
        for evidence_id in expected
    }
    gold_present_ids = sorted(
        evidence_id for evidence_id, labels in labels_by_expected.items() if labels
    )
    gold_absent_ids = sorted(expected.difference(gold_present_ids))
    present_not_found = sorted(set(gold_present_ids).difference(found))
    emitted_canonical = {
        identifier for label in emitted_labels for identifier in alias_map.get(label, [])
    }
    wrong_labels = sorted(
        label for label in emitted_labels if not expected.intersection(alias_map.get(label, []))
    )
    gold_labels = sorted(
        {label for labels in labels_by_expected.values() for label in labels},
        key=lambda label: (_label_rank(label) is None, _label_rank(label) or 0, label),
    )

    vertical = str(
        result_row.get("vertical")
        or metadata.get("vertical")
        or evaluation_row.get("vertical")
        or ""
    )
    context = _context_text(result_row, runner)
    source_prompt = _source_prompt(runner)
    finance_presence = (
        finance_metadata_presence(source_prompt, context) if vertical == "finance" else {}
    )
    generated_text = str(result_row.get("generated_text") or "")
    finance_safety_matches = [
        term for term in FINANCE_SAFETY_PATTERNS if term in generated_text.lower()
    ]
    expected_status = str(evaluation_row.get("expected_status") or "")
    observed_status = str(evaluation_row.get("observed_status") or "")
    insufficient_misuse = (
        observed_status == "insufficient_evidence" and expected_status != "insufficient_evidence"
    ) or (expected_status == "insufficient_evidence" and observed_status != "insufficient_evidence")
    semantic_terms = _semantic_terms_missing(evaluation_row)
    partial_multi = len(expected) > 1 and 0 < len(found) < len(expected)
    required_ranks = [
        rank for rank in (_label_rank(label) for label in gold_labels) if rank is not None
    ]
    wrong_ranks = [
        rank for rank in (_label_rank(label) for label in wrong_labels) if rank is not None
    ]
    ordering_issue = bool(present_not_found) and (
        (required_ranks and min(required_ranks) > 1)
        or (required_ranks and wrong_ranks and min(wrong_ranks) < min(required_ranks))
    )
    finance_metric_period_issue = _finance_metric_period_issue(
        vertical=vertical,
        gold_absent_ids=gold_absent_ids,
        finance_presence=finance_presence,
    )

    classes: list[str] = []
    if gold_absent_ids:
        classes.append("retrieved_gold_absent_from_context")
    if present_not_found:
        classes.append("evidence_present_but_not_cited")
    if partial_multi:
        classes.append("partial_multi_evidence_citation")
    if not bool(evaluation_row.get("json_validity")):
        classes.append("invalid_json")
    if not bool(evaluation_row.get("generation_contract_valid")):
        classes.append("invalid_contract")
    if bool(evaluation_row.get("safety_violation")):
        classes.append("safety_violation")
    if bool(evaluation_row.get("truncation_detected") or result_row.get("truncation_detected")):
        classes.append("truncation")
    if insufficient_misuse:
        classes.append("insufficient_evidence_wrongly_used")
    if semantic_terms:
        classes.append("answer_semantically_underdeveloped")
    if finance_metric_period_issue:
        classes.append("finance_metric_period_missing")
    if ordering_issue:
        classes.append("context_ordering_issue")

    instruction_failure = any(
        failure_class in classes
        for failure_class in (
            "evidence_present_but_not_cited",
            "invalid_json",
            "invalid_contract",
            "safety_violation",
            "insufficient_evidence_wrongly_used",
        )
    ) or bool(wrong_labels and gold_present_ids)
    if partial_multi and not gold_absent_ids:
        instruction_failure = True
    if instruction_failure:
        classes.append("model_instruction_following_failure")

    ordered_classes = [
        failure_class for failure_class in FAILURE_CLASSES if failure_class in classes
    ]
    retrieval_metadata = _retrieval_metadata(runner)
    return {
        "prompt_id": str(evaluation_row.get("prompt_id") or result_row.get("prompt_id") or ""),
        "vertical": vertical,
        "failure_classes": ordered_classes,
        "expected_evidence_ids": sorted(expected),
        "found_evidence_ids": sorted(found),
        "gold_evidence_present_ids": gold_present_ids,
        "gold_evidence_absent_ids": gold_absent_ids,
        "gold_evidence_labels": gold_labels,
        "emitted_evidence_labels": emitted_labels,
        "emitted_canonical_evidence_ids": sorted(emitted_canonical),
        "wrong_evidence_labels": wrong_labels,
        "evidence_present_but_not_cited_ids": present_not_found,
        "gold_evidence_present_in_context": len(gold_present_ids) == len(expected),
        "retrieval_gold_evidence_included_flag": retrieval_metadata.get("gold_evidence_included"),
        "retrieval_missing_gold_evidence_count": retrieval_metadata.get(
            "missing_gold_evidence_count"
        ),
        "json_validity": bool(evaluation_row.get("json_validity")),
        "generation_contract_valid": bool(evaluation_row.get("generation_contract_valid")),
        "evidence_match": bool(evaluation_row.get("evidence_match")),
        "groundedness": bool(evaluation_row.get("groundedness")),
        "safety_violation": bool(evaluation_row.get("safety_violation")),
        "safety_violation_terms": _string_list(evaluation_row.get("safety_violation_terms")),
        "truncation_detected": bool(
            evaluation_row.get("truncation_detected") or result_row.get("truncation_detected")
        ),
        "insufficient_evidence": result_row.get("insufficient_evidence"),
        "semantic_terms_missing": semantic_terms,
        "finance_metadata_presence": finance_presence,
        "finance_safety_term_matches": finance_safety_matches,
        "generated_text": generated_text,
        "prompt": str(result_row.get("prompt") or runner.get("prompt") or ""),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    for row in rows:
        class_counts.update(cast(list[str], row["failure_classes"]))
    return {
        "failed_row_count": len(rows),
        "failure_class_counts": {
            failure_class: class_counts.get(failure_class, 0) for failure_class in FAILURE_CLASSES
        },
        "rows_with_all_gold_evidence_present": sum(
            bool(row["gold_evidence_present_in_context"]) for row in rows
        ),
        "rows_with_any_gold_evidence_absent": sum(
            bool(row["gold_evidence_absent_ids"]) for row in rows
        ),
        "rows_with_wrong_evidence_cited": sum(bool(row["wrong_evidence_labels"]) for row in rows),
    }


def _finance_assessment(
    rows: list[dict[str, Any]],
    *,
    interpretation: str | None = None,
) -> dict[str, Any]:
    if not rows:
        return {}
    aggregate = _aggregate_rows(rows)
    absent = int(aggregate["rows_with_any_gold_evidence_absent"])
    ignored = sum(bool(row["evidence_present_but_not_cited_ids"]) for row in rows)
    safety = sum(bool(row["safety_violation"]) for row in rows)
    if absent > ignored and absent > safety:
        primary = "retrieval_context_snapshot"
    elif ignored:
        primary = "model_citation_selection"
    elif safety:
        primary = "safety_wording"
    else:
        primary = "mixed"
    return {
        **aggregate,
        "primary_problem": primary,
        "required_gold_in_e1_e5_count": sum(
            bool(row["gold_evidence_present_in_context"]) for row in rows
        ),
        "required_gold_absent_from_e1_e5_count": absent,
        "evidence_present_but_ignored_count": ignored,
        "wrong_evidence_cited_count": sum(bool(row["wrong_evidence_labels"]) for row in rows),
        "finance_safety_violation_count": safety,
        "finance_advice_projection_wording_count": sum(
            bool(row["finance_safety_term_matches"]) for row in rows
        ),
        "interpretation": interpretation
        or (
            "The frozen B1 Finance failures are primarily a rendered-context/workload "
            "alignment problem when required gold evidence is absent from E1-E5. Model "
            "citation selection remains a secondary issue for rows where evidence was "
            "available. This does not revise the promoted retrieval source of truth."
        ),
    }


def build_generation_quality_audit(
    *,
    evaluation_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    runner_inputs: list[dict[str, Any]] | None = None,
    block: str = "B3",
    status: str = "AUDIT_COMPLETE_QUALITY_REMAINS_BLOCKED",
    source_block: str = "B1",
    model_inference_triggered: bool = False,
    finance_interpretation: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic generation-quality audit from frozen artifacts."""

    results_by_prompt = {str(row.get("prompt_id") or ""): row for row in result_rows}
    inputs_by_prompt = {str(row.get("prompt_id") or ""): row for row in runner_inputs or []}
    failures: list[dict[str, Any]] = []
    for evaluation in evaluation_rows:
        if not is_failed_quality_row(evaluation):
            continue
        prompt_id = str(evaluation.get("prompt_id") or "")
        failures.append(
            classify_quality_failure(
                evaluation,
                results_by_prompt.get(prompt_id, {}),
                inputs_by_prompt.get(prompt_id),
            )
        )

    rows_by_vertical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failures:
        rows_by_vertical[str(row["vertical"])].append(row)
    per_vertical = {
        vertical: _aggregate_rows(rows_by_vertical.get(vertical, [])) for vertical in VERTICALS
    }
    return {
        "block": block,
        "status": status,
        "source_block": source_block,
        "row_count": len(evaluation_rows),
        "failed_row_count": len(failures),
        "passed_grounded_row_count": sum(bool(row.get("groundedness")) for row in evaluation_rows),
        "overall": _aggregate_rows(failures),
        "per_vertical": per_vertical,
        "finance_assessment": _finance_assessment(
            rows_by_vertical.get("finance", []),
            interpretation=finance_interpretation,
        ),
        "failure_rows": failures,
        "classification_method": (
            "Deterministic joins over unchanged B1 evaluator rows, raw generations, "
            "citation alias maps, rendered E1-E5 context, and frozen runner metadata."
        ),
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "model_inference_triggered": model_inference_triggered,
        "recommended_repair_block": {
            "id": "B3R1_FROZEN_WORKLOAD_CONTEXT_ALIGNMENT_REPAIR",
            "objective": (
                "Repair B1 frozen workload/runner-input alignment before changing the "
                "model, evaluator, gold data, or promoted retrieval source."
            ),
            "actions": [
                (
                    "Trace each B1 prompt from promoted retrieval output to workload "
                    "and runner export."
                ),
                "Re-export the same 100 prompt IDs with unchanged gold and evaluator semantics.",
                "Require every expected evidence ID to map to at least one rendered E1-E5 alias.",
                (
                    "Re-run this offline audit and require zero "
                    "retrieved_gold_absent_from_context rows."
                ),
                "Only then run a maximum five-prompt Finance replay to isolate citation selection.",
                "Address truncation and prohibited-phrase emission as separate one-factor changes.",
            ],
            "scale_gate": (
                "Do not increase prompt count or concurrency until context alignment is "
                "verified and the frozen B1 quality gate passes or model limits are documented."
            ),
        },
    }


def quality_audit_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten overall and vertical failure counts for CSV output."""

    rows: list[dict[str, Any]] = []
    scopes = {"overall": report["overall"], **cast(dict[str, Any], report["per_vertical"])}
    for scope, aggregate in scopes.items():
        counts = cast(dict[str, int], aggregate["failure_class_counts"])
        for failure_class in FAILURE_CLASSES:
            rows.append(
                {
                    "scope": scope,
                    "failure_class": failure_class,
                    "count": counts[failure_class],
                    "failed_row_count": aggregate["failed_row_count"],
                }
            )
    return rows


def write_generation_quality_audit_artifacts(
    *,
    report: dict[str, Any],
    report_path: str | Path,
    summary_path: str | Path,
    finance_examples_path: str | Path,
    failure_examples_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Write the four requested B3 report artifacts."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    summary_output = Path(summary_path)
    summary_rows = quality_audit_summary_rows(report)
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    failure_output = Path(failure_examples_path)
    finance_output = Path(finance_examples_path)
    failures = cast(list[dict[str, Any]], report["failure_rows"])
    with failure_output.open("w", encoding="utf-8", newline="\n") as file:
        for row in failures:
            file.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    with finance_output.open("w", encoding="utf-8", newline="\n") as file:
        for row in failures:
            if row["vertical"] == "finance":
                file.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return report_output, summary_output, finance_output, failure_output
