"""Deterministic failure diagnostics for generation-contract grounding."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

FAILURE_CLASSES = (
    "missing_required_evidence_id",
    "cited_wrong_evidence_id",
    "cited_partial_evidence_only",
    "answer_not_supported_by_cited_evidence",
    "insufficient_evidence_misuse",
    "malformed_contract",
    "semantic_under_answer",
    "multi_evidence_under_citation",
)


def classify_grounding_failure(
    evaluation_row: dict[str, Any],
    result_row: dict[str, Any],
) -> list[str]:
    """Classify one non-grounded output without semantic judge calls."""

    classes: list[str] = []
    if not evaluation_row.get("generation_contract_valid"):
        classes.append("malformed_contract")
    expected_status = str(evaluation_row.get("expected_status") or "")
    observed_status = str(evaluation_row.get("observed_status") or "")
    if (
        observed_status == "insufficient_evidence" and expected_status != "insufficient_evidence"
    ) or (
        expected_status == "insufficient_evidence" and observed_status != "insufficient_evidence"
    ):
        classes.append("insufficient_evidence_misuse")

    expected = set(evaluation_row.get("evidence_ids_expected") or [])
    found = set(evaluation_row.get("evidence_ids_found") or [])
    cited = set(result_row.get("evidence_ids") or result_row.get("citations") or [])
    missing = expected.difference(found)
    if missing:
        classes.append("missing_required_evidence_id")
        if found:
            classes.append("cited_partial_evidence_only")
        elif cited:
            classes.append("cited_wrong_evidence_id")
    if len(expected) > 1 and len(found) < len(expected):
        classes.append("multi_evidence_under_citation")

    missing_semantic_terms = [
        str(term)
        for term in evaluation_row.get("must_include_missing") or []
        if str(term) not in expected and "_" not in str(term)
    ]
    if missing_semantic_terms:
        classes.append("semantic_under_answer")
        if evaluation_row.get("evidence_match") and cited:
            classes.append("answer_not_supported_by_cited_evidence")
    return [failure_class for failure_class in FAILURE_CLASSES if failure_class in classes]


def build_grounding_failure_report(
    *,
    evaluation_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build record-level and aggregate deterministic diagnostics."""

    results_by_prompt = {str(row.get("prompt_id") or ""): row for row in result_rows}
    failures: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for evaluation in evaluation_rows:
        if evaluation.get("groundedness"):
            continue
        prompt_id = str(evaluation.get("prompt_id") or "")
        result = results_by_prompt.get(prompt_id, {})
        classes = classify_grounding_failure(evaluation, result)
        class_counts.update(classes)
        expected = set(evaluation.get("evidence_ids_expected") or [])
        found = set(evaluation.get("evidence_ids_found") or [])
        failures.append(
            {
                "prompt_id": prompt_id,
                "vertical": result.get("vertical"),
                "failure_classes": classes,
                "expected_evidence_ids": sorted(expected),
                "found_evidence_ids": sorted(found),
                "missing_evidence_ids": sorted(expected.difference(found)),
                "emitted_evidence_labels": result.get("evidence_ids")
                or result.get("citations")
                or [],
                "generation_contract_valid": evaluation.get("generation_contract_valid"),
                "evidence_match": evaluation.get("evidence_match"),
                "groundedness": evaluation.get("groundedness"),
                "must_include_missing": evaluation.get("must_include_missing") or [],
                "semantic_terms_missing": [
                    str(term)
                    for term in evaluation.get("must_include_missing") or []
                    if str(term) not in expected and "_" not in str(term)
                ],
            }
        )
    return {
        "row_count": len(evaluation_rows),
        "grounded_count": sum(bool(row.get("groundedness")) for row in evaluation_rows),
        "failure_count": len(failures),
        "failure_class_counts": dict(sorted(class_counts.items())),
        "failure_rows": failures,
        "diagnostic_method": (
            "Deterministic evaluator signals only; no LLM judge or semantic entailment "
            "model was used."
        ),
    }


def write_grounding_failure_artifacts(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """Write grounding diagnostic JSON and CSV outputs."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"failure_class": failure_class, "count": count}
        for failure_class, count in report["failure_class_counts"].items()
    ]
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["failure_class", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return report_output, summary_output
