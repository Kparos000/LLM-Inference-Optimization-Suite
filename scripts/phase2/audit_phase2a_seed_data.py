"""Audit Phase 2A seed data across all committed verticals.

This is a data QA utility only. It does not build RAG, retrieval indexes,
embeddings, prompt assembly, model calls, GPU runs, or benchmark inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-7"

DEFAULT_OUTPUT_REPORT = Path("data/generated/phase2a/phase2a_cross_vertical_qa_report.json")
DEFAULT_OUTPUT_SUMMARY_CSV = Path("data/generated/phase2a/phase2a_cross_vertical_qa_summary.csv")
DEFAULT_OUTPUT_ISSUE_LOG = Path("data/generated/phase2a/phase2a_issue_log.jsonl")

VERTICAL_TARGETS: dict[str, dict[str, Path]] = {
    "finance": {
        "prompts": Path("data/real_world_samples/finance_sample.jsonl"),
        "kb": Path("data/kb/finance/kb_sample.jsonl"),
        "gold": Path("data/eval/gold/finance_gold_sample.jsonl"),
    },
    "airline": {
        "prompts": Path("data/real_world_samples/airline_sample.jsonl"),
        "kb": Path("data/kb/airline/kb_sample.jsonl"),
        "gold": Path("data/eval/gold/airline_gold_sample.jsonl"),
    },
    "healthcare_admin": {
        "prompts": Path("data/real_world_samples/healthcare_admin_sample.jsonl"),
        "kb": Path("data/kb/healthcare_admin/kb_sample.jsonl"),
        "gold": Path("data/eval/gold/healthcare_admin_gold_sample.jsonl"),
    },
    "research_ai": {
        "prompts": Path("data/real_world_samples/research_ai_sample.jsonl"),
        "kb": Path("data/kb/research_ai/kb_sample.jsonl"),
        "gold": Path("data/eval/gold/research_ai_gold_sample.jsonl"),
    },
    "retail": {
        "prompts": Path("data/real_world_samples/retail_sample.jsonl"),
        "kb": Path("data/kb/retail/kb_sample.jsonl"),
        "gold": Path("data/eval/gold/retail_gold_sample.jsonl"),
    },
}

NEGATIVE_STATUSES = {
    "insufficient_evidence",
    "escalate",
    "out_of_scope",
    "spam_or_low_quality",
    "spam_or_fraud",
    "safety_boundary",
    "boundary_response",
}
ANSWERABLE_STATUS = "answer"
BAD_RESEARCH_AI_PHRASES = [
    "calledAgent",
    "proposeDELTA",
    "The paper frames the problem as follows",
    "Its contribution is grounded in",
    "traceable paper evidence rather than a generic summary",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object in {path} line {line_number}.")
        rows.append(parsed)
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertical",
        "prompt_count",
        "gold_count",
        "kb_count",
        "answer_count",
        "negative_status_count",
        "json_output_count",
        "markdown_table_count",
        "critical_issues",
        "warnings",
        "ready_for_250_scale",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def _next_issue_id(issues: list[dict[str, Any]]) -> str:
    return f"phase2a_qa_{len(issues) + 1:04d}"


def add_issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    vertical: str,
    file: str,
    check_name: str,
    message: str,
    recommendation: str,
    prompt_id: str | None = None,
    doc_id: str | None = None,
) -> dict[str, Any]:
    issue = {
        "issue_id": _next_issue_id(issues),
        "severity": severity,
        "vertical": vertical,
        "file": file,
        "prompt_id": prompt_id,
        "doc_id": doc_id,
        "check_name": check_name,
        "message": message,
        "recommendation": recommendation,
    }
    issues.append(issue)
    return issue


def prompt_id_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("prompt_id") or "") for row in rows)


def validate_prompt_gold_alignment(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    vertical: str,
    prompt_file: Path | str = "",
    gold_file: Path | str = "",
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    prompt_counts = prompt_id_counts(prompts)
    gold_counts = prompt_id_counts(gold)
    for prompt_id, count in prompt_counts.items():
        if not prompt_id:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(prompt_file),
                prompt_id=None,
                check_name="prompt_id_present",
                message="Prompt record is missing prompt_id.",
                recommendation="Regenerate the seed so every prompt has a stable prompt_id.",
            )
        elif count > 1:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(prompt_file),
                prompt_id=prompt_id,
                check_name="prompt_id_unique",
                message=f"Prompt id {prompt_id} appears {count} times.",
                recommendation="Regenerate or deduplicate prompt records.",
            )
    for prompt_id, count in gold_counts.items():
        if not prompt_id:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(gold_file),
                check_name="gold_prompt_id_present",
                message="Gold record is missing prompt_id.",
                recommendation="Regenerate the gold records with matching prompt_id values.",
            )
        elif count > 1:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(gold_file),
                prompt_id=prompt_id,
                check_name="one_gold_per_prompt",
                message=f"Gold prompt id {prompt_id} appears {count} times.",
                recommendation="Keep exactly one gold record per prompt.",
            )
    for prompt_id in sorted(set(prompt_counts) - set(gold_counts)):
        if prompt_id:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(gold_file),
                prompt_id=prompt_id,
                check_name="missing_gold_record",
                message=f"Prompt {prompt_id} has no matching gold record.",
                recommendation="Add or regenerate the matching gold record.",
            )
    for prompt_id in sorted(set(gold_counts) - set(prompt_counts)):
        if prompt_id:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(gold_file),
                prompt_id=prompt_id,
                check_name="orphan_gold_record",
                message=f"Gold record {prompt_id} has no matching prompt.",
                recommendation="Remove orphan gold records or add the missing prompt.",
            )
    return issues


def has_evidence_ids(record: dict[str, Any]) -> bool:
    for field in (
        "required_doc_ids",
        "required_evidence_ids",
        "required_policy_ids",
        "source_doc_ids",
    ):
        value = record.get(field)
        if isinstance(value, list) and value:
            return True
    return False


def record_evidence_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in (
        "required_doc_ids",
        "required_evidence_ids",
        "required_policy_ids",
        "source_doc_ids",
    ):
        value = record.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return ids


def prompt_task_type(record: dict[str, Any]) -> str:
    return str(
        record.get("task_type")
        or record.get("support_type")
        or record.get("metadata", {}).get("prompt_category")
        or "missing"
    )


def scan_hygiene_text(text: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    critical_patterns = [
        (r"C:\\Users", "private Windows path"),
        (r"/home/", "private Unix path"),
        (r"akpoogaga", "private username"),
        (r"\bkparo\b", "private username"),
        (r"API key", "API key reference"),
        (r"raw user_id", "raw user identifier reference"),
        (r"real customer identifier", "real customer identifier claim"),
        (r"proves vLLM is faster", "unsupported benchmark claim"),
        (r"(?i)(bearer|access|api)\s+token", "credential token reference"),
        (r"(?i)(secret|password)\s*[:=]", "credential assignment"),
    ]
    for pattern, label in critical_patterns:
        if re.search(pattern, text):
            findings.append(("critical", label))
    return findings


def scan_record_hygiene(
    records: list[dict[str, Any]],
    *,
    vertical: str,
    file: Path,
    id_field: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in records:
        row_text = flatten_text(row)
        for severity, label in scan_hygiene_text(row_text):
            add_issue(
                issues,
                severity=severity,
                vertical=vertical,
                file=str(file),
                prompt_id=str(row.get("prompt_id")) if row.get("prompt_id") else None,
                doc_id=str(row.get(id_field)) if row.get(id_field) else None,
                check_name="public_hygiene",
                message=f"Record contains {label}.",
                recommendation=(
                    "Remove private paths, identifiers, credentials, or unsupported claims."
                ),
            )
    return issues


def check_retail_specific(
    kb: list[dict[str, Any]], prompts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for record in kb:
        if record.get("document_type") != "support_policy":
            continue
        body = flatten_text(record)
        if "synthetic benchmark policy" not in body or "not Amazon policy" not in body:
            add_issue(
                issues,
                severity="critical",
                vertical="retail",
                file=str(VERTICAL_TARGETS["retail"]["kb"]),
                doc_id=str(record.get("doc_id")),
                check_name="retail_synthetic_policy_label",
                message=(
                    "Retail support policy is not clearly marked as synthetic benchmark policy "
                    "and not Amazon policy."
                ),
                recommendation=(
                    "Mark every Retail support policy as synthetic benchmark policy and not "
                    "Amazon policy."
                ),
            )
    generic_titles = [
        str(prompt.get("product_title"))
        for prompt in prompts
        if re.fullmatch(r"All_Beauty product B[0-9A-Z]{7,12}", str(prompt.get("product_title")))
    ]
    if generic_titles:
        add_issue(
            issues,
            severity="warning",
            vertical="retail",
            file=str(VERTICAL_TARGETS["retail"]["prompts"]),
            check_name="retail_generic_product_titles",
            message=f"{len(generic_titles)} Retail prompts still use generic product titles.",
            recommendation=(
                "Run targeted metadata retrieval for selected parent_asins before scaling."
            ),
        )
    return issues


def check_research_ai_specific(
    kb: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    document_types = Counter(str(row.get("document_type")) for row in kb)
    for required_type in ("paper_abstract", "paper_section"):
        if document_types.get(required_type, 0) == 0:
            add_issue(
                issues,
                severity="critical",
                vertical="research_ai",
                file=str(VERTICAL_TARGETS["research_ai"]["kb"]),
                check_name="research_ai_kb_document_type",
                message=f"Research AI KB is missing {required_type} records.",
                recommendation="Regenerate Research AI KB with abstract and section evidence.",
            )
    for record in gold:
        answer = str(record.get("reference_answer") or "")
        for phrase in BAD_RESEARCH_AI_PHRASES:
            if phrase in answer:
                add_issue(
                    issues,
                    severity="warning",
                    vertical="research_ai",
                    file=str(VERTICAL_TARGETS["research_ai"]["gold"]),
                    prompt_id=str(record.get("prompt_id")),
                    check_name="research_ai_reference_answer_style",
                    message=f"Reference answer contains known awkward phrase: {phrase}",
                    recommendation=(
                        "Regenerate Research AI gold answers with polished, evidence-specific "
                        "wording."
                    ),
                )
    return issues


def check_finance_specific(
    kb: list[dict[str, Any]], prompts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    text = flatten_text(kb)
    if "sec" not in text.lower() and "xbrl" not in text.lower():
        add_issue(
            issues,
            severity="warning",
            vertical="finance",
            file=str(VERTICAL_TARGETS["finance"]["kb"]),
            check_name="finance_provenance",
            message="Finance KB does not visibly include SEC/XBRL provenance terms.",
            recommendation=(
                "Confirm finance records preserve SEC/XBRL filing provenance before scale-up."
            ),
        )
    if len(prompts) <= 40:
        add_issue(
            issues,
            severity="info",
            vertical="finance",
            file=str(VERTICAL_TARGETS["finance"]["prompts"]),
            check_name="finance_seed_level",
            message="Finance prompt count is still at seed level.",
            recommendation="Use this only as a seed before progressive scale-up.",
        )
    add_issue(
        issues,
        severity="info",
        vertical="finance",
        file=str(VERTICAL_TARGETS["finance"]["kb"]),
        check_name="finance_raw_generated_files_local",
        message="Raw SEC generated files are not committed; this is expected for Phase 2A.",
        recommendation="Keep raw filings local/ignored unless intentionally curated later.",
    )
    return issues


def check_synthetic_vertical(
    vertical: str,
    kb: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for record in kb:
        document_type = str(record.get("document_type") or "")
        if "policy" not in document_type:
            continue
        source_type = str(record.get("source_type") or "")
        metadata_text = flatten_text(record.get("metadata", {})).lower()
        if "synthetic" not in source_type and "generator" not in metadata_text:
            add_issue(
                issues,
                severity="warning",
                vertical=vertical,
                file=str(VERTICAL_TARGETS[vertical]["kb"]),
                doc_id=str(record.get("doc_id")),
                check_name="synthetic_policy_label",
                message="Synthetic policy record is not clearly source-labeled.",
                recommendation=(
                    "Mark synthetic policies with synthetic_public_inspired or generator metadata."
                ),
            )
    return issues


def calculate_scale_up_readiness(metrics: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if metrics.get("critical_issues", 0) > 0:
        blockers.append("critical_issues")
    if metrics.get("prompt_count", 0) < 40:
        blockers.append("prompt_seed_count_below_40")
    if metrics.get("prompt_count") != metrics.get("gold_count"):
        blockers.append("prompt_gold_count_mismatch")
    if metrics.get("kb_count", 0) < 25:
        blockers.append("kb_count_below_25")
    if metrics.get("negative_status_count", 0) < 1:
        blockers.append("missing_negative_status")
    if metrics.get("answerable_without_evidence", 0) > 0:
        blockers.append("answerable_records_without_evidence")
    return {
        "has_seed_prompts": metrics.get("prompt_count", 0) >= 40,
        "has_seed_kb": metrics.get("kb_count", 0) >= 25,
        "has_seed_gold": metrics.get("gold_count", 0) >= 40,
        "has_eda_or_source_report": bool(metrics.get("has_eda_or_source_report")),
        "ready_for_250_scale": not blockers,
        "blockers": blockers,
    }


def _has_source_report(vertical: str) -> bool:
    report_paths = {
        "finance": [Path("docs/32_phase2_finance_sec_xbrl_pilot.md")],
        "airline": [Path("data/generated/airline/airline_synthetic_report.json")],
        "healthcare_admin": [
            Path("data/generated/healthcare_admin/healthcare_admin_synthetic_report.json")
        ],
        "research_ai": [Path("data/generated/research_ai/research_ai_curation_report.json")],
        "retail": [Path("data/generated/retail/retail_curation_report.json")],
    }
    return any(path.exists() for path in report_paths.get(vertical, []))


def audit_vertical(
    vertical: str, targets: dict[str, Path]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    missing_files = [name for name, path in targets.items() if not path.exists()]
    for name in missing_files:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets[name]),
            check_name="file_exists",
            message=f"Missing {name} seed file.",
            recommendation="Regenerate or restore the missing Phase 2A seed artifact.",
        )
    if missing_files:
        metrics = {
            "vertical": vertical,
            "prompt_count": 0,
            "gold_count": 0,
            "kb_count": 0,
            "critical_issues": len([row for row in issues if row["severity"] == "critical"]),
            "warnings": len([row for row in issues if row["severity"] == "warning"]),
            "negative_status_count": 0,
            "answerable_without_evidence": 0,
            "has_eda_or_source_report": _has_source_report(vertical),
        }
        metrics["scale_up_readiness"] = calculate_scale_up_readiness(metrics)
        return metrics, issues

    prompts = read_jsonl(targets["prompts"])
    kb = read_jsonl(targets["kb"])
    gold = read_jsonl(targets["gold"])
    kb_doc_ids = {str(row.get("doc_id")) for row in kb if row.get("doc_id")}
    issues.extend(
        validate_prompt_gold_alignment(
            prompts,
            gold,
            vertical=vertical,
            prompt_file=targets["prompts"],
            gold_file=targets["gold"],
        )
    )
    for kind, rows, id_field in (
        ("prompt", prompts, "prompt_id"),
        ("kb", kb, "doc_id"),
        ("gold", gold, "prompt_id"),
    ):
        issues.extend(
            scan_record_hygiene(
                rows,
                vertical=vertical,
                file=targets[kind if kind != "prompt" else "prompts"],
                id_field=id_field,
            )
        )

    if len(prompts) != len(gold):
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["gold"]),
            check_name="prompt_gold_count_match",
            message=f"Prompt/gold count mismatch: {len(prompts)} prompts vs {len(gold)} gold.",
            recommendation="Regenerate aligned prompt and gold records.",
        )
    if len(prompts) < 40:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["prompts"]),
            check_name="minimum_prompt_count",
            message=f"Only {len(prompts)} prompt records found.",
            recommendation="Regenerate at least 40 seed prompt records.",
        )
    if len(kb) < 25:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["kb"]),
            check_name="minimum_kb_count",
            message=f"Only {len(kb)} KB records found.",
            recommendation="Add enough KB/context records to reach the seed minimum.",
        )

    answerable_without_evidence = 0
    missing_doc_refs = 0
    for record in gold:
        prompt_id = str(record.get("prompt_id") or "")
        status = str(record.get("expected_status") or "")
        if status == ANSWERABLE_STATUS:
            if not has_evidence_ids(record):
                answerable_without_evidence += 1
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file=str(targets["gold"]),
                    prompt_id=prompt_id,
                    check_name="answerable_evidence_required",
                    message="Answerable gold record has no required evidence IDs.",
                    recommendation="Add required_doc_ids or required_evidence_ids.",
                )
            if not record.get("must_include"):
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file=str(targets["gold"]),
                    prompt_id=prompt_id,
                    check_name="answerable_must_include",
                    message="Answerable gold record has empty must_include.",
                    recommendation="Add evidence-specific must_include terms.",
                )
        for doc_id in record.get("required_doc_ids", []):
            if doc_id not in kb_doc_ids:
                missing_doc_refs += 1
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file=str(targets["gold"]),
                    prompt_id=prompt_id,
                    check_name="required_doc_id_exists",
                    message=f"required_doc_id {doc_id} is not present in the KB.",
                    recommendation="Fix the gold evidence reference or add the KB record.",
                )

    status_counts = Counter(str(row.get("expected_status") or "missing") for row in gold)
    negative_status_count = sum(status_counts.get(status, 0) for status in NEGATIVE_STATUSES)
    if negative_status_count == 0:
        add_issue(
            issues,
            severity="warning",
            vertical=vertical,
            file=str(targets["gold"]),
            check_name="negative_status_coverage",
            message="Vertical has no negative/status-boundary examples.",
            recommendation=(
                "Add insufficient_evidence, escalate, out_of_scope, or comparable records."
            ),
        )

    if vertical == "retail":
        issues.extend(check_retail_specific(kb, prompts))
    elif vertical == "research_ai":
        issues.extend(check_research_ai_specific(kb, gold))
    elif vertical == "finance":
        issues.extend(check_finance_specific(kb, prompts))
    elif vertical in {"airline", "healthcare_admin"}:
        issues.extend(check_synthetic_vertical(vertical, kb))

    issue_counts = Counter(row["severity"] for row in issues)
    output_format_counts = Counter(
        str(row.get("expected_output_format") or "other") for row in prompts
    )
    task_type_counts = Counter(prompt_task_type(row) for row in prompts)
    kb_document_type_counts = Counter(str(row.get("document_type") or "missing") for row in kb)
    metrics = {
        "vertical": vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb),
        "status_counts": dict(status_counts),
        "output_format_counts": dict(output_format_counts),
        "task_type_counts": dict(task_type_counts),
        "kb_document_type_counts": dict(kb_document_type_counts),
        "answer_count": status_counts.get(ANSWERABLE_STATUS, 0),
        "negative_status_count": negative_status_count,
        "json_output_count": output_format_counts.get("json", 0),
        "markdown_table_count": output_format_counts.get("markdown_table", 0),
        "critical_issues": issue_counts.get("critical", 0),
        "warnings": issue_counts.get("warning", 0),
        "info_issues": issue_counts.get("info", 0),
        "answerable_without_evidence": answerable_without_evidence,
        "missing_required_doc_refs": missing_doc_refs,
        "has_eda_or_source_report": _has_source_report(vertical),
    }
    metrics["scale_up_readiness"] = calculate_scale_up_readiness(metrics)
    return metrics, issues


def build_recommendations(issue_log: list[dict[str, Any]]) -> list[str]:
    severities = Counter(row["severity"] for row in issue_log)
    recommendations = [
        "Review warning and info issues before progressive scale-up.",
        "Keep generated/raw source data local unless explicitly curated for commit.",
        (
            "Do not start RAG, retrieval, embeddings, prompt assembly, or inference from this "
            "audit step."
        ),
    ]
    if severities.get("critical", 0):
        recommendations.insert(0, "Fix all critical issues before Phase 2A scale-up.")
    else:
        recommendations.insert(0, "No critical cross-vertical seed blockers were detected.")
    return recommendations


def renumber_issue_ids(issue_log: list[dict[str, Any]]) -> None:
    for index, issue in enumerate(issue_log, start=1):
        issue["issue_id"] = f"phase2a_qa_{index:04d}"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    all_issues: list[dict[str, Any]] = []
    per_vertical: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    output_format_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    kb_document_type_counts: Counter[str] = Counter()

    for vertical, targets in VERTICAL_TARGETS.items():
        metrics, issues = audit_vertical(vertical, targets)
        all_issues.extend(issues)
        per_vertical[vertical] = metrics
        status_counts.update(metrics.get("status_counts", {}))
        output_format_counts.update(metrics.get("output_format_counts", {}))
        task_type_counts.update(metrics.get("task_type_counts", {}))
        kb_document_type_counts.update(metrics.get("kb_document_type_counts", {}))
        readiness = metrics["scale_up_readiness"]
        summary_rows.append(
            {
                "vertical": vertical,
                "prompt_count": metrics.get("prompt_count", 0),
                "gold_count": metrics.get("gold_count", 0),
                "kb_count": metrics.get("kb_count", 0),
                "answer_count": metrics.get("answer_count", 0),
                "negative_status_count": metrics.get("negative_status_count", 0),
                "json_output_count": metrics.get("json_output_count", 0),
                "markdown_table_count": metrics.get("markdown_table_count", 0),
                "critical_issues": metrics.get("critical_issues", 0),
                "warnings": metrics.get("warnings", 0),
                "ready_for_250_scale": readiness["ready_for_250_scale"],
            }
        )

    renumber_issue_ids(all_issues)
    issue_counts_by_severity = Counter(row["severity"] for row in all_issues)
    scale_up_readiness = {
        vertical: metrics["scale_up_readiness"] for vertical, metrics in per_vertical.items()
    }
    report = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "verticals_audited": list(VERTICAL_TARGETS),
        "total_prompt_count": sum(row.get("prompt_count", 0) for row in per_vertical.values()),
        "total_gold_count": sum(row.get("gold_count", 0) for row in per_vertical.values()),
        "total_kb_count": sum(row.get("kb_count", 0) for row in per_vertical.values()),
        "status_counts": dict(status_counts),
        "output_format_counts": dict(output_format_counts),
        "task_type_counts": dict(task_type_counts),
        "kb_document_type_counts": dict(kb_document_type_counts),
        "per_vertical": per_vertical,
        "critical_issue_count": issue_counts_by_severity.get("critical", 0),
        "warning_count": issue_counts_by_severity.get("warning", 0),
        "issue_counts_by_severity": dict(issue_counts_by_severity),
        "scale_up_readiness": scale_up_readiness,
        "recommendations": build_recommendations(all_issues),
        "next_step": (
            "Proceed to Phase 2A-8 progressive scale-up planning after reviewing this audit."
        ),
    }
    write_json(Path(args.output_report), report)
    write_summary_csv(Path(args.output_summary_csv), summary_rows)
    write_jsonl(Path(args.output_issue_log), all_issues)
    summary = {
        "mode": "run_audit",
        "phase": PHASE,
        "verticals_audited": len(VERTICAL_TARGETS),
        "total_prompt_count": report["total_prompt_count"],
        "total_gold_count": report["total_gold_count"],
        "total_kb_count": report["total_kb_count"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "output_report": str(args.output_report),
        "output_summary_csv": str(args.output_summary_csv),
        "output_issue_log": str(args.output_issue_log),
        "next_step": report["next_step"],
    }
    if args.fail_on_critical and report["critical_issue_count"] > 0:
        raise RuntimeError(
            f"Phase 2A audit found {report['critical_issue_count']} critical issue(s)."
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Phase 2A cross-vertical seed data.")
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-summary-csv", default=str(DEFAULT_OUTPUT_SUMMARY_CSV))
    parser.add_argument("--output-issue-log", default=str(DEFAULT_OUTPUT_ISSUE_LOG))
    parser.add_argument("--fail-on-critical", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_audit:
        parser.error("Use --run-audit to run the Phase 2A cross-vertical QA audit.")
    try:
        summary = run_audit(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
