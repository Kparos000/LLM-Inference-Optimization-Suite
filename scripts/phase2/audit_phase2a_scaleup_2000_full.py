"""Audit the full Phase 2A 2,000-scale candidate set across all verticals.

This is data QA only. It does not build RAG, retrieval indexes, embeddings,
prompt assembly, model calls, GPU runs, or benchmark inference.
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

PHASE = "2A-15"
DATASET_NAME = "phase2a_2000_full_candidate"
TARGET_PER_VERTICAL = 2000
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
FILE_KINDS = ["prompts", "gold", "kb"]
LINGUISTIC_VARIATION_THRESHOLD = 0.60
MAX_TEMPLATE_SHARE = 0.40

DEFAULT_PARTIAL_ROOT = Path("data/generated/phase2a/scaleup")
DEFAULT_GENERATED_ROOT = Path("data/generated/phase2a/scaleup")
DEFAULT_OUTPUT_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_2000_full_qa_report.json"
)
DEFAULT_OUTPUT_SUMMARY_CSV = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_2000_full_qa_summary.csv"
)
DEFAULT_OUTPUT_ISSUE_LOG = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_2000_full_issue_log.jsonl"
)

EXPECTED_STATUS_COUNTS: dict[str, dict[str, int]] = {
    "airline": {"answer": 1800, "escalate": 160, "spam_or_fraud": 40},
    "healthcare_admin": {
        "answer": 1760,
        "escalate": 160,
        "safety_boundary": 40,
        "spam_or_fraud": 20,
        "out_of_scope": 20,
    },
    "retail": {
        "answer": 1780,
        "insufficient_evidence": 70,
        "escalate": 70,
        "spam_or_low_quality": 60,
        "out_of_scope": 20,
    },
    "finance": {"answer": 1840, "insufficient_evidence": 80, "escalate": 80},
    "research_ai": {
        "answer": 1800,
        "insufficient_evidence": 80,
        "escalate": 80,
        "out_of_scope": 40,
    },
}

EXPECTED_OUTPUT_FORMAT_COUNTS: dict[str, dict[str, int]] = {
    "airline": {"text": 1520, "json": 280, "markdown_table": 200},
    "healthcare_admin": {"text": 1560, "json": 280, "markdown_table": 160},
    "retail": {"text": 1480, "json": 320, "markdown_table": 200},
    "finance": {"text": 1240, "json": 400, "markdown_table": 360},
    "research_ai": {"text": 1440, "json": 280, "markdown_table": 280},
}

KB_TARGET_RANGES: dict[str, tuple[int, int]] = {
    "airline": (300, 500),
    "healthcare_admin": (300, 500),
    "retail": (1000, 2000),
    "finance": (1500, 2500),
    "research_ai": (1600, 2000),
}

HYGIENE_PATTERNS = [
    (re.compile(pattern, flags=re.IGNORECASE), label)
    for pattern, label in [
        (r"C:\\Users", "private Windows path"),
        (r"/home/", "private Unix path"),
        (r"akpoogaga", "private username"),
        (r"kparo", "private username"),
        (r"API key", "API key reference"),
        (r"\btoken\b", "token reference"),
        (r"\bsecret\b", "secret reference"),
        (r"\bpassword\b", "password reference"),
        (r"raw user_id", "raw user identifier"),
    ]
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object in {path} line {line_number}.")
        rows.append(parsed)
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
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
        "linguistic_variation_rate",
        "critical_issues",
        "warnings",
        "promotion_ready",
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


def source_root_for_vertical(vertical: str, partial_root: Path, generated_root: Path) -> Path:
    _ = (vertical, partial_root)
    return generated_root


def candidate_file(root: Path, vertical: str, kind: str) -> Path:
    return root / vertical / f"{vertical}_{kind}_{TARGET_PER_VERTICAL}.jsonl"


def generation_command(vertical: str) -> str:
    return (
        "python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical "
        f"--vertical {vertical} --target-per-vertical {TARGET_PER_VERTICAL}"
    )


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
) -> None:
    issues.append(
        {
            "issue_id": f"phase2a_2000_full_qa_{len(issues) + 1:04d}",
            "severity": severity,
            "vertical": vertical,
            "file": file,
            "prompt_id": prompt_id,
            "doc_id": doc_id,
            "check_name": check_name,
            "message": message,
            "recommendation": recommendation,
        }
    )


def critical_count(issues: list[dict[str, Any]]) -> int:
    return sum(1 for issue in issues if issue["severity"] == "critical")


def warning_count(issues: list[dict[str, Any]]) -> int:
    return sum(1 for issue in issues if issue["severity"] == "warning")


def normalize_question_template(question: str) -> str:
    normalized = question.lower()
    normalized = re.sub(r"\b(?:[a-z]+_)?scaleup[_-]?\d+[_-]?\d+\b", "<id>", normalized)
    normalized = re.sub(r"\b[A-Z0-9]{8,}\b", "<id>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", normalized)
    normalized = re.sub(r"\([^)]*\)", "(<entity>)", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def calculate_question_template_diversity(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    questions = [str(prompt.get("question") or prompt.get("issue") or "") for prompt in prompts]
    templates = [normalize_question_template(question) for question in questions if question]
    if not templates:
        return {
            "linguistic_variation_rate": 0.0,
            "most_common_question_template_count": 0,
            "most_common_question_template_share": 0.0,
            "unique_question_template_count": 0,
        }
    counts = Counter(templates)
    most_common = counts.most_common(1)[0][1]
    share = most_common / len(templates)
    return {
        "linguistic_variation_rate": round(1 - share, 3),
        "most_common_question_template_count": most_common,
        "most_common_question_template_share": round(share, 3),
        "unique_question_template_count": len(counts),
    }


def referenced_evidence_ids(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ["required_doc_ids", "required_evidence_ids", "required_policy_ids"]:
        raw = row.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
    return list(dict.fromkeys(values))


def load_vertical_records(
    *,
    vertical: str,
    partial_root: Path,
    generated_root: Path,
    issues: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    root = source_root_for_vertical(vertical, partial_root, generated_root)
    records: dict[str, list[dict[str, Any]]] = {}
    files: dict[str, str] = {}
    for kind in FILE_KINDS:
        path = candidate_file(root, vertical, kind)
        files[kind] = str(path)
        if not path.exists():
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(path),
                check_name="file_existence",
                message=f"Missing candidate file: {path}",
                recommendation=f"Run: {generation_command(vertical)}",
            )
            records[kind] = []
            continue
        records[kind] = read_jsonl(path)
    return records, files


def validate_counts(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> None:
    if len(prompts) != TARGET_PER_VERTICAL:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="prompts",
            check_name="prompt_count",
            message=f"Expected {TARGET_PER_VERTICAL} prompts, found {len(prompts)}.",
            recommendation=f"Regenerate with: {generation_command(vertical)}",
        )
    if len(gold) != TARGET_PER_VERTICAL:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="gold",
            check_name="gold_count",
            message=f"Expected {TARGET_PER_VERTICAL} gold records, found {len(gold)}.",
            recommendation=f"Regenerate with: {generation_command(vertical)}",
        )
    kb_min, kb_max = KB_TARGET_RANGES[vertical]
    if not kb_min <= len(kb_rows) <= kb_max:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="kb",
            check_name="kb_target_range",
            message=f"Expected KB count {kb_min}-{kb_max}, found {len(kb_rows)}.",
            recommendation="Regenerate or expand the vertical KB.",
        )


def validate_alignment(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> None:
    prompt_ids = [str(row.get("prompt_id") or "") for row in prompts]
    gold_ids = [str(row.get("prompt_id") or "") for row in gold]
    for prompt_id, count in Counter(prompt_ids).items():
        if not prompt_id or count != 1:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="prompts",
                prompt_id=prompt_id or None,
                check_name="prompt_id_uniqueness",
                message=f"Prompt ID {prompt_id!r} appears {count} time(s).",
                recommendation="Ensure every prompt ID is present exactly once.",
            )
    for prompt_id, count in Counter(gold_ids).items():
        if not prompt_id or count != 1:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="gold",
                prompt_id=prompt_id or None,
                check_name="gold_id_uniqueness",
                message=f"Gold ID {prompt_id!r} appears {count} time(s).",
                recommendation="Ensure every gold record maps to one prompt.",
            )
    for prompt_id in sorted(set(prompt_ids) - set(gold_ids))[:20]:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="gold",
            prompt_id=prompt_id,
            check_name="prompt_gold_alignment",
            message="Prompt has no matching gold record.",
            recommendation="Regenerate prompt/gold files together.",
        )
    for prompt_id in sorted(set(gold_ids) - set(prompt_ids))[:20]:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="gold",
            prompt_id=prompt_id,
            check_name="orphan_gold",
            message="Gold record has no matching prompt.",
            recommendation="Regenerate prompt/gold files together.",
        )


def validate_evidence(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> None:
    kb_doc_ids = {str(row.get("doc_id") or row.get("policy_id") or "") for row in kb_rows}
    for row in gold:
        prompt_id = str(row.get("prompt_id") or "")
        status = str(row.get("expected_status") or "")
        evidence_ids = referenced_evidence_ids(row)
        if status == "answer":
            if not evidence_ids:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file="gold",
                    prompt_id=prompt_id,
                    check_name="answerable_evidence",
                    message="Answerable gold record has no evidence IDs.",
                    recommendation="Add required_doc_ids or required_evidence_ids.",
                )
            must_include = row.get("must_include")
            if not isinstance(must_include, list) or not must_include:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file="gold",
                    prompt_id=prompt_id,
                    check_name="must_include",
                    message="Answerable gold record has empty must_include.",
                    recommendation="Add concrete required answer elements.",
                )
        else:
            must_not_include = row.get("must_not_include")
            if not isinstance(must_not_include, list) or len(must_not_include) < 2:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file="gold",
                    prompt_id=prompt_id,
                    check_name="negative_must_not_include",
                    message="Negative/boundary gold lacks meaningful must_not_include.",
                    recommendation="Add clear unsupported content guardrails.",
                )
        for evidence_id in evidence_ids:
            if evidence_id not in kb_doc_ids:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file="gold",
                    prompt_id=prompt_id,
                    doc_id=evidence_id,
                    check_name="referenced_kb_exists",
                    message=f"Referenced evidence ID {evidence_id} is missing from KB.",
                    recommendation="Regenerate KB and gold together.",
                )


def validate_distributions(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    prompts: list[dict[str, Any]],
) -> None:
    status_counts = dict(Counter(str(row.get("expected_status") or "") for row in prompts))
    output_counts = dict(Counter(str(row.get("expected_output_format") or "") for row in prompts))
    if status_counts != EXPECTED_STATUS_COUNTS[vertical]:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="prompts",
            check_name="status_distribution",
            message=f"Status distribution mismatch: {status_counts}",
            recommendation="Regenerate with the approved 2,000-scale distribution.",
        )
    if output_counts != EXPECTED_OUTPUT_FORMAT_COUNTS[vertical]:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="prompts",
            check_name="output_format_distribution",
            message=f"Output format distribution mismatch: {output_counts}",
            recommendation="Regenerate with the approved 2,000-scale distribution.",
        )


def validate_linguistic_variation(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    metrics: dict[str, Any],
) -> None:
    variation_rate = float(metrics.get("linguistic_variation_rate") or 0)
    template_share = float(metrics.get("most_common_question_template_share") or 1)
    if variation_rate < LINGUISTIC_VARIATION_THRESHOLD or template_share > MAX_TEMPLATE_SHARE:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file="prompts",
            check_name="linguistic_variation",
            message=(
                f"Linguistic variation failed: rate={variation_rate}, "
                f"most_common_share={template_share}."
            ),
            recommendation="Increase deterministic prompt phrasing variation and regenerate.",
        )


def validate_hygiene(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    rows_by_kind: dict[str, list[dict[str, Any]]],
) -> None:
    for kind, rows in rows_by_kind.items():
        for row in rows:
            text = flatten_text(row)
            record_id = str(row.get("prompt_id") or row.get("doc_id") or "")
            for pattern, label in HYGIENE_PATTERNS:
                if pattern.search(text):
                    add_issue(
                        issues,
                        severity="critical",
                        vertical=vertical,
                        file=kind,
                        prompt_id=record_id if kind != "kb" else None,
                        doc_id=record_id if kind == "kb" else None,
                        check_name="hygiene_scan",
                        message=f"Found {label}.",
                        recommendation="Remove private paths, secrets, or raw identifiers.",
                    )


def validate_domain_safety(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> None:
    prompt_answer_text = flatten_text([prompts, [row.get("reference_answer") for row in gold]])
    prompt_answer_text_lower = prompt_answer_text.lower()
    all_text_lower = flatten_text([prompts, gold, kb_rows]).lower()
    if vertical == "healthcare_admin":
        unsafe = [r"\bdiagnose (?:the|this|a) patient\b", r"\btreatment plan is\b"]
        if any(re.search(pattern, prompt_answer_text_lower) for pattern in unsafe):
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="gold",
                check_name="healthcare_clinical_advice",
                message="Healthcare Admin records appear to provide clinical advice.",
                recommendation="Keep Healthcare Admin records administrative-only.",
            )
    elif vertical == "finance":
        unsafe = [
            r"\brecommend buying\b",
            r"\brecommend selling\b",
            r"\bbuy this stock\b",
            r"\bsell this stock\b",
            r"\bprice target is\b",
            r"\binvestment advice:",
        ]
        if any(re.search(pattern, prompt_answer_text_lower) for pattern in unsafe):
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="gold",
                check_name="finance_investment_advice",
                message="Finance records appear to provide investment advice.",
                recommendation="Remove recommendations, price targets, and market commentary.",
            )
    elif vertical == "retail":
        if "user_id" in all_text_lower or "user_id_hash" in all_text_lower:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="kb",
                check_name="retail_raw_user_id",
                message="Retail records contain raw user identifier fields.",
                recommendation="Remove raw user IDs and hash fields.",
            )
        if "all_beauty product <asin>" in all_text_lower or re.search(
            r"all_beauty product [a-z0-9]+", all_text_lower
        ):
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="prompts",
                check_name="retail_generic_title",
                message="Retail records contain generic product titles.",
                recommendation="Use real sanitized product titles or neutral evidence titles.",
            )
    elif vertical == "research_ai":
        unsafe = [
            r"\bgeneral model memory\b.*\banswer\b",
            r"\bfabricated paper claim\b",
            r"\baccording to my knowledge\b",
            r"\bfull paper text dump\b",
        ]
        if any(re.search(pattern, prompt_answer_text_lower) for pattern in unsafe):
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="gold",
                check_name="research_ai_fabricated_claims",
                message="Research AI records appear to allow fabricated or memory-based claims.",
                recommendation="Keep answers cited to paper/section evidence only.",
            )
    elif vertical == "airline":
        unsafe = [r"\bguaranteed compensation\b", r"\bbypass verification\b"]
        if any(re.search(pattern, prompt_answer_text_lower) for pattern in unsafe):
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="gold",
                check_name="airline_policy_safety",
                message=(
                    "Airline records include unsupported compensation or verification language."
                ),
                recommendation="Keep Airline answers policy-cited with normal verification.",
            )


def audit_vertical(
    *,
    vertical: str,
    partial_root: Path,
    generated_root: Path,
    global_prompt_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    records, files = load_vertical_records(
        vertical=vertical,
        partial_root=partial_root,
        generated_root=generated_root,
        issues=issues,
    )
    prompts = records.get("prompts", [])
    gold = records.get("gold", [])
    kb_rows = records.get("kb", [])
    if prompts or gold or kb_rows:
        validate_counts(issues, vertical=vertical, prompts=prompts, gold=gold, kb_rows=kb_rows)
        validate_alignment(issues, vertical=vertical, prompts=prompts, gold=gold)
        validate_evidence(issues, vertical=vertical, gold=gold, kb_rows=kb_rows)
        validate_distributions(issues, vertical=vertical, prompts=prompts)
        validate_hygiene(
            issues,
            vertical=vertical,
            rows_by_kind={"prompts": prompts, "gold": gold, "kb": kb_rows},
        )
        validate_domain_safety(
            issues,
            vertical=vertical,
            prompts=prompts,
            gold=gold,
            kb_rows=kb_rows,
        )
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id") or "")
        if prompt_id in global_prompt_ids:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file="prompts",
                prompt_id=prompt_id,
                check_name="global_prompt_id_uniqueness",
                message=f"Prompt ID {prompt_id} appears in more than one vertical.",
                recommendation="Regenerate prompt IDs with vertical-specific prefixes.",
            )
        global_prompt_ids.add(prompt_id)
    linguistic_metrics = calculate_question_template_diversity(prompts)
    validate_linguistic_variation(issues, vertical=vertical, metrics=linguistic_metrics)
    per_vertical = {
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_rows),
        "status_counts": dict(Counter(str(row.get("expected_status") or "") for row in prompts)),
        "output_format_counts": dict(
            Counter(str(row.get("expected_output_format") or "") for row in prompts)
        ),
        "task_type_counts": dict(Counter(str(row.get("task_type") or "") for row in prompts)),
        "linguistic_variation": linguistic_metrics,
        "critical_issue_count": critical_count(issues),
        "warning_count": warning_count(issues),
        "promotion_ready": not issues,
        "files": files,
    }
    return per_vertical, issues


def audit_dataset(
    *,
    partial_root: Path,
    generated_root: Path,
    output_report: Path,
    output_summary_csv: Path,
    output_issue_log: Path,
) -> dict[str, Any]:
    per_vertical: dict[str, dict[str, Any]] = {}
    all_issues: list[dict[str, Any]] = []
    global_prompt_ids: set[str] = set()
    for vertical in VERTICALS:
        vertical_report, vertical_issues = audit_vertical(
            vertical=vertical,
            partial_root=partial_root,
            generated_root=generated_root,
            global_prompt_ids=global_prompt_ids,
        )
        per_vertical[vertical] = vertical_report
        all_issues.extend(vertical_issues)

    total_prompt_count = sum(row["prompt_count"] for row in per_vertical.values())
    total_gold_count = sum(row["gold_count"] for row in per_vertical.values())
    total_kb_count = sum(row["kb_count"] for row in per_vertical.values())
    if total_prompt_count != 10000:
        add_issue(
            all_issues,
            severity="critical",
            vertical="ALL",
            file="prompts",
            check_name="global_prompt_count",
            message=f"Expected 10000 prompts, found {total_prompt_count}.",
            recommendation="Regenerate all 2,000-scale verticals.",
        )
    if total_gold_count != 10000:
        add_issue(
            all_issues,
            severity="critical",
            vertical="ALL",
            file="gold",
            check_name="global_gold_count",
            message=f"Expected 10000 gold records, found {total_gold_count}.",
            recommendation="Regenerate all 2,000-scale verticals.",
        )
    for index, issue in enumerate(all_issues, start=1):
        issue["issue_id"] = f"phase2a_2000_full_qa_{index:04d}"

    global_status_counts: Counter[str] = Counter()
    global_output_format_counts: Counter[str] = Counter()
    for row in per_vertical.values():
        global_status_counts.update(row["status_counts"])
        global_output_format_counts.update(row["output_format_counts"])

    report_critical_count = critical_count(all_issues)
    report_warning_count = warning_count(all_issues)
    promotion_ready = report_critical_count == 0 and report_warning_count == 0
    report = {
        "phase": PHASE,
        "dataset_name": DATASET_NAME,
        "partial_dataset": False,
        "generated_at_utc": utc_now(),
        "included_verticals": VERTICALS,
        "total_prompt_count": total_prompt_count,
        "total_gold_count": total_gold_count,
        "total_kb_count": total_kb_count,
        "vertical_count": len(VERTICALS),
        "per_vertical": per_vertical,
        "global_status_counts": dict(global_status_counts),
        "global_output_format_counts": dict(global_output_format_counts),
        "linguistic_variation_by_vertical": {
            vertical: row["linguistic_variation"] for vertical, row in per_vertical.items()
        },
        "critical_issue_count": report_critical_count,
        "warning_count": report_warning_count,
        "issue_counts_by_severity": dict(Counter(issue["severity"] for issue in all_issues)),
        "promotion_ready": promotion_ready,
        "next_step": (
            "Promote full 2,000-scale / 10,000-record dataset."
            if promotion_ready
            else "Fix listed 2,000-scale candidate issues before full promotion."
        ),
    }

    summary_rows = []
    for vertical, row in per_vertical.items():
        status_counts = row["status_counts"]
        output_counts = row["output_format_counts"]
        summary_rows.append(
            {
                "vertical": vertical,
                "prompt_count": row["prompt_count"],
                "gold_count": row["gold_count"],
                "kb_count": row["kb_count"],
                "answer_count": status_counts.get("answer", 0),
                "negative_status_count": row["prompt_count"] - status_counts.get("answer", 0),
                "json_output_count": output_counts.get("json", 0),
                "markdown_table_count": output_counts.get("markdown_table", 0),
                "linguistic_variation_rate": row["linguistic_variation"][
                    "linguistic_variation_rate"
                ],
                "critical_issues": row["critical_issue_count"],
                "warnings": row["warning_count"],
                "promotion_ready": row["promotion_ready"],
            }
        )
    write_json(output_report, report)
    write_summary_csv(output_summary_csv, summary_rows)
    write_jsonl(output_issue_log, all_issues)
    return report


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    report = audit_dataset(
        partial_root=Path(args.partial_root),
        generated_root=Path(args.generated_root),
        output_report=Path(args.output_report),
        output_summary_csv=Path(args.output_summary_csv),
        output_issue_log=Path(args.output_issue_log),
    )
    return {
        "phase": PHASE,
        "mode": "run_audit",
        "partial_dataset": False,
        "included_verticals": VERTICALS,
        "total_prompt_count": report["total_prompt_count"],
        "total_gold_count": report["total_gold_count"],
        "total_kb_count": report["total_kb_count"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "promotion_ready": report["promotion_ready"],
        "report_path": str(args.output_report),
        "summary_csv_path": str(args.output_summary_csv),
        "issue_log_path": str(args.output_issue_log),
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--fail-on-critical", action="store_true")
    parser.add_argument("--partial-root", default=str(DEFAULT_PARTIAL_ROOT))
    parser.add_argument("--generated-root", default=str(DEFAULT_GENERATED_ROOT))
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-summary-csv", default=str(DEFAULT_OUTPUT_SUMMARY_CSV))
    parser.add_argument("--output-issue-log", default=str(DEFAULT_OUTPUT_ISSUE_LOG))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_audit:
        parser.error("Pass --run-audit to audit full 2,000-scale candidates.")
    try:
        summary = run_audit(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    if args.fail_on_critical and summary["critical_issue_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
