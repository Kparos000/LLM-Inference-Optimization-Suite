"""Audit Phase 2A 250-scale candidates across all verticals.

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

PHASE = "2A-10"
TARGET_PER_VERTICAL = 250
LINGUISTIC_VARIATION_THRESHOLD = 0.60
MAX_TEMPLATE_SHARE = 0.40

DEFAULT_OUTPUT_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_report.json"
)
DEFAULT_OUTPUT_SUMMARY_CSV = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_summary.csv"
)
DEFAULT_OUTPUT_ISSUE_LOG = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_250_issue_log.jsonl"
)

VERTICAL_TARGETS: dict[str, dict[str, Path]] = {
    "airline": {
        "prompts": Path("data/generated/phase2a/scaleup/airline/airline_prompts_250.jsonl"),
        "gold": Path("data/generated/phase2a/scaleup/airline/airline_gold_250.jsonl"),
        "kb": Path("data/generated/phase2a/scaleup/airline/airline_kb_250.jsonl"),
        "report": Path("data/generated/phase2a/scaleup_reports/airline_scaleup_250_report.json"),
    },
    "healthcare_admin": {
        "prompts": Path(
            "data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_prompts_250.jsonl"
        ),
        "gold": Path(
            "data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_gold_250.jsonl"
        ),
        "kb": Path("data/generated/phase2a/scaleup/healthcare_admin/healthcare_admin_kb_250.jsonl"),
        "report": Path(
            "data/generated/phase2a/scaleup_reports/healthcare_admin_scaleup_250_report.json"
        ),
    },
    "retail": {
        "prompts": Path("data/generated/phase2a/scaleup/retail/retail_prompts_250.jsonl"),
        "gold": Path("data/generated/phase2a/scaleup/retail/retail_gold_250.jsonl"),
        "kb": Path("data/generated/phase2a/scaleup/retail/retail_kb_250.jsonl"),
        "report": Path("data/generated/phase2a/scaleup_reports/retail_scaleup_250_report.json"),
    },
    "research_ai": {
        "prompts": Path("data/generated/phase2a/scaleup/research_ai/research_ai_prompts_250.jsonl"),
        "gold": Path("data/generated/phase2a/scaleup/research_ai/research_ai_gold_250.jsonl"),
        "kb": Path("data/generated/phase2a/scaleup/research_ai/research_ai_kb_250.jsonl"),
        "report": Path(
            "data/generated/phase2a/scaleup_reports/research_ai_scaleup_250_report.json"
        ),
    },
    "finance": {
        "prompts": Path("data/generated/phase2a/scaleup/finance/finance_prompts_250.jsonl"),
        "gold": Path("data/generated/phase2a/scaleup/finance/finance_gold_250.jsonl"),
        "kb": Path("data/generated/phase2a/scaleup/finance/finance_kb_250.jsonl"),
        "report": Path("data/generated/phase2a/scaleup_reports/finance_scaleup_250_report.json"),
    },
}

EXPECTED_STATUS_COUNTS: dict[str, dict[str, int]] = {
    "airline": {"answer": 225, "escalate": 20, "spam_or_fraud": 5},
    "healthcare_admin": {
        "answer": 220,
        "escalate": 20,
        "safety_boundary": 5,
        "spam_or_fraud": 3,
        "out_of_scope": 2,
    },
    "retail": {
        "answer": 222,
        "insufficient_evidence": 9,
        "escalate": 9,
        "spam_or_low_quality": 7,
        "out_of_scope": 3,
    },
    "research_ai": {
        "answer": 225,
        "insufficient_evidence": 10,
        "escalate": 10,
        "out_of_scope": 5,
    },
    "finance": {"answer": 230, "insufficient_evidence": 10, "escalate": 10},
}

EXPECTED_OUTPUT_FORMAT_COUNTS: dict[str, dict[str, int]] = {
    "airline": {"text": 190, "json": 35, "markdown_table": 25},
    "healthcare_admin": {"text": 195, "json": 35, "markdown_table": 20},
    "retail": {"text": 185, "json": 40, "markdown_table": 25},
    "research_ai": {"text": 180, "json": 35, "markdown_table": 35},
    "finance": {"text": 155, "json": 50, "markdown_table": 45},
}

NEGATIVE_STATUSES = {
    "insufficient_evidence",
    "escalate",
    "out_of_scope",
    "spam_or_low_quality",
    "spam_or_fraud",
    "safety_boundary",
}
BOUNDARY_TERMS = [
    "insufficient",
    "escalate",
    "decline",
    "ignore",
    "fraud",
    "boundary",
    "out of scope",
    "out_of_scope",
    "outside",
    "not support",
    "do not",
    "cannot",
]
HYGIENE_PATTERNS = [
    (re.compile(pattern, flags=re.IGNORECASE), label)
    for pattern, label in [
        (r"C:\\Users", "private Windows path"),
        (r"/home/", "private Unix path"),
        (r"akpoogaga", "private username"),
        (r"\bkparo\b", "private username"),
        (r"API key", "API key reference"),
        (r"\btoken\b", "token reference"),
        (r"\bsecret\b", "secret reference"),
        (r"\bpassword\b", "password reference"),
        (r"raw user_id", "raw user identifier"),
        (r"\bTODO\b", "TODO placeholder"),
        (r"\bFIXME\b", "FIXME placeholder"),
        (r"lorem ipsum", "placeholder text"),
    ]
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return parsed


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


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
            "issue_id": f"phase2a_250_qa_{len(issues) + 1:04d}",
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


def renumber_issue_ids(issues: list[dict[str, Any]]) -> None:
    for index, issue in enumerate(issues, start=1):
        issue["issue_id"] = f"phase2a_250_qa_{index:04d}"


def generation_command(vertical: str) -> str:
    return (
        "python scripts/phase2/generate_phase2a_scaleup.py --generate-vertical "
        f"--vertical {vertical} --target-per-vertical 250"
    )


def find_missing_candidate_files(
    targets: dict[str, dict[str, Path]] | None = None,
) -> list[dict[str, Any]]:
    target_map = targets or VERTICAL_TARGETS
    missing: list[dict[str, Any]] = []
    for vertical, files in target_map.items():
        for kind in ("prompts", "gold", "kb"):
            path = files[kind]
            if not path.exists():
                missing.append(
                    {
                        "vertical": vertical,
                        "kind": kind,
                        "path": str(path),
                        "command": generation_command(vertical),
                    }
                )
    return missing


def missing_files_error(missing_files: list[dict[str, Any]]) -> str:
    commands = sorted({str(item["command"]) for item in missing_files})
    missing_lines = "\n".join(
        f"- {item['vertical']} {item['kind']}: {item['path']}" for item in missing_files
    )
    command_lines = "\n".join(commands)
    return (
        "Missing generated 250-scale candidate file(s):\n"
        f"{missing_lines}\n\n"
        "Generate the missing candidates with:\n"
        f"{command_lines}"
    )


def prompt_id_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("prompt_id") or "") for row in rows)


def evidence_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in ("required_doc_ids", "required_evidence_ids", "required_policy_ids"):
        value = record.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return ids


def support_ids(record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in ("required_chunk_ids", "required_citations"):
        value = record.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return ids


def answer_text(record: dict[str, Any]) -> str:
    fields = [
        record.get("question"),
        record.get("reference_answer"),
        record.get("answer"),
        record.get("expected_answer"),
    ]
    return " ".join(flatten_text(field) for field in fields if field is not None)


def validate_counts(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    targets: dict[str, Path],
) -> None:
    if len(prompts) != TARGET_PER_VERTICAL:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["prompts"]),
            check_name="prompt_count",
            message=f"Expected 250 prompts; found {len(prompts)}.",
            recommendation="Regenerate the vertical 250-scale candidate prompts.",
        )
    if len(gold) != TARGET_PER_VERTICAL:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["gold"]),
            check_name="gold_count",
            message=f"Expected 250 gold records; found {len(gold)}.",
            recommendation="Regenerate the vertical 250-scale candidate gold records.",
        )
    if not kb_rows:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["kb"]),
            check_name="kb_count",
            message="Expected a positive KB record count; found 0.",
            recommendation="Regenerate the vertical candidate KB file.",
        )


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
                check_name="prompt_id_present",
                message="Prompt record is missing prompt_id.",
                recommendation="Regenerate prompts with stable prompt IDs.",
            )
        elif count > 1:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(prompt_file),
                prompt_id=prompt_id,
                check_name="prompt_id_unique_per_vertical",
                message=f"Prompt id {prompt_id} appears {count} times in {vertical}.",
                recommendation="Deduplicate prompt IDs before promotion.",
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
                recommendation="Regenerate gold records with matching prompt IDs.",
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


def validate_global_prompt_ids(
    issues: list[dict[str, Any]],
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
) -> None:
    seen: dict[str, str] = {}
    for vertical, prompts in prompts_by_vertical.items():
        for prompt in prompts:
            prompt_id = str(prompt.get("prompt_id") or "")
            if not prompt_id:
                continue
            previous = seen.get(prompt_id)
            if previous and previous != vertical:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file=str(VERTICAL_TARGETS[vertical]["prompts"]),
                    prompt_id=prompt_id,
                    check_name="prompt_id_unique_globally",
                    message=f"Prompt id {prompt_id} also appears in {previous}.",
                    recommendation="Regenerate candidates with globally unique prompt IDs.",
                )
            seen[prompt_id] = vertical


def validate_evidence_coverage(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    *,
    vertical: str,
    targets: dict[str, Path],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    kb_doc_ids = {str(row.get("doc_id") or "") for row in kb_rows if row.get("doc_id")}
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        expected_status = str(
            gold_row.get("expected_status") or prompt.get("expected_status") or ""
        )
        combined_ids = evidence_ids(gold_row) or evidence_ids(prompt)
        combined_support = support_ids(gold_row) or support_ids(prompt)
        must_include = gold_row.get("must_include")
        if expected_status == "answer":
            if not combined_ids:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file=str(targets["gold"]),
                    prompt_id=prompt_id,
                    check_name="answerable_evidence_required",
                    message="Answerable record has no required evidence IDs.",
                    recommendation="Add required_doc_ids or required_evidence_ids.",
                )
            if not combined_support:
                add_issue(
                    issues,
                    severity="warning",
                    vertical=vertical,
                    file=str(targets["gold"]),
                    prompt_id=prompt_id,
                    check_name="answerable_support_locator_recommended",
                    message="Answerable record has no chunk IDs or citations.",
                    recommendation="Add required_chunk_ids or required_citations.",
                )
            if not isinstance(must_include, list) or not must_include:
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
        for doc_id in gold_row.get("required_doc_ids", []):
            if str(doc_id) not in kb_doc_ids:
                add_issue(
                    issues,
                    severity="critical",
                    vertical=vertical,
                    file=str(targets["gold"]),
                    prompt_id=prompt_id,
                    check_name="required_doc_id_exists",
                    message=f"required_doc_id {doc_id} is not present in the KB.",
                    recommendation="Fix the evidence reference or add the KB record.",
                )
    return issues


def validate_negative_records(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    vertical: str,
    targets: dict[str, Path],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    negative_count = 0
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        status = str(gold_row.get("expected_status") or prompt.get("expected_status") or "")
        if status == "answer":
            continue
        negative_count += 1
        must_not_include = gold_row.get("must_not_include")
        if not isinstance(must_not_include, list) or not must_not_include:
            add_issue(
                issues,
                severity="critical",
                vertical=vertical,
                file=str(targets["gold"]),
                prompt_id=prompt_id,
                check_name="negative_must_not_include",
                message="Negative/status-boundary record has empty must_not_include.",
                recommendation="Add meaningful unsupported-content exclusions.",
            )
        reference_answer = str(gold_row.get("reference_answer") or "").lower()
        expected_escalation = bool(gold_row.get("expected_escalation"))
        if (
            reference_answer
            and not expected_escalation
            and not any(term in reference_answer for term in BOUNDARY_TERMS)
        ):
            add_issue(
                issues,
                severity="warning",
                vertical=vertical,
                file=str(targets["gold"]),
                prompt_id=prompt_id,
                check_name="negative_boundary_answer",
                message="Negative/status-boundary answer does not state a clear boundary.",
                recommendation=(
                    "Make the reference answer clearly decline, escalate, or limit scope."
                ),
            )
    if negative_count == 0:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["gold"]),
            check_name="negative_status_coverage",
            message="Vertical has no non-answer records.",
            recommendation="Regenerate with negative/status-boundary examples.",
        )
    return issues


def validate_distribution(
    issues: list[dict[str, Any]],
    *,
    vertical: str,
    prompts: list[dict[str, Any]],
    targets: dict[str, Path],
) -> None:
    status_counts = dict(Counter(str(row.get("expected_status") or "missing") for row in prompts))
    output_counts = dict(
        Counter(str(row.get("expected_output_format") or "missing") for row in prompts)
    )
    if status_counts != EXPECTED_STATUS_COUNTS[vertical]:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["prompts"]),
            check_name="status_distribution",
            message=(
                f"Status distribution mismatch: expected {EXPECTED_STATUS_COUNTS[vertical]}, "
                f"found {status_counts}."
            ),
            recommendation="Regenerate candidates with the approved status distribution.",
        )
    if output_counts != EXPECTED_OUTPUT_FORMAT_COUNTS[vertical]:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(targets["prompts"]),
            check_name="output_format_distribution",
            message=(
                "Output format distribution mismatch: expected "
                f"{EXPECTED_OUTPUT_FORMAT_COUNTS[vertical]}, found {output_counts}."
            ),
            recommendation="Regenerate candidates with the approved output format distribution.",
        )


def dynamic_question_values(prompt: dict[str, Any]) -> list[str]:
    values: list[str] = []
    direct_fields = [
        "airline",
        "category",
        "company",
        "filing_form",
        "issue_type",
        "paper_title",
        "product_id",
        "product_title",
        "prompt_id",
        "route",
        "support_type",
        "task_type",
        "ticker",
        "ticket_id",
        "vertical",
    ]
    for field in direct_fields:
        value = prompt.get(field)
        if isinstance(value, str) and value:
            values.append(value)
    metadata = prompt.get("metadata")
    if isinstance(metadata, dict):
        for value in metadata.values():
            if isinstance(value, str) and value:
                values.append(value)
            elif isinstance(value, list):
                values.extend(str(item) for item in value if isinstance(item, str) and item)
    for field in ("required_doc_ids", "required_evidence_ids", "required_policy_ids"):
        value = prompt.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return values


def normalize_question_template(prompt: dict[str, Any]) -> str:
    text = str(prompt.get("question") or "").lower()
    text = re.sub(r"\b\d+(?:[.,]\d+)*\b", "<num>", text)
    for value in sorted(dynamic_question_values(prompt), key=len, reverse=True):
        normalized_value = str(value).lower().strip()
        if len(normalized_value) >= 2:
            text = text.replace(normalized_value, "<value>")
    text = re.sub(r"\b[a-z]{2,6}-[a-z0-9-]{2,}\b", "<id>", text)
    text = re.sub(r"\b[a-z0-9]{8,}\b", "<id>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_question_template_diversity(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    if not prompts:
        return {
            "linguistic_variation_rate": 0.0,
            "most_common_question_template_count": 0,
            "most_common_question_template_share": 0.0,
            "unique_question_template_count": 0,
        }
    counts = Counter(normalize_question_template(prompt) for prompt in prompts)
    most_common = counts.most_common(1)[0][1]
    total = len(prompts)
    most_common_share = most_common / total
    return {
        "linguistic_variation_rate": round(1 - most_common_share, 3),
        "most_common_question_template_count": most_common,
        "most_common_question_template_share": round(most_common_share, 3),
        "unique_question_template_count": len(counts),
    }


def linguistic_metrics_from_report_or_prompts(
    report: dict[str, Any],
    prompts: list[dict[str, Any]],
) -> dict[str, Any]:
    required_fields = {
        "linguistic_variation_rate",
        "most_common_question_template_count",
        "most_common_question_template_share",
        "unique_question_template_count",
    }
    if required_fields.issubset(report):
        return {
            "linguistic_variation_rate": float(report["linguistic_variation_rate"]),
            "most_common_question_template_count": int(
                report["most_common_question_template_count"]
            ),
            "most_common_question_template_share": float(
                report["most_common_question_template_share"]
            ),
            "unique_question_template_count": int(report["unique_question_template_count"]),
        }
    return calculate_question_template_diversity(prompts)


def validate_linguistic_variation(
    prompts: list[dict[str, Any]],
    *,
    vertical: str,
    report: dict[str, Any] | None = None,
    file: Path | str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = linguistic_metrics_from_report_or_prompts(report or {}, prompts)
    issues: list[dict[str, Any]] = []
    rate = float(metrics["linguistic_variation_rate"])
    share = float(metrics["most_common_question_template_share"])
    if rate < LINGUISTIC_VARIATION_THRESHOLD or share > MAX_TEMPLATE_SHARE:
        add_issue(
            issues,
            severity="critical",
            vertical=vertical,
            file=str(file),
            check_name="linguistic_variation_gate",
            message=(
                f"Linguistic variation failed: rate={rate:.3f}, "
                f"most_common_template_share={share:.3f}."
            ),
            recommendation="Regenerate candidate prompts with deterministic wording variation.",
        )
    return metrics, issues


def scan_hygiene_text(text: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for pattern, label in HYGIENE_PATTERNS:
        if pattern.search(text):
            findings.append(("critical", label))
    return findings


def scan_record_hygiene(
    records: list[dict[str, Any]],
    *,
    vertical: str,
    file: Path | str,
    id_field: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for record in records:
        row_text = flatten_text(record)
        for severity, label in scan_hygiene_text(row_text):
            add_issue(
                issues,
                severity=severity,
                vertical=vertical,
                file=str(file),
                prompt_id=str(record.get("prompt_id")) if record.get("prompt_id") else None,
                doc_id=str(record.get(id_field)) if record.get(id_field) else None,
                check_name="public_hygiene",
                message=f"Record contains {label}.",
                recommendation="Remove private paths, credentials, placeholders, or raw IDs.",
            )
    return issues


def check_healthcare_specific(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    targets: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    target_files = targets or VERTICAL_TARGETS["healthcare_admin"]
    clinical_advice_patterns = [
        r"(?<!no )(?<!without )\bdiagnos(?:e|is|ed)\b",
        r"\btreatment advice\b",
        r"\btreatment instructions\b",
        r"\btake \d+",
        r"\bmedication dosage\b",
    ]
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        text = answer_text(gold_row).lower()
        for pattern in clinical_advice_patterns:
            if re.search(pattern, text) and "no clinical advice" not in text:
                add_issue(
                    issues,
                    severity="critical",
                    vertical="healthcare_admin",
                    file=str(target_files["gold"]),
                    prompt_id=prompt_id,
                    check_name="healthcare_clinical_advice",
                    message="Healthcare reference answer appears to provide clinical advice.",
                    recommendation="Keep healthcare_admin answers administrative only.",
                )
        prompt = prompts_by_id.get(prompt_id, {})
        prompt_text = flatten_text(
            [prompt.get("question"), prompt.get("issue"), prompt.get("support_type")]
        ).lower()
        status = str(gold_row.get("expected_status") or prompt.get("expected_status") or "")
        if any(
            term in prompt_text
            for term in ("urgent", "clinical", "symptom", "diagnosis", "treatment")
        ) and status not in {"safety_boundary", "escalate"}:
            add_issue(
                issues,
                severity="critical",
                vertical="healthcare_admin",
                file=str(target_files["prompts"]),
                prompt_id=prompt_id,
                check_name="healthcare_safety_status",
                message=(
                    "Urgent, safety, or clinical prompt is not marked safety_boundary/escalate."
                ),
                recommendation=(
                    "Route urgent or clinical requests to safety boundary or escalation."
                ),
            )
    return issues


def check_finance_specific(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    targets: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    target_files = targets or VERTICAL_TARGETS["finance"]
    recommendation_patterns = [
        r"\binvestment advice\b",
        r"\brecommend(?:s|ed|ing)? (?:buying|selling|holding)\b",
        r"\bshould (?:buy|sell|hold|invest)\b",
        r"\bbuy recommendation\b",
        r"\bsell recommendation\b",
        r"\bhold recommendation\b",
    ]
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        text = f"{answer_text(prompt)} {answer_text(gold_row)}".lower()
        for pattern in recommendation_patterns:
            if re.search(pattern, text):
                add_issue(
                    issues,
                    severity="critical",
                    vertical="finance",
                    file=str(target_files["gold"]),
                    prompt_id=prompt_id,
                    check_name="finance_investment_advice",
                    message="Finance candidate appears to provide investment advice.",
                    recommendation="Keep Finance answers limited to SEC/XBRL evidence.",
                )
        if re.search(r"\d", str(gold_row.get("reference_answer") or "")) and not evidence_ids(
            gold_row
        ):
            add_issue(
                issues,
                severity="critical",
                vertical="finance",
                file=str(target_files["gold"]),
                prompt_id=prompt_id,
                check_name="finance_numeric_evidence",
                message="Finance answer contains numeric claims without required evidence IDs.",
                recommendation="Attach SEC/XBRL evidence IDs to numeric finance answers.",
            )
    return issues


def check_retail_specific(
    prompts: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    *,
    targets: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    target_files = targets or VERTICAL_TARGETS["retail"]
    for row in prompts:
        text = answer_text(row)
        product_title = str(row.get("product_title") or "")
        if "raw user_id" in text.lower() or "user_id" in row:
            add_issue(
                issues,
                severity="critical",
                vertical="retail",
                file=str(target_files["prompts"]),
                prompt_id=str(row.get("prompt_id")) if row.get("prompt_id") else None,
                check_name="retail_raw_user_id",
                message="Retail prompt exposes a raw user identifier.",
                recommendation="Remove raw user IDs from Retail generated records.",
            )
        if re.fullmatch(r"All_Beauty product B[0-9A-Z]{7,12}", product_title):
            add_issue(
                issues,
                severity="critical",
                vertical="retail",
                file=str(target_files["prompts"]),
                prompt_id=str(row.get("prompt_id")) if row.get("prompt_id") else None,
                check_name="retail_generic_product_title",
                message="Retail prompt uses a generic All_Beauty product title.",
                recommendation="Use reviewed product metadata rather than generic titles.",
            )
    for row in kb_rows:
        if str(row.get("document_type") or "") != "support_policy":
            continue
        body = flatten_text(row)
        if "synthetic benchmark policy" not in body or "not Amazon policy" not in body:
            add_issue(
                issues,
                severity="critical",
                vertical="retail",
                file=str(target_files["kb"]),
                doc_id=str(row.get("doc_id")) if row.get("doc_id") else None,
                check_name="retail_synthetic_policy_label",
                message=(
                    "Retail support policy is not clearly marked as synthetic benchmark policy."
                ),
                recommendation=(
                    "State that support policies are synthetic benchmark policy, not Amazon policy."
                ),
            )
    return issues


def check_research_ai_specific(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    targets: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    target_files = targets or VERTICAL_TARGETS["research_ai"]
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        status = str(gold_row.get("expected_status") or prompt.get("expected_status") or "")
        text = f"{answer_text(prompt)} {answer_text(gold_row)}".lower()
        if "general model memory" in text and "must_not_include" not in text:
            add_issue(
                issues,
                severity="critical",
                vertical="research_ai",
                file=str(target_files["gold"]),
                prompt_id=prompt_id,
                check_name="research_ai_general_memory",
                message="Research AI answer appears to rely on general model memory.",
                recommendation="Ground Research AI answers in cited paper evidence.",
            )
        if "fabricated paper claim" in text or "fabricated citation" in text:
            add_issue(
                issues,
                severity="critical",
                vertical="research_ai",
                file=str(target_files["gold"]),
                prompt_id=prompt_id,
                check_name="research_ai_fabricated_claim",
                message="Research AI candidate contains a fabricated-claim marker.",
                recommendation="Remove fabricated claims and cite KB/paper evidence only.",
            )
        if status == "answer" and not evidence_ids(gold_row):
            add_issue(
                issues,
                severity="critical",
                vertical="research_ai",
                file=str(target_files["gold"]),
                prompt_id=prompt_id,
                check_name="research_ai_answerable_evidence",
                message="Answerable Research AI gold row has no KB/paper evidence IDs.",
                recommendation="Attach required_doc_ids and citations to answerable records.",
            )
    return issues


def check_airline_specific(
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    targets: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    target_files = targets or VERTICAL_TARGETS["airline"]
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    risky_patterns = [
        (r"\bguarantee(?:d)? compensation\b", "airline_compensation_promise"),
        (r"\bskip verification\b", "airline_verification_bypass"),
        (r"\bbypass verification\b", "airline_verification_bypass"),
        (r"\bpolicy exception without citation\b", "airline_uncited_policy_exception"),
    ]
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        text = f"{answer_text(prompt)} {answer_text(gold_row)}".lower()
        for pattern, check_name in risky_patterns:
            if re.search(pattern, text):
                add_issue(
                    issues,
                    severity="critical",
                    vertical="airline",
                    file=str(target_files["gold"]),
                    prompt_id=prompt_id,
                    check_name=check_name,
                    message="Airline candidate contains unsafe policy handling language.",
                    recommendation="Keep Airline answers grounded in cited policy evidence.",
                )
    return issues


def run_domain_specific_checks(
    vertical: str,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    *,
    targets: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    if vertical == "healthcare_admin":
        return check_healthcare_specific(prompts, gold, targets=targets)
    if vertical == "finance":
        return check_finance_specific(prompts, gold, targets=targets)
    if vertical == "retail":
        return check_retail_specific(prompts, kb_rows, targets=targets)
    if vertical == "research_ai":
        return check_research_ai_specific(prompts, gold, targets=targets)
    if vertical == "airline":
        return check_airline_specific(prompts, gold, targets=targets)
    return []


def audit_vertical(
    vertical: str,
    targets: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    prompts = read_jsonl(targets["prompts"])
    gold = read_jsonl(targets["gold"])
    kb_rows = read_jsonl(targets["kb"])
    scaleup_report = read_json_if_exists(targets["report"])

    validate_counts(
        issues,
        vertical=vertical,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_rows,
        targets=targets,
    )
    issues.extend(
        validate_prompt_gold_alignment(
            prompts,
            gold,
            vertical=vertical,
            prompt_file=targets["prompts"],
            gold_file=targets["gold"],
        )
    )
    issues.extend(
        validate_evidence_coverage(
            prompts,
            gold,
            kb_rows,
            vertical=vertical,
            targets=targets,
        )
    )
    issues.extend(validate_negative_records(prompts, gold, vertical=vertical, targets=targets))
    validate_distribution(issues, vertical=vertical, prompts=prompts, targets=targets)
    linguistic_metrics, linguistic_issues = validate_linguistic_variation(
        prompts,
        vertical=vertical,
        report=scaleup_report,
        file=targets["report"] if targets["report"].exists() else targets["prompts"],
    )
    issues.extend(linguistic_issues)

    for kind, rows, id_field in (
        ("prompts", prompts, "prompt_id"),
        ("gold", gold, "prompt_id"),
        ("kb", kb_rows, "doc_id"),
    ):
        issues.extend(
            scan_record_hygiene(
                rows,
                vertical=vertical,
                file=targets[kind],
                id_field=id_field,
            )
        )
    issues.extend(run_domain_specific_checks(vertical, prompts, gold, kb_rows, targets=targets))

    status_counts = Counter(str(row.get("expected_status") or "missing") for row in prompts)
    output_counts = Counter(str(row.get("expected_output_format") or "missing") for row in prompts)
    task_type_counts = Counter(str(row.get("task_type") or "missing") for row in prompts)
    issue_counts = Counter(row["severity"] for row in issues)
    metrics = {
        "vertical": vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_rows),
        "status_counts": dict(status_counts),
        "output_format_counts": dict(output_counts),
        "task_type_counts": dict(task_type_counts),
        "answer_count": status_counts.get("answer", 0),
        "negative_status_count": sum(status_counts.get(status, 0) for status in NEGATIVE_STATUSES),
        "json_output_count": output_counts.get("json", 0),
        "markdown_table_count": output_counts.get("markdown_table", 0),
        "linguistic_variation": linguistic_metrics,
        "critical_issues": issue_counts.get("critical", 0),
        "warnings": issue_counts.get("warning", 0),
        "promotion_ready": issue_counts.get("critical", 0) == 0
        and issue_counts.get("warning", 0) == 0,
    }
    return metrics, issues, prompts


def build_report(
    per_vertical: dict[str, dict[str, Any]],
    all_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_counts_by_severity = Counter(row["severity"] for row in all_issues)
    total_prompt_count = sum(row["prompt_count"] for row in per_vertical.values())
    total_gold_count = sum(row["gold_count"] for row in per_vertical.values())
    total_kb_count = sum(row["kb_count"] for row in per_vertical.values())
    global_status_counts: Counter[str] = Counter()
    global_output_format_counts: Counter[str] = Counter()
    global_task_type_counts: Counter[str] = Counter()
    for metrics in per_vertical.values():
        global_status_counts.update(metrics["status_counts"])
        global_output_format_counts.update(metrics["output_format_counts"])
        global_task_type_counts.update(metrics["task_type_counts"])
    critical_issue_count = issue_counts_by_severity.get("critical", 0)
    warning_count = issue_counts_by_severity.get("warning", 0)
    promotion_ready = (
        critical_issue_count == 0
        and warning_count == 0
        and total_prompt_count == 1250
        and total_gold_count == 1250
        and len(per_vertical) == 5
    )
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "verticals_audited": list(per_vertical),
        "vertical_count": len(per_vertical),
        "total_prompt_count": total_prompt_count,
        "total_gold_count": total_gold_count,
        "total_kb_count": total_kb_count,
        "per_vertical": per_vertical,
        "global_status_counts": dict(global_status_counts),
        "global_output_format_counts": dict(global_output_format_counts),
        "global_task_type_counts": dict(global_task_type_counts),
        "linguistic_variation_by_vertical": {
            vertical: metrics["linguistic_variation"] for vertical, metrics in per_vertical.items()
        },
        "critical_issue_count": critical_issue_count,
        "warning_count": warning_count,
        "issue_counts_by_severity": {
            "critical": critical_issue_count,
            "warning": warning_count,
            "info": issue_counts_by_severity.get("info", 0),
        },
        "promotion_ready": promotion_ready,
        "next_step": (
            "Proceed to Phase 2A-11 promotion of the full 250-scale dataset."
            if promotion_ready
            else "Fix listed candidate issues before promotion."
        ),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    missing_files = find_missing_candidate_files()
    if missing_files:
        raise RuntimeError(missing_files_error(missing_files))

    all_issues: list[dict[str, Any]] = []
    per_vertical: dict[str, dict[str, Any]] = {}
    prompts_by_vertical: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for vertical, targets in VERTICAL_TARGETS.items():
        metrics, issues, prompts = audit_vertical(vertical, targets)
        per_vertical[vertical] = metrics
        all_issues.extend(issues)
        prompts_by_vertical[vertical] = prompts

    validate_global_prompt_ids(all_issues, prompts_by_vertical)
    renumber_issue_ids(all_issues)
    report = build_report(per_vertical, all_issues)
    for vertical, metrics in per_vertical.items():
        summary_rows.append(
            {
                "vertical": vertical,
                "prompt_count": metrics["prompt_count"],
                "gold_count": metrics["gold_count"],
                "kb_count": metrics["kb_count"],
                "answer_count": metrics["answer_count"],
                "negative_status_count": metrics["negative_status_count"],
                "json_output_count": metrics["json_output_count"],
                "markdown_table_count": metrics["markdown_table_count"],
                "linguistic_variation_rate": metrics["linguistic_variation"][
                    "linguistic_variation_rate"
                ],
                "critical_issues": metrics["critical_issues"],
                "warnings": metrics["warnings"],
                "promotion_ready": metrics["promotion_ready"],
            }
        )

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
        "promotion_ready": report["promotion_ready"],
        "output_report": str(args.output_report),
        "output_summary_csv": str(args.output_summary_csv),
        "output_issue_log": str(args.output_issue_log),
        "next_step": report["next_step"],
    }
    if args.fail_on_critical and report["critical_issue_count"] > 0:
        raise RuntimeError(
            f"Phase 2A-10 audit found {report['critical_issue_count']} critical issue(s)."
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Phase 2A 250-scale candidates.")
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--fail-on-critical", action="store_true")
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-summary-csv", default=str(DEFAULT_OUTPUT_SUMMARY_CSV))
    parser.add_argument("--output-issue-log", default=str(DEFAULT_OUTPUT_ISSUE_LOG))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_audit:
        parser.error("Use --run-audit to run the Phase 2A-10 250-scale QA audit.")
    try:
        summary = run_audit(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
