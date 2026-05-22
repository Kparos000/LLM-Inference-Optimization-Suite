"""Review and optionally promote a Phase 2A scale-up candidate dataset.

This script performs data QA only. It does not build RAG, retrieval indexes,
embeddings, prompt assembly, model calls, GPU runs, or benchmark inference.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-9A"
DEFAULT_VERTICAL = "airline"
DEFAULT_TARGET_COUNT = 250
DEFAULT_CANDIDATE_PROMPTS = Path("data/generated/phase2a/scaleup/airline/airline_prompts_250.jsonl")
DEFAULT_CANDIDATE_GOLD = Path("data/generated/phase2a/scaleup/airline/airline_gold_250.jsonl")
DEFAULT_CANDIDATE_KB = Path("data/generated/phase2a/scaleup/airline/airline_kb_250.jsonl")
DEFAULT_REVIEW_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/airline_250_candidate_review_report.json"
)
DEFAULT_PROMOTED_OUTPUT_DIR = Path("data/scaleup/airline")

EXPECTED_STATUS_COUNTS = {"answer": 225, "escalate": 20, "spam_or_fraud": 5}
EXPECTED_OUTPUT_FORMAT_COUNTS = {"text": 190, "json": 35, "markdown_table": 25}
PRIVATE_HYGIENE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"C:\\Users",
        r"/home/",
        r"akpoogaga",
        r"kparo",
        r"API key",
        r"\btoken\b",
        r"\bsecret\b",
        r"\bpassword\b",
        r"raw user_id",
    ]
]
PLACEHOLDER_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [r"\bTODO\b", r"\bFIXME\b", r"lorem ipsum"]
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
    check_name: str,
    message: str,
    recommendation: str,
    prompt_id: str | None = None,
    doc_id: str | None = None,
) -> None:
    issues.append(
        {
            "issue_id": f"airline_250_review_{len(issues) + 1:04d}",
            "severity": severity,
            "check_name": check_name,
            "message": message,
            "recommendation": recommendation,
            "prompt_id": prompt_id,
            "doc_id": doc_id,
        }
    )


def validate_counts(
    issues: list[dict[str, Any]],
    *,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    target_count: int,
) -> None:
    expected = {
        "prompt_count": (len(prompts), target_count),
        "gold_count": (len(gold), target_count),
        "kb_count": (len(kb_rows), 25),
    }
    for check_name, (actual, minimum_or_exact) in expected.items():
        if check_name == "kb_count":
            if actual < minimum_or_exact:
                add_issue(
                    issues,
                    severity="critical",
                    check_name=check_name,
                    message=f"Expected at least {minimum_or_exact} KB records; found {actual}.",
                    recommendation="Regenerate or repair the candidate KB file.",
                )
        elif actual != minimum_or_exact:
            add_issue(
                issues,
                severity="critical",
                check_name=check_name,
                message=f"Expected {minimum_or_exact} records; found {actual}.",
                recommendation="Regenerate the candidate files before promotion.",
            )


def validate_prompt_gold_alignment(
    issues: list[dict[str, Any]],
    *,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> None:
    prompt_counts = Counter(str(row.get("prompt_id") or "") for row in prompts)
    gold_counts = Counter(str(row.get("prompt_id") or "") for row in gold)
    for prompt_id, count in prompt_counts.items():
        if not prompt_id:
            add_issue(
                issues,
                severity="critical",
                check_name="prompt_id_present",
                message="Prompt record is missing prompt_id.",
                recommendation="Regenerate the candidate prompts with stable IDs.",
            )
        elif count > 1:
            add_issue(
                issues,
                severity="critical",
                check_name="prompt_id_unique",
                message=f"Prompt id {prompt_id} appears {count} times.",
                recommendation="Deduplicate prompt IDs before promotion.",
                prompt_id=prompt_id,
            )
    for prompt_id, count in gold_counts.items():
        if not prompt_id:
            add_issue(
                issues,
                severity="critical",
                check_name="gold_prompt_id_present",
                message="Gold record is missing prompt_id.",
                recommendation="Regenerate the candidate gold file with stable IDs.",
            )
        elif count > 1:
            add_issue(
                issues,
                severity="critical",
                check_name="gold_prompt_id_unique",
                message=f"Gold prompt id {prompt_id} appears {count} times.",
                recommendation="Deduplicate gold records before promotion.",
                prompt_id=prompt_id,
            )
    for prompt_id in prompt_counts:
        if prompt_id and gold_counts.get(prompt_id, 0) != 1:
            add_issue(
                issues,
                severity="critical",
                check_name="prompt_gold_alignment",
                message=f"Prompt {prompt_id} does not have exactly one gold record.",
                recommendation="Ensure every prompt has exactly one matching gold record.",
                prompt_id=prompt_id,
            )
    for prompt_id in gold_counts:
        if prompt_id and prompt_counts.get(prompt_id, 0) != 1:
            add_issue(
                issues,
                severity="critical",
                check_name="orphan_gold_record",
                message=f"Gold record {prompt_id} has no matching prompt.",
                recommendation="Remove orphan gold records or add the matching prompt.",
                prompt_id=prompt_id,
            )


def validate_evidence(
    issues: list[dict[str, Any]],
    *,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> None:
    kb_doc_ids = {str(row.get("doc_id") or "") for row in kb_rows}
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        expected_status = str(
            gold_row.get("expected_status") or prompt.get("expected_status") or ""
        )
        required_doc_ids = [str(item) for item in gold_row.get("required_doc_ids", []) if item]
        prompt_evidence_ids = [
            str(item)
            for item in (
                prompt.get("required_evidence_ids")
                or prompt.get("required_policy_ids")
                or gold_row.get("required_chunk_ids")
                or []
            )
            if item
        ]
        if expected_status == "answer":
            if not required_doc_ids and not prompt_evidence_ids:
                add_issue(
                    issues,
                    severity="critical",
                    check_name="answerable_evidence_present",
                    message=f"Answerable prompt {prompt_id} has no required evidence IDs.",
                    recommendation="Add required_doc_ids or required_evidence_ids.",
                    prompt_id=prompt_id,
                )
            if not gold_row.get("must_include"):
                add_issue(
                    issues,
                    severity="critical",
                    check_name="answerable_must_include_present",
                    message=f"Answerable prompt {prompt_id} has empty must_include.",
                    recommendation="Add grounded required terms to must_include.",
                    prompt_id=prompt_id,
                )
        for doc_id in [*required_doc_ids, *prompt_evidence_ids]:
            if doc_id and doc_id not in kb_doc_ids:
                add_issue(
                    issues,
                    severity="critical",
                    check_name="evidence_id_in_kb",
                    message=f"Evidence id {doc_id} for prompt {prompt_id} is missing from KB.",
                    recommendation="Ensure required evidence IDs exist as KB doc_id values.",
                    prompt_id=prompt_id,
                    doc_id=doc_id,
                )


def validate_distributions(
    issues: list[dict[str, Any]],
    *,
    status_counts: dict[str, int],
    output_format_counts: dict[str, int],
) -> None:
    if status_counts != EXPECTED_STATUS_COUNTS:
        add_issue(
            issues,
            severity="critical",
            check_name="expected_status_distribution",
            message=(
                f"Status distribution mismatch. Expected {EXPECTED_STATUS_COUNTS}; "
                f"found {status_counts}."
            ),
            recommendation="Regenerate the candidate with the approved Airline 250 status mix.",
        )
    if output_format_counts != EXPECTED_OUTPUT_FORMAT_COUNTS:
        add_issue(
            issues,
            severity="critical",
            check_name="expected_output_format_distribution",
            message=(
                f"Output format distribution mismatch. Expected {EXPECTED_OUTPUT_FORMAT_COUNTS}; "
                f"found {output_format_counts}."
            ),
            recommendation="Regenerate the candidate with the approved Airline 250 format mix.",
        )


def validate_hygiene_and_quality(
    issues: list[dict[str, Any]],
    *,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> None:
    for row in [*prompts, *gold, *kb_rows]:
        row_id = str(row.get("prompt_id") or row.get("doc_id") or "unknown")
        text = flatten_text(row)
        for pattern in PRIVATE_HYGIENE_PATTERNS:
            if pattern.search(text):
                add_issue(
                    issues,
                    severity="critical",
                    check_name="hygiene_scan",
                    message=f"Disallowed hygiene pattern found in {row_id}: {pattern.pattern}",
                    recommendation=(
                        "Remove private paths, secrets, raw identifiers, or unsafe text."
                    ),
                    prompt_id=row.get("prompt_id"),
                    doc_id=row.get("doc_id"),
                )
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                add_issue(
                    issues,
                    severity="critical",
                    check_name="placeholder_scan",
                    message=f"Placeholder text found in {row_id}: {pattern.pattern}",
                    recommendation="Replace placeholder content with finalized deterministic data.",
                    prompt_id=row.get("prompt_id"),
                    doc_id=row.get("doc_id"),
                )
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id") or "")
        question = str(prompt.get("question") or prompt.get("issue") or "").strip()
        if not question:
            add_issue(
                issues,
                severity="critical",
                check_name="question_present",
                message=f"Prompt {prompt_id} has no question or issue text.",
                recommendation="Regenerate the prompt with a usable question.",
                prompt_id=prompt_id,
            )
    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        reference_answer = str(gold_row.get("reference_answer") or "").strip()
        if not reference_answer:
            add_issue(
                issues,
                severity="critical",
                check_name="reference_answer_present",
                message=f"Gold record {prompt_id} has an empty reference_answer.",
                recommendation="Regenerate the gold record with a grounded reference answer.",
                prompt_id=prompt_id,
            )
    question_counts = Counter(
        str(prompt.get("question") or prompt.get("issue") or "").strip() for prompt in prompts
    )
    duplicate_questions = [
        question for question, count in question_counts.items() if question and count > 3
    ]
    if duplicate_questions:
        add_issue(
            issues,
            severity="warning",
            check_name="duplicate_question_text",
            message=(
                "Some question text appears more than three times; "
                f"duplicate groups: {len(duplicate_questions)}."
            ),
            recommendation="Review template diversity before scaling beyond 250 records.",
        )


def validate_negative_examples(
    issues: list[dict[str, Any]], *, status_counts: dict[str, int]
) -> None:
    negative_count = sum(status_counts.get(status, 0) for status in ["escalate", "spam_or_fraud"])
    if negative_count == 0:
        add_issue(
            issues,
            severity="critical",
            check_name="negative_examples_present",
            message="Candidate has no negative/escalation examples.",
            recommendation="Regenerate with the approved Airline negative-status mix.",
        )


def build_review_report(
    *,
    vertical: str,
    target_count: int,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    promoted_output_dir: Path | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    status_counts = dict(Counter(str(row.get("expected_status") or "") for row in prompts))
    output_format_counts = dict(
        Counter(str(row.get("expected_output_format") or "") for row in prompts)
    )
    task_type_counts = dict(Counter(str(row.get("task_type") or "") for row in prompts))
    validate_counts(
        issues,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_rows,
        target_count=target_count,
    )
    validate_prompt_gold_alignment(issues, prompts=prompts, gold=gold)
    validate_evidence(issues, prompts=prompts, gold=gold, kb_rows=kb_rows)
    validate_distributions(
        issues,
        status_counts=status_counts,
        output_format_counts=output_format_counts,
    )
    validate_hygiene_and_quality(issues, prompts=prompts, gold=gold, kb_rows=kb_rows)
    validate_negative_examples(issues, status_counts=status_counts)
    critical_issue_count = sum(1 for issue in issues if issue["severity"] == "critical")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    promotion_ready = critical_issue_count == 0 and warning_count == 0
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "vertical": vertical,
        "target_count": target_count,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_rows),
        "status_counts": status_counts,
        "output_format_counts": output_format_counts,
        "task_type_counts": task_type_counts,
        "critical_issue_count": critical_issue_count,
        "warning_count": warning_count,
        "issue_log": issues,
        "promotion_ready": promotion_ready,
        "promoted_output_dir": str(promoted_output_dir) if promoted_output_dir else None,
        "next_step": (
            "Promote the Airline 250 candidate and extend 250 generation to the next vertical."
            if promotion_ready
            else "Fix candidate review issues before promotion."
        ),
    }


def load_candidate(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        read_jsonl(Path(args.candidate_prompts)),
        read_jsonl(Path(args.candidate_gold)),
        read_jsonl(Path(args.candidate_kb)),
    )


def review_candidate(args: argparse.Namespace) -> dict[str, Any]:
    if args.vertical != DEFAULT_VERTICAL or int(args.target_count) != DEFAULT_TARGET_COUNT:
        raise RuntimeError("Phase 2A-9A currently supports only airline target-count 250.")
    prompts, gold, kb_rows = load_candidate(args)
    report = build_review_report(
        vertical=args.vertical,
        target_count=int(args.target_count),
        prompts=prompts,
        gold=gold,
        kb_rows=kb_rows,
    )
    write_json(Path(args.review_report), report)
    return {
        "phase": PHASE,
        "mode": "review_candidate",
        "vertical": args.vertical,
        "target_count": int(args.target_count),
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "promotion_ready": report["promotion_ready"],
        "review_report": str(args.review_report),
        "next_step": report["next_step"],
    }


def promote_if_clean(args: argparse.Namespace) -> dict[str, Any]:
    if args.vertical != DEFAULT_VERTICAL or int(args.target_count) != DEFAULT_TARGET_COUNT:
        raise RuntimeError("Phase 2A-9A currently supports only airline target-count 250.")
    prompts, gold, kb_rows = load_candidate(args)
    promoted_output_dir = Path(args.promoted_output_dir)
    report = build_review_report(
        vertical=args.vertical,
        target_count=int(args.target_count),
        prompts=prompts,
        gold=gold,
        kb_rows=kb_rows,
        promoted_output_dir=promoted_output_dir,
    )
    if not report["promotion_ready"]:
        write_json(Path(args.review_report), report)
        return {
            "phase": PHASE,
            "mode": "promote_if_clean",
            "vertical": args.vertical,
            "target_count": int(args.target_count),
            "critical_issue_count": report["critical_issue_count"],
            "warning_count": report["warning_count"],
            "promotion_ready": False,
            "promoted": False,
            "review_report": str(args.review_report),
            "next_step": report["next_step"],
        }
    promoted_output_dir.mkdir(parents=True, exist_ok=True)
    destination_prompts = promoted_output_dir / "airline_prompts_250.jsonl"
    destination_gold = promoted_output_dir / "airline_gold_250.jsonl"
    destination_kb = promoted_output_dir / "airline_kb_250.jsonl"
    shutil.copyfile(Path(args.candidate_prompts), destination_prompts)
    shutil.copyfile(Path(args.candidate_gold), destination_gold)
    shutil.copyfile(Path(args.candidate_kb), destination_kb)
    report["promoted_output_dir"] = str(promoted_output_dir)
    report["next_step"] = (
        "Promoted Airline 250 scale-up files. Extend reviewed 250 generation to "
        "healthcare, retail, research_ai, and finance."
    )
    write_json(Path(args.review_report), report)
    return {
        "phase": PHASE,
        "mode": "promote_if_clean",
        "vertical": args.vertical,
        "target_count": int(args.target_count),
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "promotion_ready": True,
        "promoted": True,
        "promoted_output_dir": str(promoted_output_dir),
        "review_report": str(args.review_report),
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-candidate", action="store_true")
    parser.add_argument("--promote-if-clean", action="store_true")
    parser.add_argument("--vertical", default=DEFAULT_VERTICAL)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    parser.add_argument("--candidate-prompts", type=Path, default=DEFAULT_CANDIDATE_PROMPTS)
    parser.add_argument("--candidate-gold", type=Path, default=DEFAULT_CANDIDATE_GOLD)
    parser.add_argument("--candidate-kb", type=Path, default=DEFAULT_CANDIDATE_KB)
    parser.add_argument("--review-report", type=Path, default=DEFAULT_REVIEW_REPORT)
    parser.add_argument("--promoted-output-dir", type=Path, default=DEFAULT_PROMOTED_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    mode_count = int(bool(args.review_candidate)) + int(bool(args.promote_if_clean))
    if mode_count != 1:
        parser.error("Choose exactly one mode: --review-candidate or --promote-if-clean.")
    try:
        if args.review_candidate:
            summary = review_candidate(args)
            exit_code = 0
        else:
            summary = promote_if_clean(args)
            exit_code = 0 if summary["promoted"] else 1
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
