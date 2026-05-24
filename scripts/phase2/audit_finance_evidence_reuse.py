"""Audit Finance 250 evidence reuse before 1,000-scale generation.

This Phase 2A-12D audit reads the promoted Finance 250 prompt/gold/KB files,
checks evidence reuse and coverage, and writes local readiness reports. It does
not generate prompts, build RAG, retrieval indexes, embeddings, model calls, GPU
runs, or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-12D"

DEFAULT_PROMPTS_PATH = Path("data/scaleup/finance/finance_prompts_250.jsonl")
DEFAULT_GOLD_PATH = Path("data/scaleup/finance/finance_gold_250.jsonl")
DEFAULT_KB_PATH = Path("data/scaleup/finance/finance_kb_250.jsonl")
DEFAULT_OUTPUT_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/finance_evidence_reuse_audit_report.json"
)
DEFAULT_OUTPUT_CSV = Path(
    "data/generated/phase2a/scaleup_reports/finance_evidence_reuse_by_doc.csv"
)

HIGH_RISK_SHARE_THRESHOLD = 0.20
MEDIUM_RISK_SHARE_THRESHOLD = 0.10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object in {path} on line {line_number}")
        rows.append(parsed)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and item != ""]
    if value is not None and value != "":
        return [str(value)]
    return []


def required_doc_ids(row: dict[str, Any]) -> list[str]:
    ids = as_str_list(row.get("required_doc_ids"))
    if ids:
        return ids
    ids = as_str_list(row.get("required_evidence_ids"))
    if ids:
        return ids
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        ids = as_str_list(metadata.get("required_evidence_ids"))
        if ids:
            return ids
    return []


def reuse_risk_from_share(max_doc_reuse_share: float) -> str:
    if max_doc_reuse_share > HIGH_RISK_SHARE_THRESHOLD:
        return "high"
    if max_doc_reuse_share >= MEDIUM_RISK_SHARE_THRESHOLD:
        return "medium"
    return "low"


def counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def least_used(counter: Counter[str], *, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {"value": value, "count": count}
        for value, count in sorted(counter.items(), key=lambda item: (item[1], item[0]))[:limit]
    ]


def build_doc_reuse_rows(
    prompts: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    prompts_by_id = {str(row.get("prompt_id") or ""): row for row in prompts}
    prompt_ids_by_doc: dict[str, set[str]] = defaultdict(set)
    ticker_counts_by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    form_counts_by_doc: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts_by_doc: dict[str, Counter[str]] = defaultdict(Counter)

    for gold_row in gold:
        prompt_id = str(gold_row.get("prompt_id") or "")
        prompt = prompts_by_id.get(prompt_id, {})
        doc_ids = required_doc_ids(gold_row) or required_doc_ids(prompt)
        for doc_id in set(doc_ids):
            prompt_ids_by_doc[doc_id].add(prompt_id)
            ticker = str(prompt.get("ticker") or "unknown")
            filing_form = str(prompt.get("filing_form") or "unknown")
            task_type = str(prompt.get("task_type") or gold_row.get("task_type") or "unknown")
            ticker_counts_by_doc[doc_id][ticker] += 1
            form_counts_by_doc[doc_id][filing_form] += 1
            task_counts_by_doc[doc_id][task_type] += 1

    rows: list[dict[str, Any]] = []
    total_prompts = max(1, len(prompts))
    for doc_id, prompt_ids in prompt_ids_by_doc.items():
        prompt_count = len(prompt_ids)
        rows.append(
            {
                "doc_id": doc_id,
                "prompt_count": prompt_count,
                "prompt_share": round(prompt_count / total_prompts, 6),
                "sample_prompt_ids": ";".join(sorted(prompt_ids)[:10]),
                "top_tickers": ";".join(
                    f"{ticker}:{count}"
                    for ticker, count in ticker_counts_by_doc[doc_id].most_common(3)
                ),
                "top_filing_forms": ";".join(
                    f"{form}:{count}" for form, count in form_counts_by_doc[doc_id].most_common(3)
                ),
                "top_task_types": ";".join(
                    f"{task}:{count}" for task, count in task_counts_by_doc[doc_id].most_common(3)
                ),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["prompt_count"]), str(row["doc_id"])))


def build_recommendations(
    *,
    risk: str,
    max_doc_reuse_share: float,
    underused_tickers: list[dict[str, Any]],
    underused_forms: list[dict[str, Any]],
) -> list[str]:
    recommendations = [
        "Use the current Finance 250 evidence mix as the baseline for 1,000-scale planning.",
        "Track required_doc_ids during generation so no single SEC/XBRL evidence record dominates.",
        (
            "Keep investment advice, price targets, projections, and unsupported claims out of "
            "gold answers."
        ),
    ]
    if risk == "high":
        recommendations.insert(
            0,
            (
                "Reduce reuse of the top Finance evidence record before implementing "
                "1,000-scale generation."
            ),
        )
    elif risk == "medium":
        recommendations.insert(
            0,
            (
                "Proceed cautiously: reserve additional SEC/XBRL contexts for the most reused "
                "evidence family."
            ),
        )
    else:
        recommendations.insert(
            0,
            (
                "Evidence reuse concentration is low enough for Finance 1,000-scale generator "
                "implementation."
            ),
        )
    if max_doc_reuse_share >= MEDIUM_RISK_SHARE_THRESHOLD:
        recommendations.append(
            "Add per-document caps or diversify filing sections if reuse rises during "
            "1,000 generation."
        )
    if underused_tickers:
        recommendations.append(
            "Review ticker balance before 1,000 generation; least-used tickers are "
            + ", ".join(f"{item['value']} ({item['count']})" for item in underused_tickers[:3])
            + "."
        )
    if underused_forms:
        recommendations.append(
            "Review filing-form balance before 1,000 generation; least-used forms are "
            + ", ".join(f"{item['value']} ({item['count']})" for item in underused_forms[:3])
            + "."
        )
    return recommendations


def build_audit_report(
    *,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    output_csv: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    doc_reuse_rows = build_doc_reuse_rows(prompts, gold)
    max_doc_reuse_count = int(doc_reuse_rows[0]["prompt_count"]) if doc_reuse_rows else 0
    max_doc_reuse_share = round(max_doc_reuse_count / len(prompts), 6) if prompts else 0.0
    risk = reuse_risk_from_share(max_doc_reuse_share)
    ticker_counter = Counter(str(row.get("ticker") or "unknown") for row in prompts)
    filing_form_counter = Counter(str(row.get("filing_form") or "unknown") for row in prompts)
    task_type_counter = Counter(str(row.get("task_type") or "unknown") for row in prompts)
    overused_evidence_ids = [
        row for row in doc_reuse_rows if float(row["prompt_share"]) >= MEDIUM_RISK_SHARE_THRESHOLD
    ]
    underused_tickers = least_used(ticker_counter)
    underused_forms = least_used(filing_form_counter)
    ready = bool(prompts and gold and kb_rows and risk != "high")
    report = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "total_prompts": len(prompts),
        "total_gold": len(gold),
        "total_kb": len(kb_rows),
        "unique_required_doc_ids": len(doc_reuse_rows),
        "evidence_reuse_top_10": doc_reuse_rows[:10],
        "max_doc_reuse_count": max_doc_reuse_count,
        "max_doc_reuse_share": max_doc_reuse_share,
        "ticker_counts": counter_to_sorted_dict(ticker_counter),
        "filing_form_counts": counter_to_sorted_dict(filing_form_counter),
        "task_type_counts": counter_to_sorted_dict(task_type_counter),
        "overused_evidence_ids": overused_evidence_ids,
        "underused_companies_or_tickers": underused_tickers,
        "underused_filing_forms": underused_forms,
        "evidence_reuse_risk": risk,
        "ready_for_1000_finance_generation": ready,
        "recommendations": build_recommendations(
            risk=risk,
            max_doc_reuse_share=max_doc_reuse_share,
            underused_tickers=underused_tickers,
            underused_forms=underused_forms,
        ),
        "output_csv": str(output_csv),
    }
    return report, doc_reuse_rows


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    prompts = read_jsonl(Path(args.prompts))
    gold = read_jsonl(Path(args.gold))
    kb_rows = read_jsonl(Path(args.kb))
    output_report = Path(args.output_report)
    output_csv = Path(args.output_csv)
    report, doc_rows = build_audit_report(
        prompts=prompts,
        gold=gold,
        kb_rows=kb_rows,
        output_csv=output_csv,
    )
    write_json(output_report, report)
    write_csv(
        output_csv,
        doc_rows,
        [
            "doc_id",
            "prompt_count",
            "prompt_share",
            "sample_prompt_ids",
            "top_tickers",
            "top_filing_forms",
            "top_task_types",
        ],
    )
    return {
        "mode": "run_audit",
        "phase": PHASE,
        "total_prompts": report["total_prompts"],
        "total_gold": report["total_gold"],
        "total_kb": report["total_kb"],
        "unique_required_doc_ids": report["unique_required_doc_ids"],
        "max_doc_reuse_count": report["max_doc_reuse_count"],
        "max_doc_reuse_share": report["max_doc_reuse_share"],
        "evidence_reuse_risk": report["evidence_reuse_risk"],
        "ready_for_1000_finance_generation": report["ready_for_1000_finance_generation"],
        "output_report": str(output_report),
        "output_csv": str(output_csv),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-audit", action="store_true")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS_PATH))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD_PATH))
    parser.add_argument("--kb", default=str(DEFAULT_KB_PATH))
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_audit:
        parser.error("Use --run-audit to write the Finance evidence reuse audit.")
    try:
        summary = run_audit(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
