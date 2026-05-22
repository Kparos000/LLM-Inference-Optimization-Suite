"""Write the Phase 2A progressive scale-up planning report.

This script is planning-only. It does not generate scaled prompt datasets,
build RAG/retrieval indexes, create embeddings, assemble inference prompts,
call models, run GPUs, or execute benchmark inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-8"
DEFAULT_SCALEUP_PLAN = Path("data/sources/phase2a_scaleup_plan.json")
DEFAULT_QA_REPORT = Path("data/generated/phase2a/phase2a_cross_vertical_qa_report.json")
DEFAULT_OUTPUT_REPORT = Path("data/generated/phase2a/phase2a_scaleup_plan_report.json")
DEFAULT_OUTPUT_MATRIX_CSV = Path("data/generated/phase2a/phase2a_scaleup_matrix.csv")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return parsed


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertical",
        "checkpoint",
        "prompts",
        "total_prompts",
        "kb_min",
        "kb_max",
        "purpose",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_qa_readiness(qa_report_path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    blockers: list[str] = []
    if not qa_report_path.exists():
        warnings.append("Phase 2A-7 QA report is missing; run the cross-vertical audit first.")
        return {}, warnings, ["missing_phase2a_qa_report"]
    qa_report = read_json(qa_report_path)
    readiness = qa_report.get("scale_up_readiness", {})
    if not isinstance(readiness, dict):
        blockers.append("qa_report_missing_scale_up_readiness")
        return qa_report, warnings, blockers
    for vertical, status in readiness.items():
        if not isinstance(status, dict) or not status.get("ready_for_250_scale"):
            blockers.append(f"{vertical}_not_ready_for_250_scale")
    if qa_report.get("critical_issue_count", 0):
        blockers.append("qa_report_has_critical_issues")
    return qa_report, warnings, blockers


def checkpoint_prompt_targets(plan: dict[str, Any]) -> dict[str, int]:
    checkpoints = plan["checkpoints"]
    return {
        name: int(values["prompts_per_vertical"])
        for name, values in checkpoints.items()
        if name != "checkpoint_seed"
    }


def build_matrix_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoints = plan["checkpoints"]
    for vertical, strategy in plan["vertical_scale_strategy"].items():
        for checkpoint in ("checkpoint_250", "checkpoint_1000", "checkpoint_2000"):
            kb_key = checkpoint.replace("checkpoint_", "kb_target_")
            kb_range = strategy[kb_key]
            rows.append(
                {
                    "vertical": vertical,
                    "checkpoint": checkpoint,
                    "prompts": strategy[checkpoint],
                    "total_prompts": checkpoints[checkpoint]["total_prompts"],
                    "kb_min": kb_range["min"],
                    "kb_max": kb_range["max"],
                    "purpose": checkpoints[checkpoint]["purpose"],
                }
            )
    return rows


def estimated_kb_record_ranges(plan: dict[str, Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for checkpoint in ("250", "1000", "2000"):
        min_total = 0
        max_total = 0
        for strategy in plan["vertical_scale_strategy"].values():
            kb_range = strategy[f"kb_target_{checkpoint}"]
            min_total += int(kb_range["min"])
            max_total += int(kb_range["max"])
        totals[f"checkpoint_{checkpoint}"] = {"min": min_total, "max": max_total}
    return totals


def build_report(
    plan: dict[str, Any], qa_report: dict[str, Any], warnings: list[str], blockers: list[str]
) -> dict[str, Any]:
    all_ready = not blockers
    checkpoints = plan["checkpoints"]
    report = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "current_seed_status": checkpoints["checkpoint_seed"],
        "approved_checkpoints": checkpoints,
        "total_target_prompts": checkpoints["checkpoint_2000"]["total_prompts"],
        "total_target_gold_records": checkpoints["checkpoint_2000"]["total_prompts"],
        "estimated_kb_record_ranges": estimated_kb_record_ranges(plan),
        "per_vertical_targets": plan["vertical_scale_strategy"],
        "gold_review_subset_plan": plan["gold_strategy"]["gold_review_subset"],
        "deep_review_subset_plan": plan["gold_strategy"]["deep_review_subset"],
        "readiness_from_qa": qa_report.get("scale_up_readiness", {}),
        "blockers": blockers,
        "warnings": warnings,
        "recommend_generation": all_ready,
        "next_step": (
            "Proceed to Phase 2A-9 250-record scale-up generator foundation."
            if all_ready
            else "Resolve blockers and rerun Phase 2A-7 before generating checkpoint_250."
        ),
    }
    if not qa_report:
        report["next_step"] = "Run Phase 2A-7 cross-vertical QA audit before scale-up generation."
    return report


def write_report(args: argparse.Namespace) -> dict[str, Any]:
    plan = read_json(Path(args.scaleup_plan))
    qa_report, warnings, blockers = load_qa_readiness(Path(args.qa_report))
    report = build_report(plan, qa_report, warnings, blockers)
    matrix_rows = build_matrix_rows(plan)
    write_json(Path(args.output_report), report)
    write_matrix_csv(Path(args.output_matrix_csv), matrix_rows)
    return {
        "mode": "write_report",
        "phase": PHASE,
        "recommend_generation": report["recommend_generation"],
        "blocker_count": len(report["blockers"]),
        "warning_count": len(report["warnings"]),
        "total_target_prompts": report["total_target_prompts"],
        "total_target_gold_records": report["total_target_gold_records"],
        "output_report": str(args.output_report),
        "output_matrix_csv": str(args.output_matrix_csv),
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--scaleup-plan", type=Path, default=DEFAULT_SCALEUP_PLAN)
    parser.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--output-matrix-csv", type=Path, default=DEFAULT_OUTPUT_MATRIX_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.write_report:
        parser.error("Pass --write-report to write the Phase 2A scale-up planning report.")
    try:
        summary = write_report(args)
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
