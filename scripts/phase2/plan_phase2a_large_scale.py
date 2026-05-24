"""Plan future Phase 2A 4,000/5,000 scale-up checkpoints.

This is planning scaffolding only. It does not generate records, build RAG,
create retrieval indexes, create embeddings, call models, run inference, or run
GPU experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-16A"
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
DEFAULT_PROMOTED_2000_ROOT = Path("data/scaleup_2000_full")
DEFAULT_OUTPUT_REPORT = Path("data/generated/phase2a/large_scale/phase2a_large_scale_plan.json")
DEFAULT_OUTPUT_MATRIX_CSV = Path(
    "data/generated/phase2a/large_scale/phase2a_large_scale_matrix.csv"
)

KB_TARGETS = {
    "airline": {
        "4000": {"min": 600, "max": 900},
        "5000": {"min": 800, "max": 1200},
    },
    "healthcare_admin": {
        "4000": {"min": 600, "max": 900},
        "5000": {"min": 800, "max": 1200},
    },
    "retail": {
        "4000": {"min": 2000, "max": 4000},
        "5000": {"min": 2500, "max": 5000},
    },
    "finance": {
        "4000": {"min": 2500, "max": 4500},
        "5000": {"min": 3500, "max": 6000},
    },
    "research_ai": {
        "4000": {"min": 1600, "max": 2800},
        "5000": {"min": 2000, "max": 3500},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object in {path} line {line_number}.")
        rows.append(parsed)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertical",
        "current_2000_prompt_count",
        "current_2000_gold_count",
        "current_2000_kb_count",
        "target_4000_additional_prompts_needed",
        "target_5000_additional_prompts_needed",
        "recommended_kb_target_4000",
        "recommended_kb_target_5000",
        "source_expansion_required_before_4000",
        "source_expansion_required_before_5000",
        "generator_implemented_for_4000",
        "generator_implemented_for_5000",
        "ready_for_4000_generator_implementation",
        "ready_for_5000_generator_implementation",
        "blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_research_ai_approved_papers() -> int:
    rows = read_jsonl_if_exists(Path("data/sources/research_ai_approved_papers.jsonl"))
    return sum(
        1
        for row in rows
        if str(row.get("approval_status") or "approved").lower() == "approved"
        and not bool(row.get("missing_pdf_or_section_text"))
    )


def count_research_ai_sections() -> int:
    return count_jsonl(Path("data/processed/research_ai/paper_sections_manifest.jsonl"))


def file_counts(root: Path, vertical: str) -> dict[str, int]:
    return {
        "prompts": count_jsonl(root / vertical / f"{vertical}_prompts_2000.jsonl"),
        "gold": count_jsonl(root / vertical / f"{vertical}_gold_2000.jsonl"),
        "kb": count_jsonl(root / vertical / f"{vertical}_kb_2000.jsonl"),
    }


def range_text(target: dict[str, int]) -> str:
    return f"{target['min']}-{target['max']}"


def vertical_notes(vertical: str, approved_count: int, section_count: int) -> list[str]:
    if vertical == "research_ai":
        return [
            (
                "Current source pool has 60 approved papers and 2,590 extracted sections "
                "when local processed artifacts are present."
            ),
            (
                f"Detected approved papers: {approved_count}; detected extracted sections: "
                f"{section_count}."
            ),
            "The promoted benchmark KB is a selected gold-linked subset.",
            (
                "Future RAG/context engineering should use the full retrieval corpus, "
                "not only promoted benchmark KB."
            ),
            "4,000-scale may be possible using more of the existing section pool.",
            (
                "5,000-scale may require nearly all suitable sections or more papers "
                "if EDA shows evidence reuse concentration."
            ),
        ]
    if vertical in {"airline", "healthcare_admin"}:
        return [
            (
                "Synthetic policy-bound vertical; source expansion is not expected "
                "before generator implementation."
            )
        ]
    if vertical == "retail":
        return [
            "Use EDA to confirm category/product evidence reuse before 4,000 and 5,000 scale-up."
        ]
    return ["Use EDA to confirm SEC/XBRL evidence reuse before 4,000 and 5,000 scale-up."]


def build_plan(promoted_root: Path) -> dict[str, Any]:
    approved_count = count_research_ai_approved_papers()
    section_count = count_research_ai_sections()
    per_vertical: dict[str, dict[str, Any]] = {}
    matrix_rows: list[dict[str, Any]] = []

    for vertical in VERTICALS:
        counts = file_counts(promoted_root, vertical)
        blockers: list[str] = []
        source_expansion_4000 = False
        source_expansion_5000 = vertical in {"retail", "finance"}
        if vertical == "research_ai" and section_count and section_count < 2000:
            source_expansion_4000 = True
            source_expansion_5000 = True
            blockers.append("research_ai_section_pool_below_large_scale_guidance")
        elif vertical == "research_ai":
            source_expansion_5000 = False

        ready_4000 = (
            counts["prompts"] == 2000 and counts["gold"] == 2000 and not source_expansion_4000
        )
        ready_5000 = (
            counts["prompts"] == 2000 and counts["gold"] == 2000 and not source_expansion_5000
        )
        row = {
            "current_2000_prompt_count": counts["prompts"],
            "current_2000_gold_count": counts["gold"],
            "current_2000_kb_count": counts["kb"],
            "target_4000_additional_prompts_needed": 2000,
            "target_5000_additional_prompts_needed": 3000,
            "recommended_kb_target_4000": KB_TARGETS[vertical]["4000"],
            "recommended_kb_target_5000": KB_TARGETS[vertical]["5000"],
            "source_expansion_required_before_4000": source_expansion_4000,
            "source_expansion_required_before_5000": source_expansion_5000,
            "generator_implemented_for_4000": False,
            "generator_implemented_for_5000": False,
            "ready_for_4000_generator_implementation": ready_4000,
            "ready_for_5000_generator_implementation": ready_5000,
            "blockers": blockers,
            "notes": vertical_notes(vertical, approved_count, section_count),
        }
        per_vertical[vertical] = row
        matrix_rows.append(
            {
                "vertical": vertical,
                "current_2000_prompt_count": counts["prompts"],
                "current_2000_gold_count": counts["gold"],
                "current_2000_kb_count": counts["kb"],
                "target_4000_additional_prompts_needed": 2000,
                "target_5000_additional_prompts_needed": 3000,
                "recommended_kb_target_4000": range_text(KB_TARGETS[vertical]["4000"]),
                "recommended_kb_target_5000": range_text(KB_TARGETS[vertical]["5000"]),
                "source_expansion_required_before_4000": source_expansion_4000,
                "source_expansion_required_before_5000": source_expansion_5000,
                "generator_implemented_for_4000": False,
                "generator_implemented_for_5000": False,
                "ready_for_4000_generator_implementation": ready_4000,
                "ready_for_5000_generator_implementation": ready_5000,
                "blockers": ";".join(blockers),
            }
        )

    total_current_prompts = sum(row["current_2000_prompt_count"] for row in per_vertical.values())
    total_current_gold = sum(row["current_2000_gold_count"] for row in per_vertical.values())
    total_current_kb = sum(row["current_2000_kb_count"] for row in per_vertical.values())
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "current_checkpoint": "checkpoint_2000",
        "future_checkpoints": {
            "checkpoint_4000": {
                "prompts_per_vertical": 4000,
                "total_prompts": 20000,
                "purpose": "GPU stress tier",
            },
            "checkpoint_5000": {
                "prompts_per_vertical": 5000,
                "total_prompts": 25000,
                "purpose": "max expanded benchmark capacity",
            },
        },
        "total_current_prompts": total_current_prompts,
        "total_current_gold": total_current_gold,
        "total_current_kb": total_current_kb,
        "research_ai_source_pool": {
            "approved_paper_count_guidance": 60,
            "extracted_section_count_guidance": 2590,
            "detected_approved_paper_count": approved_count,
            "detected_extracted_section_count": section_count,
            "benchmark_kb_is_selected_subset": True,
            "full_retrieval_corpus_needed_for_phase2b": True,
        },
        "per_vertical": per_vertical,
        "matrix_rows": matrix_rows,
        "can_plan_4000": True,
        "can_plan_5000": True,
        "should_generate_now": False,
        "next_step": (
            "Run EDA and Phase 2B corpus checks before generating 4,000/5,000 stress-tier datasets."
        ),
    }


def write_report(args: argparse.Namespace) -> dict[str, Any]:
    report = build_plan(Path(args.promoted_2000_root))
    write_json(Path(args.output_report), report)
    write_matrix(Path(args.output_matrix_csv), report["matrix_rows"])
    return {
        "phase": PHASE,
        "mode": "write_report",
        "current_checkpoint": report["current_checkpoint"],
        "future_checkpoints": report["future_checkpoints"],
        "total_current_prompts": report["total_current_prompts"],
        "total_current_gold": report["total_current_gold"],
        "total_current_kb": report["total_current_kb"],
        "can_plan_4000": report["can_plan_4000"],
        "can_plan_5000": report["can_plan_5000"],
        "should_generate_now": report["should_generate_now"],
        "report_path": str(args.output_report),
        "matrix_csv_path": str(args.output_matrix_csv),
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--promoted-2000-root", default=str(DEFAULT_PROMOTED_2000_ROOT))
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-matrix-csv", default=str(DEFAULT_OUTPUT_MATRIX_CSV))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.write_report:
        parser.error("Pass --write-report to write the large-scale plan.")
    try:
        summary = write_report(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
