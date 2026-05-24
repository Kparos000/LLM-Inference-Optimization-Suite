"""Promote the clean full Phase 2A 2,000-scale dataset.

This script copies already-generated, already-audited candidate files only. It
does not build RAG, retrieval indexes, embeddings, prompt assembly, model calls,
GPU runs, or benchmark inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-15"
DATASET_NAME = "phase2a_2000_full"
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
FILE_KINDS = ["prompts", "gold", "kb"]
TARGET_PER_VERTICAL = 2000

DEFAULT_QA_REPORT = Path("data/generated/phase2a/scaleup_reports/phase2a_2000_full_qa_report.json")
DEFAULT_PARTIAL_ROOT = Path("data/generated/phase2a/scaleup")
DEFAULT_GENERATED_ROOT = Path("data/generated/phase2a/scaleup")
DEFAULT_PROMOTED_ROOT = Path("data/scaleup_2000_full")
DEFAULT_PROMOTION_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_2000_full_promotion_report.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return parsed


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def count_jsonl(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def source_root_for_vertical(vertical: str, partial_root: Path, generated_root: Path) -> Path:
    _ = (vertical, partial_root)
    return generated_root


def source_file(root: Path, vertical: str, kind: str) -> Path:
    return root / vertical / f"{vertical}_{kind}_{TARGET_PER_VERTICAL}.jsonl"


def promoted_file(root: Path, vertical: str, kind: str) -> Path:
    return root / vertical / f"{vertical}_{kind}_{TARGET_PER_VERTICAL}.jsonl"


def validate_clean_qa_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing Phase 2A-15 full QA report: {path}. Run "
            "python scripts/phase2/audit_phase2a_scaleup_2000_full.py --run-audit"
        )
    qa_report = read_json(path)
    if qa_report.get("partial_dataset") is not False:
        raise RuntimeError("Phase 2A-15 QA report is not marked full dataset.")
    if qa_report.get("promotion_ready") is not True:
        raise RuntimeError("Phase 2A-15 QA report is not promotion_ready.")
    if int(qa_report.get("critical_issue_count", -1)) != 0:
        raise RuntimeError("Phase 2A-15 QA report has critical issues.")
    if int(qa_report.get("warning_count", -1)) != 0:
        raise RuntimeError("Phase 2A-15 QA report has warnings.")
    if list(qa_report.get("included_verticals", [])) != VERTICALS:
        raise RuntimeError("Phase 2A-15 QA report does not match all verticals.")
    return qa_report


def expected_source_files(partial_root: Path, generated_root: Path) -> list[Path]:
    paths: list[Path] = []
    for vertical in VERTICALS:
        root = source_root_for_vertical(vertical, partial_root, generated_root)
        paths.extend(source_file(root, vertical, kind) for kind in FILE_KINDS)
    return paths


def validate_source_files(partial_root: Path, generated_root: Path) -> None:
    missing = [
        path for path in expected_source_files(partial_root, generated_root) if not path.exists()
    ]
    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "Missing full 2,000-scale source file(s):\n"
            f"{missing_lines}\n"
            "Run the Phase 2A-14 generators and full QA before promotion."
        )


def copy_source_files(
    *,
    partial_root: Path,
    generated_root: Path,
    promoted_root: Path,
) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for vertical in VERTICALS:
        root = source_root_for_vertical(vertical, partial_root, generated_root)
        destination_dir = promoted_root / vertical
        destination_dir.mkdir(parents=True, exist_ok=True)
        for kind in FILE_KINDS:
            source = source_file(root, vertical, kind)
            destination = promoted_file(promoted_root, vertical, kind)
            shutil.copyfile(source, destination)
            copied.append(
                {
                    "vertical": vertical,
                    "kind": kind,
                    "source": str(source),
                    "destination": str(destination),
                }
            )
    return copied


def build_per_vertical_counts(promoted_root: Path) -> dict[str, dict[str, Any]]:
    per_vertical: dict[str, dict[str, Any]] = {}
    for vertical in VERTICALS:
        prompt_path = promoted_file(promoted_root, vertical, "prompts")
        gold_path = promoted_file(promoted_root, vertical, "gold")
        kb_path = promoted_file(promoted_root, vertical, "kb")
        per_vertical[vertical] = {
            "prompt_count": count_jsonl(prompt_path),
            "gold_count": count_jsonl(gold_path),
            "kb_count": count_jsonl(kb_path),
            "files": {
                "prompts": str(prompt_path),
                "gold": str(gold_path),
                "kb": str(kb_path),
            },
        }
    return per_vertical


def build_manifest(
    *,
    qa_report: dict[str, Any],
    qa_report_path: Path,
    partial_root: Path,
    generated_root: Path,
    promoted_root: Path,
) -> dict[str, Any]:
    per_vertical = build_per_vertical_counts(promoted_root)
    return {
        "phase": PHASE,
        "dataset_name": DATASET_NAME,
        "partial_dataset": False,
        "generated_at_utc": utc_now(),
        "verticals": VERTICALS,
        "total_prompt_count": sum(row["prompt_count"] for row in per_vertical.values()),
        "total_gold_count": sum(row["gold_count"] for row in per_vertical.values()),
        "total_kb_count": sum(row["kb_count"] for row in per_vertical.values()),
        "per_vertical": per_vertical,
        "partial_source_root": str(partial_root),
        "generated_source_root": str(generated_root),
        "promoted_root": str(promoted_root),
        "qa_report_path": str(qa_report_path),
        "quality_summary": {
            "critical_issue_count": int(qa_report.get("critical_issue_count", 0)),
            "warning_count": int(qa_report.get("warning_count", 0)),
            "promotion_ready": bool(qa_report.get("promotion_ready")),
        },
        "scaleup_notes": [
            "no RAG",
            "no inference",
            "no embeddings",
            "generated from Phase 2A deterministic local generators",
            "full five-vertical 2,000-scale checkpoint",
        ],
        "next_step": "Run comprehensive Phase 2A EDA before Phase 2B context engineering.",
    }


def scaleup_readme() -> str:
    return """# Phase 2A Full 2,000-Scale Dataset

This directory contains the promoted full Phase 2A 2,000-scale dataset,
`phase2a_2000_full`.

It contains 2,000 prompts per vertical and 10,000 prompts total across Airline,
Healthcare Admin, Retail, Finance, and Research AI.

## Layout

Each vertical has three JSONL files:

- `<vertical>_prompts_2000.jsonl`
- `<vertical>_gold_2000.jsonl`
- `<vertical>_kb_2000.jsonl`

The manifest is `phase2a_2000_full_manifest.json`.

## Scope

This checkpoint contains promoted deterministic data records only. It does not
include inference results, RAG indexes, retrieval artifacts, embeddings, prompt
assembly outputs, model calls, GPU runs, or benchmark logs.

## Next Step

Begin 2,000-per-vertical generator planning for the 10,000-record target.
"""


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    qa_report_path = Path(args.qa_report)
    partial_root = Path(args.partial_root)
    generated_root = Path(args.generated_root)
    promoted_root = Path(args.promoted_root)
    promotion_report_path = Path(args.promotion_report)

    qa_report = validate_clean_qa_report(qa_report_path)
    validate_source_files(partial_root, generated_root)
    copied_files = copy_source_files(
        partial_root=partial_root,
        generated_root=generated_root,
        promoted_root=promoted_root,
    )
    manifest = build_manifest(
        qa_report=qa_report,
        qa_report_path=qa_report_path,
        partial_root=partial_root,
        generated_root=generated_root,
        promoted_root=promoted_root,
    )
    manifest_path = promoted_root / "phase2a_2000_full_manifest.json"
    readme_path = promoted_root / "README.md"
    write_json(manifest_path, manifest)
    readme_path.write_text(scaleup_readme(), encoding="utf-8")

    report = {
        "phase": PHASE,
        "dataset_name": DATASET_NAME,
        "generated_at_utc": utc_now(),
        "copied_file_count": len(copied_files),
        "copied_files": copied_files,
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "qa_report_path": str(qa_report_path),
        "quality_summary": manifest["quality_summary"],
        "total_prompt_count": manifest["total_prompt_count"],
        "total_gold_count": manifest["total_gold_count"],
        "total_kb_count": manifest["total_kb_count"],
        "next_step": manifest["next_step"],
    }
    write_json(promotion_report_path, report)

    return {
        "mode": "promote",
        "phase": PHASE,
        "dataset_name": DATASET_NAME,
        "copied_file_count": len(copied_files),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "promotion_report": str(promotion_report_path),
        "total_prompt_count": manifest["total_prompt_count"],
        "total_gold_count": manifest["total_gold_count"],
        "total_kb_count": manifest["total_kb_count"],
        "promotion_ready": manifest["quality_summary"]["promotion_ready"],
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--qa-report", default=str(DEFAULT_QA_REPORT))
    parser.add_argument("--partial-root", default=str(DEFAULT_PARTIAL_ROOT))
    parser.add_argument("--generated-root", default=str(DEFAULT_GENERATED_ROOT))
    parser.add_argument("--promoted-root", default=str(DEFAULT_PROMOTED_ROOT))
    parser.add_argument("--promotion-report", default=str(DEFAULT_PROMOTION_REPORT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.promote:
        parser.error("Pass --promote to promote the full 2,000-scale dataset.")
    try:
        summary = run_promotion(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
