"""Promote the clean Phase 2A 250-scale candidate dataset.

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

PHASE = "2A-11"
DATASET_NAME = "phase2a_250_scaleup"
VERTICALS = ["airline", "healthcare_admin", "retail", "research_ai", "finance"]
FILE_KINDS = ["prompts", "gold", "kb"]
DEFAULT_QA_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_250_cross_vertical_qa_report.json"
)
DEFAULT_GENERATED_ROOT = Path("data/generated/phase2a/scaleup")
DEFAULT_PROMOTED_ROOT = Path("data/scaleup")
DEFAULT_PROMOTION_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_250_promotion_report.json"
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
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def candidate_file(root: Path, vertical: str, kind: str) -> Path:
    return root / vertical / f"{vertical}_{kind}_250.jsonl"


def promoted_file(root: Path, vertical: str, kind: str) -> Path:
    return root / vertical / f"{vertical}_{kind}_250.jsonl"


def validate_clean_qa_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing Phase 2A-10 QA report: {path}. Run "
            "python scripts/phase2/audit_phase2a_scaleup_250.py --run-audit"
        )
    qa_report = read_json(path)
    if qa_report.get("promotion_ready") is not True:
        raise RuntimeError("Phase 2A-10 QA report is not promotion_ready.")
    if int(qa_report.get("critical_issue_count", -1)) != 0:
        raise RuntimeError("Phase 2A-10 QA report has critical issues.")
    if int(qa_report.get("warning_count", -1)) != 0:
        raise RuntimeError("Phase 2A-10 QA report has warnings.")
    return qa_report


def expected_candidate_files(generated_root: Path) -> list[Path]:
    return [
        candidate_file(generated_root, vertical, kind)
        for vertical in VERTICALS
        for kind in FILE_KINDS
    ]


def validate_candidate_files(generated_root: Path) -> None:
    missing = [path for path in expected_candidate_files(generated_root) if not path.exists()]
    if missing:
        missing_lines = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "Missing generated 250-scale candidate file(s):\n"
            f"{missing_lines}\n"
            "Run the Phase 2A-9 generator for each missing vertical before promotion."
        )


def copy_candidate_files(generated_root: Path, promoted_root: Path) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for vertical in VERTICALS:
        destination_dir = promoted_root / vertical
        destination_dir.mkdir(parents=True, exist_ok=True)
        for kind in FILE_KINDS:
            source = candidate_file(generated_root, vertical, kind)
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
    generated_root: Path,
    promoted_root: Path,
) -> dict[str, Any]:
    per_vertical = build_per_vertical_counts(promoted_root)
    return {
        "phase": PHASE,
        "dataset_name": DATASET_NAME,
        "generated_at_utc": utc_now(),
        "verticals": VERTICALS,
        "total_prompt_count": sum(row["prompt_count"] for row in per_vertical.values()),
        "total_gold_count": sum(row["gold_count"] for row in per_vertical.values()),
        "total_kb_count": sum(row["kb_count"] for row in per_vertical.values()),
        "per_vertical": per_vertical,
        "source_candidate_root": str(generated_root),
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
        ],
    }


def scaleup_readme() -> str:
    return """# Phase 2A 250-Scale Dataset

This directory contains the promoted Phase 2A 250-scale dataset,
`phase2a_250_scaleup`.

It is the first promoted scale-up checkpoint: 250 prompts per vertical and
1,250 prompts total across Airline, Healthcare Admin, Retail, Research AI, and
Finance.

## Layout

Each vertical has three JSONL files:

- `<vertical>_prompts_250.jsonl`
- `<vertical>_gold_250.jsonl`
- `<vertical>_kb_250.jsonl`

The manifest is `phase2a_250_manifest.json`.

## Scope

This is not the 1,000, 2,000, 4,000, or 5,000 per-vertical dataset. It contains
promoted deterministic data records only. It does not include inference results,
RAG indexes, retrieval artifacts, embeddings, prompt assembly outputs, model
calls, GPU runs, or benchmark logs.

## Next Step

The next planned work is extending the deterministic generators to the
1,000-per-vertical checkpoint after this 250-scale dataset is reviewed as the
baseline promoted set.
"""


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    qa_report_path = Path(args.qa_report)
    generated_root = Path(args.generated_root)
    promoted_root = Path(args.promoted_root)
    promotion_report_path = Path(args.promotion_report)

    qa_report = validate_clean_qa_report(qa_report_path)
    validate_candidate_files(generated_root)
    copied_files = copy_candidate_files(generated_root, promoted_root)

    manifest = build_manifest(
        qa_report=qa_report,
        qa_report_path=qa_report_path,
        generated_root=generated_root,
        promoted_root=promoted_root,
    )
    manifest_path = promoted_root / "phase2a_250_manifest.json"
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
        "next_step": "Begin 1,000-per-vertical generator extensions after review.",
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
    parser = argparse.ArgumentParser(description="Promote Phase 2A 250-scale candidates.")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--qa-report", default=str(DEFAULT_QA_REPORT))
    parser.add_argument("--generated-root", default=str(DEFAULT_GENERATED_ROOT))
    parser.add_argument("--promoted-root", default=str(DEFAULT_PROMOTED_ROOT))
    parser.add_argument("--promotion-report", default=str(DEFAULT_PROMOTION_REPORT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.promote:
        parser.error("Use --promote to promote the Phase 2A 250-scale dataset.")
    try:
        summary = run_promotion(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
