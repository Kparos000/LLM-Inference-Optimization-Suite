"""Plan Phase 2A 1,000-per-vertical scale-up readiness.

This script writes planning/readiness artifacts only. It does not generate
records, build RAG, retrieval indexes, embeddings, prompt assembly, model calls,
GPU runs, or benchmark inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-12A"
TARGET_PER_VERTICAL = 1000
TOTAL_TARGET_PROMPTS = 5000
PREVIOUS_CHECKPOINT = "checkpoint_250"
TARGET_CHECKPOINT = "checkpoint_1000"

DEFAULT_SCALEUP_PLAN = Path("data/sources/phase2a_scaleup_plan.json")
DEFAULT_PROMOTED_250_MANIFEST = Path("data/scaleup/phase2a_250_manifest.json")
DEFAULT_OUTPUT_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_1000_scaleup_readiness_report.json"
)
DEFAULT_OUTPUT_MATRIX_CSV = Path(
    "data/generated/phase2a/scaleup_reports/phase2a_1000_scaleup_matrix.csv"
)
DEFAULT_RETAIL_MULTICATEGORY_REPORT = Path(
    "data/generated/retail/multicategory/retail_multicategory_source_report.json"
)
DEFAULT_RESEARCH_AI_EXPANSION_REPORT = Path(
    "data/generated/research_ai/research_ai_40_paper_expansion_report.json"
)
DEFAULT_FINANCE_EVIDENCE_REUSE_REPORT = Path(
    "data/generated/phase2a/scaleup_reports/finance_evidence_reuse_audit_report.json"
)

RECOMMENDED_STRATEGIES = {
    "airline": "Extend deterministic synthetic policy/ticket generator.",
    "healthcare_admin": "Extend deterministic synthetic admin generator.",
    "retail": "Prepare a larger sampled review/metadata set and category expansion plan.",
    "research_ai": (
        "Expand to about 40 papers or increase section coverage before 1,000-scale generation."
    ),
    "finance": (
        "Use the current 8-company SEC/XBRL corpus first, with evidence-reuse checks "
        "to avoid repetitive prompts."
    ),
}
SOURCE_EXPANSION_REQUIRED = {
    "airline": False,
    "healthcare_admin": False,
    "retail": True,
    "research_ai": True,
    "finance": False,
}


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


def write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertical",
        "current_250_prompts",
        "target_1000_prompts",
        "additional_prompts_needed",
        "current_kb_count",
        "target_kb_min",
        "target_kb_max",
        "source_ready_for_1000",
        "source_expansion_required",
        "source_expansion_ready",
        "generator_implemented_for_1000",
        "generator_implementation_required",
        "ready_for_1000_generator_implementation",
        "ready_for_1000_generation",
        "recommended_generator_strategy",
        "blockers",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_promoted_250_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(
            f"Missing promoted 250 manifest: {path}. Run Phase 2A-11 promotion first."
        )
    manifest = read_json(path)
    quality = manifest.get("quality_summary", {})
    if manifest.get("dataset_name") != "phase2a_250_scaleup":
        raise RuntimeError("Promoted manifest does not describe phase2a_250_scaleup.")
    if manifest.get("total_prompt_count") != 1250 or manifest.get("total_gold_count") != 1250:
        raise RuntimeError("Promoted 250 manifest does not have 1,250 prompts and gold records.")
    if not isinstance(quality, dict) or quality.get("promotion_ready") is not True:
        raise RuntimeError("Promoted 250 manifest is not marked promotion_ready.")
    per_vertical = manifest.get("per_vertical")
    if not isinstance(per_vertical, dict):
        raise RuntimeError("Promoted 250 manifest is missing per_vertical counts.")
    for vertical, metrics in per_vertical.items():
        if not isinstance(metrics, dict):
            raise RuntimeError(f"Promoted 250 manifest has malformed metrics for {vertical}.")
        if metrics.get("prompt_count") != 250 or metrics.get("gold_count") != 250:
            raise RuntimeError(f"Promoted 250 manifest is incomplete for {vertical}.")
        if int(metrics.get("kb_count", 0)) <= 0:
            raise RuntimeError(f"Promoted 250 manifest has no KB records for {vertical}.")
    return manifest


def target_kb_range(plan: dict[str, Any], vertical: str) -> dict[str, int]:
    strategies = plan.get("vertical_scale_strategy", {})
    if not isinstance(strategies, dict):
        return {"min": 0, "max": 0}
    vertical_plan = strategies.get(vertical, {})
    if not isinstance(vertical_plan, dict):
        return {"min": 0, "max": 0}
    raw_range = vertical_plan.get("kb_target_1000", {})
    if not isinstance(raw_range, dict):
        return {"min": 0, "max": 0}
    return {
        "min": int(raw_range.get("min", 0)),
        "max": int(raw_range.get("max", 0)),
    }


def source_requirements(plan: dict[str, Any], vertical: str) -> list[str]:
    strategies = plan.get("vertical_scale_strategy", {})
    if not isinstance(strategies, dict):
        return []
    vertical_plan = strategies.get(vertical, {})
    if not isinstance(vertical_plan, dict):
        return []
    raw_requirements = vertical_plan.get("source_expansion", [])
    if not isinstance(raw_requirements, list):
        return []
    return [str(item) for item in raw_requirements]


def retail_source_expansion_ready(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    return bool(report.get("retail_ready_for_1000_source_expansion"))


def research_ai_source_expansion_ready(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    return bool(report.get("expansion_ready_for_1000"))


def research_ai_missing_requirements(report: dict[str, Any] | None) -> list[str]:
    if not isinstance(report, dict):
        return [
            "missing_research_ai_40_paper_expansion_report",
            "additional_approved_papers_needed:unknown",
            "section_coverage_below_target_min:unknown",
        ]
    missing = report.get("missing_requirements", [])
    if not isinstance(missing, list):
        return []
    return [str(item) for item in missing]


def finance_evidence_reuse_ready(report: dict[str, Any] | None) -> bool:
    if not isinstance(report, dict):
        return False
    return bool(report.get("ready_for_1000_finance_generation"))


def finance_evidence_reuse_risk(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict):
        return "missing"
    return str(report.get("evidence_reuse_risk") or "unknown")


def vertical_blockers(
    vertical: str,
    source_expansion_required: bool,
    source_expansion_ready: bool,
    *,
    finance_evidence_reuse_report: dict[str, Any] | None = None,
) -> list[str]:
    blockers: list[str] = []
    if source_expansion_required and not source_expansion_ready:
        blockers.append("source_expansion_required_before_1000_generation")
    if vertical == "research_ai" and not source_expansion_ready:
        blockers.append("expand_research_ai_to_40_papers_or_equivalent_section_coverage")
    if vertical == "retail" and not source_expansion_ready:
        blockers.append("expand_retail_review_metadata_sample_and_categories")
    if (
        vertical == "finance"
        and finance_evidence_reuse_risk(finance_evidence_reuse_report) == "high"
    ):
        blockers.append("finance_evidence_reuse_high_risk")
    return blockers


def build_per_vertical_readiness(
    *,
    scaleup_plan: dict[str, Any],
    promoted_manifest: dict[str, Any],
    retail_multicategory_report: dict[str, Any] | None = None,
    research_ai_expansion_report: dict[str, Any] | None = None,
    finance_evidence_reuse_report: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    per_vertical_manifest = promoted_manifest["per_vertical"]
    readiness: dict[str, dict[str, Any]] = {}
    for vertical in promoted_manifest["verticals"]:
        metrics = per_vertical_manifest[vertical]
        source_required = SOURCE_EXPANSION_REQUIRED.get(vertical, True)
        if vertical == "retail":
            source_ready = retail_source_expansion_ready(retail_multicategory_report)
        elif vertical == "research_ai":
            source_ready = research_ai_source_expansion_ready(research_ai_expansion_report)
        else:
            source_ready = not source_required
        blockers = vertical_blockers(
            vertical,
            source_required,
            source_ready,
            finance_evidence_reuse_report=finance_evidence_reuse_report,
        )
        kb_range = target_kb_range(scaleup_plan, vertical)
        source_ready_for_1000 = not blockers
        generator_implemented_for_1000 = False
        generator_required = True
        readiness[vertical] = {
            "current_250_prompts": int(metrics["prompt_count"]),
            "target_1000_prompts": TARGET_PER_VERTICAL,
            "additional_prompts_needed": TARGET_PER_VERTICAL - int(metrics["prompt_count"]),
            "current_kb_count": int(metrics["kb_count"]),
            "target_kb_range": kb_range,
            "source_ready_for_1000": source_ready_for_1000,
            "source_expansion_required": source_required,
            "source_expansion_ready": source_ready,
            "source_expansion_missing_requirements": (
                research_ai_missing_requirements(research_ai_expansion_report)
                if vertical == "research_ai" and not source_ready
                else []
            ),
            "generator_implemented_for_1000": generator_implemented_for_1000,
            "generator_implementation_required": generator_required,
            "ready_for_1000_generator_implementation": source_ready_for_1000,
            "recommended_generator_strategy": RECOMMENDED_STRATEGIES[vertical],
            "source_requirements": source_requirements(scaleup_plan, vertical),
            "evidence_reuse_audit_ready": (
                finance_evidence_reuse_ready(finance_evidence_reuse_report)
                if vertical == "finance"
                else None
            ),
            "evidence_reuse_risk": (
                finance_evidence_reuse_risk(finance_evidence_reuse_report)
                if vertical == "finance"
                else None
            ),
            "blockers": blockers,
            "ready_for_1000_generation": (source_ready_for_1000 and generator_implemented_for_1000),
        }
    return readiness


def build_review_subset_plan() -> dict[str, Any]:
    return {
        "gold_review_subset_target": "500 to 1000",
        "deep_review_subset_target": "150 to 300",
        "stratify_by": [
            "vertical",
            "task_type",
            "status",
            "output_format",
            "difficulty",
            "evidence_type",
        ],
    }


def next_step_for_blockers(blockers: list[str]) -> str:
    if blockers:
        blocked_verticals = {blocker.split(":", maxsplit=1)[0] for blocker in blockers}
        if blocked_verticals == {"research_ai"}:
            return (
                "Implement 1,000 generators for source-ready verticals while resolving "
                "Research AI source expansion."
            )
        return (
            "Implement 1,000 generators for source-ready verticals while resolving "
            "remaining source readiness blockers."
        )
    return "Implement 1,000 generators for source-ready verticals."


def build_report(
    *,
    scaleup_plan: dict[str, Any],
    promoted_manifest: dict[str, Any],
    promoted_manifest_path: Path,
    retail_multicategory_report: dict[str, Any] | None = None,
    retail_multicategory_report_path: Path | None = None,
    research_ai_expansion_report: dict[str, Any] | None = None,
    research_ai_expansion_report_path: Path | None = None,
    finance_evidence_reuse_report: dict[str, Any] | None = None,
    finance_evidence_reuse_report_path: Path | None = None,
) -> dict[str, Any]:
    per_vertical = build_per_vertical_readiness(
        scaleup_plan=scaleup_plan,
        promoted_manifest=promoted_manifest,
        retail_multicategory_report=retail_multicategory_report,
        research_ai_expansion_report=research_ai_expansion_report,
        finance_evidence_reuse_report=finance_evidence_reuse_report,
    )
    source_expansion_requirements = {
        vertical: metrics["source_requirements"] for vertical, metrics in per_vertical.items()
    }
    kb_expansion_targets = {
        vertical: metrics["target_kb_range"] for vertical, metrics in per_vertical.items()
    }
    blockers = [
        f"{vertical}:{blocker}"
        for vertical, metrics in per_vertical.items()
        for blocker in metrics["blockers"]
    ]
    source_ready_verticals = [
        vertical for vertical, metrics in per_vertical.items() if metrics["source_ready_for_1000"]
    ]
    generator_implementation_ready_verticals = [
        vertical
        for vertical, metrics in per_vertical.items()
        if metrics["ready_for_1000_generator_implementation"]
    ]
    blocked_verticals = [
        vertical for vertical, metrics in per_vertical.items() if metrics["blockers"]
    ]
    generation_ready_verticals = [
        vertical
        for vertical, metrics in per_vertical.items()
        if metrics["ready_for_1000_generation"]
    ]
    warnings = ["full_5000_generation_should_wait_for_blocker_resolution"]
    if not finance_evidence_reuse_ready(finance_evidence_reuse_report):
        warnings.insert(0, "finance:evidence_reuse_audit_required_before_generation")
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "target_per_vertical": TARGET_PER_VERTICAL,
        "total_target_prompts": TOTAL_TARGET_PROMPTS,
        "previous_checkpoint": PREVIOUS_CHECKPOINT,
        "target_checkpoint": TARGET_CHECKPOINT,
        "required_previous_manifest": str(promoted_manifest_path),
        "promoted_250_found": True,
        "retail_multicategory_source_report_path": (
            str(retail_multicategory_report_path) if retail_multicategory_report_path else None
        ),
        "retail_source_expansion_ready": retail_source_expansion_ready(retail_multicategory_report),
        "research_ai_expansion_report_path": (
            str(research_ai_expansion_report_path) if research_ai_expansion_report_path else None
        ),
        "research_ai_source_expansion_ready": research_ai_source_expansion_ready(
            research_ai_expansion_report
        ),
        "research_ai_missing_requirements": research_ai_missing_requirements(
            research_ai_expansion_report
        )
        if not research_ai_source_expansion_ready(research_ai_expansion_report)
        else [],
        "finance_evidence_reuse_audit_report_path": (
            str(finance_evidence_reuse_report_path) if finance_evidence_reuse_report_path else None
        ),
        "finance_evidence_reuse_audit_ready": finance_evidence_reuse_ready(
            finance_evidence_reuse_report
        ),
        "finance_evidence_reuse_risk": finance_evidence_reuse_risk(finance_evidence_reuse_report),
        "per_vertical_readiness": per_vertical,
        "source_ready_verticals": source_ready_verticals,
        "generator_implementation_ready_verticals": generator_implementation_ready_verticals,
        "blocked_verticals": blocked_verticals,
        "generation_ready_verticals": generation_ready_verticals,
        "can_start_1000_generator_implementation": bool(source_ready_verticals),
        "source_expansion_requirements": source_expansion_requirements,
        "kb_expansion_targets": kb_expansion_targets,
        "gold_generation_strategy": {
            "all_prompts_require_gold": True,
            "answerable_records_require_evidence_ids": True,
            "negative_records_require_must_not_include": True,
            "reuse_phase2a_distribution_contracts": True,
        },
        "review_subset_plan": build_review_subset_plan(),
        "blockers": blockers,
        "warnings": warnings,
        "recommend_generation": bool(generation_ready_verticals),
        "next_step": next_step_for_blockers(blockers),
    }


def matrix_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vertical, metrics in report["per_vertical_readiness"].items():
        kb_range = metrics["target_kb_range"]
        rows.append(
            {
                "vertical": vertical,
                "current_250_prompts": metrics["current_250_prompts"],
                "target_1000_prompts": metrics["target_1000_prompts"],
                "additional_prompts_needed": metrics["additional_prompts_needed"],
                "current_kb_count": metrics["current_kb_count"],
                "target_kb_min": kb_range["min"],
                "target_kb_max": kb_range["max"],
                "source_ready_for_1000": metrics["source_ready_for_1000"],
                "source_expansion_required": metrics["source_expansion_required"],
                "source_expansion_ready": metrics["source_expansion_ready"],
                "generator_implemented_for_1000": metrics["generator_implemented_for_1000"],
                "generator_implementation_required": metrics["generator_implementation_required"],
                "ready_for_1000_generator_implementation": metrics[
                    "ready_for_1000_generator_implementation"
                ],
                "ready_for_1000_generation": metrics["ready_for_1000_generation"],
                "recommended_generator_strategy": metrics["recommended_generator_strategy"],
                "blockers": ";".join(metrics["blockers"]),
            }
        )
    return rows


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    scaleup_plan_path = Path(args.scaleup_plan)
    promoted_manifest_path = Path(args.promoted_250_manifest)
    if not scaleup_plan_path.exists():
        raise RuntimeError(f"Missing scale-up plan: {scaleup_plan_path}")
    scaleup_plan = read_json(scaleup_plan_path)
    promoted_manifest = require_promoted_250_manifest(promoted_manifest_path)
    retail_report_path = Path(args.retail_multicategory_report)
    retail_report = read_json(retail_report_path) if retail_report_path.exists() else None
    research_ai_report_path = Path(args.research_ai_expansion_report)
    research_ai_report = (
        read_json(research_ai_report_path) if research_ai_report_path.exists() else None
    )
    finance_report_path = Path(args.finance_evidence_reuse_report)
    finance_report = read_json(finance_report_path) if finance_report_path.exists() else None
    report = build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=promoted_manifest,
        promoted_manifest_path=promoted_manifest_path,
        retail_multicategory_report=retail_report,
        retail_multicategory_report_path=retail_report_path if retail_report else None,
        research_ai_expansion_report=research_ai_report,
        research_ai_expansion_report_path=(research_ai_report_path if research_ai_report else None),
        finance_evidence_reuse_report=finance_report,
        finance_evidence_reuse_report_path=finance_report_path if finance_report else None,
    )
    write_json(Path(args.output_report), report)
    write_matrix_csv(Path(args.output_matrix_csv), matrix_rows(report))
    return {
        "mode": "write_report",
        "phase": PHASE,
        "target_per_vertical": TARGET_PER_VERTICAL,
        "total_target_prompts": TOTAL_TARGET_PROMPTS,
        "previous_checkpoint": PREVIOUS_CHECKPOINT,
        "promoted_250_found": True,
        "recommend_generation": report["recommend_generation"],
        "blocker_count": len(report["blockers"]),
        "warning_count": len(report["warnings"]),
        "output_report": str(args.output_report),
        "output_matrix_csv": str(args.output_matrix_csv),
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Phase 2A 1,000-scale readiness.")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--scaleup-plan", default=str(DEFAULT_SCALEUP_PLAN))
    parser.add_argument("--promoted-250-manifest", default=str(DEFAULT_PROMOTED_250_MANIFEST))
    parser.add_argument(
        "--retail-multicategory-report", default=str(DEFAULT_RETAIL_MULTICATEGORY_REPORT)
    )
    parser.add_argument(
        "--research-ai-expansion-report", default=str(DEFAULT_RESEARCH_AI_EXPANSION_REPORT)
    )
    parser.add_argument(
        "--finance-evidence-reuse-report", default=str(DEFAULT_FINANCE_EVIDENCE_REUSE_REPORT)
    )
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--output-matrix-csv", default=str(DEFAULT_OUTPUT_MATRIX_CSV))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.write_report:
        parser.error("Use --write-report to write the Phase 2A-12A readiness report.")
    try:
        summary = run_plan(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
