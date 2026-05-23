"""Generate Phase 2A scale-up planning artifacts and local pilot candidates.

This script is data-generation scaffolding only. It does not build RAG,
retrieval indexes, embeddings, prompt assembly, model calls, GPU runs, or
benchmark inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-9"
GENERATOR_NAME = "phase2a_9_scaleup_local_candidates"

DEFAULT_SCALEUP_PLAN = Path("data/sources/phase2a_scaleup_plan.json")
DEFAULT_QA_REPORT = Path("data/generated/phase2a/phase2a_cross_vertical_qa_report.json")
DEFAULT_OUTPUT_DIR = Path("data/generated/phase2a/scaleup")
DEFAULT_REPORT_DIR = Path("data/generated/phase2a/scaleup_reports")

VERTICALS = ["finance", "airline", "healthcare_admin", "research_ai", "retail"]
SUPPORTED_TARGETS = [250, 1000, 2000, 4000, 5000]
TARGET_TO_CHECKPOINT = {
    250: "checkpoint_250",
    1000: "checkpoint_1000",
    2000: "checkpoint_2000",
    4000: "checkpoint_4000",
    5000: "checkpoint_5000",
}
IMPLEMENTED_GENERATION_TARGETS = {
    "finance": {250},
    "airline": {250},
    "healthcare_admin": {250},
    "research_ai": {250},
    "retail": {250},
}
VERTICAL_FILES: dict[str, dict[str, Path]] = {
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

STATUS_DISTRIBUTION_BASIS_POINTS: dict[str, dict[str, int]] = {
    "finance": {"answer": 9200, "insufficient_evidence": 400, "escalate": 400},
    "airline": {"answer": 9000, "escalate": 800, "spam_or_fraud": 200},
    "healthcare_admin": {
        "answer": 8800,
        "escalate": 800,
        "safety_boundary": 200,
        "spam_or_fraud": 100,
        "out_of_scope": 100,
    },
    "research_ai": {
        "answer": 9000,
        "insufficient_evidence": 400,
        "escalate": 400,
        "out_of_scope": 200,
    },
    "retail": {
        "answer": 8880,
        "insufficient_evidence": 360,
        "escalate": 360,
        "spam_or_low_quality": 280,
        "out_of_scope": 120,
    },
}

TASK_DISTRIBUTION_250: dict[str, dict[str, int]] = {
    "finance": {
        "answer_grounded": 95,
        "extract_structured": 45,
        "compare_filings": 35,
        "calculation": 35,
        "escalation_response": 20,
        "evidence_citation_lookup": 20,
    },
    "airline": {
        "policy_lookup": 135,
        "answer_grounded": 55,
        "extract_structured": 25,
        "compare_options": 10,
        "escalation_response": 20,
        "quality_boundary": 5,
    },
    "healthcare_admin": {
        "answer_grounded": 120,
        "policy_reasoning": 55,
        "extract_structured": 30,
        "escalation_response": 25,
        "safety_boundary": 10,
        "quality_boundary": 10,
    },
    "research_ai": {
        "answer_grounded": 90,
        "paper_method": 45,
        "results_evaluation": 35,
        "extract_structured": 30,
        "compare_papers": 25,
        "literature_table": 15,
        "escalation_response": 10,
    },
    "retail": {
        "answer_grounded": 95,
        "issue_identification": 45,
        "compare_products": 25,
        "extract_structured": 35,
        "policy_reasoning": 30,
        "quality_boundary": 10,
        "escalation_response": 10,
    },
}

OUTPUT_FORMAT_BASIS_POINTS: dict[str, dict[str, int]] = {
    "finance": {"text": 6200, "json": 2000, "markdown_table": 1800},
    "airline": {"text": 7600, "json": 1400, "markdown_table": 1000},
    "healthcare_admin": {"text": 7800, "json": 1400, "markdown_table": 800},
    "research_ai": {"text": 7200, "json": 1400, "markdown_table": 1400},
    "retail": {"text": 7400, "json": 1600, "markdown_table": 1000},
}

DIFFICULTY_BASIS_POINTS = {"easy": 3200, "medium": 5200, "hard": 1600}

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
LINGUISTIC_VARIATION_THRESHOLD = 0.60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return parsed


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object in {path} line {line_number}.")
        rows.append(parsed)
    return rows


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


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


def choose_phrase_variant(index: int, variants: list[str]) -> str:
    if not variants:
        raise ValueError("At least one phrase variant is required.")
    return variants[index % len(variants)]


def dynamic_question_values(prompt: dict[str, Any]) -> list[str]:
    values: list[str] = []
    direct_fields = [
        "airline",
        "department",
        "expected_queue",
        "issue_type",
        "company",
        "filing_form",
        "fiscal_period",
        "fiscal_year",
        "product_id",
        "product_title",
        "prompt_id",
        "route",
        "support_type",
        "ticker",
        "ticket_id",
        "travel_type",
    ]
    for field in direct_fields:
        value = prompt.get(field)
        if isinstance(value, str) and len(value) >= 4:
            values.append(value)
            values.append(value.replace("_", " "))

    metadata = prompt.get("metadata")
    if isinstance(metadata, dict):
        for field in [
            "prompt_category",
            "source_titles",
            "source_parent_asins",
            "topics",
        ]:
            value = metadata.get(field)
            if isinstance(value, str):
                values.append(value)
                values.append(value.replace("_", " "))
            elif isinstance(value, list):
                values.extend(str(item) for item in value if len(str(item)) >= 4)

    for field in [
        "required_chunk_ids",
        "required_citations",
        "required_doc_ids",
        "required_evidence_ids",
        "required_paper_ids",
        "source_parent_asins",
        "source_paper_ids",
        "source_product_ids",
    ]:
        value = prompt.get(field)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if len(str(item)) >= 4)

    return sorted(set(values), key=len, reverse=True)


def normalized_question_template(prompt: dict[str, Any]) -> str:
    question = str(prompt.get("question") or prompt.get("issue") or "")
    normalized = question.lower()
    for value in dynamic_question_values(prompt):
        lowered = value.lower()
        if len(lowered) >= 4:
            normalized = re.sub(re.escape(lowered), "<value>", normalized)

    normalized = re.sub(r"\b[a-z]{3}-[a-z]{3}\b", "<route>", normalized)
    normalized = re.sub(r"\b[a-z]{2,8}-[a-z0-9-]{2,}\b", "<id>", normalized)
    normalized = re.sub(r"\bresearch_ai_[a-z0-9_]+\b", "<id>", normalized)
    normalized = re.sub(r"\bretail_[a-z0-9_]+\b", "<id>", normalized)
    normalized = re.sub(r"\bmch-pol-\d+\b", "<id>", normalized)
    normalized = re.sub(r"\bca-scale-\d+\b", "<id>", normalized)
    normalized = re.sub(r"\b[a-z0-9]{10}\b", "<id>", normalized)
    normalized = re.sub(r"\d+(?:\.\d+)?", "<num>", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def calculate_question_template_diversity(prompts: list[dict[str, Any]]) -> dict[str, Any]:
    templates = [normalized_question_template(prompt) for prompt in prompts]
    if not templates:
        return {
            "linguistic_variation_rate": 0.0,
            "most_common_question_template_count": 0,
            "most_common_question_template_share": 0.0,
            "unique_question_template_count": 0,
        }
    template_counts = Counter(templates)
    most_common_count = template_counts.most_common(1)[0][1]
    most_common_share = most_common_count / len(templates)
    return {
        "linguistic_variation_rate": round(1.0 - most_common_share, 6),
        "most_common_question_template_count": most_common_count,
        "most_common_question_template_share": round(most_common_share, 6),
        "unique_question_template_count": len(template_counts),
    }


def supported_targets_message() -> str:
    return ", ".join(str(target) for target in SUPPORTED_TARGETS)


def validate_target(target_per_vertical: int) -> None:
    if target_per_vertical not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target_per_vertical: {target_per_vertical}. "
            f"Supported targets: {supported_targets_message()}."
        )


def get_checkpoint_for_target(target_per_vertical: int) -> str:
    validate_target(target_per_vertical)
    return TARGET_TO_CHECKPOINT[target_per_vertical]


def calculate_total_prompts(target_per_vertical: int, vertical_count: int = 5) -> int:
    validate_target(target_per_vertical)
    return target_per_vertical * vertical_count


def recommended_previous_checkpoint(target_per_vertical: int) -> str | None:
    validate_target(target_per_vertical)
    ordered = SUPPORTED_TARGETS
    index = ordered.index(target_per_vertical)
    if index == 0:
        return "checkpoint_seed"
    return get_checkpoint_for_target(ordered[index - 1])


def next_checkpoint(target_per_vertical: int) -> str | None:
    validate_target(target_per_vertical)
    ordered = SUPPORTED_TARGETS
    index = ordered.index(target_per_vertical)
    if index == len(ordered) - 1:
        return None
    return get_checkpoint_for_target(ordered[index + 1])


def target_warnings(target_per_vertical: int) -> list[str]:
    if target_per_vertical in {4000, 5000}:
        return [
            "This target is a scaffolding tier. Run smaller checkpoints and QA before "
            "attempting large local generation."
        ]
    if target_per_vertical == 2000:
        return ["This is the near-term main target. Generate and review smaller checkpoints first."]
    return []


def scale_counts(base_counts: dict[str, int], target_total: int) -> dict[str, int]:
    base_total = sum(base_counts.values())
    if target_total <= 0:
        raise ValueError("target_per_vertical must be positive.")
    if base_total == target_total:
        return dict(base_counts)
    scaled: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, value in base_counts.items():
        exact = value * target_total / base_total
        whole = int(exact)
        scaled[key] = whole
        remainders.append((exact - whole, key))
    remaining = target_total - sum(scaled.values())
    for _, key in sorted(remainders, reverse=True)[:remaining]:
        scaled[key] += 1
    return scaled


def percentage_counts(
    percentages_basis_points: dict[str, int], target_total: int
) -> dict[str, int]:
    if target_total <= 0:
        raise ValueError("target_per_vertical must be positive.")
    if sum(percentages_basis_points.values()) != 10000:
        raise ValueError("Distribution basis points must sum to 10000.")
    scaled: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    for index, (key, basis_points) in enumerate(percentages_basis_points.items()):
        numerator = basis_points * target_total
        whole = numerator // 10000
        scaled[key] = whole
        remainders.append((numerator % 10000, -index, key))
    remaining = target_total - sum(scaled.values())
    for _, _, key in sorted(remainders, reverse=True)[:remaining]:
        scaled[key] += 1
    return scaled


def calculate_distribution_counts(
    vertical: str, target_per_vertical: int = 250
) -> dict[str, dict[str, int]]:
    validate_target(target_per_vertical)
    if vertical not in VERTICALS:
        raise ValueError(f"Unsupported vertical: {vertical}")
    return {
        "expected_status": percentage_counts(
            STATUS_DISTRIBUTION_BASIS_POINTS[vertical], target_per_vertical
        ),
        "task_type": scale_counts(TASK_DISTRIBUTION_250[vertical], target_per_vertical),
        "expected_output_format": scale_counts(
            percentage_counts(OUTPUT_FORMAT_BASIS_POINTS[vertical], 250),
            target_per_vertical,
        ),
        "difficulty": percentage_counts(DIFFICULTY_BASIS_POINTS, target_per_vertical),
    }


def expand_count_sequence(counts: dict[str, int]) -> list[str]:
    sequence: list[str] = []
    for key, count in counts.items():
        sequence.extend([key] * count)
    return sequence


def selected_verticals(vertical: str) -> list[str]:
    if vertical == "all":
        return list(VERTICALS)
    if vertical not in VERTICALS:
        raise ValueError(f"Unsupported vertical: {vertical}")
    return [vertical]


def load_qa_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "qa_report_exists": False,
            "critical_issue_count": None,
            "scale_up_readiness": {},
            "warnings": [
                "Phase 2A-7 QA report is missing; run --run-audit before committing scale-up."
            ],
        }
    report = load_json(path)
    return {
        "qa_report_exists": True,
        "critical_issue_count": int(report.get("critical_issue_count", 0)),
        "scale_up_readiness": report.get("scale_up_readiness", {}),
        "warnings": [],
    }


def qa_ready_for_vertical(qa_status: dict[str, Any], vertical: str) -> bool | None:
    readiness = qa_status.get("scale_up_readiness", {})
    if not qa_status.get("qa_report_exists"):
        return None
    if qa_status.get("critical_issue_count", 0):
        return False
    if not isinstance(readiness, dict):
        return False
    vertical_status = readiness.get(vertical)
    return bool(isinstance(vertical_status, dict) and vertical_status.get("ready_for_250_scale"))


def source_readiness(vertical: str, target_per_vertical: int) -> dict[str, Any]:
    validate_target(target_per_vertical)
    seed_files = VERTICAL_FILES[vertical]
    missing_seed_files = [str(path) for path in seed_files.values() if not path.exists()]
    blockers = [f"missing_seed_file:{path}" for path in missing_seed_files]
    planning_notes: list[str] = []
    optional_artifacts: dict[str, bool] = {}

    if vertical == "finance":
        optional_artifacts = {
            "selected_filings_manifest": Path(
                "data/processed/finance/sec/selected_filings_manifest.jsonl"
            ).exists(),
            "selected_filing_documents_manifest": Path(
                "data/processed/finance/sec/selected_filing_documents_manifest.jsonl"
            ).exists(),
            "sec_filing_text_manifest": Path(
                "data/processed/finance/sec/filing_text_manifest.jsonl"
            ).exists(),
            "sec_filing_sections_manifest": Path(
                "data/processed/finance/sec/filing_sections_manifest.jsonl"
            ).exists(),
            "xbrl_concept_inventory": Path(
                "data/processed/finance/sec/xbrl_concept_inventory.jsonl"
            ).exists(),
            "section_quality_report": Path(
                "data/processed/finance/sec/finance_section_quality_report.json"
            ).exists(),
        }
    elif vertical == "airline":
        optional_artifacts = {
            "synthetic_seed_generator": Path(
                "scripts/phase2/generate_airline_synthetic.py"
            ).exists()
        }
    elif vertical == "healthcare_admin":
        optional_artifacts = {
            "synthetic_seed_generator": Path(
                "scripts/phase2/generate_healthcare_admin_synthetic.py"
            ).exists()
        }
    elif vertical == "research_ai":
        optional_artifacts = {
            "approved_paper_registry": Path(
                "data/sources/research_ai_approved_papers.jsonl"
            ).exists(),
            "enriched_paper_registry": Path(
                "data/generated/research_ai/enriched_paper_registry.jsonl"
            ).exists(),
            "paper_text_manifest": Path(
                "data/processed/research_ai/paper_text_manifest.jsonl"
            ).exists(),
            "paper_sections_manifest": Path(
                "data/processed/research_ai/paper_sections_manifest.jsonl"
            ).exists(),
            "section_quality_report": Path(
                "data/generated/research_ai/research_ai_section_quality_report.json"
            ).exists(),
        }
    elif vertical == "retail":
        optional_artifacts = {
            "retail_review_sample": Path(
                "data/generated/retail/amazon_reviews_sample.jsonl"
            ).exists(),
            "targeted_metadata_sample": Path(
                "data/generated/retail/retail_targeted_metadata_sample.jsonl"
            ).exists(),
            "targeted_metadata_report": Path(
                "data/generated/retail/retail_targeted_metadata_enrichment_report.json"
            ).exists(),
        }

    generation_implemented = target_per_vertical in IMPLEMENTED_GENERATION_TARGETS.get(
        vertical, set()
    )
    source_artifacts_ready = not missing_seed_files
    if not generation_implemented:
        blockers.append("generation_not_implemented_for_vertical_target")
        planning_notes.append(
            "Planning is available, but actual record generation requires explicit implementation."
        )
    if target_per_vertical > 250 and not generation_implemented:
        blockers.append("large_target_generation_requires_checkpoint_review")
        planning_notes.append(
            "Large target generation requires successful review of prior checkpoints."
        )
    actual_generation_ready = source_artifacts_ready and generation_implemented
    return {
        "vertical": vertical,
        "target_per_vertical": target_per_vertical,
        "seed_files": {name: str(path) for name, path in seed_files.items()},
        "missing_seed_files": missing_seed_files,
        "optional_local_artifacts": optional_artifacts,
        "source_artifacts_ready": source_artifacts_ready,
        "ready_for_actual_generation": actual_generation_ready,
        "generation_implemented": generation_implemented,
        "generation_scope": (
            "local_candidate_generation" if generation_implemented else "planning_only"
        ),
        "blockers": blockers,
        "planning_notes": planning_notes,
    }


def build_generation_manifest(
    *,
    vertical: str,
    target_per_vertical: int,
    qa_status: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    validate_target(target_per_vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    total_target_prompts = calculate_total_prompts(target_per_vertical)
    readiness = source_readiness(vertical, target_per_vertical)
    qa_ready = qa_ready_for_vertical(qa_status, vertical)
    blockers = list(readiness["blockers"])
    planning_notes = list(readiness["planning_notes"])
    if qa_ready is False:
        blockers.append(f"phase2a_qa_not_ready_for_{target_per_vertical}_scale")
        planning_notes.append("Phase 2A QA must pass before actual record generation.")
    checkpoint = plan.get("checkpoints", {}).get(checkpoint_name, {})
    kb_range = (
        plan.get("vertical_scale_strategy", {})
        .get(vertical, {})
        .get(f"kb_target_{target_per_vertical}", {})
    )
    generation_implemented = bool(readiness["generation_implemented"])
    warnings = target_warnings(target_per_vertical)
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "vertical": vertical,
        "target_per_vertical": target_per_vertical,
        "total_target_prompts": total_target_prompts,
        "checkpoint": checkpoint_name,
        "checkpoint_purpose": checkpoint.get("purpose", "QA-scale deterministic dataset"),
        "approved_targets": SUPPORTED_TARGETS,
        "distributions": calculate_distribution_counts(vertical, target_per_vertical),
        "expected_kb_range": {
            "min": kb_range.get("min"),
            "max": kb_range.get("max"),
        },
        "source_readiness": readiness,
        "qa_ready_for_250_scale": qa_ready,
        "qa_ready_for_scale": qa_ready,
        "source_artifacts_ready": readiness["source_artifacts_ready"],
        "generation_scope": (
            "local_candidate_generation" if generation_implemented else "planning_only"
        ),
        "generation_implemented": generation_implemented,
        "ready_for_actual_generation": (
            qa_ready is not False and readiness["ready_for_actual_generation"]
        ),
        "recommended_previous_checkpoint": recommended_previous_checkpoint(target_per_vertical),
        "promotion_required_before_next_checkpoint": next_checkpoint(target_per_vertical),
        "warnings": warnings,
        "blockers": blockers,
        "planning_notes": planning_notes,
        "source_inputs": readiness["seed_files"],
        "outputs_are_local_and_ignored": True,
        "next_step": (
            f"Generate local {vertical} {target_per_vertical}-scale candidates with "
            "--generate-vertical."
            if generation_implemented and not blockers
            else "Resolve blockers or extend the generator before producing local candidates."
        ),
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    plan = load_json(Path(args.scaleup_plan))
    qa_status = load_qa_status(Path(args.qa_report))
    verticals = selected_verticals(args.vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    manifests = {
        vertical: build_generation_manifest(
            vertical=vertical,
            target_per_vertical=target_per_vertical,
            qa_status=qa_status,
            plan=plan,
        )
        for vertical in verticals
    }
    return {
        "phase": PHASE,
        "mode": "dry_run",
        "target_per_vertical": target_per_vertical,
        "planned_total_prompts": calculate_total_prompts(target_per_vertical, len(verticals)),
        "full_checkpoint_total_prompts": calculate_total_prompts(target_per_vertical),
        "checkpoint": checkpoint_name,
        "checkpoint_purpose": plan["checkpoints"][checkpoint_name]["purpose"],
        "approved_targets": SUPPORTED_TARGETS,
        "qa_report_exists": qa_status["qa_report_exists"],
        "qa_warnings": qa_status["warnings"],
        "warnings": target_warnings(target_per_vertical),
        "verticals": manifests,
        "outputs_are_local_and_ignored": True,
        "writes_generated_records": False,
        "next_step": (
            "Run --generate-plan, then use implemented vertical generators only after "
            "checkpoint review."
        ),
    }


def generate_plan(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    plan = load_json(Path(args.scaleup_plan))
    qa_status = load_qa_status(Path(args.qa_report))
    report_dir = Path(args.report_dir)
    verticals = selected_verticals(args.vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    manifests: dict[str, dict[str, Any]] = {}
    for vertical in verticals:
        manifest = build_generation_manifest(
            vertical=vertical,
            target_per_vertical=target_per_vertical,
            qa_status=qa_status,
            plan=plan,
        )
        manifest_path = report_dir / f"{vertical}_scaleup_{target_per_vertical}_manifest.json"
        write_json(manifest_path, manifest)
        manifests[vertical] = {
            "manifest_path": str(manifest_path),
            "blocker_count": len(manifest["blockers"]),
            "blockers": manifest["blockers"],
            "ready_for_actual_generation": manifest["ready_for_actual_generation"],
            "generation_implemented": manifest["generation_implemented"],
            "generation_scope": manifest["generation_scope"],
        }
    aggregate_path = report_dir / f"phase2a_scaleup_generation_plan_{target_per_vertical}.json"
    aggregate = {
        "phase": PHASE,
        "mode": "generate_plan",
        "generated_at_utc": utc_now(),
        "target_per_vertical": target_per_vertical,
        "total_target_prompts": calculate_total_prompts(target_per_vertical),
        "planned_total_prompts": calculate_total_prompts(target_per_vertical, len(verticals)),
        "checkpoint": checkpoint_name,
        "checkpoint_purpose": plan["checkpoints"][checkpoint_name]["purpose"],
        "approved_targets": SUPPORTED_TARGETS,
        "recommended_previous_checkpoint": recommended_previous_checkpoint(target_per_vertical),
        "promotion_required_before_next_checkpoint": next_checkpoint(target_per_vertical),
        "verticals": manifests,
        "warnings": [*qa_status["warnings"], *target_warnings(target_per_vertical)],
        "outputs_are_local_and_ignored": True,
        "writes_generated_records": False,
        "next_step": (
            "Use --generate-vertical only for explicitly implemented target/vertical pairs."
        ),
    }
    write_json(aggregate_path, aggregate)
    return {
        "phase": PHASE,
        "mode": "generate_plan",
        "target_per_vertical": target_per_vertical,
        "total_target_prompts": aggregate["total_target_prompts"],
        "checkpoint": checkpoint_name,
        "approved_targets": SUPPORTED_TARGETS,
        "manifest_count": len(verticals),
        "aggregate_report": str(aggregate_path),
        "verticals": manifests,
        "next_step": aggregate["next_step"],
    }


def compact_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value)


def finance_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def finance_status_task_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pairs.extend([("answer", "answer_grounded")] * 95)
    pairs.extend([("answer", "calculation")] * 35)
    pairs.extend([("answer", "compare_filings")] * 35)
    pairs.extend([("answer", "extract_structured")] * 45)
    pairs.extend([("answer", "evidence_citation_lookup")] * 20)
    pairs.extend([("insufficient_evidence", "escalation_response")] * 10)
    pairs.extend([("escalate", "escalation_response")] * 10)
    return pairs


def finance_output_for_task(*, task_type: str, output_counts: dict[str, int]) -> str:
    if task_type == "calculation" and output_counts["json"] < 35:
        output_counts["json"] += 1
        return "json"
    if task_type == "compare_filings" and output_counts["markdown_table"] < 35:
        output_counts["markdown_table"] += 1
        return "markdown_table"
    if task_type == "extract_structured" and output_counts["json"] < 50:
        output_counts["json"] += 1
        return "json"
    if task_type == "extract_structured" and output_counts["markdown_table"] < 45:
        output_counts["markdown_table"] += 1
        return "markdown_table"
    output_counts["text"] += 1
    return "text"


def finance_prompt_facts(seed_prompt: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = finance_metadata(seed_prompt)
    facts: list[dict[str, Any]] = []
    for key in ["fact", "revenue_fact", "net_income_fact"]:
        value = metadata.get(key)
        if isinstance(value, dict):
            facts.append(value)
    value = metadata.get("facts")
    if isinstance(value, list):
        facts.extend(item for item in value if isinstance(item, dict))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (
            str(fact.get("concept") or ""),
            str(fact.get("end") or ""),
            str(fact.get("value") or ""),
        )
        if key not in seen and key[0]:
            seen.add(key)
            deduped.append(fact)
    return deduped


def finance_fact_label(seed_prompt: dict[str, Any], concept: str) -> str:
    metadata = finance_metadata(seed_prompt)
    labels = metadata.get("humanized_concept_labels")
    if isinstance(labels, dict) and concept in labels:
        return str(labels[concept])
    if metadata.get("raw_xbrl_concept") == concept and metadata.get("humanized_concept_label"):
        return str(metadata["humanized_concept_label"])
    return concept


def finance_fact_doc_id(seed_prompt: dict[str, Any]) -> str:
    ticker = str(seed_prompt.get("ticker") or "MULTI")
    prompt_id = str(seed_prompt.get("prompt_id") or "finance_seed")
    return f"finance_kb_xbrl_{ticker}_{prompt_id}_facts"


def build_finance_xbrl_fact_rows(
    seed_prompts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    doc_id_by_prompt_id: dict[str, str] = {}
    for seed_prompt in seed_prompts:
        facts = finance_prompt_facts(seed_prompt)
        if not facts:
            continue
        prompt_id = str(seed_prompt.get("prompt_id") or "")
        ticker = str(seed_prompt.get("ticker") or "MULTI")
        company = str(seed_prompt.get("company") or ticker)
        doc_id = finance_fact_doc_id(seed_prompt)
        fact_summaries = []
        concepts: list[str] = []
        accession_numbers: list[str] = []
        for fact in facts:
            concept = str(fact.get("concept") or "")
            concepts.append(concept)
            accession = str(fact.get("accn") or "")
            if accession:
                accession_numbers.append(accession)
            label = finance_fact_label(seed_prompt, concept)
            fact_summaries.append(
                f"{label} ({concept}) = {fact.get('value')} {fact.get('unit', '')} "
                f"for fiscal year {fact.get('fy')} period {fact.get('fp')} "
                f"from Form {fact.get('form')} accession {accession}"
            )
        body = (
            f"Curated XBRL fact evidence for {company} ({ticker}): "
            + "; ".join(fact_summaries)
            + ". Use these values only as cited SEC/XBRL facts; do not infer forecasts "
            "or analyst conclusions."
        )
        rows.append(
            {
                "allowed_to_commit": False,
                "body": body,
                "doc_id": doc_id,
                "document_type": "xbrl_fact_evidence",
                "metadata": {
                    "accession_numbers": sorted(set(accession_numbers)),
                    "company_name": company,
                    "concepts": sorted(set(concepts)),
                    "source_prompt_id": prompt_id,
                    "ticker": ticker,
                },
                "source_id": "finance_sec_edgar_xbrl",
                "source_type": "derived",
                "tags": ["finance", "sec", "xbrl", "fact-evidence"],
                "title": f"{ticker} curated XBRL fact evidence",
                "version": "phase2a-9e-scaleup-v1",
                "vertical": "finance",
            }
        )
        doc_id_by_prompt_id[prompt_id] = doc_id
    return rows, doc_id_by_prompt_id


def build_finance_8k_event_rows() -> list[dict[str, Any]]:
    selected_filings = load_jsonl_if_exists(
        Path("data/processed/finance/sec/selected_filings_manifest.jsonl")
    )
    rows: list[dict[str, Any]] = []
    counts_by_ticker: Counter[str] = Counter()
    for filing in selected_filings:
        if str(filing.get("form") or "") != "8-K":
            continue
        ticker = str(filing.get("ticker") or "")
        if not ticker or counts_by_ticker[ticker] >= 3:
            continue
        accession = str(filing.get("accession_number") or "")
        doc_id = f"finance_kb_sec_{ticker}_8K_{compact_identifier(accession)}_filing_event"
        items = str(filing.get("items") or "not specified")
        company = str(filing.get("company_name") or ticker)
        filing_date = str(filing.get("filing_date") or "")
        report_date = str(filing.get("report_date") or filing_date)
        selection_reason = str(filing.get("selection_reason") or "Selected Form 8-K filing.")
        body = (
            f"{company} ({ticker}) filed Form 8-K on {filing_date} "
            f"(report date {report_date}) with SEC item(s) {items}. "
            f"The selected filing manifest notes: {selection_reason} "
            "This evidence supports filing-event identification only and should not be "
            "used to infer financial outcomes beyond the cited filing metadata."
        )
        rows.append(
            {
                "allowed_to_commit": False,
                "body": body,
                "doc_id": doc_id,
                "document_type": "sec_filing_event",
                "metadata": {
                    "accession_number": accession,
                    "company_name": company,
                    "filing_date": filing_date,
                    "form": "8-K",
                    "items": items,
                    "provenance_url": filing.get("derived_filing_url"),
                    "report_date": report_date,
                    "source_manifest_record_id": filing.get("record_id"),
                    "ticker": ticker,
                },
                "provenance_url": filing.get("derived_filing_url"),
                "source_id": "finance_sec_edgar_xbrl",
                "source_type": "derived",
                "tags": ["8-k", "finance", "sec", "filing-event"],
                "title": f"{ticker} 8-K filing event ({filing_date})",
                "version": "phase2a-9e-scaleup-v1",
                "vertical": "finance",
            }
        )
        counts_by_ticker[ticker] += 1
    return rows


def sanitize_finance_kb_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(row)
    body = str(sanitized.get("body") or "")
    body = re.sub(
        r"before deciding to purchase, hold,? or sell shares of our common stock",
        "before making investment decisions",
        body,
        flags=re.IGNORECASE,
    )
    sanitized["body"] = body
    return sanitized


def finance_context_from_seed_prompt(
    seed_prompt: dict[str, Any], fact_doc_id_by_prompt_id: dict[str, str]
) -> dict[str, Any]:
    prompt_id = str(seed_prompt.get("prompt_id") or "")
    metadata = finance_metadata(seed_prompt)
    source_doc_ids = [str(item) for item in seed_prompt.get("source_doc_ids", []) if item]
    fact_doc_id = fact_doc_id_by_prompt_id.get(prompt_id)
    return {
        "company": str(seed_prompt.get("company") or seed_prompt.get("ticker") or "Finance"),
        "comparison_tickers": [
            str(item) for item in metadata.get("comparison_tickers", []) if item
        ],
        "doc_ids": [fact_doc_id] if fact_doc_id else source_doc_ids,
        "fact_doc_id": fact_doc_id,
        "facts": finance_prompt_facts(seed_prompt),
        "filing_form": str(seed_prompt.get("filing_form") or ""),
        "fiscal_period": str(seed_prompt.get("fiscal_period") or ""),
        "fiscal_year": str(seed_prompt.get("fiscal_year") or ""),
        "metric_label": str(
            metadata.get("metric_label")
            or metadata.get("humanized_concept_label")
            or "selected financial metric"
        ),
        "prompt_category": str(metadata.get("prompt_category") or ""),
        "seed_task_type": str(seed_prompt.get("task_type") or ""),
        "source_doc_ids": source_doc_ids,
        "ticker": str(seed_prompt.get("ticker") or "MULTI"),
        "type": "seed_xbrl",
    }


def finance_section_context(row: dict[str, Any]) -> dict[str, Any]:
    metadata = finance_metadata(row)
    return {
        "company": str(metadata.get("company_name") or metadata.get("ticker") or "Finance"),
        "doc_ids": [str(row.get("doc_id") or "")],
        "filing_date": str(metadata.get("filing_date") or ""),
        "filing_form": str(metadata.get("form") or ""),
        "report_date": str(metadata.get("report_date") or ""),
        "section_record_id": str(metadata.get("section_record_id") or row.get("doc_id") or ""),
        "section_type": str(metadata.get("section_type") or ""),
        "ticker": str(metadata.get("ticker") or ""),
        "title": str(row.get("title") or ""),
        "type": "sec_section",
    }


def finance_event_context(row: dict[str, Any]) -> dict[str, Any]:
    metadata = finance_metadata(row)
    return {
        "company": str(metadata.get("company_name") or metadata.get("ticker") or "Finance"),
        "doc_ids": [str(row.get("doc_id") or "")],
        "filing_date": str(metadata.get("filing_date") or ""),
        "filing_form": "8-K",
        "items": str(metadata.get("items") or "not specified"),
        "report_date": str(metadata.get("report_date") or ""),
        "ticker": str(metadata.get("ticker") or ""),
        "title": str(row.get("title") or ""),
        "type": "sec_event",
    }


def finance_context_pools(
    *,
    seed_prompts: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    fact_doc_id_by_prompt_id: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    seed_contexts = [
        finance_context_from_seed_prompt(seed_prompt, fact_doc_id_by_prompt_id)
        for seed_prompt in seed_prompts
    ]
    section_contexts = [
        finance_section_context(row)
        for row in kb_rows
        if str(row.get("document_type") or "") == "sec_filing_section"
    ]
    event_contexts = [
        finance_event_context(row)
        for row in kb_rows
        if str(row.get("document_type") or "") == "sec_filing_event"
    ]
    return {
        "calculation": [
            context
            for context in seed_contexts
            if context["seed_task_type"] == "calculation_answer" and context["doc_ids"]
        ],
        "compare_companies": [
            context
            for context in seed_contexts
            if context["seed_task_type"] == "compare_companies" and context["doc_ids"]
        ],
        "direct_numeric": [
            context
            for context in seed_contexts
            if context["seed_task_type"] == "answer_short" and context["doc_ids"]
        ],
        "events": event_contexts,
        "sections": section_contexts,
        "sections_10k": [
            context for context in section_contexts if context.get("filing_form") == "10-K"
        ],
        "sections_10q": [
            context for context in section_contexts if context.get("filing_form") == "10-Q"
        ],
        "trend": [
            context
            for context in seed_contexts
            if context["seed_task_type"] == "trend_summary" and context["doc_ids"]
        ],
    }


def finance_select_contexts(
    *,
    status: str,
    task_type: str,
    index: int,
    pools: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if status in {"insufficient_evidence", "escalate"}:
        candidates = pools["direct_numeric"] or pools["sections"]
        return [candidates[index % len(candidates)]]
    if task_type == "calculation":
        candidates = pools["calculation"] or pools["direct_numeric"]
        return [candidates[index % len(candidates)]]
    if task_type == "extract_structured":
        candidates = pools["events"] if index % 4 == 0 and pools["events"] else pools["sections"]
        return [candidates[index % len(candidates)]]
    if task_type == "evidence_citation_lookup":
        candidates = pools["sections"] if index % 2 == 0 else pools["direct_numeric"]
        return [candidates[index % len(candidates)]]
    if task_type == "compare_filings":
        if index % 3 == 0 and pools["sections_10k"] and pools["sections_10q"]:
            q_context = pools["sections_10q"][index % len(pools["sections_10q"])]
            matching_10k = [
                context
                for context in pools["sections_10k"]
                if context["ticker"] == q_context["ticker"]
            ]
            return [matching_10k[index % len(matching_10k)], q_context]
        if index % 3 == 1 and pools["compare_companies"]:
            return [pools["compare_companies"][index % len(pools["compare_companies"])]]
        return [
            pools["sections_10k"][index % len(pools["sections_10k"])],
            pools["sections_10k"][(index + 1) % len(pools["sections_10k"])],
        ]
    if pools["events"] and index % 7 == 0:
        return [pools["events"][index % len(pools["events"])]]
    if pools["trend"] and index % 5 == 0:
        return [pools["trend"][index % len(pools["trend"])]]
    if pools["direct_numeric"] and index % 5 == 1:
        return [pools["direct_numeric"][index % len(pools["direct_numeric"])]]
    return [pools["sections"][index % len(pools["sections"])]]


def finance_doc_ids(contexts: list[dict[str, Any]]) -> list[str]:
    doc_ids: list[str] = []
    for context in contexts:
        for doc_id in context.get("doc_ids", []):
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(str(doc_id))
    return doc_ids


def finance_chunk_ids(
    required_doc_ids: list[str], kb_by_doc_id: dict[str, dict[str, Any]]
) -> list[str]:
    chunk_ids: list[str] = []
    for doc_id in required_doc_ids:
        metadata = finance_metadata(kb_by_doc_id.get(doc_id, {}))
        chunk_ids.append(
            str(
                metadata.get("section_record_id")
                or metadata.get("source_manifest_record_id")
                or metadata.get("source_prompt_id")
                or doc_id
            )
        )
    return chunk_ids


def finance_citations(
    required_doc_ids: list[str], kb_by_doc_id: dict[str, dict[str, Any]]
) -> list[str]:
    citations: list[str] = []
    for doc_id in required_doc_ids:
        row = kb_by_doc_id.get(doc_id, {})
        metadata = finance_metadata(row)
        accession = metadata.get("accession_number")
        accessions = metadata.get("accession_numbers")
        ticker = str(metadata.get("ticker") or "finance")
        if accession:
            citations.append(f"sec://{accession}#{doc_id}")
        elif isinstance(accessions, list) and accessions:
            citations.append(f"xbrl://{ticker}/{accessions[0]}#{doc_id}")
        else:
            citations.append(f"finance://{ticker}#{doc_id}")
    return citations


def finance_context_label(contexts: list[dict[str, Any]]) -> str:
    if len(contexts) == 1:
        context = contexts[0]
        ticker = str(context.get("ticker") or "")
        company = str(context.get("company") or ticker)
        form = str(context.get("filing_form") or "")
        metric = str(context.get("metric_label") or "")
        if form:
            return f"{company} ({ticker}) {form}".strip()
        if metric:
            return f"{company} ({ticker}) {metric}".strip()
        return f"{company} ({ticker})".strip()
    return " and ".join(
        f"{context.get('company', context.get('ticker', 'Finance'))} "
        f"({context.get('ticker', '')})".strip()
        for context in contexts
    )


def finance_question_text(
    *,
    prompt_number: int,
    status: str,
    task_type: str,
    contexts: list[dict[str, Any]],
    required_doc_ids: list[str],
) -> str:
    label = finance_context_label(contexts)
    evidence = ", ".join(required_doc_ids[:3])
    if status == "insufficient_evidence":
        variants = [
            (
                f"Using public SEC/XBRL evidence only, can the records for {label} reveal "
                "a confidential internal margin target? State the evidence boundary."
            ),
            (
                f"Determine whether cited Finance evidence {evidence} is sufficient to "
                "answer a request for unannounced internal forecasts."
            ),
            (
                f"A user asks for private financial planning details about {label}; "
                "respond from the public-filing evidence boundary."
            ),
            (
                f"For scenario {prompt_number}, decide whether the available SEC records "
                "support a confidential budget claim."
            ),
            (
                f"What should the answer say when public evidence for {label} does not "
                "support the requested private finance detail?"
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "escalate":
        variants = [
            (
                f"An analyst asks whether the cited evidence for {label} is enough for a "
                "high-stakes finance conclusion. Should it be escalated for review?"
            ),
            (
                f"Using only SEC/XBRL evidence {evidence}, determine whether this finance "
                "request needs analyst review before an answer."
            ),
            (
                f"A finance reviewer needs a decision about {label}; explain the evidence "
                "limits and escalation path."
            ),
            (
                f"For scenario {prompt_number}, decide whether the filing evidence can be "
                "answered directly or requires analyst review."
            ),
            (
                f"What should staff do when the cited Finance records for {label} do not "
                "support a complete conclusion?"
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "calculation":
        variants = [
            (
                f"Using cited XBRL facts for {label}, calculate the requested margin "
                "and show the formula."
            ),
            (
                f"What calculation can be made from the SEC/XBRL facts for {label}? "
                "Include the formula."
            ),
            f"Compute the finance ratio supported by records {evidence}; avoid projections.",
            (
                f"For scenario {prompt_number}, calculate the grounded metric from "
                "the cited XBRL inputs."
            ),
            (
                f"Show a calculation using only the cited facts for {label}, with no "
                "forecasted values."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "compare_filings":
        variants = [
            f"Compare the cited Finance evidence for {label} in a compact markdown table.",
            f"Using only SEC/XBRL records {evidence}, create a grounded filing comparison.",
            (
                f"What differs between the cited filings or companies for {label}? "
                "Keep to the evidence."
            ),
            (
                f"For scenario {prompt_number}, compare 10-K, 10-Q, or company "
                "evidence without adding claims."
            ),
            f"Create a table that contrasts the available filing evidence for {label}.",
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "extract_structured":
        variants = [
            (
                f"Return JSON for {label} with ticker, filing_form, filing_date, "
                "evidence_ids, and boundary."
            ),
            f"Using only cited Finance evidence, extract a structured record for {label}.",
            f"Create a JSON evidence note for records {evidence}; do not add unsupported fields.",
            (
                f"For scenario {prompt_number}, structure the SEC filing evidence "
                "into a compact JSON object."
            ),
            f"Extract key filing metadata and the grounded support action for {label}.",
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "evidence_citation_lookup":
        variants = [
            (
                f"Which cited Finance record supports an answer about {label}, and "
                "what is its evidence ID?"
            ),
            f"Identify the SEC or XBRL citation that should support a grounded answer for {label}.",
            "Using only cited records, name the document and chunk ID behind the Finance claim.",
            f"For scenario {prompt_number}, point to the evidence record that supports the answer.",
            f"What citation should be used before answering a question about {label}?",
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    variants = [
        f"Using only cited SEC filing evidence, summarize the finance-relevant point for {label}.",
        f"What does the cited Finance evidence say about {label}? Stay within the records.",
        f"Summarize the selected SEC/XBRL evidence for {label} without making projections.",
        (
            f"For scenario {prompt_number}, answer the single-filing Finance question "
            f"using records {evidence}."
        ),
        f"Explain the cited filing section or XBRL fact for {label} in finance-ready language.",
    ]
    return choose_phrase_variant(prompt_number - 1, variants)


def finance_calculation_summary(context: dict[str, Any]) -> str | None:
    facts = context.get("facts", [])
    if not isinstance(facts, list):
        return None
    revenue_fact = next(
        (
            fact
            for fact in facts
            if "revenue" in str(fact.get("concept") or "").lower()
            and isinstance(fact.get("value"), int | float)
        ),
        None,
    )
    net_income_fact = next(
        (
            fact
            for fact in facts
            if str(fact.get("concept") or "") in {"NetIncomeLoss", "ProfitLoss"}
            and isinstance(fact.get("value"), int | float)
        ),
        None,
    )
    if not revenue_fact or not net_income_fact:
        return None
    revenue = float(revenue_fact["value"])
    net_income = float(net_income_fact["value"])
    if revenue == 0:
        return None
    margin = net_income / revenue * 100
    return (
        "Calculation: net income divided by revenue, multiplied by 100. "
        f"Using cited XBRL values {int(net_income)} and {int(revenue)}, "
        f"the margin is {margin:.2f}%."
    )


def finance_evidence_focus(contexts: list[dict[str, Any]]) -> str:
    if len(contexts) > 1:
        return "the cited filings or company records on each side of the comparison"
    context = contexts[0]
    context_type = str(context.get("type") or "")
    filing_form = str(context.get("filing_form") or "").strip()
    metric_label = str(context.get("metric_label") or "").strip()
    section_type = str(context.get("section_type") or "").replace("_", " ").strip()
    if context_type == "sec_event":
        return "the Form 8-K filing-event record"
    if context_type == "seed_xbrl":
        metric_phrase = metric_label if metric_label else "selected metric"
        form_phrase = f" from the {filing_form}" if filing_form else ""
        return f"the XBRL fact record for {metric_phrase}{form_phrase}"
    if context_type == "sec_section":
        section_phrase = section_type if section_type else "selected filing section"
        form_phrase = f" in the {filing_form}" if filing_form else ""
        return f"the {section_phrase} section{form_phrase}"
    return "the selected Finance evidence record"


def finance_reference_answer(
    *,
    status: str,
    task_type: str,
    output_format: str,
    contexts: list[dict[str, Any]],
    required_doc_ids: list[str],
) -> str:
    label = finance_context_label(contexts)
    focus = finance_evidence_focus(contexts)
    evidence = ", ".join(required_doc_ids) if required_doc_ids else "the cited records"
    if status == "insufficient_evidence":
        summary = (
            f"The available Finance evidence for {label} is not enough to answer the "
            "request reliably. The correct response is to state insufficient_evidence, "
            f"point to the evidence boundary around {evidence}, and avoid guessing, "
            "projecting, or adding unsupported financial claims."
        )
    elif status == "escalate":
        summary = (
            f"The {label} request should be escalated for analyst review. The selected "
            f"records ({evidence}) may support a narrow filing-grounded discussion, "
            "but they are not enough for a complete conclusion without review."
        )
    elif task_type == "calculation":
        calculation_summary = finance_calculation_summary(contexts[0])
        if calculation_summary:
            summary = (
                f"{calculation_summary} The response should cite {evidence}, show the "
                "formula clearly, and avoid inferring trends beyond the selected values."
            )
        else:
            summary = (
                f"The calculation for {label} should be based only on {evidence}. "
                "The answer should show the formula or comparison clearly and avoid "
                "using unstated values."
            )
    elif task_type == "compare_filings":
        summary = (
            f"The comparison for {label} should use only {evidence}. Identify the "
            f"specific company, filing form, period or section in {focus}, and evidence record for "
            "each side of the comparison; do not add investment conclusions or external "
            "market commentary."
        )
    elif task_type == "extract_structured":
        summary = (
            f"The structured answer should extract the company, ticker, filing form, "
            f"period, metric or section, and evidence ID for {label} from {focus} "
            f"({evidence}). "
            "It should not add fields that are not present in the cited record."
        )
    elif task_type == "evidence_citation_lookup":
        summary = (
            f"The answer should identify {evidence} as the Finance evidence supporting "
            f"{label}. It should name the document or chunk IDs for {focus} before making any "
            "filing-grounded statement."
        )
    else:
        summary = (
            f"The cited filing evidence supports a grounded answer for {label}. The "
            f"response should cite {evidence}, focus on {focus}, and avoid projections, "
            "valuation targets, or unsupported "
            "financial claims."
        )

    if output_format == "json":
        return json.dumps(
            {
                "status": status,
                "task_type": task_type,
                "subject": label,
                "evidence_ids": required_doc_ids,
                "answer_boundary": summary,
            },
            sort_keys=True,
        )
    if output_format == "markdown_table":
        rows = [
            "| Subject | Task | Evidence | Boundary |",
            "| --- | --- | --- | --- |",
        ]
        for context in contexts:
            rows.append(
                f"| {finance_context_label([context])} | {task_type} | "
                f"{', '.join(context.get('doc_ids', []))} | Cite the listed records; "
                "avoid projections and market commentary |"
            )
        return "\n".join(rows)
    return summary


def build_finance_pilot_records(
    *,
    target_per_vertical: int,
    seed: int,
    seed_prompts: list[dict[str, Any]],
    seed_gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if target_per_vertical != 250:
        raise RuntimeError("Finance generation is currently implemented only for 250.")
    seed_prompt_ids = {str(row.get("prompt_id") or "") for row in seed_prompts}
    seed_gold_ids = {str(row.get("prompt_id") or "") for row in seed_gold}
    if not seed_prompt_ids or seed_prompt_ids != seed_gold_ids:
        raise RuntimeError("Finance seed prompt/gold IDs are not aligned.")

    distributions = calculate_distribution_counts("finance", target_per_vertical)
    status_task_pairs = finance_status_task_pairs()
    if len(status_task_pairs) != target_per_vertical:
        raise RuntimeError("Finance status/task sequence does not match the target count.")
    if Counter(status for status, _ in status_task_pairs) != distributions["expected_status"]:
        raise RuntimeError("Finance status sequence does not match the approved distribution.")
    if Counter(task for _, task in status_task_pairs) != distributions["task_type"]:
        raise RuntimeError("Finance task sequence does not match the approved distribution.")

    fact_rows, fact_doc_id_by_prompt_id = build_finance_xbrl_fact_rows(seed_prompts)
    event_rows = build_finance_8k_event_rows()
    kb_copy = [sanitize_finance_kb_row(row) for row in kb_rows]
    existing_doc_ids = {str(row.get("doc_id") or "") for row in kb_copy}
    for row in [*fact_rows, *event_rows]:
        if str(row.get("doc_id") or "") not in existing_doc_ids:
            kb_copy.append(row)
            existing_doc_ids.add(str(row.get("doc_id") or ""))

    pools = finance_context_pools(
        seed_prompts=seed_prompts,
        kb_rows=kb_copy,
        fact_doc_id_by_prompt_id=fact_doc_id_by_prompt_id,
    )
    if not pools["sections"] or not pools["direct_numeric"] or not pools["calculation"]:
        raise RuntimeError("Finance seed data does not expose enough SEC/XBRL contexts.")

    difficulty_sequence = expand_count_sequence(distributions["difficulty"])
    output_counts = {"text": 0, "json": 0, "markdown_table": 0}
    kb_by_doc_id = {str(row.get("doc_id") or ""): row for row in kb_copy}

    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, (status, task_type) in enumerate(status_task_pairs):
        prompt_number = index + 1
        contexts = finance_select_contexts(
            status=status,
            task_type=task_type,
            index=index,
            pools=pools,
        )
        required_doc_ids = [
            doc_id for doc_id in finance_doc_ids(contexts) if doc_id in kb_by_doc_id
        ]
        if status == "answer" and not required_doc_ids:
            raise RuntimeError(f"Finance prompt {prompt_number} has no valid evidence IDs.")

        output_format = finance_output_for_task(task_type=task_type, output_counts=output_counts)
        difficulty = difficulty_sequence[index]
        prompt_id = f"finance_scaleup_{target_per_vertical}_{prompt_number:04d}"
        question = finance_question_text(
            prompt_number=prompt_number,
            status=status,
            task_type=task_type,
            contexts=contexts,
            required_doc_ids=required_doc_ids,
        )
        company = finance_context_label(contexts)
        ticker = str(contexts[0].get("ticker") or "MULTI")
        filing_form = str(contexts[0].get("filing_form") or "")
        required_chunk_ids = finance_chunk_ids(required_doc_ids, kb_by_doc_id)
        required_citations = finance_citations(required_doc_ids, kb_by_doc_id)
        prompt = {
            "company": company,
            "expected_action": "answer_with_citations"
            if status == "answer"
            else ("escalate_for_review" if status == "escalate" else "state_insufficient_evidence"),
            "expected_output_format": output_format,
            "expected_status": status,
            "filing_form": filing_form,
            "metadata": {
                "difficulty": difficulty,
                "evidence_type": "sec_xbrl_filing_evidence",
                "generator": GENERATOR_NAME,
                "prompt_category": task_type if status == "answer" else status,
                "requires_citation": bool(required_doc_ids),
                "scaleup_candidate": True,
                "seed": seed,
                "target_per_vertical": target_per_vertical,
            },
            "prompt_id": prompt_id,
            "question": question,
            "required_chunk_ids": required_chunk_ids,
            "required_citations": required_citations,
            "required_doc_ids": required_doc_ids,
            "required_evidence_ids": required_doc_ids,
            "task_type": task_type,
            "ticker": ticker,
            "vertical": "finance",
        }
        reference_answer = finance_reference_answer(
            status=status,
            task_type=task_type,
            output_format=output_format,
            contexts=contexts,
            required_doc_ids=required_doc_ids,
        )
        must_not_include = [
            "unsupported financial claims",
            "investment recommendation",
            "price target",
            "private internal targets",
            "fabricated citations",
            "unverifiable projections",
            "guaranteed outcome",
        ]
        if status != "answer":
            must_not_include.extend(["private budgets", "unannounced forecasts"])
        gold_row = {
            "expected_escalation": status == "escalate",
            "expected_status": status,
            "metadata": {
                "expected_output_format": output_format,
                "prompt_category": task_type if status == "answer" else status,
                "required_evidence_ids": required_doc_ids,
                "source_subject": company,
            },
            "must_include": [ticker, *required_doc_ids[:2]],
            "must_not_include": must_not_include,
            "prompt_id": prompt_id,
            "reference_answer": reference_answer,
            "required_chunk_ids": required_chunk_ids,
            "required_citations": required_citations,
            "required_doc_ids": required_doc_ids,
            "task_type": task_type,
            "vertical": "finance",
        }
        prompts.append(prompt)
        gold.append(gold_row)

    if output_counts != distributions["expected_output_format"]:
        raise RuntimeError("Finance output format sequence does not match approved distribution.")
    return prompts, gold, kb_copy


def build_airline_policy_map(seed_prompts: list[dict[str, Any]]) -> dict[str, list[str]]:
    counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for row in seed_prompts:
        support_type = str(row.get("support_type") or "general_support")
        policy_ids = tuple(str(item) for item in row.get("required_policy_ids", []) if item)
        if policy_ids:
            counts[support_type][policy_ids] += 1
    policy_map: dict[str, list[str]] = {}
    for support_type, support_counts in counts.items():
        policy_map[support_type] = list(support_counts.most_common(1)[0][0])
    if not policy_map:
        raise RuntimeError("Airline seed prompts do not expose required policy IDs.")
    return policy_map


def airline_action_for_status(status: str) -> str:
    if status == "answer":
        return "answer_policy"
    if status == "spam_or_fraud":
        return "fraud_review"
    return "manual_review"


def airline_task_for_status(status: str, fallback_task: str) -> str:
    if status == "spam_or_fraud":
        return "quality_boundary"
    if status == "escalate":
        return "escalation_response"
    return fallback_task


def airline_issue_text(
    *,
    prompt_number: int,
    status: str,
    support_type: str,
    route: str,
    policy_ids: list[str],
) -> str:
    support_label = support_type.replace("_", " ")
    evidence = ", ".join(policy_ids)
    if status == "spam_or_fraud":
        variants = [
            (
                "A Canada Air support agent is reviewing a suspicious "
                f"{support_label} request for scenario {prompt_number}. Use policy "
                f"records {evidence} to decide the fraud-review path."
            ),
            (
                f"What should support do when a traveler presents a possible "
                f"{support_label} issue on {route}? Ground the response in {evidence}."
            ),
            (
                f"A passenger asks for help with {support_label}, but the case may need "
                f"fraud review. Apply only the cited Canada Air policies: {evidence}."
            ),
            (
                "Using only Canada Air policy records, classify this suspicious "
                f"{support_label} scenario and name the next support action."
            ),
            (
                f"For scenario {prompt_number}, determine whether the {support_label} "
                "message should be routed away from normal support and into fraud review."
            ),
        ]
        question = choose_phrase_variant(prompt_number - 1, variants)
        if f"scenario {prompt_number}" not in question.lower():
            question = f"{question} Scenario {prompt_number}."
        return question
    elif status == "escalate":
        variants = [
            (
                f"A passenger on route {route} needs help with {support_label}. Decide "
                f"whether the cited Canada Air records {evidence} require manual review."
            ),
            (
                f"Using only the cited policy records, explain why this {support_label} "
                "case should be escalated before promising an outcome."
            ),
            (
                f"A Canada Air customer is asking about {support_label}; what should the "
                "support agent do when the request needs manual review?"
            ),
            (
                f"Review the {support_label} request in scenario {prompt_number} and "
                "state the escalation path supported by the cited policies."
            ),
            (
                f"What is the grounded support response for a {support_label} case on "
                f"{route} when policy evidence points to manual review?"
            ),
        ]
        question = choose_phrase_variant(prompt_number - 1, variants)
        if f"scenario {prompt_number}" not in question.lower():
            question = f"{question} Scenario {prompt_number}."
        return question
    else:
        variants = [
            (
                f"A traveler needs help with {support_label} on Canada Air route {route}. "
                "Answer using only the cited policy evidence."
            ),
            (
                f"A Canada Air customer is asking about {support_label}. Explain the "
                f"policy-backed support action using records {evidence}."
            ),
            (
                f"Using only the cited policy records, explain how support should handle "
                f"this {support_label} request."
            ),
            (
                f"A passenger on route {route} wants to know what Canada Air can do for "
                f"{support_label}. Keep the answer grounded in policy evidence."
            ),
            (
                f"What should the support agent do when a traveler reports {support_label} "
                f"in scenario {prompt_number}? Cite the relevant Canada Air records."
            ),
        ]
        question = choose_phrase_variant(prompt_number - 1, variants)
        if f"scenario {prompt_number}" not in question.lower():
            question = f"{question} Scenario {prompt_number}."
        return question


def airline_reference_answer(
    *,
    status: str,
    output_format: str,
    support_type: str,
    action: str,
    policy_ids: list[str],
) -> str:
    support_label = support_type.replace("_", " ")
    evidence = ", ".join(policy_ids)
    if status == "spam_or_fraud":
        return (
            f"Treat this {support_label} scenario as a fraud or chargeback review. "
            f"Use Canada Air policy evidence {evidence}, avoid approving benefits from "
            "unverified claims, and route the case to the fraud review path."
        )
    if status == "escalate":
        return (
            f"Escalate this {support_label} case for manual review. The cited Canada Air "
            f"policy records {evidence} should be checked before making a final support "
            "decision, and the answer should not promise refunds, waivers, or exceptions."
        )
    if output_format == "json":
        return json.dumps(
            {
                "airline": "Canada Air",
                "support_type": support_type,
                "recommended_action": action,
                "evidence_ids": policy_ids,
            },
            sort_keys=True,
        )
    if output_format == "markdown_table":
        return (
            "| Field | Grounded answer |\n"
            "| --- | --- |\n"
            f"| Support type | {support_label} |\n"
            f"| Recommended action | {action} |\n"
            f"| Evidence | {evidence} |"
        )
    return (
        f"Answer the {support_label} request using Canada Air policy records {evidence}. "
        f"The grounded action is {action}; do not add exceptions, compensation, or identity "
        "bypasses that are not supported by the cited policies."
    )


def build_airline_pilot_records(
    *,
    target_per_vertical: int,
    seed: int,
    seed_prompts: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    distributions = calculate_distribution_counts("airline", target_per_vertical)
    status_sequence = expand_count_sequence(distributions["expected_status"])
    task_sequence = expand_count_sequence(distributions["task_type"])
    output_sequence = expand_count_sequence(distributions["expected_output_format"])
    difficulty_sequence = expand_count_sequence(distributions["difficulty"])
    policy_map = build_airline_policy_map(seed_prompts)
    support_types = sorted(policy_map)
    route_cycle = ["YVR-NRT", "YYZ-LHR", "YUL-CDG", "YYC-MEX", "YOW-YHZ", "YEG-YVR"]
    travel_cycle = ["domestic", "transborder", "international"]

    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index in range(target_per_vertical):
        prompt_number = index + 1
        status = status_sequence[index]
        support_type = (
            "fraud_or_chargeback"
            if status == "spam_or_fraud" and "fraud_or_chargeback" in policy_map
            else support_types[index % len(support_types)]
        )
        policy_ids = policy_map[support_type]
        output_format = output_sequence[index]
        task_type = airline_task_for_status(status, task_sequence[index])
        action = airline_action_for_status(status)
        difficulty = difficulty_sequence[index]
        prompt_id = f"airline_scaleup_{target_per_vertical}_{prompt_number:04d}"
        ticket_id = f"CA-SCALE-{prompt_number:04d}"
        route = route_cycle[index % len(route_cycle)]
        issue = airline_issue_text(
            prompt_number=prompt_number,
            status=status,
            support_type=support_type,
            route=route,
            policy_ids=policy_ids,
        )
        prompt = {
            "airline": "Canada Air",
            "expected_action": action,
            "expected_output_format": output_format,
            "expected_status": status,
            "issue": issue,
            "metadata": {
                "difficulty": difficulty,
                "generator": GENERATOR_NAME,
                "prompt_category": support_type,
                "scaleup_candidate": True,
                "seed": seed,
                "target_per_vertical": target_per_vertical,
            },
            "partner_airline_involved": support_type in {"partner_airline", "codeshare"},
            "prompt_id": prompt_id,
            "question": issue,
            "required_evidence_ids": policy_ids,
            "required_policy_ids": policy_ids,
            "route": route,
            "support_type": support_type,
            "task_type": task_type,
            "ticket_id": ticket_id,
            "travel_type": travel_cycle[index % len(travel_cycle)],
            "vertical": "airline",
        }
        reference_answer = airline_reference_answer(
            status=status,
            output_format=output_format,
            support_type=support_type,
            action=action,
            policy_ids=policy_ids,
        )
        gold_row = {
            "expected_action": action,
            "expected_status": status,
            "metadata": {
                "difficulty": difficulty,
                "expected_action": action,
                "generator": GENERATOR_NAME,
                "prompt_category": support_type,
                "required_policy_ids": policy_ids,
                "support_type": support_type,
                "ticket_id": ticket_id,
            },
            "must_include": ["Canada Air", action, *policy_ids],
            "must_not_include": [
                "unsupported compensation promise",
                "verification bypass",
                "guaranteed refund outside policy",
                "uncited policy exception",
            ],
            "prompt_id": prompt_id,
            "reference_answer": reference_answer,
            "required_citations": [{"doc_id": policy_id} for policy_id in policy_ids],
            "required_chunk_ids": policy_ids,
            "required_doc_ids": policy_ids,
            "task_type": task_type,
            "vertical": "airline",
        }
        prompts.append(prompt)
        gold.append(gold_row)
    kb_copy = [dict(row) for row in kb_rows]
    return prompts, gold, kb_copy


def build_healthcare_policy_context(
    seed_prompts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    actions: dict[str, Counter[str]] = defaultdict(Counter)
    queues: dict[str, Counter[str]] = defaultdict(Counter)
    boundaries: dict[str, Counter[str]] = defaultdict(Counter)
    for row in seed_prompts:
        support_type = str(row.get("support_type") or "general_admin")
        policy_ids = tuple(str(item) for item in row.get("required_policy_ids", []) if item)
        if policy_ids:
            counts[support_type][policy_ids] += 1
        actions[support_type][str(row.get("expected_action") or "answer_policy")] += 1
        queues[support_type][str(row.get("expected_queue") or "general_admin")] += 1
        boundaries[support_type][str(row.get("safety_boundary") or "administrative_only")] += 1

    policy_context: dict[str, dict[str, Any]] = {}
    for support_type, support_counts in counts.items():
        policy_context[support_type] = {
            "policy_ids": list(support_counts.most_common(1)[0][0]),
            "expected_action": actions[support_type].most_common(1)[0][0],
            "expected_queue": queues[support_type].most_common(1)[0][0],
            "safety_boundary": boundaries[support_type].most_common(1)[0][0],
        }
    if not policy_context:
        raise RuntimeError("Healthcare seed prompts do not expose required policy IDs.")
    return policy_context


def healthcare_status_task_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pairs.extend([("answer", "answer_grounded")] * 120)
    pairs.extend([("answer", "policy_reasoning")] * 55)
    pairs.extend([("answer", "extract_structured")] * 30)
    pairs.extend([("answer", "escalation_response")] * 15)
    pairs.extend([("escalate", "escalation_response")] * 10)
    pairs.extend([("escalate", "quality_boundary")] * 5)
    pairs.extend([("escalate", "safety_boundary")] * 5)
    pairs.extend([("safety_boundary", "safety_boundary")] * 5)
    pairs.extend([("spam_or_fraud", "quality_boundary")] * 3)
    pairs.extend([("out_of_scope", "quality_boundary")] * 2)
    return pairs


def healthcare_context_for_status(
    *,
    status: str,
    index: int,
    policy_context: dict[str, dict[str, Any]],
) -> tuple[str, list[str], str, str, str]:
    support_types = sorted(policy_context)
    if status == "safety_boundary":
        support_type = (
            "lab_result_availability"
            if "lab_result_availability" in policy_context
            else support_types[index % len(support_types)]
        )
        return (
            support_type,
            ["MCH-POL-023", "MCH-POL-024"],
            "urgent_clinical_redirect",
            "clinical_staff_review",
            "urgent_clinical_redirect",
        )
    if status == "spam_or_fraud":
        support_type = "portal_access" if "portal_access" in policy_context else support_types[0]
        return (
            support_type,
            ["MCH-POL-025"],
            "ignore_spam_or_fraud",
            "portal_support",
            "administrative_only",
        )
    if status == "out_of_scope":
        support_type = (
            "clinic_location_hours"
            if "clinic_location_hours" in policy_context
            else support_types[index % len(support_types)]
        )
        return (
            support_type,
            ["MCH-POL-025"],
            "decline_out_of_scope",
            "general_admin",
            "administrative_only",
        )
    if status == "escalate":
        preferred = [
            "privacy_request",
            "medical_records_request",
            "insurance_verification",
            "billing_question",
            "prior_authorization_status",
        ]
        available = [item for item in preferred if item in policy_context] or support_types
        support_type = available[index % len(available)]
        context = policy_context[support_type]
        return (
            support_type,
            list(context["policy_ids"]),
            "escalate_manual_review",
            str(context["expected_queue"]),
            str(context["safety_boundary"]),
        )

    support_type = support_types[index % len(support_types)]
    context = policy_context[support_type]
    return (
        support_type,
        list(context["policy_ids"]),
        str(context["expected_action"]),
        str(context["expected_queue"]),
        str(context["safety_boundary"]),
    )


def healthcare_issue_text(
    *,
    prompt_number: int,
    status: str,
    support_type: str,
    expected_queue: str,
) -> str:
    support_label = support_type.replace("_", " ")
    if status == "safety_boundary":
        variants = [
            (
                "A patient asks MapleCare Health for urgent clinical guidance. Respond "
                "only with the administrative urgent clinical boundary workflow."
            ),
            (
                "Using the admin policy evidence, determine how staff should handle an "
                "urgent clinical boundary request without giving clinical advice."
            ),
            (
                f"A caller raises an urgent {support_label} concern in scenario "
                f"{prompt_number}; route it through the admin boundary process."
            ),
            (
                "What should staff do when a message asks for urgent care guidance but "
                "the support team can provide only administrative routing?"
            ),
            (
                "Should this urgent clinic request be handled directly or redirected? "
                "Use only the MapleCare Health administrative boundary policy."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "spam_or_fraud":
        variants = [
            (
                "A portal message includes suspicious billing or credential claims. "
                "Decide how the administrative queue should handle it."
            ),
            (
                "Using the admin policy evidence, classify this low-quality or fraud-like "
                "portal request before any staff action."
            ),
            (
                f"A caller needs help with {support_label}, but the message includes "
                "signals that require fraud or spam review."
            ),
            (
                "What should staff do when a portal request looks unreliable and asks "
                "for billing or credential handling?"
            ),
            (
                f"For scenario {prompt_number}, determine whether the suspicious "
                "administrative message should be handled as normal support."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "out_of_scope":
        variants = [
            (
                "A requester asks MapleCare Health an unrelated non-healthcare question. "
                "Apply the support boundary policy."
            ),
            (
                "Using the admin policy evidence, determine how to decline a question "
                "outside the clinic support corpus."
            ),
            (
                f"A caller needs help with a topic unrelated to {support_label}; respond "
                "from the support boundary, not general knowledge."
            ),
            (
                "What should staff do when an administrative channel receives a request "
                "that is outside MapleCare Health support?"
            ),
            (
                f"Should scenario {prompt_number} be answered directly or marked "
                "out-of-scope under the administrative support policy?"
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "escalate":
        variants = [
            (
                f"A patient asks the clinic admin team about {support_label}; decide "
                f"whether {expected_queue} review is required before answering."
            ),
            (
                f"Using the admin policy evidence, determine the escalation path for a "
                f"{support_label} request."
            ),
            (
                f"A caller needs help with {support_label}, but the case may require "
                f"{expected_queue} review. What should staff do?"
            ),
            (
                f"What should staff do when a {support_label} request cannot be resolved "
                "directly by the front administrative team?"
            ),
            (
                f"Should this {support_label} request be handled directly or escalated "
                f"to {expected_queue}? Use only MapleCare Health policy evidence."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    variants = [
        (
            f"A patient asks MapleCare Health about {support_label}. Provide an "
            "administrative answer using only the cited policy evidence."
        ),
        (
            f"Using the admin policy evidence, determine how staff should respond to a "
            f"{support_label} request."
        ),
        (
            f"A caller needs help with {support_label}; identify the policy-backed "
            "administrative action and queue."
        ),
        (f"What should staff do when an existing patient asks about {support_label}?"),
        (
            f"Should the clinic admin team answer this {support_label} request directly, "
            "and what policy evidence supports that action?"
        ),
    ]
    return choose_phrase_variant(prompt_number - 1, variants)


def healthcare_reference_answer(
    *,
    status: str,
    output_format: str,
    support_type: str,
    action: str,
    expected_queue: str,
    policy_ids: list[str],
) -> str:
    support_label = support_type.replace("_", " ")
    evidence = ", ".join(policy_ids)
    if status == "safety_boundary":
        summary = (
            "This request crosses MapleCare Health's administrative-only boundary. "
            f"Use policy evidence {evidence} to route through the urgent clinical "
            "redirect workflow; do not provide diagnosis, treatment, dosage, or "
            "clinical reassurance."
        )
    elif status == "spam_or_fraud":
        summary = (
            f"Treat this {support_label} message as spam or fraud review. Use policy "
            f"evidence {evidence}, avoid relying on the message as valid patient "
            "evidence, and route it to the configured review workflow."
        )
    elif status == "out_of_scope":
        summary = (
            "The question is outside the Healthcare Admin support corpus. Use policy "
            f"evidence {evidence} to decline the unrelated request and avoid answering "
            "from general knowledge."
        )
    elif status == "escalate":
        summary = (
            f"Escalate the {support_label} request to the {expected_queue} queue. "
            f"Policy evidence {evidence} supports administrative review before any "
            "response; do not provide clinical interpretation or unsupported approvals."
        )
    else:
        summary = (
            f"Use MapleCare Health policy evidence {evidence} to answer the "
            f"{support_label} request. The administrative action is {action} through "
            f"the {expected_queue} queue, with no diagnosis, treatment guidance, or "
            "identity-verification bypass."
        )

    if output_format == "json":
        return json.dumps(
            {
                "provider": "MapleCare Health",
                "support_type": support_type,
                "recommended_action": action,
                "expected_queue": expected_queue,
                "evidence_ids": policy_ids,
                "admin_boundary": "no clinical advice",
            },
            sort_keys=True,
        )
    if output_format == "markdown_table":
        return (
            "| Field | Grounded answer |\n"
            "| --- | --- |\n"
            f"| Support type | {support_label} |\n"
            f"| Recommended action | {action} |\n"
            f"| Queue | {expected_queue} |\n"
            f"| Evidence | {evidence} |\n"
            "| Boundary | Administrative support only; no clinical advice |"
        )
    return summary


def build_healthcare_pilot_records(
    *,
    target_per_vertical: int,
    seed: int,
    seed_prompts: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if target_per_vertical != 250:
        raise RuntimeError("Healthcare Admin generation is currently implemented only for 250.")
    distributions = calculate_distribution_counts("healthcare_admin", target_per_vertical)
    status_task_pairs = healthcare_status_task_pairs()
    if len(status_task_pairs) != target_per_vertical:
        raise RuntimeError("Healthcare status/task sequence does not match the target count.")
    if Counter(status for status, _ in status_task_pairs) != distributions["expected_status"]:
        raise RuntimeError("Healthcare status sequence does not match the approved distribution.")
    if Counter(task for _, task in status_task_pairs) != distributions["task_type"]:
        raise RuntimeError("Healthcare task sequence does not match the approved distribution.")

    output_sequence = expand_count_sequence(distributions["expected_output_format"])
    difficulty_sequence = expand_count_sequence(distributions["difficulty"])
    policy_context = build_healthcare_policy_context(seed_prompts)
    channel_cycle = ["secure_message", "portal", "web_form", "phone_note"]
    patient_type_cycle = ["existing_patient", "new_patient", "care_partner"]

    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, (status, task_type) in enumerate(status_task_pairs):
        prompt_number = index + 1
        support_type, policy_ids, action, expected_queue, safety_boundary = (
            healthcare_context_for_status(
                status=status,
                index=index,
                policy_context=policy_context,
            )
        )
        output_format = output_sequence[index]
        difficulty = difficulty_sequence[index]
        prompt_id = f"healthcare_admin_scaleup_{target_per_vertical}_{prompt_number:04d}"
        ticket_id = f"MCH-SCALE-{prompt_number:04d}"
        question = healthcare_issue_text(
            prompt_number=prompt_number,
            status=status,
            support_type=support_type,
            expected_queue=expected_queue,
        )
        prompt = {
            "channel": channel_cycle[index % len(channel_cycle)],
            "department": expected_queue,
            "expected_action": action,
            "expected_output_format": output_format,
            "expected_queue": expected_queue,
            "expected_status": status,
            "issue": question,
            "metadata": {
                "difficulty": difficulty,
                "generator": GENERATOR_NAME,
                "prompt_category": support_type,
                "scaleup_candidate": True,
                "seed": seed,
                "target_per_vertical": target_per_vertical,
            },
            "patient_type": patient_type_cycle[index % len(patient_type_cycle)],
            "privacy_sensitive": support_type in {"privacy_request", "medical_records_request"},
            "prompt_id": prompt_id,
            "question": question,
            "required_evidence_ids": policy_ids,
            "required_policy_ids": policy_ids,
            "safety_boundary": safety_boundary,
            "support_type": support_type,
            "task_type": task_type,
            "ticket_id": ticket_id,
            "vertical": "healthcare_admin",
        }
        reference_answer = healthcare_reference_answer(
            status=status,
            output_format=output_format,
            support_type=support_type,
            action=action,
            expected_queue=expected_queue,
            policy_ids=policy_ids,
        )
        gold_row = {
            "expected_action": action,
            "expected_queue": expected_queue,
            "expected_status": status,
            "metadata": {
                "difficulty": difficulty,
                "expected_action": action,
                "expected_queue": expected_queue,
                "generator": GENERATOR_NAME,
                "prompt_category": support_type,
                "required_policy_ids": policy_ids,
                "safety_boundary": safety_boundary,
                "support_type": support_type,
                "ticket_id": ticket_id,
            },
            "must_include": ["MapleCare Health", action, *policy_ids],
            "must_not_include": [
                "diagnosis",
                "treatment instructions",
                "medication dosage advice",
                "clinical reassurance",
                "medical advice",
                "identity verification bypass",
            ],
            "privacy_sensitive": prompt["privacy_sensitive"],
            "prompt_id": prompt_id,
            "reference_answer": reference_answer,
            "required_citations": [{"doc_id": policy_id} for policy_id in policy_ids],
            "required_chunk_ids": policy_ids,
            "required_doc_ids": policy_ids,
            "task_type": task_type,
            "vertical": "healthcare_admin",
        }
        prompts.append(prompt)
        gold.append(gold_row)

    kb_copy = [dict(row) for row in kb_rows]
    return prompts, gold, kb_copy


def retail_contexts_by_role(seed_prompts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_prompts:
        context = {
            "category": row.get("category") or "All_Beauty",
            "expected_action": row.get("expected_action") or "answer",
            "issue_type": row.get("issue_type") or row.get("metadata", {}).get("prompt_category"),
            "product_id": row.get("product_id") or row.get("source_product_ids", ["retail"])[0],
            "product_title": row.get("product_title")
            or row.get("metadata", {}).get("source_titles", ["Retail product evidence"])[0],
            "required_doc_ids": list(
                row.get("required_doc_ids") or row.get("required_evidence_ids") or []
            ),
            "source_parent_asins": list(
                row.get("source_parent_asins")
                or row.get("metadata", {}).get("source_parent_asins")
                or []
            ),
            "source_product_ids": list(row.get("source_product_ids") or []),
            "seed_task_type": row.get("task_type"),
            "seed_status": row.get("expected_status"),
        }
        task_type = str(row.get("task_type") or "")
        issue_type = str(row.get("issue_type") or "")
        status = str(row.get("expected_status") or "")
        if status == "answer":
            contexts["answer"].append(context)
        if task_type == "compare_products":
            contexts["compare_products"].append(context)
        if task_type == "extract_structured":
            contexts["extract_structured"].append(context)
        if task_type == "policy_reasoning":
            contexts["policy_reasoning"].append(context)
        if issue_type == "quality_complaint":
            contexts["issue_identification"].append(context)
        if status == "spam_or_low_quality" or issue_type == "suspicious_review":
            contexts["spam_or_low_quality"].append(context)
        if status in {"insufficient_evidence", "escalate"}:
            contexts["escalation"].append(context)
        if status == "out_of_scope":
            contexts["out_of_scope"].append(context)

    if not contexts["answer"]:
        raise RuntimeError("Retail seed prompts do not expose answerable product evidence.")
    return dict(contexts)


def retail_status_task_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pairs.extend([("answer", "answer_grounded")] * 95)
    pairs.extend([("answer", "issue_identification")] * 45)
    pairs.extend([("answer", "extract_structured")] * 35)
    pairs.extend([("answer", "policy_reasoning")] * 27)
    pairs.extend([("answer", "compare_products")] * 20)
    pairs.extend([("insufficient_evidence", "policy_reasoning")] * 3)
    pairs.extend([("insufficient_evidence", "escalation_response")] * 6)
    pairs.extend([("escalate", "escalation_response")] * 4)
    pairs.extend([("escalate", "compare_products")] * 5)
    pairs.extend([("spam_or_low_quality", "quality_boundary")] * 7)
    pairs.extend([("out_of_scope", "quality_boundary")] * 3)
    return pairs


def retail_output_for_task(
    *,
    task_type: str,
    output_counts: dict[str, int],
) -> str:
    if task_type == "extract_structured" and output_counts["json"] < 35:
        output_counts["json"] += 1
        return "json"
    if task_type == "policy_reasoning" and output_counts["json"] < 40:
        output_counts["json"] += 1
        return "json"
    if task_type == "compare_products" and output_counts["markdown_table"] < 25:
        output_counts["markdown_table"] += 1
        return "markdown_table"
    output_counts["text"] += 1
    return "text"


def retail_context_for_pair(
    *,
    status: str,
    task_type: str,
    index: int,
    contexts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    if status == "spam_or_low_quality":
        candidates = contexts.get("spam_or_low_quality") or contexts["answer"]
    elif status in {"insufficient_evidence", "escalate"}:
        candidates = (
            contexts.get("escalation") or contexts.get("policy_reasoning") or contexts["answer"]
        )
    elif status == "out_of_scope":
        candidates = contexts.get("out_of_scope") or contexts["answer"]
    elif task_type in contexts:
        candidates = contexts[task_type]
    else:
        candidates = contexts["answer"]
    return dict(candidates[index % len(candidates)])


def retail_required_docs_for_status(
    *,
    status: str,
    task_type: str,
    context: dict[str, Any],
) -> list[str]:
    required_doc_ids = [str(item) for item in context.get("required_doc_ids", []) if item]
    if status == "spam_or_low_quality":
        return ["retail_policy_low_quality_review_handling", *required_doc_ids]
    if status == "out_of_scope":
        return ["retail_policy_out_of_scope_rules"]
    if status in {"insufficient_evidence", "escalate"}:
        if "retail_policy_escalation_rules" not in required_doc_ids:
            return ["retail_policy_escalation_rules", *required_doc_ids]
    if task_type == "policy_reasoning" and not any(
        doc_id.startswith("retail_policy_") for doc_id in required_doc_ids
    ):
        return ["retail_policy_return_refund_triage", *required_doc_ids]
    return required_doc_ids


def retail_issue_text(
    *,
    prompt_number: int,
    status: str,
    task_type: str,
    product_id: str,
    product_title: str,
) -> str:
    if status == "spam_or_low_quality":
        variants = [
            (
                f"Should the cited review evidence for {product_title} ({product_id}) "
                "be treated as reliable support evidence or spam-like content?"
            ),
            (
                f"A support agent is reviewing a suspicious complaint about "
                f"{product_title}. Determine whether moderation is needed first."
            ),
            (
                f"Using only the product and review evidence, decide whether "
                f"{product_id} has low-quality review signals."
            ),
            (
                f"Assess the reliability of the cited review for {product_title} before "
                "using it to support a customer-facing answer."
            ),
            (
                f"For scenario {prompt_number}, classify whether the available retail "
                "evidence should be filtered as spam-like or low quality."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "insufficient_evidence":
        variants = [
            (
                f"Is the selected evidence sufficient to resolve a support request for "
                f"{product_title} ({product_id})?"
            ),
            (
                f"A support agent is reviewing limited evidence about {product_title}; "
                "state what is missing before answering."
            ),
            (
                f"Using only product and review evidence, determine whether the "
                f"{product_id} request can be answered without guessing."
            ),
            (
                f"Based on the review evidence, decide if the support team has enough "
                f"context to respond about {product_title}."
            ),
            (
                f"For scenario {prompt_number}, mark whether the retail evidence is "
                "insufficient and identify the boundary."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "escalate":
        variants = [
            (
                f"Should the cited evidence for {product_title} ({product_id}) be "
                "handled directly or escalated to support review?"
            ),
            (
                f"A support agent is reviewing a complaint about {product_title}; "
                "determine whether policy review is required."
            ),
            (
                f"Using only the product, review, and policy evidence, identify the "
                f"escalation path for {product_id}."
            ),
            (
                f"Based on the available evidence for {product_title}, decide whether "
                "support should avoid promising a resolution."
            ),
            (
                f"For scenario {prompt_number}, state why the retail support case needs "
                "escalation before a final answer."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "out_of_scope":
        variants = [
            (
                "A user asks an unrelated question outside the selected retail product "
                "and support-policy corpus. Apply the out-of-scope boundary."
            ),
            (
                "Using only the retail support policy evidence, determine how to decline "
                "a request that is not about the cited product."
            ),
            (
                f"A support agent receives a question unrelated to {product_title}; "
                "respond from the support boundary rather than outside knowledge."
            ),
            (
                "What should support do when the request is outside the retail product "
                "and review evidence?"
            ),
            (
                f"For scenario {prompt_number}, decide whether the user request should "
                "be marked out-of-scope."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "issue_identification":
        variants = [
            (
                f"Based on the review evidence, identify the main support issue themes "
                f"for {product_title} ({product_id})."
            ),
            (
                f"A support agent is reviewing feedback about {product_title}; extract "
                "the customer issue categories."
            ),
            (
                f"Using only the product and review evidence, name the support problems "
                f"connected to {product_id}."
            ),
            (f"What support issue is most visible in the cited evidence for {product_title}?"),
            (
                f"For scenario {prompt_number}, summarize the retail complaint themes "
                "without adding unsupported product claims."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "extract_structured":
        variants = [
            (
                f"Extract a JSON support record for {product_title} ({product_id}) with "
                "issue type, summary, action, and evidence IDs."
            ),
            (
                f"Using only the product/review evidence, create a structured support "
                f"case for {product_title}."
            ),
            (
                f"A support agent needs a JSON triage note for {product_id}; include the "
                "evidence boundary and recommended action."
            ),
            (
                f"Convert the cited retail evidence for {product_title} into a compact "
                "structured support record."
            ),
            (
                f"For scenario {prompt_number}, output the product issue, evidence "
                "summary, and action as JSON."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "policy_reasoning":
        variants = [
            (
                f"Apply the synthetic benchmark support policy to the cited review "
                f"evidence for {product_title} ({product_id})."
            ),
            (
                f"A support agent is checking policy fit for {product_title}; explain "
                "what the benchmark policy allows."
            ),
            (
                f"Using only product, review, and policy evidence, determine the support "
                f"action for {product_id}."
            ),
            (
                f"Based on the available evidence for {product_title}, reason through "
                "the synthetic support-policy boundary."
            ),
            (
                f"For scenario {prompt_number}, explain the policy-backed retail "
                "response without treating it as Amazon policy."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "compare_products":
        variants = [
            (
                f"Compare the available retail evidence for {product_title} "
                f"({product_id}) in a compact support-ready table."
            ),
            (
                f"A support agent needs a side-by-side evidence view for {product_title}; "
                "prepare the comparison table."
            ),
            (
                f"Using only cited product and review records, compare the support "
                f"signals for {product_id}."
            ),
            (
                f"What does the cited evidence show when comparing retail signals for "
                f"{product_title}?"
            ),
            (
                f"For scenario {prompt_number}, create a table comparing the relevant "
                "retail evidence and caveats."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    variants = [
        (
            f"Based on the review evidence, summarize the available support signal for "
            f"{product_title} ({product_id})."
        ),
        (
            f"A support agent is reviewing a complaint about {product_title}; write a "
            "grounded answer from the cited evidence."
        ),
        (f"Using only the product/review evidence, explain what can be said about {product_id}."),
        (
            f"What should support say about {product_title} when limited to the selected "
            "retail evidence?"
        ),
        (
            f"For scenario {prompt_number}, summarize the product evidence in "
            "support-ready language."
        ),
    ]
    return choose_phrase_variant(prompt_number - 1, variants)


def retail_reference_answer(
    *,
    status: str,
    task_type: str,
    output_format: str,
    product_id: str,
    product_title: str,
    issue_type: str,
    required_doc_ids: list[str],
) -> str:
    evidence = ", ".join(required_doc_ids)
    if output_format == "json":
        return json.dumps(
            {
                "product_id": product_id,
                "product_title": product_title,
                "issue_type": issue_type,
                "evidence_summary": (
                    "Use only the cited sanitized product, review, and policy evidence."
                ),
                "recommended_action": status,
                "evidence_ids": required_doc_ids,
            },
            sort_keys=True,
        )
    if output_format == "markdown_table":
        return (
            "| Field | Grounded retail answer |\n"
            "| --- | --- |\n"
            f"| Product | {product_title} ({product_id}) |\n"
            f"| Task | {task_type} |\n"
            f"| Issue type | {issue_type} |\n"
            f"| Evidence | {evidence} |"
        )
    if status == "spam_or_low_quality":
        return (
            f"Flag the cited evidence for {product_title} ({product_id}) as low-quality "
            f"or spam-like. Use evidence {evidence}, and do not treat the review as "
            "strong product evidence without moderation or support review."
        )
    if status == "insufficient_evidence":
        return (
            f"The selected evidence for {product_title} ({product_id}) is insufficient "
            f"to resolve the support request. Use evidence {evidence}, ask for the "
            "missing order or product context, and do not guess."
        )
    if status == "escalate":
        return (
            f"Escalate the {product_title} ({product_id}) support request because the "
            f"selected evidence {evidence} requires policy or account review. Do not "
            "promise refunds, replacements, or safety conclusions."
        )
    if status == "out_of_scope":
        return (
            "The question is outside the Retail support corpus. A grounded system should "
            f"use evidence {evidence} to mark it out_of_scope and should not answer from "
            "general model memory or fabricate citations."
        )
    if task_type == "policy_reasoning":
        return (
            f"Apply the synthetic benchmark policy to {product_title} ({product_id}) "
            f"using evidence {evidence}. The answer may recommend support review, but "
            "must not claim to be Amazon policy or guarantee a resolution."
        )
    return (
        f"Use the cited sanitized evidence {evidence} to answer about {product_title} "
        f"({product_id}). Keep the answer limited to observed review or metadata signals "
        "and avoid unsupported product claims."
    )


def build_retail_pilot_records(
    *,
    target_per_vertical: int,
    seed: int,
    seed_prompts: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if target_per_vertical != 250:
        raise RuntimeError("Retail generation is currently implemented only for 250.")
    distributions = calculate_distribution_counts("retail", target_per_vertical)
    status_task_pairs = retail_status_task_pairs()
    if len(status_task_pairs) != target_per_vertical:
        raise RuntimeError("Retail status/task sequence does not match the target count.")
    if Counter(status for status, _ in status_task_pairs) != distributions["expected_status"]:
        raise RuntimeError("Retail status sequence does not match the approved distribution.")
    if Counter(task for _, task in status_task_pairs) != distributions["task_type"]:
        raise RuntimeError("Retail task sequence does not match the approved distribution.")

    difficulty_sequence = expand_count_sequence(distributions["difficulty"])
    contexts = retail_contexts_by_role(seed_prompts)
    output_counts = {"text": 0, "json": 0, "markdown_table": 0}
    kb_doc_ids = {str(row.get("doc_id") or "") for row in kb_rows}

    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, (status, task_type) in enumerate(status_task_pairs):
        prompt_number = index + 1
        context = retail_context_for_pair(
            status=status,
            task_type=task_type,
            index=index,
            contexts=contexts,
        )
        product_id = str(context.get("product_id") or "retail_policy_only")
        product_title = str(context.get("product_title") or "Synthetic Retail policy evidence")
        issue_type = str(context.get("issue_type") or task_type)
        required_doc_ids = [
            doc_id
            for doc_id in retail_required_docs_for_status(
                status=status,
                task_type=task_type,
                context=context,
            )
            if doc_id in kb_doc_ids
        ]
        if not required_doc_ids:
            raise RuntimeError(f"Retail prompt {prompt_number} has no valid evidence IDs.")
        output_format = retail_output_for_task(
            task_type=task_type,
            output_counts=output_counts,
        )
        difficulty = difficulty_sequence[index]
        prompt_id = f"retail_scaleup_{target_per_vertical}_{prompt_number:04d}"
        question = retail_issue_text(
            prompt_number=prompt_number,
            status=status,
            task_type=task_type,
            product_id=product_id,
            product_title=product_title,
        )
        source_parent_asins = list(context.get("source_parent_asins") or [product_id])
        source_product_ids = list(context.get("source_product_ids") or [product_id])
        prompt = {
            "category": context.get("category") or "All_Beauty",
            "expected_action": "answer" if status == "answer" else status,
            "expected_output_format": output_format,
            "expected_status": status,
            "issue_type": issue_type,
            "metadata": {
                "category": context.get("category") or "All_Beauty",
                "difficulty": difficulty,
                "evidence_type": "retail_scaleup_candidate",
                "generator": GENERATOR_NAME,
                "prompt_category": issue_type,
                "requires_citation": True,
                "scaleup_candidate": True,
                "seed": seed,
                "source_parent_asins": source_parent_asins,
                "source_titles": [product_title],
                "synthetic_policy_not_amazon_policy": task_type == "policy_reasoning",
                "target_per_vertical": target_per_vertical,
            },
            "product_id": product_id,
            "product_title": product_title,
            "prompt_id": prompt_id,
            "question": question,
            "required_doc_ids": required_doc_ids,
            "required_evidence_ids": required_doc_ids,
            "source_parent_asins": source_parent_asins,
            "source_product_ids": source_product_ids,
            "task_type": task_type,
            "vertical": "retail",
        }
        reference_answer = retail_reference_answer(
            status=status,
            task_type=task_type,
            output_format=output_format,
            product_id=product_id,
            product_title=product_title,
            issue_type=issue_type,
            required_doc_ids=required_doc_ids,
        )
        must_not_include = [
            "unsupported claims",
            "raw user IDs",
            "customer identifiers",
            "Amazon policy guarantee",
            "claims outside selected product evidence",
        ]
        if status != "answer":
            must_not_include.extend(["guessing missing details", "general model memory"])
        gold_row = {
            "expected_escalation": status in {"insufficient_evidence", "escalate"},
            "expected_status": status,
            "metadata": {
                "evidence_types": ["review", "metadata", "policy"],
                "expected_output_format": output_format,
                "prompt_category": issue_type,
                "required_evidence_ids": required_doc_ids,
                "required_parent_asins": source_parent_asins,
                "source_titles": [product_title],
            },
            "must_include": [product_id, product_title.split()[0], issue_type, *required_doc_ids],
            "must_not_include": must_not_include,
            "prompt_id": prompt_id,
            "reference_answer": reference_answer,
            "required_chunk_ids": required_doc_ids,
            "required_citations": [
                f"retail://All_Beauty/{product_id}#{doc_id}" for doc_id in required_doc_ids
            ],
            "required_doc_ids": required_doc_ids,
            "task_type": task_type,
            "vertical": "retail",
        }
        prompts.append(prompt)
        gold.append(gold_row)

    if output_counts != distributions["expected_output_format"]:
        raise RuntimeError("Retail output format sequence does not match approved distribution.")
    kb_copy = [dict(row) for row in kb_rows]
    return prompts, gold, kb_copy


def research_ai_status_task_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pairs.extend([("answer", "answer_grounded")] * 85)
    pairs.extend([("answer", "paper_method")] * 45)
    pairs.extend([("answer", "results_evaluation")] * 35)
    pairs.extend([("answer", "extract_structured")] * 30)
    pairs.extend([("answer", "compare_papers")] * 25)
    pairs.extend([("answer", "literature_table")] * 5)
    pairs.extend([("insufficient_evidence", "literature_table")] * 10)
    pairs.extend([("escalate", "escalation_response")] * 10)
    pairs.extend([("out_of_scope", "answer_grounded")] * 5)
    return pairs


def research_ai_output_for_task(
    *,
    task_type: str,
    output_counts: dict[str, int],
) -> str:
    if task_type == "extract_structured" and output_counts["json"] < 30:
        output_counts["json"] += 1
        return "json"
    if task_type == "paper_method" and output_counts["json"] < 35:
        output_counts["json"] += 1
        return "json"
    if task_type == "compare_papers" and output_counts["markdown_table"] < 20:
        output_counts["markdown_table"] += 1
        return "markdown_table"
    if task_type == "literature_table" and output_counts["markdown_table"] < 35:
        output_counts["markdown_table"] += 1
        return "markdown_table"
    output_counts["text"] += 1
    return "text"


def research_ai_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def research_ai_contexts_by_paper(kb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts_by_paper: dict[str, dict[str, Any]] = {}
    for row in kb_rows:
        metadata = research_ai_metadata(row)
        paper_id = str(metadata.get("paper_id") or row.get("paper_id") or "")
        doc_id = str(row.get("doc_id") or "")
        if not paper_id or not doc_id:
            continue
        context = contexts_by_paper.setdefault(
            paper_id,
            {
                "authors": [],
                "citation_by_doc_id": {},
                "chunk_by_doc_id": {},
                "doc_ids_by_section_type": defaultdict(list),
                "paper_id": paper_id,
                "provenance_url": "",
                "section_type_by_doc_id": {},
                "title": "",
                "topic": "",
                "venue": "",
                "year": "",
            },
        )
        title = str(metadata.get("title") or row.get("title") or context["title"])
        provenance_url = str(
            metadata.get("provenance_url") or row.get("provenance_url") or context["provenance_url"]
        )
        context["title"] = title
        context["provenance_url"] = provenance_url
        context["topic"] = str(metadata.get("topic") or context["topic"])
        context["venue"] = str(metadata.get("venue") or context["venue"])
        context["year"] = str(metadata.get("year") or context["year"])
        authors = metadata.get("authors")
        if isinstance(authors, list):
            context["authors"] = [str(author) for author in authors if author]

        section_type = str(metadata.get("section_type") or "").lower()
        if not section_type:
            document_type = str(row.get("document_type") or "")
            evidence_type = str(metadata.get("evidence_type") or "")
            if document_type == "paper_abstract" or evidence_type == "abstract":
                section_type = "abstract"
            elif document_type == "paper_metadata" or evidence_type == "metadata":
                section_type = "metadata"
            else:
                section_type = evidence_type or document_type or "evidence"
        context["doc_ids_by_section_type"][section_type].append(doc_id)
        context["section_type_by_doc_id"][doc_id] = section_type
        chunk_id = str(metadata.get("section_record_id") or doc_id)
        context["chunk_by_doc_id"][doc_id] = chunk_id
        context["citation_by_doc_id"][doc_id] = (
            f"{provenance_url}#{chunk_id}" if provenance_url else chunk_id
        )

    contexts = [
        context
        for context in contexts_by_paper.values()
        if context.get("title") and context.get("doc_ids_by_section_type")
    ]
    return sorted(contexts, key=lambda item: (str(item["title"]), str(item["paper_id"])))


def research_ai_select_doc_ids(
    context: dict[str, Any],
    preferred_section_types: list[str],
    *,
    max_docs: int,
) -> list[str]:
    selected: list[str] = []
    doc_ids_by_section_type = context["doc_ids_by_section_type"]
    for section_type in preferred_section_types:
        for doc_id in doc_ids_by_section_type.get(section_type, []):
            if doc_id not in selected:
                selected.append(str(doc_id))
            if len(selected) >= max_docs:
                return selected
    fallback_types = [
        "abstract",
        "introduction",
        "method",
        "results",
        "evaluation",
        "experiments",
        "limitations",
        "metadata",
    ]
    for section_type in fallback_types:
        for doc_id in doc_ids_by_section_type.get(section_type, []):
            if doc_id not in selected:
                selected.append(str(doc_id))
            if len(selected) >= max_docs:
                return selected
    for doc_ids in doc_ids_by_section_type.values():
        for doc_id in doc_ids:
            if doc_id not in selected:
                selected.append(str(doc_id))
            if len(selected) >= max_docs:
                return selected
    return selected


def research_ai_context_window(
    *,
    contexts: list[dict[str, Any]],
    index: int,
    size: int,
) -> list[dict[str, Any]]:
    if not contexts:
        raise RuntimeError("Research AI KB does not expose paper contexts.")
    return [contexts[(index + offset) % len(contexts)] for offset in range(size)]


def research_ai_doc_ids_for_task(
    *,
    status: str,
    task_type: str,
    contexts: list[dict[str, Any]],
) -> list[str]:
    if status == "out_of_scope":
        return []
    if task_type == "paper_method":
        return research_ai_select_doc_ids(
            contexts[0], ["method", "approach", "abstract"], max_docs=2
        )
    if task_type == "results_evaluation":
        return research_ai_select_doc_ids(
            contexts[0], ["results", "evaluation", "experiments", "abstract"], max_docs=2
        )
    if task_type == "extract_structured":
        return research_ai_select_doc_ids(
            contexts[0],
            ["method", "results", "evaluation", "experiments", "limitations", "abstract"],
            max_docs=4,
        )
    if task_type == "compare_papers":
        doc_ids: list[str] = []
        for context in contexts[:2]:
            doc_ids.extend(
                research_ai_select_doc_ids(
                    context,
                    ["method", "results", "evaluation", "experiments", "abstract"],
                    max_docs=2,
                )
            )
        return doc_ids
    if task_type == "literature_table":
        doc_ids = []
        for context in contexts[:3]:
            doc_ids.extend(
                research_ai_select_doc_ids(
                    context,
                    ["method", "results", "evaluation", "experiments", "abstract"],
                    max_docs=2,
                )
            )
        return doc_ids[:5]
    if status == "escalate":
        return research_ai_select_doc_ids(contexts[0], ["abstract", "metadata"], max_docs=1)
    return research_ai_select_doc_ids(
        contexts[0], ["abstract", "introduction", "method"], max_docs=2
    )


def research_ai_title_terms(title: str) -> list[str]:
    terms = [
        term
        for term in re.findall(r"[A-Za-z0-9]+", title)
        if len(term) > 2 and term.lower() not in {"the", "and", "for", "with", "via"}
    ]
    return terms[:3] or [title.split()[0]]


def research_ai_prompt_category(
    *,
    status: str,
    task_type: str,
    index: int,
) -> str:
    if status == "insufficient_evidence":
        return "insufficient_evidence"
    if status == "escalate":
        return "escalation"
    if status == "out_of_scope":
        return "out_of_scope"
    if task_type == "answer_grounded" and index % 7 == 0:
        return "evidence_citation_lookup"
    return {
        "answer_grounded": "concept_explanation",
        "paper_method": "paper_method",
        "results_evaluation": "results_evaluation",
        "extract_structured": "structured_extraction",
        "compare_papers": "compare_papers",
        "literature_table": "literature_table",
        "escalation_response": "escalation",
    }.get(task_type, task_type)


def research_ai_question_text(
    *,
    prompt_number: int,
    status: str,
    task_type: str,
    prompt_category: str,
    contexts: list[dict[str, Any]],
    required_doc_ids: list[str],
) -> str:
    if status == "out_of_scope":
        variants = [
            (
                "A user asks for a current sports schedule and ticket advice in the "
                "Research AI channel. Apply the out-of-scope boundary."
            ),
            (
                "Using only the Research AI corpus boundary, determine how to decline a "
                "request that is not about the cited papers."
            ),
            (
                f"Scenario {prompt_number} asks for non-paper current-event advice; "
                "mark it out-of-scope rather than answering from outside knowledge."
            ),
            (
                "What should the benchmark answer do when a question falls outside the "
                "Research AI paper evidence?"
            ),
            (
                "A request asks for ticket guidance instead of paper analysis. State the "
                "out-of-scope boundary without inventing citations."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)

    title = str(contexts[0]["title"])
    if status == "insufficient_evidence":
        cited = ", ".join(required_doc_ids)
        variants = [
            (
                "Can the available Research AI paper evidence prove that the cited papers "
                f"will outperform all future production systems? Cite only {cited}."
            ),
            (
                f"Using only records {cited}, decide whether the evidence is sufficient "
                "to make a future-production superiority claim."
            ),
            (
                "A reviewer asks for a guarantee about future systems. State whether the "
                f"cited Research AI evidence is sufficient: {cited}."
            ),
            (
                "Does the selected paper evidence support a broad deployment prediction, "
                "or should the answer mark insufficient evidence?"
            ),
            (
                f"For scenario {prompt_number}, explain why the cited records cannot be "
                "used to prove future production performance."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if status == "escalate":
        variants = [
            (
                f"A benchmark owner asks whether {title} should be certified for "
                "high-stakes production deployment. Use only the cited paper evidence."
            ),
            (
                f"Using the cited paper evidence, determine whether a deployment decision "
                f"about {title} needs escalation."
            ),
            (
                f"A reviewer wants to approve {title} for production use; explain the "
                "evidence boundary and escalation need."
            ),
            (
                f"What should the benchmark answer do when evidence about {title} is not "
                "enough for certification?"
            ),
            (
                f"For scenario {prompt_number}, decide whether the Research AI evidence "
                "supports direct approval or manual review."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "paper_method":
        variants = [
            (f"Using only the cited method evidence, explain the research method used by {title}."),
            (
                f"What method does the paper describe for {title}, and which evidence "
                "records support that summary?"
            ),
            (
                f"Extract the key method evidence about {title} without adding claims "
                "outside the cited sections."
            ),
            (
                f"A reviewer needs a grounded method summary for {title}; cite the "
                "selected evidence records."
            ),
            (
                f"For scenario {prompt_number}, describe the paper's method using only "
                "the provided Research AI KB records."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "results_evaluation":
        variants = [
            (
                f"Using only the cited results or evaluation evidence, summarize what "
                f"{title} reports about evaluation."
            ),
            (
                f"What evaluation evidence does {title} provide, and what should the "
                "answer avoid overstating?"
            ),
            (
                f"Extract the key evidence about results for {title} without adding "
                "unsupported numeric claims."
            ),
            (
                f"A reviewer asks for a grounded evaluation summary of {title}; use only "
                "the selected sections."
            ),
            (
                f"For scenario {prompt_number}, state what the cited results evidence "
                "supports and what remains outside the evidence."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "extract_structured":
        variants = [
            (
                f"Extract a JSON object for {title} with paper_title, method_or_setup, "
                "result_or_evaluation, limitation_or_caveat, and evidence_ids."
            ),
            (f"Using the cited paper evidence, create a structured record for {title}."),
            (
                f"Convert the selected sections for {title} into JSON with method, "
                "evaluation, caveat, and evidence IDs."
            ),
            (
                f"A reviewer needs structured evidence about {title}; include only fields "
                "supported by cited records."
            ),
            (
                f"For scenario {prompt_number}, extract the key method and result "
                "evidence into a compact JSON answer."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "compare_papers":
        other_title = str(contexts[1]["title"])
        variants = [
            (
                f"Create a grounded comparison between {title} and {other_title} using "
                "only cited method and evaluation evidence."
            ),
            (
                f"Compare the available evidence for {title} and {other_title}; keep the "
                "answer limited to listed records."
            ),
            (
                f"What differs between {title} and {other_title} according to the cited "
                "Research AI sections?"
            ),
            (
                "Using only the selected paper evidence, prepare a compact comparison of "
                f"{title} versus {other_title}."
            ),
            (
                f"For scenario {prompt_number}, compare the two papers and name the "
                "evidence boundary for each."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if task_type == "literature_table":
        titles = ", ".join(str(context["title"]) for context in contexts[:3])
        variants = [
            (
                "Create a compact literature table for these Research AI papers using "
                f"only the cited evidence: {titles}."
            ),
            (f"Using selected paper evidence, build a literature table covering: {titles}."),
            (f"Extract table-ready method and evaluation evidence for these papers: {titles}."),
            (
                "A reviewer needs a grounded literature table. Include only the cited "
                f"records for: {titles}."
            ),
            (
                f"For scenario {prompt_number}, summarize the cited Research AI papers "
                "in table form with evidence IDs."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    if prompt_category == "evidence_citation_lookup":
        variants = [
            (
                f"Which cited section supports a claim about {title}, and what should a "
                "grounded answer say about the evidence boundary?"
            ),
            (
                f"Identify the evidence record for {title} that can support the answer, "
                "then state the citation boundary."
            ),
            (
                f"Using the cited paper evidence, name one record that supports a claim "
                f"about {title}."
            ),
            (
                f"A reviewer asks where the claim about {title} comes from; cite the "
                "supporting Research AI record."
            ),
            (
                f"For scenario {prompt_number}, point to the cited section that supports "
                "the paper-level claim."
            ),
        ]
        return choose_phrase_variant(prompt_number - 1, variants)
    variants = [
        (
            f"Using the cited paper evidence, explain the main research problem and "
            f"contribution of {title} in plain language."
        ),
        (f"What does {title} contribute, according to the selected Research AI evidence?"),
        (f"Extract the key evidence about the problem and contribution in {title}."),
        (
            f"A reviewer needs a grounded concept explanation for {title}; use only the "
            "cited records."
        ),
        (
            f"For scenario {prompt_number}, summarize the paper's contribution without "
            "using outside knowledge."
        ),
    ]
    return choose_phrase_variant(prompt_number - 1, variants)


def research_ai_required_chunks_and_citations(
    *,
    contexts: list[dict[str, Any]],
    required_doc_ids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    chunk_ids: list[str] = []
    citations: list[str] = []
    section_types: list[str] = []
    for doc_id in required_doc_ids:
        for context in contexts:
            chunk_by_doc_id = context["chunk_by_doc_id"]
            if doc_id not in chunk_by_doc_id:
                continue
            chunk_ids.append(str(chunk_by_doc_id[doc_id]))
            citations.append(str(context["citation_by_doc_id"][doc_id]))
            section_types.append(str(context["section_type_by_doc_id"].get(doc_id, "evidence")))
            break
    return chunk_ids, citations, section_types


def research_ai_reference_answer(
    *,
    status: str,
    task_type: str,
    output_format: str,
    contexts: list[dict[str, Any]],
    required_doc_ids: list[str],
    section_types: list[str],
) -> str:
    titles = [str(context["title"]) for context in contexts]
    evidence = ", ".join(required_doc_ids) if required_doc_ids else "no in-corpus evidence"
    sections = ", ".join(dict.fromkeys(section_types)) if section_types else "none"
    if status == "out_of_scope":
        summary = (
            "This request is outside the Research AI paper corpus. The answer should mark "
            "the request out_of_scope, avoid current-events or ticket advice, and not "
            "invent paper citations."
        )
    elif status == "insufficient_evidence":
        summary = (
            f"The available Research AI evidence ({evidence}) is insufficient to prove "
            "future production superiority. The answer should state insufficient_evidence, "
            "cite the available records, and avoid unsupported projections."
        )
    elif status == "escalate":
        summary = (
            f"Escalate the deployment-certification request for {titles[0]}. Evidence "
            f"{evidence} can support paper-level discussion, but it is not enough by "
            "itself to approve high-stakes production use."
        )
    else:
        summary = (
            f"Use {titles[0]} and cited {sections} evidence ({evidence}) to answer the "
            f"{task_type} request. The answer should name the paper, cite the required "
            "records, and avoid claims outside the selected paper evidence."
        )

    if output_format == "json":
        return json.dumps(
            {
                "status": status,
                "task_type": task_type,
                "paper_titles": titles,
                "evidence_ids": required_doc_ids,
                "grounding_rule": "use only cited Research AI KB evidence",
                "answer_boundary": summary,
            },
            sort_keys=True,
        )
    if output_format == "markdown_table":
        rows = [
            "| Paper | Task | Evidence | Boundary |",
            "| --- | --- | --- | --- |",
        ]
        for context in contexts:
            title = str(context["title"])
            context_doc_ids = [
                doc_id for doc_id in required_doc_ids if doc_id in context["section_type_by_doc_id"]
            ]
            rows.append(
                f"| {title} | {task_type} | {', '.join(context_doc_ids)} | "
                "Cite only selected evidence |"
            )
        return "\n".join(rows)
    return summary


def build_research_ai_pilot_records(
    *,
    target_per_vertical: int,
    seed: int,
    seed_prompts: list[dict[str, Any]],
    seed_gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if target_per_vertical != 250:
        raise RuntimeError("Research AI generation is currently implemented only for 250.")
    seed_prompt_ids = {str(row.get("prompt_id") or "") for row in seed_prompts}
    seed_gold_ids = {str(row.get("prompt_id") or "") for row in seed_gold}
    if not seed_prompt_ids or seed_prompt_ids != seed_gold_ids:
        raise RuntimeError("Research AI seed prompt/gold IDs are not aligned.")

    distributions = calculate_distribution_counts("research_ai", target_per_vertical)
    status_task_pairs = research_ai_status_task_pairs()
    if len(status_task_pairs) != target_per_vertical:
        raise RuntimeError("Research AI status/task sequence does not match the target count.")
    if Counter(status for status, _ in status_task_pairs) != distributions["expected_status"]:
        raise RuntimeError("Research AI status sequence does not match the approved distribution.")
    if Counter(task for _, task in status_task_pairs) != distributions["task_type"]:
        raise RuntimeError("Research AI task sequence does not match the approved distribution.")

    difficulty_sequence = expand_count_sequence(distributions["difficulty"])
    output_counts = {"text": 0, "json": 0, "markdown_table": 0}
    paper_contexts = research_ai_contexts_by_paper(kb_rows)
    if len(paper_contexts) < 10:
        raise RuntimeError("Research AI KB does not expose enough paper contexts for 250 scale.")
    kb_doc_ids = {str(row.get("doc_id") or "") for row in kb_rows}

    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for index, (status, task_type) in enumerate(status_task_pairs):
        prompt_number = index + 1
        context_count = 3 if task_type == "literature_table" else 2
        contexts = research_ai_context_window(
            contexts=paper_contexts,
            index=index,
            size=context_count,
        )
        if task_type not in {"compare_papers", "literature_table"}:
            contexts = contexts[:1]
        required_doc_ids = [
            doc_id
            for doc_id in research_ai_doc_ids_for_task(
                status=status,
                task_type=task_type,
                contexts=contexts,
            )
            if doc_id in kb_doc_ids
        ]
        if status == "answer" and not required_doc_ids:
            raise RuntimeError(f"Research AI prompt {prompt_number} has no valid evidence IDs.")

        output_format = research_ai_output_for_task(
            task_type=task_type,
            output_counts=output_counts,
        )
        difficulty = difficulty_sequence[index]
        prompt_category = research_ai_prompt_category(
            status=status,
            task_type=task_type,
            index=index,
        )
        prompt_id = f"research_ai_scaleup_{target_per_vertical}_{prompt_number:04d}"
        required_chunk_ids, required_citations, section_types = (
            research_ai_required_chunks_and_citations(
                contexts=contexts,
                required_doc_ids=required_doc_ids,
            )
        )
        source_titles = (
            [] if status == "out_of_scope" else [str(context["title"]) for context in contexts]
        )
        source_paper_ids = (
            [] if status == "out_of_scope" else [str(context["paper_id"]) for context in contexts]
        )
        topics = (
            []
            if status == "out_of_scope"
            else sorted({str(context["topic"]) for context in contexts if context["topic"]})
        )
        question = research_ai_question_text(
            prompt_number=prompt_number,
            status=status,
            task_type=task_type,
            prompt_category=prompt_category,
            contexts=contexts,
            required_doc_ids=required_doc_ids,
        )
        expected_action = {
            "answer": "answer_with_citations",
            "insufficient_evidence": "state_insufficient_evidence",
            "escalate": "escalate_for_review",
            "out_of_scope": "decline_out_of_scope",
        }[status]
        prompt = {
            "expected_action": expected_action,
            "expected_output_format": output_format,
            "expected_status": status,
            "metadata": {
                "difficulty": difficulty,
                "evidence_type": section_types,
                "generator": GENERATOR_NAME,
                "prompt_category": prompt_category,
                "requires_citation": bool(required_doc_ids),
                "scaleup_candidate": True,
                "seed": seed,
                "source_titles": source_titles,
                "target_per_vertical": target_per_vertical,
                "topics": topics,
            },
            "prompt_id": prompt_id,
            "question": question,
            "required_chunk_ids": required_chunk_ids,
            "required_citations": required_citations,
            "required_doc_ids": required_doc_ids,
            "required_evidence_ids": required_doc_ids,
            "required_paper_ids": source_paper_ids,
            "source_paper_ids": source_paper_ids,
            "task_type": task_type,
            "topic": topics[0] if topics else "research_ai",
            "vertical": "research_ai",
        }
        reference_answer = research_ai_reference_answer(
            status=status,
            task_type=task_type,
            output_format=output_format,
            contexts=contexts,
            required_doc_ids=required_doc_ids,
            section_types=section_types,
        )
        must_include = (
            [*research_ai_title_terms(source_titles[0]), *required_doc_ids[:2]]
            if status == "answer"
            else [status, *required_doc_ids[:1]]
        )
        must_not_include = [
            "unsupported claims",
            "uncited claims",
            "fabricated citations",
            "claims outside selected paper evidence",
            "private file paths",
        ]
        if status in {"insufficient_evidence", "escalate"}:
            must_not_include.extend(
                [
                    "future production superiority guarantee",
                    "deployment approval from paper evidence alone",
                    "unsupported numeric claims",
                ]
            )
        if status == "out_of_scope":
            must_not_include.extend(
                [
                    "sports schedule answer",
                    "ticket purchasing advice",
                    "invented Research AI citation",
                ]
            )
        gold_row = {
            "expected_action": expected_action,
            "expected_escalation": status == "escalate",
            "expected_status": status,
            "metadata": {
                "expected_output_format": output_format,
                "prompt_category": prompt_category,
                "required_evidence_ids": required_doc_ids,
                "required_paper_ids": source_paper_ids,
                "required_section_types": section_types,
                "source_titles": source_titles,
            },
            "must_include": must_include,
            "must_not_include": must_not_include,
            "prompt_id": prompt_id,
            "reference_answer": reference_answer,
            "required_chunk_ids": required_chunk_ids,
            "required_citations": required_citations,
            "required_doc_ids": required_doc_ids,
            "task_type": task_type,
            "vertical": "research_ai",
        }
        prompts.append(prompt)
        gold.append(gold_row)

    if output_counts != distributions["expected_output_format"]:
        raise RuntimeError(
            "Research AI output format sequence does not match approved distribution."
        )
    kb_copy = [dict(row) for row in kb_rows]
    return prompts, gold, kb_copy


def validate_prompt_gold_alignment(
    prompts: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    prompt_ids = [str(row.get("prompt_id") or "") for row in prompts]
    gold_ids = [str(row.get("prompt_id") or "") for row in gold]
    prompt_counts = Counter(prompt_ids)
    gold_counts = Counter(gold_ids)
    for prompt_id, count in prompt_counts.items():
        if not prompt_id:
            issues.append("prompt_missing_prompt_id")
        elif count > 1:
            issues.append(f"duplicate_prompt_id:{prompt_id}")
    for prompt_id, count in gold_counts.items():
        if not prompt_id:
            issues.append("gold_missing_prompt_id")
        elif count > 1:
            issues.append(f"duplicate_gold_prompt_id:{prompt_id}")
    for prompt_id in prompt_counts:
        if prompt_id and gold_counts.get(prompt_id, 0) != 1:
            issues.append(f"prompt_missing_single_gold:{prompt_id}")
    for prompt_id in gold_counts:
        if prompt_id and prompt_counts.get(prompt_id, 0) != 1:
            issues.append(f"orphan_gold_prompt_id:{prompt_id}")
    return issues


def validate_evidence_coverage(
    gold: list[dict[str, Any]], kb_rows: list[dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    kb_doc_ids = {str(row.get("doc_id") or "") for row in kb_rows}
    for row in gold:
        prompt_id = str(row.get("prompt_id") or "")
        required_doc_ids = [str(item) for item in row.get("required_doc_ids", []) if item]
        if row.get("expected_status") == "answer" and not required_doc_ids:
            issues.append(f"answerable_missing_required_doc_ids:{prompt_id}")
        for doc_id in required_doc_ids:
            if doc_id not in kb_doc_ids:
                issues.append(f"missing_kb_doc_id:{prompt_id}:{doc_id}")
    return issues


def validate_no_private_hygiene_terms(rows: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for row in rows:
        row_id = str(row.get("prompt_id") or row.get("doc_id") or "unknown")
        text = flatten_text(row)
        for pattern in PRIVATE_HYGIENE_PATTERNS:
            if pattern.search(text):
                issues.append(f"hygiene_term_found:{row_id}:{pattern.pattern}")
    return issues


def write_scaleup_report(
    path: Path,
    *,
    vertical: str,
    target_per_vertical: int,
    prompts: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    blockers: list[str],
    warnings: list[str],
    checkpoint: str,
    generation_scope: str,
    generation_implemented: bool,
) -> dict[str, Any]:
    report_warnings = list(warnings)
    validation_issues = (
        validate_prompt_gold_alignment(prompts, gold)
        + validate_evidence_coverage(gold, kb_rows)
        + validate_no_private_hygiene_terms(prompts + gold + kb_rows)
    )
    linguistic_metrics = calculate_question_template_diversity(prompts)
    if prompts and linguistic_metrics["linguistic_variation_rate"] < LINGUISTIC_VARIATION_THRESHOLD:
        issue = (
            "linguistic_variation_warning:"
            f"rate_below_{LINGUISTIC_VARIATION_THRESHOLD:.2f}:"
            f"{linguistic_metrics['linguistic_variation_rate']:.3f}"
        )
        validation_issues.append(issue)
        report_warnings.append(issue)
    report = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "vertical": vertical,
        "target_per_vertical": target_per_vertical,
        "total_target_prompts": calculate_total_prompts(target_per_vertical),
        "checkpoint": checkpoint,
        "approved_targets": SUPPORTED_TARGETS,
        "generation_scope": generation_scope,
        "generation_implemented": generation_implemented,
        "recommended_previous_checkpoint": recommended_previous_checkpoint(target_per_vertical),
        "promotion_required_before_next_checkpoint": next_checkpoint(target_per_vertical),
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_rows),
        "linguistic_variation_rate": linguistic_metrics["linguistic_variation_rate"],
        "most_common_question_template_count": linguistic_metrics[
            "most_common_question_template_count"
        ],
        "most_common_question_template_share": linguistic_metrics[
            "most_common_question_template_share"
        ],
        "unique_question_template_count": linguistic_metrics["unique_question_template_count"],
        "status_counts": dict(Counter(str(row.get("expected_status")) for row in prompts)),
        "task_type_counts": dict(Counter(str(row.get("task_type")) for row in prompts)),
        "output_format_counts": dict(
            Counter(str(row.get("expected_output_format")) for row in prompts)
        ),
        "critical_issue_count": len(validation_issues) + len(blockers),
        "warning_count": len(report_warnings),
        "validation_issues": validation_issues,
        "blockers": blockers,
        "warnings": report_warnings,
        "next_step": (
            f"Review local {vertical} {target_per_vertical}-scale candidates before "
            "promoting or extending generation."
            if not validation_issues and not blockers
            else "Fix blockers or validation issues before using these candidates."
        ),
    }
    write_json(path, report)
    return report


def generate_finance_vertical(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    if target_per_vertical not in IMPLEMENTED_GENERATION_TARGETS["finance"]:
        raise RuntimeError(
            f"Generation for finance at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    qa_status = load_qa_status(Path(args.qa_report))
    qa_ready = qa_ready_for_vertical(qa_status, "finance")
    blockers: list[str] = []
    warnings = list(qa_status["warnings"])
    if qa_ready is False:
        blockers.append(f"phase2a_qa_not_ready_for_finance_{target_per_vertical}_scale")
    readiness = source_readiness("finance", target_per_vertical)
    blockers.extend(readiness["missing_seed_files"])
    report_path = Path(args.report_dir) / f"finance_scaleup_{target_per_vertical}_report.json"
    if blockers:
        write_scaleup_report(
            report_path,
            vertical="finance",
            target_per_vertical=target_per_vertical,
            prompts=[],
            gold=[],
            kb_rows=[],
            blockers=blockers,
            warnings=warnings,
            checkpoint=checkpoint_name,
            generation_scope="local_candidate_generation",
            generation_implemented=True,
        )
        return {
            "phase": PHASE,
            "mode": "generate_vertical",
            "vertical": "finance",
            "blockers": blockers,
            "report_path": str(report_path),
            "next_step": "Resolve blockers and rerun generation.",
        }

    seed_prompts = load_jsonl(VERTICAL_FILES["finance"]["prompts"])
    seed_gold = load_jsonl(VERTICAL_FILES["finance"]["gold"])
    kb_rows = load_jsonl(VERTICAL_FILES["finance"]["kb"])
    prompts, gold, kb_copy = build_finance_pilot_records(
        target_per_vertical=target_per_vertical,
        seed=int(args.seed),
        seed_prompts=seed_prompts,
        seed_gold=seed_gold,
        kb_rows=kb_rows,
    )
    output_dir = Path(args.output_dir) / "finance"
    prompts_path = output_dir / f"finance_prompts_{target_per_vertical}.jsonl"
    gold_path = output_dir / f"finance_gold_{target_per_vertical}.jsonl"
    kb_path = output_dir / f"finance_kb_{target_per_vertical}.jsonl"
    write_jsonl(prompts_path, prompts)
    write_jsonl(gold_path, gold)
    write_jsonl(kb_path, kb_copy)
    report = write_scaleup_report(
        report_path,
        vertical="finance",
        target_per_vertical=target_per_vertical,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_copy,
        blockers=[],
        warnings=warnings,
        checkpoint=checkpoint_name,
        generation_scope="local_candidate_generation",
        generation_implemented=True,
    )
    return {
        "phase": PHASE,
        "mode": "generate_vertical",
        "vertical": "finance",
        "target_per_vertical": target_per_vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_copy),
        "status_counts": report["status_counts"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "prompts_path": str(prompts_path),
        "gold_path": str(gold_path),
        "kb_path": str(kb_path),
        "report_path": str(report_path),
        "next_step": report["next_step"],
    }


def generate_airline_vertical(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    if target_per_vertical not in IMPLEMENTED_GENERATION_TARGETS["airline"]:
        raise RuntimeError(
            f"Generation for airline at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    qa_status = load_qa_status(Path(args.qa_report))
    qa_ready = qa_ready_for_vertical(qa_status, "airline")
    blockers: list[str] = []
    warnings = list(qa_status["warnings"])
    if qa_ready is False:
        blockers.append(f"phase2a_qa_not_ready_for_airline_{target_per_vertical}_scale")
    readiness = source_readiness("airline", target_per_vertical)
    blockers.extend(readiness["missing_seed_files"])
    if blockers:
        report_path = Path(args.report_dir) / f"airline_scaleup_{target_per_vertical}_report.json"
        write_scaleup_report(
            report_path,
            vertical="airline",
            target_per_vertical=target_per_vertical,
            prompts=[],
            gold=[],
            kb_rows=[],
            blockers=blockers,
            warnings=warnings,
            checkpoint=checkpoint_name,
            generation_scope="local_candidate_generation",
            generation_implemented=True,
        )
        return {
            "phase": PHASE,
            "mode": "generate_vertical",
            "vertical": "airline",
            "blockers": blockers,
            "report_path": str(report_path),
            "next_step": "Resolve blockers and rerun generation.",
        }

    seed_prompts = load_jsonl(VERTICAL_FILES["airline"]["prompts"])
    kb_rows = load_jsonl(VERTICAL_FILES["airline"]["kb"])
    prompts, gold, kb_copy = build_airline_pilot_records(
        target_per_vertical=target_per_vertical,
        seed=int(args.seed),
        seed_prompts=seed_prompts,
        kb_rows=kb_rows,
    )
    output_dir = Path(args.output_dir) / "airline"
    report_dir = Path(args.report_dir)
    prompts_path = output_dir / f"airline_prompts_{target_per_vertical}.jsonl"
    gold_path = output_dir / f"airline_gold_{target_per_vertical}.jsonl"
    kb_path = output_dir / f"airline_kb_{target_per_vertical}.jsonl"
    report_path = report_dir / f"airline_scaleup_{target_per_vertical}_report.json"
    write_jsonl(prompts_path, prompts)
    write_jsonl(gold_path, gold)
    write_jsonl(kb_path, kb_copy)
    report = write_scaleup_report(
        report_path,
        vertical="airline",
        target_per_vertical=target_per_vertical,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_copy,
        blockers=[],
        warnings=warnings,
        checkpoint=checkpoint_name,
        generation_scope="local_candidate_generation",
        generation_implemented=True,
    )
    return {
        "phase": PHASE,
        "mode": "generate_vertical",
        "vertical": "airline",
        "target_per_vertical": target_per_vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_copy),
        "status_counts": report["status_counts"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "prompts_path": str(prompts_path),
        "gold_path": str(gold_path),
        "kb_path": str(kb_path),
        "report_path": str(report_path),
        "next_step": report["next_step"],
    }


def generate_healthcare_admin_vertical(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    if target_per_vertical not in IMPLEMENTED_GENERATION_TARGETS["healthcare_admin"]:
        raise RuntimeError(
            f"Generation for healthcare_admin at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    qa_status = load_qa_status(Path(args.qa_report))
    qa_ready = qa_ready_for_vertical(qa_status, "healthcare_admin")
    blockers: list[str] = []
    warnings = list(qa_status["warnings"])
    if qa_ready is False:
        blockers.append(f"phase2a_qa_not_ready_for_healthcare_admin_{target_per_vertical}_scale")
    readiness = source_readiness("healthcare_admin", target_per_vertical)
    blockers.extend(readiness["missing_seed_files"])
    report_path = (
        Path(args.report_dir) / f"healthcare_admin_scaleup_{target_per_vertical}_report.json"
    )
    if blockers:
        write_scaleup_report(
            report_path,
            vertical="healthcare_admin",
            target_per_vertical=target_per_vertical,
            prompts=[],
            gold=[],
            kb_rows=[],
            blockers=blockers,
            warnings=warnings,
            checkpoint=checkpoint_name,
            generation_scope="local_candidate_generation",
            generation_implemented=True,
        )
        return {
            "phase": PHASE,
            "mode": "generate_vertical",
            "vertical": "healthcare_admin",
            "blockers": blockers,
            "report_path": str(report_path),
            "next_step": "Resolve blockers and rerun generation.",
        }

    seed_prompts = load_jsonl(VERTICAL_FILES["healthcare_admin"]["prompts"])
    kb_rows = load_jsonl(VERTICAL_FILES["healthcare_admin"]["kb"])
    prompts, gold, kb_copy = build_healthcare_pilot_records(
        target_per_vertical=target_per_vertical,
        seed=int(args.seed),
        seed_prompts=seed_prompts,
        kb_rows=kb_rows,
    )
    output_dir = Path(args.output_dir) / "healthcare_admin"
    prompts_path = output_dir / f"healthcare_admin_prompts_{target_per_vertical}.jsonl"
    gold_path = output_dir / f"healthcare_admin_gold_{target_per_vertical}.jsonl"
    kb_path = output_dir / f"healthcare_admin_kb_{target_per_vertical}.jsonl"
    write_jsonl(prompts_path, prompts)
    write_jsonl(gold_path, gold)
    write_jsonl(kb_path, kb_copy)
    report = write_scaleup_report(
        report_path,
        vertical="healthcare_admin",
        target_per_vertical=target_per_vertical,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_copy,
        blockers=[],
        warnings=warnings,
        checkpoint=checkpoint_name,
        generation_scope="local_candidate_generation",
        generation_implemented=True,
    )
    return {
        "phase": PHASE,
        "mode": "generate_vertical",
        "vertical": "healthcare_admin",
        "target_per_vertical": target_per_vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_copy),
        "status_counts": report["status_counts"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "prompts_path": str(prompts_path),
        "gold_path": str(gold_path),
        "kb_path": str(kb_path),
        "report_path": str(report_path),
        "next_step": report["next_step"],
    }


def generate_research_ai_vertical(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    if target_per_vertical not in IMPLEMENTED_GENERATION_TARGETS["research_ai"]:
        raise RuntimeError(
            f"Generation for research_ai at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    qa_status = load_qa_status(Path(args.qa_report))
    qa_ready = qa_ready_for_vertical(qa_status, "research_ai")
    blockers: list[str] = []
    warnings = list(qa_status["warnings"])
    if qa_ready is False:
        blockers.append(f"phase2a_qa_not_ready_for_research_ai_{target_per_vertical}_scale")
    readiness = source_readiness("research_ai", target_per_vertical)
    blockers.extend(readiness["missing_seed_files"])
    report_path = Path(args.report_dir) / f"research_ai_scaleup_{target_per_vertical}_report.json"
    if blockers:
        write_scaleup_report(
            report_path,
            vertical="research_ai",
            target_per_vertical=target_per_vertical,
            prompts=[],
            gold=[],
            kb_rows=[],
            blockers=blockers,
            warnings=warnings,
            checkpoint=checkpoint_name,
            generation_scope="local_candidate_generation",
            generation_implemented=True,
        )
        return {
            "phase": PHASE,
            "mode": "generate_vertical",
            "vertical": "research_ai",
            "blockers": blockers,
            "report_path": str(report_path),
            "next_step": "Resolve blockers and rerun generation.",
        }

    seed_prompts = load_jsonl(VERTICAL_FILES["research_ai"]["prompts"])
    seed_gold = load_jsonl(VERTICAL_FILES["research_ai"]["gold"])
    kb_rows = load_jsonl(VERTICAL_FILES["research_ai"]["kb"])
    prompts, gold, kb_copy = build_research_ai_pilot_records(
        target_per_vertical=target_per_vertical,
        seed=int(args.seed),
        seed_prompts=seed_prompts,
        seed_gold=seed_gold,
        kb_rows=kb_rows,
    )
    output_dir = Path(args.output_dir) / "research_ai"
    prompts_path = output_dir / f"research_ai_prompts_{target_per_vertical}.jsonl"
    gold_path = output_dir / f"research_ai_gold_{target_per_vertical}.jsonl"
    kb_path = output_dir / f"research_ai_kb_{target_per_vertical}.jsonl"
    write_jsonl(prompts_path, prompts)
    write_jsonl(gold_path, gold)
    write_jsonl(kb_path, kb_copy)
    report = write_scaleup_report(
        report_path,
        vertical="research_ai",
        target_per_vertical=target_per_vertical,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_copy,
        blockers=[],
        warnings=warnings,
        checkpoint=checkpoint_name,
        generation_scope="local_candidate_generation",
        generation_implemented=True,
    )
    return {
        "phase": PHASE,
        "mode": "generate_vertical",
        "vertical": "research_ai",
        "target_per_vertical": target_per_vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_copy),
        "status_counts": report["status_counts"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "prompts_path": str(prompts_path),
        "gold_path": str(gold_path),
        "kb_path": str(kb_path),
        "report_path": str(report_path),
        "next_step": report["next_step"],
    }


def generate_retail_vertical(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    checkpoint_name = get_checkpoint_for_target(target_per_vertical)
    if target_per_vertical not in IMPLEMENTED_GENERATION_TARGETS["retail"]:
        raise RuntimeError(
            f"Generation for retail at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    qa_status = load_qa_status(Path(args.qa_report))
    qa_ready = qa_ready_for_vertical(qa_status, "retail")
    blockers: list[str] = []
    warnings = list(qa_status["warnings"])
    if qa_ready is False:
        blockers.append(f"phase2a_qa_not_ready_for_retail_{target_per_vertical}_scale")
    readiness = source_readiness("retail", target_per_vertical)
    blockers.extend(readiness["missing_seed_files"])
    report_path = Path(args.report_dir) / f"retail_scaleup_{target_per_vertical}_report.json"
    if blockers:
        write_scaleup_report(
            report_path,
            vertical="retail",
            target_per_vertical=target_per_vertical,
            prompts=[],
            gold=[],
            kb_rows=[],
            blockers=blockers,
            warnings=warnings,
            checkpoint=checkpoint_name,
            generation_scope="local_candidate_generation",
            generation_implemented=True,
        )
        return {
            "phase": PHASE,
            "mode": "generate_vertical",
            "vertical": "retail",
            "blockers": blockers,
            "report_path": str(report_path),
            "next_step": "Resolve blockers and rerun generation.",
        }

    seed_prompts = load_jsonl(VERTICAL_FILES["retail"]["prompts"])
    kb_rows = load_jsonl(VERTICAL_FILES["retail"]["kb"])
    prompts, gold, kb_copy = build_retail_pilot_records(
        target_per_vertical=target_per_vertical,
        seed=int(args.seed),
        seed_prompts=seed_prompts,
        kb_rows=kb_rows,
    )
    output_dir = Path(args.output_dir) / "retail"
    prompts_path = output_dir / f"retail_prompts_{target_per_vertical}.jsonl"
    gold_path = output_dir / f"retail_gold_{target_per_vertical}.jsonl"
    kb_path = output_dir / f"retail_kb_{target_per_vertical}.jsonl"
    write_jsonl(prompts_path, prompts)
    write_jsonl(gold_path, gold)
    write_jsonl(kb_path, kb_copy)
    report = write_scaleup_report(
        report_path,
        vertical="retail",
        target_per_vertical=target_per_vertical,
        prompts=prompts,
        gold=gold,
        kb_rows=kb_copy,
        blockers=[],
        warnings=warnings,
        checkpoint=checkpoint_name,
        generation_scope="local_candidate_generation",
        generation_implemented=True,
    )
    return {
        "phase": PHASE,
        "mode": "generate_vertical",
        "vertical": "retail",
        "target_per_vertical": target_per_vertical,
        "prompt_count": len(prompts),
        "gold_count": len(gold),
        "kb_count": len(kb_copy),
        "status_counts": report["status_counts"],
        "critical_issue_count": report["critical_issue_count"],
        "warning_count": report["warning_count"],
        "prompts_path": str(prompts_path),
        "gold_path": str(gold_path),
        "kb_path": str(kb_path),
        "report_path": str(report_path),
        "next_step": report["next_step"],
    }


def generate_vertical(args: argparse.Namespace) -> dict[str, Any]:
    target_per_vertical = int(args.target_per_vertical)
    validate_target(target_per_vertical)
    if args.vertical == "all":
        raise RuntimeError("Pass a single --vertical value for --generate-vertical.")
    if target_per_vertical not in IMPLEMENTED_GENERATION_TARGETS.get(args.vertical, set()):
        raise RuntimeError(
            f"Generation for {args.vertical} at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    if target_per_vertical > 250 and not args.allow_large_local_generation:
        raise RuntimeError(
            f"Generation for {args.vertical} at {target_per_vertical} requires explicit "
            "implementation and prior checkpoint review."
        )
    if args.vertical == "finance":
        return generate_finance_vertical(args)
    if args.vertical == "airline":
        return generate_airline_vertical(args)
    if args.vertical == "healthcare_admin":
        return generate_healthcare_admin_vertical(args)
    if args.vertical == "research_ai":
        return generate_research_ai_vertical(args)
    if args.vertical == "retail":
        return generate_retail_vertical(args)
    raise RuntimeError(
        f"Generation for {args.vertical} at {target_per_vertical} requires explicit "
        "implementation and prior checkpoint review."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate-plan", action="store_true")
    parser.add_argument("--generate-vertical", action="store_true")
    parser.add_argument("--vertical", choices=[*VERTICALS, "all"], default="all")
    parser.add_argument("--target-per-vertical", type=int, default=250)
    parser.add_argument("--scaleup-plan", type=Path, default=DEFAULT_SCALEUP_PLAN)
    parser.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-large-local-generation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_modes = sum(
        bool(value) for value in [args.dry_run, args.generate_plan, args.generate_vertical]
    )
    if selected_modes != 1:
        parser.error("Choose exactly one mode: --dry-run, --generate-plan, or --generate-vertical.")
    try:
        validate_target(int(args.target_per_vertical))
        if args.dry_run:
            summary = dry_run(args)
        elif args.generate_plan:
            summary = generate_plan(args)
        else:
            summary = generate_vertical(args)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
