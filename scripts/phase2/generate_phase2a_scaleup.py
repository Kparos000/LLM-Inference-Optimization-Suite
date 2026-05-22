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
GENERATOR_NAME = "phase2a_9_scaleup_airline_pilot"

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
    "airline": {250},
    "healthcare_admin": {250},
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
            "sec_filing_text_manifest": Path(
                "data/processed/finance/sec/filing_text_manifest.jsonl"
            ).exists(),
            "sec_filing_sections_manifest": Path(
                "data/processed/finance/sec/filing_sections_manifest.jsonl"
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
            "paper_text_manifest": Path(
                "data/processed/research_ai/paper_text_manifest.jsonl"
            ).exists(),
            "paper_sections_manifest": Path(
                "data/processed/research_ai/paper_sections_manifest.jsonl"
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
        support_label = support_type.replace("_", " ")
        issue = (
            f"Traveler asks Canada Air for {support_label} help in scale-up scenario "
            f"{prompt_number}. Use only the cited policy evidence."
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
            "route": route_cycle[index % len(route_cycle)],
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
        return (
            "Requester asks MapleCare Health for urgent clinical guidance in scale-up "
            f"scenario {prompt_number}. Respond only with the administrative urgent "
            "clinical boundary workflow and do not provide clinical advice."
        )
    if status == "spam_or_fraud":
        return (
            "A portal message includes suspicious billing or credential claims in "
            f"scale-up scenario {prompt_number}. Decide how the administrative support "
            "queue should handle the low-quality or fraud-like message."
        )
    if status == "out_of_scope":
        return (
            "Requester asks MapleCare Health an unrelated non-healthcare question in "
            f"scale-up scenario {prompt_number}. Use the support boundary policy rather "
            "than answering from general knowledge."
        )
    if status == "escalate":
        return (
            f"Patient asks about {support_label} in scale-up scenario {prompt_number}, "
            f"but the request needs {expected_queue} review before staff can answer. "
            "Use only MapleCare Health administrative policy evidence."
        )
    return (
        f"Patient asks MapleCare Health about {support_label} in scale-up scenario "
        f"{prompt_number}. Provide an administrative answer using only the cited "
        "policy evidence."
    )


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
        return (
            f"Assess whether the cited review evidence for {product_title} ({product_id}) "
            "should be treated as low-quality or spam-like before using it as product "
            f"support evidence in scenario {prompt_number}."
        )
    if status == "insufficient_evidence":
        return (
            f"Determine whether the selected evidence is sufficient to resolve a support "
            f"request for {product_title} ({product_id}) in scenario {prompt_number}."
        )
    if status == "escalate":
        return (
            f"Decide whether the cited product and policy evidence for {product_title} "
            f"({product_id}) requires support escalation in scenario {prompt_number}."
        )
    if status == "out_of_scope":
        return (
            "A user asks an unrelated question outside the selected retail product and "
            f"support-policy corpus in scenario {prompt_number}. Apply the out-of-scope "
            "boundary instead of answering from general memory."
        )
    if task_type == "issue_identification":
        return (
            f"Identify the main support issue themes in the cited review evidence for "
            f"{product_title} ({product_id})."
        )
    if task_type == "extract_structured":
        return (
            f"Extract a JSON support record for {product_title} ({product_id}) with the "
            "issue type, evidence summary, recommended action, and evidence IDs."
        )
    if task_type == "policy_reasoning":
        return (
            f"Apply the synthetic benchmark support policy to the cited review evidence "
            f"for {product_title} ({product_id}); do not treat it as Amazon policy."
        )
    if task_type == "compare_products":
        return (
            f"Compare the cited retail evidence for {product_title} ({product_id}) using "
            "a compact support-ready table."
        )
    return (
        f"Summarize the available retail review evidence for {product_title} "
        f"({product_id}) in support-ready language."
    )


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
    validation_issues = (
        validate_prompt_gold_alignment(prompts, gold)
        + validate_evidence_coverage(gold, kb_rows)
        + validate_no_private_hygiene_terms(prompts + gold + kb_rows)
    )
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
        "status_counts": dict(Counter(str(row.get("expected_status")) for row in prompts)),
        "task_type_counts": dict(Counter(str(row.get("task_type")) for row in prompts)),
        "output_format_counts": dict(
            Counter(str(row.get("expected_output_format")) for row in prompts)
        ),
        "critical_issue_count": len(validation_issues) + len(blockers),
        "warning_count": len(warnings),
        "validation_issues": validation_issues,
        "blockers": blockers,
        "warnings": warnings,
        "next_step": (
            f"Review local {vertical} {target_per_vertical}-scale candidates before "
            "promoting or extending generation."
            if not validation_issues and not blockers
            else "Fix blockers or validation issues before using these candidates."
        ),
    }
    write_json(path, report)
    return report


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
    if args.vertical == "airline":
        return generate_airline_vertical(args)
    if args.vertical == "healthcare_admin":
        return generate_healthcare_admin_vertical(args)
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
