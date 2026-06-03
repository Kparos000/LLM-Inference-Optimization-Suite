"""Offline retrieval root-cause analysis for Phase 3 quality gates.

This module reads existing retrieval diagnostics and promoted-dataset records.
It does not run retrieval, model inference, GPU work, or external API calls.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")
STRICT_ABLATION_MODES = ("prompt_text_only", "prompt_plus_metadata")
ROOT_CAUSES = (
    "prompt_missing_entity",
    "prompt_missing_metric",
    "prompt_missing_period",
    "metadata_missing_entity",
    "metadata_missing_metric",
    "metadata_missing_period",
    "gold_not_in_corpus",
    "gold_absent_from_candidate_pool",
    "gold_in_candidates_not_final_top5",
    "duplicate_or_near_duplicate_confusion",
    "chunk_too_broad",
    "chunk_too_narrow",
    "weak_dense_similarity",
    "weak_lexical_match",
    "reranker_miscalibrated",
    "evidence_label_too_narrow",
    "multi_hop_or_comparison_query",
    "unknown",
)
RECOMMENDATION_BY_CAUSE = {
    "prompt_missing_entity": "prompt/gold repair",
    "prompt_missing_metric": "prompt/gold repair",
    "prompt_missing_period": "prompt/gold repair",
    "metadata_missing_entity": "metadata enrichment",
    "metadata_missing_metric": "metadata enrichment",
    "metadata_missing_period": "metadata enrichment",
    "gold_not_in_corpus": "corpus/gold alignment repair",
    "gold_absent_from_candidate_pool": "candidate retrieval",
    "gold_in_candidates_not_final_top5": "reranking/final selection",
    "duplicate_or_near_duplicate_confusion": "reranking/final selection",
    "chunk_too_broad": "corpus/chunk repair",
    "chunk_too_narrow": "corpus/chunk repair",
    "weak_dense_similarity": "embedding model/vector index",
    "weak_lexical_match": "candidate retrieval",
    "reranker_miscalibrated": "reranking/final selection",
    "evidence_label_too_narrow": "prompt/gold repair",
    "multi_hop_or_comparison_query": "candidate retrieval",
    "unknown": "candidate retrieval",
}
SUMMARY_FIELDS = [
    "split",
    "ablation_mode",
    "memory_mode",
    "vertical",
    "record_count",
    "failure_count",
    "recall_at_5",
    "mrr",
    "candidate_recall_at_100",
    "candidate_vs_final_recall_gap",
    "primary_root_cause",
    "primary_root_cause_count",
    "root_cause_counts_json",
    "recommended_fix_area",
    "top_recommended_fix",
    "candidate_retrieval_blocker",
    "reranking_blocker",
    "prompt_gold_repair_required",
    "slo_target",
    "slo_margin",
    "slo_status",
]
FINANCE_METRIC_SYNONYMS = {
    "revenue": {"revenue", "sales", "net sales"},
    "operating_income": {"operating income", "operating profit"},
    "net_income": {"net income", "earnings"},
    "margin": {"margin", "gross margin", "operating margin"},
    "capex": {"capex", "capital expenditure", "capital expenditures"},
    "cash_flow": {"cash flow", "operating cash flow", "free cash flow"},
    "research_development": {"r&d", "research and development", "research development"},
    "risk": {"risk", "risk factor", "risk factors"},
    "segment": {"segment", "segments"},
    "guidance": {"guidance", "outlook", "forecast"},
}
FINANCE_TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
QUARTER_PATTERN = re.compile(r"\b(?:q[1-4]|quarter|quarterly|annual|fiscal year|fy)\b", re.I)
FORM_PATTERN = re.compile(r"\b(?:10-k|10-q|8-k)\b", re.I)
XBRL_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z]+(?:Revenue|Income|Expense|Assets|Liabilities|Cash|Sales)\b"
)
RETAIL_ISSUE_TERMS = {
    "return",
    "refund",
    "defect",
    "broken",
    "damaged",
    "quality",
    "rating",
    "review",
    "sentiment",
    "support",
}
RESEARCH_SECTION_TERMS = {
    "abstract",
    "introduction",
    "method",
    "methods",
    "experiment",
    "experiments",
    "result",
    "results",
    "limitation",
    "limitations",
    "appendix",
}
POLICY_TERMS = {
    "policy",
    "refund",
    "cancellation",
    "accessibility",
    "baggage",
    "identity",
    "verification",
    "escalation",
}
HEALTHCARE_TERMS = {
    "appointment",
    "scheduling",
    "privacy",
    "identity",
    "administrative",
    "procedure",
    "safety",
    "clinical",
    "escalation",
}


@dataclass(frozen=True)
class DatasetRecords:
    """Promoted-dataset records indexed by prompt ID and vertical."""

    prompts_by_id: dict[str, dict[str, Any]]
    gold_by_id: dict[str, dict[str, Any]]
    corpus_ids_by_vertical: dict[str, set[str]]


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        msg = f"Expected JSON object at {path}"
        raise ValueError(msg)
    return payload


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read CSV rows."""

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a sorted, indented JSON report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write CSV rows."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write JSONL rows."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    return output


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a field to float."""

    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a field to int."""

    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalized_text(*values: Any) -> str:
    """Return lower-cased searchable text from nested values."""

    parts: list[str] = []

    def append(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for nested in value.values():
                append(nested)
        elif isinstance(value, list):
            for nested in value:
                append(nested)
        else:
            parts.append(str(value))

    for value in values:
        append(value)
    return " ".join(parts).lower()


def load_dataset_records(dataset_root: str | Path) -> DatasetRecords:
    """Load promoted prompt/gold records and benchmark KB ID coverage."""

    root = Path(dataset_root)
    prompts_by_id: dict[str, dict[str, Any]] = {}
    gold_by_id: dict[str, dict[str, Any]] = {}
    corpus_ids_by_vertical: dict[str, set[str]] = {vertical: set() for vertical in VERTICALS}
    for vertical in VERTICALS:
        vertical_root = root / vertical
        prompt_path = vertical_root / f"{vertical}_prompts_2000.jsonl"
        gold_path = vertical_root / f"{vertical}_gold_2000.jsonl"
        kb_path = vertical_root / f"{vertical}_kb_2000.jsonl"
        if prompt_path.exists():
            for row in read_jsonl(prompt_path):
                prompt_id = str(row.get("prompt_id") or "")
                if prompt_id:
                    prompts_by_id[prompt_id] = row
        if gold_path.exists():
            for row in read_jsonl(gold_path):
                prompt_id = str(row.get("prompt_id") or "")
                if prompt_id:
                    gold_by_id[prompt_id] = row
        if kb_path.exists():
            for row in read_jsonl(kb_path):
                add_corpus_ids(corpus_ids_by_vertical[vertical], row)
    return DatasetRecords(
        prompts_by_id=prompts_by_id,
        gold_by_id=gold_by_id,
        corpus_ids_by_vertical=corpus_ids_by_vertical,
    )


def add_corpus_ids(corpus_ids: set[str], row: dict[str, Any]) -> None:
    """Add useful evidence identifiers from a benchmark KB row."""

    for field_name in ("doc_id", "source_id", "chunk_id", "parent_id"):
        value = row.get(field_name)
        if value:
            corpus_ids.add(str(value))
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for value in metadata.values():
            if isinstance(value, str):
                corpus_ids.add(value)
            elif isinstance(value, list):
                corpus_ids.update(str(item) for item in value if item)


def load_slo_targets(path: str | Path) -> dict[str, float]:
    """Load retrieval SLO targets, with safe defaults if the file is absent."""

    defaults = {
        "prompt_text_only": 0.70,
        "prompt_plus_metadata": 0.80,
        "prompt_plus_source_hints": 0.95,
        "finance_prompt_text_only": 0.65,
        "finance_prompt_plus_metadata": 0.80,
    }
    slo_path = Path(path)
    if not slo_path.exists():
        return defaults
    payload = yaml.safe_load(slo_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return defaults
    retrieval = payload.get("retrieval", {})
    if not isinstance(retrieval, dict):
        return defaults
    return {
        "prompt_text_only": safe_float(
            retrieval.get("overall_prompt_text_only_hybrid_recall_at_5"),
            defaults["prompt_text_only"],
        ),
        "prompt_plus_metadata": safe_float(
            retrieval.get("overall_prompt_plus_metadata_hybrid_recall_at_5"),
            defaults["prompt_plus_metadata"],
        ),
        "prompt_plus_source_hints": safe_float(
            retrieval.get("source_hint_assisted_hybrid_recall_at_5"),
            defaults["prompt_plus_source_hints"],
        ),
        "finance_prompt_text_only": safe_float(
            retrieval.get("finance_prompt_text_only_hybrid_recall_at_5"),
            defaults["finance_prompt_text_only"],
        ),
        "finance_prompt_plus_metadata": safe_float(
            retrieval.get("finance_prompt_plus_metadata_hybrid_recall_at_5"),
            defaults["finance_prompt_plus_metadata"],
        ),
    }


def detect_finance_metric(text: str) -> tuple[bool, str | None]:
    """Detect a finance metric family from prompt-visible text."""

    for family, terms in FINANCE_METRIC_SYNONYMS.items():
        if any(term in text for term in terms):
            return True, family
    return False, None


def detect_entity(prompt: dict[str, Any], vertical: str) -> bool:
    """Return whether prompt-visible entity metadata is present."""

    text = normalized_text(prompt.get("question"), prompt)
    if vertical == "finance":
        if prompt.get("ticker") or prompt.get("company"):
            return True
        return any(len(token) <= 5 for token in FINANCE_TICKER_PATTERN.findall(str(prompt)))
    if vertical == "retail":
        return bool(
            prompt.get("product_title") or prompt.get("product_id") or prompt.get("category")
        )
    if vertical == "research_ai":
        raw_metadata = prompt.get("metadata")
        metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        return bool(prompt.get("topic") or metadata.get("source_titles"))
    if vertical == "airline":
        return bool(
            prompt.get("support_type")
            or prompt.get("route")
            or any(term in text for term in POLICY_TERMS)
        )
    if vertical == "healthcare_admin":
        return bool(
            prompt.get("support_type")
            or prompt.get("department")
            or any(term in text for term in HEALTHCARE_TERMS)
        )
    return bool(text.strip())


def feature_detection(prompt: dict[str, Any], vertical: str) -> dict[str, Any]:
    """Return vertical-aware query feature detection flags."""

    text = normalized_text(prompt.get("question"), prompt)
    entity_success = detect_entity(prompt, vertical)
    metric_success = True
    metric_family: str | None = None
    period_success = True
    xbrl_success = False
    form_section_success = False
    issue_success = False
    section_success = False
    policy_success = False
    safety_procedure_success = False
    category_success = False
    product_title_success = False

    if vertical == "finance":
        metric_success, metric_family = detect_finance_metric(text)
        period_success = bool(YEAR_PATTERN.search(text) or QUARTER_PATTERN.search(text))
        xbrl_success = bool(XBRL_PATTERN.search(str(prompt)))
        form_section_success = bool(FORM_PATTERN.search(text) or prompt.get("filing_form"))
    elif vertical == "retail":
        category_success = bool(prompt.get("category"))
        product_title_success = bool(prompt.get("product_title"))
        issue_success = any(term in text for term in RETAIL_ISSUE_TERMS)
    elif vertical == "research_ai":
        raw_metadata = prompt.get("metadata")
        metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        section_success = any(term in text for term in RESEARCH_SECTION_TERMS)
        issue_success = bool(metadata.get("source_titles") or prompt.get("topic"))
    elif vertical == "airline":
        policy_success = any(term in text for term in POLICY_TERMS) or bool(
            prompt.get("support_type")
        )
        safety_procedure_success = any(
            term in text for term in ("fraud", "identity", "accessibility", "escalation")
        )
    elif vertical == "healthcare_admin":
        policy_success = bool(prompt.get("support_type") or prompt.get("department"))
        safety_procedure_success = any(term in text for term in HEALTHCARE_TERMS)

    return {
        "entity_detection_success": entity_success,
        "metric_detection_success": metric_success,
        "metric_family": metric_family,
        "period_detection_success": period_success,
        "xbrl_concept_detection_success": xbrl_success,
        "form_section_detection_success": form_section_success,
        "product_title_detection_success": product_title_success,
        "category_detection_success": category_success,
        "review_issue_detection_success": issue_success,
        "paper_section_detection_success": section_success,
        "policy_type_detection_success": policy_success,
        "safety_or_procedure_detection_success": safety_procedure_success,
    }


def classify_failure(
    row: dict[str, Any],
    *,
    prompt_record: dict[str, Any] | None = None,
    gold_in_corpus: bool = True,
) -> dict[str, Any]:
    """Classify one failed retrieval row or example."""

    vertical = str(row.get("vertical") or (prompt_record.get("vertical") if prompt_record else ""))
    recall_at_5 = safe_float(row.get("recall_at_5"))
    candidate_recall_at_100 = safe_float(
        row.get("candidate_recall_at_100"),
        safe_float(row.get("candidate_recall_at_50")),
    )
    candidate_recall_at_50 = safe_float(row.get("candidate_recall_at_50"))
    candidate_recall_at_20 = safe_float(row.get("candidate_recall_at_20"))
    candidate_recall_at_10 = safe_float(row.get("candidate_recall_at_10"))
    reasons = [str(reason) for reason in row.get("failure_reasons", []) if reason]
    features = feature_detection(prompt_record or {}, vertical) if prompt_record else {}

    if not gold_in_corpus or row.get("gold_not_in_corpus"):
        primary = "gold_not_in_corpus"
    elif (
        vertical == "finance"
        and prompt_record
        and not features.get("metric_detection_success", True)
    ):
        primary = "prompt_missing_metric"
    elif (
        vertical == "finance"
        and prompt_record
        and not features.get("period_detection_success", True)
    ):
        primary = "prompt_missing_period"
    elif prompt_record and not features.get("entity_detection_success", True):
        primary = "prompt_missing_entity"
    elif "gold_not_in_candidate_pool" in reasons:
        primary = "gold_absent_from_candidate_pool"
    elif candidate_recall_at_100 <= 0.0 and (
        row.get("gold_evidence_ids") or row.get("record_count")
    ):
        primary = "gold_absent_from_candidate_pool"
    elif "gold_in_top50_not_top5" in reasons or "gold_in_top100_not_top5" in reasons:
        primary = "gold_in_candidates_not_final_top5"
    elif candidate_recall_at_100 > 0.0 and recall_at_5 < 1.0:
        primary = "gold_in_candidates_not_final_top5"
    elif "chunk_too_broad" in reasons:
        primary = "chunk_too_broad"
    elif "chunk_too_narrow" in reasons:
        primary = "chunk_too_narrow"
    elif "bad_query_terms" in reasons:
        primary = "weak_lexical_match"
    elif "poor_scoring" in reasons:
        primary = "reranker_miscalibrated"
    else:
        primary = "unknown"

    return {
        "primary_root_cause": primary,
        "recommended_fix_area": RECOMMENDATION_BY_CAUSE[primary],
        "candidate_recall_at_10": candidate_recall_at_10,
        "candidate_recall_at_20": candidate_recall_at_20,
        "candidate_recall_at_50": candidate_recall_at_50,
        "candidate_recall_at_100": candidate_recall_at_100,
        "gold_was_in_top_10": candidate_recall_at_10 > 0,
        "gold_was_in_top_20": candidate_recall_at_20 > 0,
        "gold_was_in_top_50": candidate_recall_at_50 > 0 or "gold_in_top50_not_top5" in reasons,
        "final_selector_dropped_gold": (
            candidate_recall_at_50 > 0
            or "gold_in_top50_not_top5" in reasons
            or "gold_in_top100_not_top5" in reasons
        )
        and recall_at_5 < 1.0,
        **features,
    }


def aggregate_root_causes(
    row: dict[str, Any],
    *,
    gold_audit_by_vertical: dict[str, Any],
) -> Counter[str]:
    """Build non-mutually-exclusive root-cause counts for one aggregate row."""

    vertical = str(row.get("vertical") or "")
    ablation_mode = str(row.get("ablation_mode") or "")
    record_count = safe_int(row.get("record_count"))
    failure_count = safe_int(row.get("failure_count"), record_count)
    absent_count = round(record_count * safe_float(row.get("gold_absent_from_top100_rate")))
    final_drop_count = round(record_count * safe_float(row.get("gold_in_top100_but_not_top5_rate")))
    counts: Counter[str] = Counter()
    if absent_count:
        counts["gold_absent_from_candidate_pool"] = absent_count
    if final_drop_count:
        counts["gold_in_candidates_not_final_top5"] = final_drop_count
    if safe_float(row.get("candidate_recall_at_100")) < 0.65 and record_count:
        counts["weak_dense_similarity"] += max(0, failure_count - final_drop_count)
    if safe_float(row.get("mrr")) < 0.5 and safe_float(row.get("candidate_recall_at_100")) > 0.5:
        counts["reranker_miscalibrated"] += final_drop_count or failure_count

    audit = gold_audit_by_vertical.get(vertical, {})
    if isinstance(audit, dict):
        gold_not_in_corpus = safe_int(audit.get("gold_not_in_corpus_count"))
        if gold_not_in_corpus:
            counts["gold_not_in_corpus"] = gold_not_in_corpus
        if ablation_mode in STRICT_ABLATION_MODES:
            prompt_missing_entity = safe_int(audit.get("prompt_missing_entity_count"))
            prompt_missing_metric = safe_int(audit.get("prompt_missing_metric_count"))
            prompt_missing_period = safe_int(audit.get("prompt_missing_period_count"))
            if prompt_missing_entity:
                counts["prompt_missing_entity"] = min(prompt_missing_entity, failure_count)
            if prompt_missing_metric:
                counts["prompt_missing_metric"] = min(prompt_missing_metric, failure_count)
            if prompt_missing_period:
                counts["prompt_missing_period"] = min(prompt_missing_period, failure_count)
    if not counts and failure_count:
        counts["unknown"] = failure_count
    return counts


def primary_cause_from_counts(counts: Counter[str]) -> str:
    """Return the most important root cause from counts with deterministic tie-breaking."""

    if not counts:
        return "unknown"
    priority = {
        "gold_not_in_corpus": 0,
        "prompt_missing_period": 1,
        "prompt_missing_metric": 2,
        "prompt_missing_entity": 3,
        "gold_in_candidates_not_final_top5": 4,
        "reranker_miscalibrated": 5,
        "gold_absent_from_candidate_pool": 6,
        "weak_dense_similarity": 7,
        "chunk_too_broad": 8,
        "chunk_too_narrow": 9,
    }
    return sorted(counts, key=lambda cause: (-counts[cause], priority.get(cause, 100), cause))[0]


def recommended_fix_for_cause(primary: str, row: dict[str, Any]) -> str:
    """Return a concrete recommended fix."""

    if primary in {"prompt_missing_entity", "prompt_missing_metric", "prompt_missing_period"}:
        return (
            "Repair prompt/gold metadata so strict retrieval has visible entity, metric, "
            "and period cues."
        )
    if primary == "gold_not_in_corpus":
        return "Repair corpus/gold alignment before changing retriever scoring."
    if primary == "gold_absent_from_candidate_pool":
        return "Improve candidate retrieval, indexing, and chunking before final selector work."
    if primary == "gold_in_candidates_not_final_top5":
        return (
            "Tune reranking and final top-5 evidence selection using candidate-window diagnostics."
        )
    if primary in {"chunk_too_broad", "chunk_too_narrow"}:
        return "Repair chunk granularity for this vertical before reranking."
    if primary == "weak_dense_similarity":
        return "Improve embedding/indexed text and lexical expansion for strict ablation modes."
    if safe_float(row.get("candidate_recall_at_100")) - safe_float(row.get("recall_at_5")) > 0.20:
        return (
            "Focus on reranking/final selection because candidate recall substantially "
            "exceeds final recall."
        )
    return (
        "Inspect sampled failures and improve candidate retrieval before another optimization pass."
    )


def target_for_row(row: dict[str, Any], slo_targets: dict[str, float]) -> float:
    """Return the applicable SLO threshold for a retrieval summary row."""

    ablation_mode = str(row.get("ablation_mode") or "")
    vertical = str(row.get("vertical") or "")
    if vertical == "finance" and ablation_mode == "prompt_text_only":
        return slo_targets["finance_prompt_text_only"]
    if vertical == "finance" and ablation_mode == "prompt_plus_metadata":
        return slo_targets["finance_prompt_plus_metadata"]
    return slo_targets.get(ablation_mode, 0.0)


def build_summary_rows(
    retrieval_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    *,
    gold_audit_by_vertical: dict[str, Any],
    slo_targets: dict[str, float],
) -> list[dict[str, Any]]:
    """Build vertical/ablation/memory summary rows."""

    diagnostic_index = {
        (
            str(row.get("split")),
            str(row.get("ablation_mode")),
            str(row.get("memory_mode")),
            str(row.get("vertical")),
        ): row
        for row in diagnostic_rows
    }
    summary_rows: list[dict[str, Any]] = []
    for retrieval_row in retrieval_rows:
        if str(retrieval_row.get("split")) != "final_10000":
            continue
        if str(retrieval_row.get("memory_mode")) == "mm0_no_context":
            continue
        key = (
            str(retrieval_row.get("split")),
            str(retrieval_row.get("ablation_mode")),
            str(retrieval_row.get("memory_mode")),
            str(retrieval_row.get("vertical")),
        )
        merged = {**retrieval_row, **diagnostic_index.get(key, {})}
        counts = aggregate_root_causes(merged, gold_audit_by_vertical=gold_audit_by_vertical)
        primary = primary_cause_from_counts(counts)
        target = target_for_row(merged, slo_targets)
        recall_at_5 = safe_float(merged.get("recall_at_5"))
        candidate_recall = safe_float(merged.get("candidate_recall_at_100"))
        final_gap = round(max(0.0, candidate_recall - recall_at_5), 6)
        row = {
            "split": merged.get("split", ""),
            "ablation_mode": merged.get("ablation_mode", ""),
            "memory_mode": merged.get("memory_mode", ""),
            "vertical": merged.get("vertical", ""),
            "record_count": safe_int(merged.get("record_count")),
            "failure_count": safe_int(merged.get("failure_count")),
            "recall_at_5": recall_at_5,
            "mrr": safe_float(merged.get("mrr")),
            "candidate_recall_at_100": candidate_recall,
            "candidate_vs_final_recall_gap": final_gap,
            "primary_root_cause": primary,
            "primary_root_cause_count": counts[primary],
            "root_cause_counts_json": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            "recommended_fix_area": RECOMMENDATION_BY_CAUSE[primary],
            "top_recommended_fix": recommended_fix_for_cause(primary, merged),
            "candidate_retrieval_blocker": counts["gold_absent_from_candidate_pool"]
            >= counts["gold_in_candidates_not_final_top5"],
            "reranking_blocker": counts["gold_in_candidates_not_final_top5"]
            > counts["gold_absent_from_candidate_pool"],
            "prompt_gold_repair_required": any(
                counts[cause] > 0
                for cause in (
                    "prompt_missing_entity",
                    "prompt_missing_metric",
                    "prompt_missing_period",
                    "gold_not_in_corpus",
                )
            ),
            "slo_target": target,
            "slo_margin": round(recall_at_5 - target, 6),
            "slo_status": "PASSED" if recall_at_5 >= target else "FAILED",
        }
        summary_rows.append(row)
    return summary_rows


def gold_ids_for_prompt(prompt_id: str, dataset_records: DatasetRecords) -> list[str]:
    """Return required gold IDs for a prompt."""

    gold = dataset_records.gold_by_id.get(prompt_id, {})
    ids: list[str] = []
    for field in ("required_doc_ids", "required_evidence_ids", "required_chunk_ids"):
        value = gold.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    metadata = gold.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("required_evidence_ids")
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return list(dict.fromkeys(ids))


def gold_in_corpus(prompt_id: str, vertical: str, dataset_records: DatasetRecords) -> bool:
    """Return whether any required gold evidence ID appears in benchmark KB identifiers."""

    ids = gold_ids_for_prompt(prompt_id, dataset_records)
    if not ids:
        return True
    corpus_ids = dataset_records.corpus_ids_by_vertical.get(vertical, set())
    return any(evidence_id in corpus_ids for evidence_id in ids)


def collect_failure_examples(
    diagnostic_report: dict[str, Any],
    dataset_records: DatasetRecords,
) -> list[dict[str, Any]]:
    """Collect and classify sampled failure examples."""

    raw_examples: list[dict[str, Any]] = []
    sample_examples = diagnostic_report.get("sample_failure_examples", [])
    if isinstance(sample_examples, list):
        raw_examples.extend(row for row in sample_examples if isinstance(row, dict))
    finance_specific = diagnostic_report.get("finance_specific", {})
    if isinstance(finance_specific, dict):
        by_ablation = finance_specific.get("failure_examples_by_ablation", {})
        if isinstance(by_ablation, dict):
            for examples in by_ablation.values():
                if isinstance(examples, list):
                    for row in examples:
                        if isinstance(row, dict):
                            raw_examples.append({"vertical": "finance", **row})

    deduped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for example in raw_examples:
        key = (
            str(example.get("prompt_id")),
            str(example.get("ablation_mode")),
            str(example.get("memory_mode")),
            str(example.get("vertical")),
        )
        deduped.setdefault(key, example)

    selected: list[dict[str, Any]] = []
    vertical_counts: Counter[str] = Counter()
    finance_count = 0
    for example in deduped.values():
        vertical = str(example.get("vertical") or "")
        if vertical == "finance":
            if finance_count >= 30:
                continue
            finance_count += 1
        else:
            if vertical_counts[vertical] >= 20:
                continue
            vertical_counts[vertical] += 1
        prompt_id = str(example.get("prompt_id") or "")
        prompt = dataset_records.prompts_by_id.get(prompt_id, {})
        in_corpus = gold_in_corpus(prompt_id, vertical, dataset_records)
        classification = classify_failure(example, prompt_record=prompt, gold_in_corpus=in_corpus)
        gold_ids = gold_ids_for_prompt(prompt_id, dataset_records)
        selected.append(
            {
                **example,
                "root_cause": classification["primary_root_cause"],
                "recommended_fix_area": classification["recommended_fix_area"],
                "task_type": prompt.get("task_type", ""),
                "company": prompt.get("company", ""),
                "ticker": prompt.get("ticker", ""),
                "metric_family": classification.get("metric_family"),
                "period_detection_success": classification.get("period_detection_success"),
                "entity_detection_success": classification.get("entity_detection_success"),
                "metric_detection_success": classification.get("metric_detection_success"),
                "xbrl_concept_detection_success": classification.get(
                    "xbrl_concept_detection_success"
                ),
                "form_section_detection_success": classification.get(
                    "form_section_detection_success"
                ),
                "gold_not_in_corpus": not in_corpus,
                "gold_evidence_ids": gold_ids or example.get("gold_evidence_ids", []),
                "gold_was_in_top_10": classification["gold_was_in_top_10"],
                "gold_was_in_top_20": classification["gold_was_in_top_20"],
                "gold_was_in_top_50": classification["gold_was_in_top_50"],
                "final_selector_dropped_gold": classification["final_selector_dropped_gold"],
            }
        )
    return selected


def summarize_by_field(
    examples: list[dict[str, Any]], field_name: str
) -> dict[str, dict[str, int]]:
    """Count example root causes by an example field."""

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        value = str(example.get(field_name) or "unknown")
        grouped[value][str(example.get("root_cause") or "unknown")] += 1
    return {key: dict(counter) for key, counter in sorted(grouped.items())}


def main_cause_by_vertical(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the dominant strict-mode cause for each vertical."""

    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in summary_rows:
        if row["ablation_mode"] not in STRICT_ABLATION_MODES:
            continue
        counts = json.loads(str(row["root_cause_counts_json"]))
        if isinstance(counts, dict):
            for cause, count in counts.items():
                grouped[str(row["vertical"])][str(cause)] += safe_int(count)
    output: dict[str, dict[str, Any]] = {}
    for vertical, counter in grouped.items():
        primary = primary_cause_from_counts(counter)
        output[vertical] = {
            "main_root_cause": primary,
            "recommended_fix_area": RECOMMENDATION_BY_CAUSE[primary],
            "root_cause_counts": dict(counter),
            "top_recommended_fix": recommended_fix_for_cause(primary, {"vertical": vertical}),
        }
    return output


def blocker_assessment(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess whether candidate retrieval or final selection is the bigger blocker."""

    candidate_count = 0
    reranking_count = 0
    prompt_repair_count = 0
    for row in summary_rows:
        if row["ablation_mode"] not in STRICT_ABLATION_MODES:
            continue
        counts = json.loads(str(row["root_cause_counts_json"]))
        if not isinstance(counts, dict):
            continue
        candidate_count += safe_int(counts.get("gold_absent_from_candidate_pool"))
        reranking_count += safe_int(counts.get("gold_in_candidates_not_final_top5"))
        prompt_repair_count += sum(
            safe_int(counts.get(cause))
            for cause in ("prompt_missing_entity", "prompt_missing_metric", "prompt_missing_period")
        )
    if prompt_repair_count >= max(candidate_count, reranking_count):
        primary = "prompt/gold repair"
    elif reranking_count > candidate_count:
        primary = "reranking/final selection"
    else:
        primary = "candidate retrieval"
    return {
        "candidate_retrieval_failure_count": candidate_count,
        "reranking_final_selection_failure_count": reranking_count,
        "prompt_gold_repair_signal_count": prompt_repair_count,
        "primary_blocker": primary,
        "prompt_gold_repair_required": prompt_repair_count > 0,
    }


def build_retrieval_root_cause_report(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    slo_config: str | Path,
    output_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build and write retrieval root-cause reports."""

    context_path = Path(context_root)
    dataset_records = load_dataset_records(dataset_root)
    retrieval_rows = read_csv_rows(context_path / "retrieval_evaluation_summary.csv")
    diagnostic_rows = read_csv_rows(context_path / "retrieval_diagnostic_summary.csv")
    diagnostic_report = read_json(context_path / "retrieval_diagnostic_report.json")
    gold_audit_report = read_json(context_path / "gold_evidence_audit_report.json")
    evidence_selection_report = read_json(context_path / "evidence_selection_report.json")
    reranker_calibration_report = read_json(context_path / "reranker_calibration_report.json")
    corpus_build_report = read_json(context_path / "corpus_build_report.json")
    corpus_registry = read_json(context_path / "corpus_registry.json")
    slo_targets = load_slo_targets(slo_config)
    gold_audit_by_vertical = gold_audit_report.get("by_vertical", {})
    if not isinstance(gold_audit_by_vertical, dict):
        gold_audit_by_vertical = {}

    summary_rows = build_summary_rows(
        retrieval_rows,
        diagnostic_rows,
        gold_audit_by_vertical=gold_audit_by_vertical,
        slo_targets=slo_targets,
    )
    examples = collect_failure_examples(diagnostic_report, dataset_records)
    by_vertical = main_cause_by_vertical(summary_rows)
    blocker = blocker_assessment(summary_rows)
    finance_text = next(
        (
            row
            for row in summary_rows
            if row["vertical"] == "finance"
            and row["ablation_mode"] == "prompt_text_only"
            and row["memory_mode"] == "mm2_hybrid_top5"
        ),
        {},
    )
    finance_metadata = next(
        (
            row
            for row in summary_rows
            if row["vertical"] == "finance"
            and row["ablation_mode"] == "prompt_plus_metadata"
            and row["memory_mode"] == "mm2_hybrid_top5"
        ),
        {},
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "offline_retrieval_root_cause_no_inference_no_gpu_no_api",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_paid_api_call_triggered": True,
        "retrieval_scoring_modified": False,
        "slo_targets": slo_targets,
        "input_reports": {
            "retrieval_evaluation_report": str(context_path / "retrieval_evaluation_report.json"),
            "retrieval_diagnostic_report": str(context_path / "retrieval_diagnostic_report.json"),
            "gold_evidence_audit_report": str(context_path / "gold_evidence_audit_report.json"),
            "evidence_selection_report": str(context_path / "evidence_selection_report.json"),
            "reranker_calibration_report": str(context_path / "reranker_calibration_report.json"),
            "corpus_registry": str(context_path / "corpus_registry.json"),
            "corpus_build_report": str(context_path / "corpus_build_report.json"),
        },
        "diagnostic_counts_are_not_mutually_exclusive": True,
        "by_vertical": by_vertical,
        "by_task_type": summarize_by_field(examples, "task_type"),
        "by_company_or_ticker": summarize_by_field(examples, "ticker"),
        "by_metric_family": summarize_by_field(examples, "metric_family"),
        "blocker_assessment": blocker,
        "finance_prompt_text_only": finance_text,
        "finance_prompt_plus_metadata": finance_metadata,
        "top_finance_failure_examples": [
            example for example in examples if example.get("vertical") == "finance"
        ][:30],
        "top_failure_examples_by_vertical": {
            vertical: [example for example in examples if example.get("vertical") == vertical][:20]
            for vertical in VERTICALS
        },
        "candidate_recall_vs_final_recall_gap": {
            f"{row['ablation_mode']}|{row['memory_mode']}|{row['vertical']}": row[
                "candidate_vs_final_recall_gap"
            ]
            for row in summary_rows
        },
        "evidence_selection_contract_valid": evidence_selection_report.get(
            "contract_excludes_gold_labels"
        ),
        "reranker_backend": reranker_calibration_report.get("reranker_backend"),
        "corpus_records_validated": corpus_build_report.get("all_context_records_validated"),
        "corpus_registry_entry_count": len(corpus_registry.get("entries", []))
        if isinstance(corpus_registry.get("entries"), list)
        else 0,
        "recommended_next_block": (
            "Repair prompt/gold metadata for finance period/metric cues, then tune final top-5 "
            "selection where candidate recall already exceeds final recall."
            if blocker["prompt_gold_repair_required"]
            else "Tune reranking/final selection before another candidate retrieval pass."
        ),
    }

    output_path = Path(output_root)
    write_json(output_path / "retrieval_root_cause_report.json", report)
    write_csv(output_path / "retrieval_root_cause_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_jsonl(output_path / "retrieval_failure_examples.jsonl", examples)
    return report, summary_rows, examples
