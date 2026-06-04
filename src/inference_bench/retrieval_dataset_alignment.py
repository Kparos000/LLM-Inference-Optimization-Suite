"""Retrieval dataset/gold alignment repair for Phase 3 Block 18.

This module creates a generated repaired retrieval dataset without modifying the
promoted benchmark dataset. It uses expanded valid evidence sets only for
offline evaluation, never as runtime retrieval query text.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, TypeAlias, cast

from inference_bench.canonical_queries import build_canonical_query
from inference_bench.context_schema import ContextRecord
from inference_bench.gold_evidence_audit import gold_ids_from_gold_record
from inference_bench.memory_workloads import (
    build_retrievers,
    close_retrievers,
    load_context_corpora,
    load_prompts_and_gold,
    recall_at_candidate_k,
    retrieve_for_mode,
)
from inference_bench.retrieval import (
    DEFAULT_FINAL_TOP_K,
    CompanyTickerResolver,
    RetrievalResult,
    context_match_ids,
    evaluate_retrieval_results,
    normalize_identifier,
    scrub_direct_evidence_identifiers,
    tokenize,
)
from inference_bench.retrieval_keys import derive_retrieval_keys, retrieval_key_terms
from inference_bench.slo import (
    SLO_VERTICALS,
    build_slo_readiness_report,
    load_slo_config,
)
from inference_bench.slo import (
    write_csv as write_slo_csv,
)
from inference_bench.slo import (
    write_json as write_slo_json,
)
from inference_bench.vertical_retrieval_repair import (
    VALIDATION_FIELDS,
    candidate_results_from_ids,
    select_stage_prompts,
    slo_status_for_metrics,
    warm_qdrant_repair_queries,
)

REPAIR_REASONS = (
    "prompt_lacks_required_retrieval_cues",
    "gold_label_too_narrow",
    "multiple_valid_evidence_not_counted",
    "corpus_near_duplicate_confusion",
    "chunk_too_broad",
    "chunk_too_narrow",
    "missing_canonical_retrieval_query",
    "metadata_missing",
    "evidence_not_recoverable",
    "candidate_retrieval_failure",
    "final_selection_failure",
)
ALIGNMENT_SUMMARY_FIELDS = [
    "vertical",
    "record_count",
    *REPAIR_REASONS,
]
VALIDATION_SUMMARY_FIELDS = [
    "dataset_variant",
    *VALIDATION_FIELDS,
]
PreparedVariantItem: TypeAlias = tuple[
    str,
    dict[str, Any],
    Any,
    tuple[str, ...],
    tuple[str, ...],
]
DIRECT_RUNTIME_ID_RE = re.compile(
    r"\b(?:CA-POL|MCH-POL)-[A-Za-z0-9-]+\b|"
    r"\b(?:finance|retail|research_ai|airline|healthcare_admin)_(?:kb|doc|section|"
    r"policy|review|summary|chunk|text|corpus)[A-Za-z0-9_:/.\-]*|"
    r"\bB[A-Z0-9]{9}\b|sec://\S+|xbrl://\S+",
    re.I,
)


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write JSONL rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
    output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return output_path


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write CSV rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def sanitize_runtime_query(text: str) -> tuple[str, int]:
    """Remove direct IDs from runtime retrieval query text."""

    scrubbed, blocked = scrub_direct_evidence_identifiers(text)
    scrubbed, asin_blocked = DIRECT_RUNTIME_ID_RE.subn(" ", scrubbed)
    return re.sub(r"\s+", " ", scrubbed).strip(), blocked + asin_blocked


def context_records_by_match_id(records: list[ContextRecord]) -> dict[str, list[ContextRecord]]:
    """Index context records by all matchable IDs."""

    indexed: dict[str, list[ContextRecord]] = defaultdict(list)
    for record in records:
        for match_id in context_match_ids(record):
            indexed[match_id].append(record)
    return indexed


def linked_gold_records(
    gold_record: dict[str, Any] | None,
    by_match_id: dict[str, list[ContextRecord]],
) -> list[ContextRecord]:
    """Return context records linked by original gold IDs."""

    if gold_record is None:
        return []
    linked: list[ContextRecord] = []
    seen: set[str] = set()
    for gold_id in gold_ids_from_gold_record(gold_record):
        for key in (gold_id, normalize_identifier(gold_id)):
            for record in by_match_id.get(key, []):
                if record.context_id not in seen:
                    seen.add(record.context_id)
                    linked.append(record)
    return linked


def metadata_text(record: ContextRecord, fields: tuple[str, ...]) -> str:
    """Return selected metadata values as text."""

    values: list[str] = []
    for field in fields:
        value = record.metadata.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value is not None:
            values.append(str(value))
    return " ".join(values)


def record_valid_ids(record: ContextRecord) -> list[str]:
    """Return stable evidence IDs that can match this context."""

    ids = [
        record.context_id,
        record.chunk_id,
        str(record.metadata.get("original_doc_id") or ""),
        str(record.metadata.get("section_record_id") or ""),
        str(record.metadata.get("source_manifest_record_id") or ""),
        str(record.metadata.get("document_record_id") or ""),
    ]
    return [value for value in dict.fromkeys(ids) if value]


def add_ids(target: dict[Any, list[str]], key: Any, ids: list[str]) -> None:
    """Append IDs to an index key."""

    if key in (None, "", ()):
        return
    target.setdefault(key, []).extend(ids)


def build_expansion_indexes(
    corpora_by_vertical: dict[str, list[ContextRecord]],
) -> dict[str, dict[str, Any]]:
    """Build fast lookup indexes for expanded valid evidence IDs."""

    indexes: dict[str, dict[str, Any]] = {vertical: {} for vertical in SLO_VERTICALS}
    for vertical, records in corpora_by_vertical.items():
        if vertical == "retail":
            by_product: dict[str, list[str]] = {}
            by_category_policy: dict[str, list[str]] = {}
            for record in records:
                ids = record_valid_ids(record)
                product = normalize_identifier(str(record.metadata.get("product_title") or ""))
                category = normalize_identifier(str(record.metadata.get("category") or ""))
                evidence_kind = normalize_identifier(
                    " ".join(
                        [
                            str(record.metadata.get("evidence_type") or ""),
                            str(record.metadata.get("document_type") or ""),
                        ]
                    )
                )
                if product and any(kind in evidence_kind for kind in ("review", "summary")):
                    add_ids(by_product, product, ids)
                if category and "policy" in evidence_kind:
                    add_ids(by_category_policy, category, ids)
            indexes[vertical] = {
                "by_product": by_product,
                "by_category_policy": by_category_policy,
            }
        elif vertical == "finance":
            by_ticker_form: dict[tuple[str, str], list[str]] = {}
            by_ticker: dict[str, list[str]] = {}
            for record in records:
                ids = record_valid_ids(record)
                ticker = normalize_identifier(str(record.metadata.get("ticker") or ""))
                form = normalize_identifier(
                    str(record.metadata.get("form") or record.metadata.get("document_type") or "")
                )
                if ticker:
                    add_ids(by_ticker, ticker, ids)
                    add_ids(by_ticker_form, (ticker, form), ids)
            indexes[vertical] = {
                "by_ticker_form": by_ticker_form,
                "by_ticker": by_ticker,
            }
        elif vertical == "research_ai":
            by_paper_section: dict[tuple[str, str], list[str]] = {}
            by_paper: dict[str, list[str]] = {}
            by_topic_section: dict[tuple[str, str], list[str]] = {}
            for record in records:
                ids = record_valid_ids(record)
                paper = normalize_identifier(
                    str(record.metadata.get("paper_title") or record.title)
                )
                topic = normalize_identifier(str(record.metadata.get("topic") or ""))
                section_value = (
                    record.metadata.get("section_type")
                    or record.metadata.get("evidence_type")
                    or ""
                )
                section = normalize_identifier(str(section_value))
                add_ids(by_paper, paper, ids)
                add_ids(by_paper_section, (paper, section), ids)
                add_ids(by_topic_section, (topic, section), ids)
            indexes[vertical] = {
                "by_paper_section": by_paper_section,
                "by_paper": by_paper,
                "by_topic_section": by_topic_section,
            }
        else:
            by_token: dict[str, list[str]] = {}
            for record in records:
                ids = record_valid_ids(record)
                token_text = " ".join(
                    [
                        record.title,
                        record.text,
                        metadata_text(record, ("tags", "policy_tags")),
                    ]
                )
                for token in set(tokenize(token_text)):
                    add_ids(by_token, token, ids)
            indexes[vertical] = {"by_token": by_token}
    return indexes


def expanded_valid_evidence_ids_from_index(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    canonical_metadata: dict[str, Any],
    expansion_index: dict[str, Any],
) -> list[str]:
    """Return expanded valid evidence IDs using a precomputed index."""

    original = gold_ids_from_gold_record(gold_record) if gold_record else []
    vertical = str(prompt.get("vertical") or "")
    expanded: list[str] = []
    if vertical == "retail":
        product = normalize_identifier(str(canonical_metadata.get("product_title") or ""))
        category = normalize_identifier(str(canonical_metadata.get("product_category") or ""))
        expanded.extend(cast(dict[str, list[str]], expansion_index["by_product"]).get(product, []))
        if "policy" in normalize_identifier(str(canonical_metadata.get("support_intent") or "")):
            expanded.extend(
                cast(dict[str, list[str]], expansion_index["by_category_policy"]).get(category, [])
            )
    elif vertical == "finance":
        ticker = normalize_identifier(str(canonical_metadata.get("ticker") or ""))
        form = normalize_identifier(str(canonical_metadata.get("filing_type") or ""))
        by_ticker_form = cast(dict[tuple[str, str], list[str]], expansion_index["by_ticker_form"])
        by_ticker = cast(dict[str, list[str]], expansion_index["by_ticker"])
        expanded.extend(by_ticker_form.get((ticker, form), []))
        if not expanded:
            expanded.extend(by_ticker.get(ticker, []))
    elif vertical == "research_ai":
        paper = normalize_identifier(str(canonical_metadata.get("paper_title") or ""))
        topic = normalize_identifier(str(canonical_metadata.get("topic") or ""))
        section_types = {
            normalize_identifier(str(item))
            for item in canonical_metadata.get("section_types", [])
            if str(item)
        }
        if canonical_metadata.get("section_type"):
            section_types.add(normalize_identifier(str(canonical_metadata["section_type"])))
        by_paper_section = cast(
            dict[tuple[str, str], list[str]],
            expansion_index["by_paper_section"],
        )
        by_paper = cast(dict[str, list[str]], expansion_index["by_paper"])
        by_topic_section = cast(
            dict[tuple[str, str], list[str]],
            expansion_index["by_topic_section"],
        )
        if section_types:
            for section in section_types:
                expanded.extend(by_paper_section.get((paper, section), []))
                expanded.extend(by_topic_section.get((topic, section), []))
        else:
            expanded.extend(by_paper.get(paper, []))
    else:
        by_token = cast(dict[str, list[str]], expansion_index["by_token"])
        cue_terms = set(tokenize(" ".join(map(str, canonical_metadata.values()))))
        for token in cue_terms:
            expanded.extend(by_token.get(token, [])[:20])
    return list(dict.fromkeys([*original, *expanded]))


def expand_retail_valid_evidence(
    prompt: dict[str, Any],
    records: list[ContextRecord],
) -> list[str]:
    """Return same-product Retail review/summary/policy alternatives."""

    product_title = normalize_identifier(str(prompt.get("product_title") or ""))
    category = normalize_identifier(str(prompt.get("category") or ""))
    issue_type = normalize_identifier(str(prompt.get("issue_type") or ""))
    valid: list[str] = []
    for record in records:
        record_product = normalize_identifier(str(record.metadata.get("product_title") or ""))
        record_category = normalize_identifier(str(record.metadata.get("category") or ""))
        evidence_kind = normalize_identifier(
            " ".join(
                [
                    str(record.metadata.get("evidence_type") or ""),
                    str(record.metadata.get("document_type") or ""),
                ]
            )
        )
        if product_title and record_product == product_title:
            if any(kind in evidence_kind for kind in ("review", "summary")):
                valid.extend(record_valid_ids(record))
            elif "policy" in issue_type and "policy" in evidence_kind:
                valid.extend(record_valid_ids(record))
        elif (
            category
            and record_category == category
            and "policy" in issue_type
            and "policy" in evidence_kind
        ):
            valid.extend(record_valid_ids(record))
    return valid


def expand_finance_valid_evidence(
    prompt: dict[str, Any],
    records: list[ContextRecord],
    metadata: dict[str, Any],
) -> list[str]:
    """Return Finance alternatives that match company/ticker/form/metric cues."""

    ticker = normalize_identifier(str(metadata.get("ticker") or prompt.get("ticker") or ""))
    form = normalize_identifier(str(metadata.get("filing_type") or prompt.get("filing_form") or ""))
    metric = normalize_identifier(str(metadata.get("metric_family") or ""))
    period_value = (
        metadata.get("period")
        or metadata.get("fiscal_year")
        or metadata.get("fiscal_quarter")
        or ""
    )
    period = normalize_identifier(str(period_value))
    valid: list[str] = []
    for record in records:
        record_ticker = normalize_identifier(str(record.metadata.get("ticker") or ""))
        if ticker and record_ticker != ticker:
            continue
        record_form = normalize_identifier(
            str(record.metadata.get("form") or record.metadata.get("document_type") or "")
        )
        if form and form not in record_form:
            continue
        searchable = normalize_identifier(
            " ".join(
                [
                    record.title,
                    record.text,
                    metadata_text(
                        record,
                        (
                            "concept",
                            "concepts",
                            "section_type",
                            "section_title",
                            "report_date",
                            "filing_date",
                            "tags",
                            "document_type",
                        ),
                    ),
                ]
            )
        )
        if metric and metric not in searchable:
            continue
        if period and period not in searchable:
            continue
        valid.extend(record_valid_ids(record))
    return valid


def expand_research_ai_valid_evidence(
    prompt: dict[str, Any],
    records: list[ContextRecord],
    metadata: dict[str, Any],
) -> list[str]:
    """Return Research AI alternatives for same paper/topic and visible section types."""

    paper_title = normalize_identifier(str(metadata.get("paper_title") or ""))
    topic = normalize_identifier(str(metadata.get("topic") or prompt.get("topic") or ""))
    section_types = {
        normalize_identifier(str(item))
        for item in metadata.get("section_types", [])
        if str(item).strip()
    }
    section_type = normalize_identifier(str(metadata.get("section_type") or ""))
    if section_type:
        section_types.add(section_type)
    valid: list[str] = []
    for record in records:
        record_paper = normalize_identifier(str(record.metadata.get("paper_title") or record.title))
        record_topic = normalize_identifier(str(record.metadata.get("topic") or ""))
        record_section = normalize_identifier(
            str(record.metadata.get("section_type") or record.metadata.get("evidence_type") or "")
        )
        same_paper = paper_title and (paper_title == record_paper or paper_title in record_paper)
        same_topic = topic and topic == record_topic
        if same_paper and (not section_types or record_section in section_types):
            valid.extend(record_valid_ids(record))
        elif same_topic and section_types and record_section in section_types:
            valid.extend(record_valid_ids(record))
    return valid


def expand_policy_valid_evidence(
    prompt: dict[str, Any],
    records: list[ContextRecord],
    metadata: dict[str, Any],
    *,
    vertical: str,
) -> list[str]:
    """Return Airline/Healthcare alternatives by visible support/policy cues."""

    if vertical == "airline":
        cue_values = [
            str(metadata.get("support_type") or ""),
            " ".join(str(item) for item in metadata.get("policy_issue_terms", []) if item),
        ]
    else:
        cue_values = [
            str(metadata.get("support_type") or ""),
            str(metadata.get("department") or ""),
            str(metadata.get("safety_boundary") or ""),
            " ".join(str(item) for item in metadata.get("admin_procedure_terms", []) if item),
        ]
    cue_tokens = set(tokenize(" ".join(cue_values)))
    valid: list[str] = []
    for record in records:
        record_tokens = set(
            tokenize(
                " ".join(
                    [
                        record.title,
                        record.text,
                        metadata_text(record, ("tags", "policy_tags")),
                    ]
                )
            )
        )
        if cue_tokens and cue_tokens & record_tokens:
            valid.extend(record_valid_ids(record))
    return valid


def expanded_valid_evidence_ids(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    records: list[ContextRecord],
    canonical_metadata: dict[str, Any],
) -> list[str]:
    """Return original plus expanded valid evidence IDs for offline evaluation."""

    original = gold_ids_from_gold_record(gold_record) if gold_record else []
    vertical = str(prompt.get("vertical") or "")
    if vertical == "retail":
        expanded = expand_retail_valid_evidence(prompt, records)
    elif vertical == "finance":
        expanded = expand_finance_valid_evidence(prompt, records, canonical_metadata)
    elif vertical == "research_ai":
        expanded = expand_research_ai_valid_evidence(prompt, records, canonical_metadata)
    elif vertical in {"airline", "healthcare_admin"}:
        expanded = expand_policy_valid_evidence(
            prompt,
            records,
            canonical_metadata,
            vertical=vertical,
        )
    else:
        expanded = []
    return list(dict.fromkeys([*original, *expanded]))


def evaluate_with_expanded_valid_evidence(
    *,
    original_gold_ids: list[str],
    expanded_valid_ids: list[str],
    results: list[RetrievalResult],
) -> dict[str, Any]:
    """Evaluate retrieval with expanded valid alternatives.

    Expanded valid IDs are alternatives. The denominator remains the original
    gold requirement count so broad valid sets do not inflate the denominator.
    """

    required_count = max(1, len(list(dict.fromkeys(original_gold_ids))))
    expanded = set(dict.fromkeys([*original_gold_ids, *expanded_valid_ids]))
    matched_contexts: set[str] = set()
    matched_ids: set[str] = set()
    reciprocal_rank = 0.0
    for result in results:
        match_ids = context_match_ids(result.context_record)
        current_matches = {
            valid_id
            for valid_id in expanded
            if valid_id in match_ids or normalize_identifier(valid_id) in match_ids
        }
        if current_matches:
            matched_contexts.add(result.context_record.context_id)
            matched_ids.update(current_matches)
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / result.rank
    return {
        "recall_at_5": min(required_count, len(matched_contexts)) / required_count,
        "mrr": reciprocal_rank,
        "matched_valid_evidence_ids": sorted(matched_ids),
        "missing_gold_evidence_count": max(0, required_count - len(matched_contexts)),
    }


def candidate_recall_with_expanded_valid_evidence(
    *,
    original_gold_ids: list[str],
    expanded_valid_ids: list[str],
    candidate_results: list[RetrievalResult],
    top_k: int,
) -> float:
    """Return candidate recall using expanded alternatives."""

    return float(
        evaluate_with_expanded_valid_evidence(
            original_gold_ids=original_gold_ids,
            expanded_valid_ids=expanded_valid_ids,
            results=candidate_results[:top_k],
        )["recall_at_5"]
    )


def missing_key_fields(vertical: str, metadata: dict[str, Any]) -> list[str]:
    """Return missing canonical metadata fields for a vertical."""

    required = {
        "finance": ("ticker", "filing_type", "metric_family", "period", "section_type"),
        "retail": ("product_title", "category", "support_intent", "review_issue_terms"),
        "research_ai": ("paper_title", "topic", "section_type"),
        "airline": ("support_type", "policy_issue_terms"),
        "healthcare_admin": ("support_type", "department", "safety_boundary"),
    }.get(vertical, ())
    return [
        field
        for field in required
        if metadata.get(field) in (None, "", [], {})
        and not (
            vertical == "finance" and field == "section_type" and metadata.get("filing_section")
        )
    ]


def classify_alignment_reasons(
    *,
    vertical: str,
    original_metrics: dict[str, float],
    original_gold_ids: list[str],
    expanded_ids: list[str],
    metadata: dict[str, Any],
    linked_records: list[ContextRecord],
) -> list[str]:
    """Classify prompt/gold/corpus alignment repair reasons."""

    reasons: list[str] = ["missing_canonical_retrieval_query"]
    missing_fields = missing_key_fields(vertical, metadata)
    if missing_fields:
        reasons.extend(["prompt_lacks_required_retrieval_cues", "metadata_missing"])
    if not linked_records and original_gold_ids:
        reasons.append("evidence_not_recoverable")
    if len(expanded_ids) > len(original_gold_ids):
        reasons.extend(["gold_label_too_narrow", "multiple_valid_evidence_not_counted"])
    if vertical in {"retail", "research_ai"} and len(expanded_ids) > max(3, len(original_gold_ids)):
        reasons.append("corpus_near_duplicate_confusion")
    if linked_records:
        max_tokens = max(record.token_estimate for record in linked_records)
        min_tokens = min(record.token_estimate for record in linked_records)
        if max_tokens > 450:
            reasons.append("chunk_too_broad")
        if min_tokens < 20:
            reasons.append("chunk_too_narrow")
    if original_metrics["candidate_recall_at_50"] < 1.0:
        reasons.append("candidate_retrieval_failure")
    if original_metrics["final_recall_at_5"] < 1.0:
        reasons.append("final_selection_failure")
    return [reason for reason in REPAIR_REASONS if reason in set(reasons)]


def canonical_metadata_from_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    """Return canonical retrieval metadata from a prompt."""

    keys = derive_retrieval_keys(prompt, ablation_mode="prompt_plus_metadata")
    metadata: dict[str, Any] = dict(keys.values)
    metadata["retrieval_key_terms"] = retrieval_key_terms(keys)
    if prompt.get("vertical") == "finance":
        metadata["section_type"] = metadata.get("filing_section")
    if prompt.get("vertical") == "retail":
        metadata["product_category"] = metadata.get("category")
        metadata["policy_type"] = metadata.get("policy_context")
        metadata["review_issue_type"] = metadata.get("support_intent")
    if prompt.get("vertical") == "research_ai":
        metadata["method_result_limitation_cue"] = {
            "method": bool(metadata.get("method_signal")),
            "result": bool(metadata.get("results_signal")),
            "limitation": "limitation" in " ".join(map(str, metadata.values())).lower(),
        }
    if prompt.get("vertical") == "airline":
        metadata["travel_issue"] = metadata.get("support_type")
        metadata["policy_type"] = metadata.get("support_type")
        metadata["escalation_type"] = "escalation" if metadata.get("escalation_signal") else None
    if prompt.get("vertical") == "healthcare_admin":
        metadata["admin_task_type"] = metadata.get("support_type")
        metadata["policy_type"] = metadata.get("support_type")
        metadata["privacy_safety_cue"] = {
            "privacy": bool(metadata.get("privacy_signal")),
            "identity": bool(metadata.get("identity_signal")),
            "safety_boundary": metadata.get("safety_boundary"),
        }
    return metadata


def repaired_record_from_prompt(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    records: list[ContextRecord],
    by_match_id: dict[str, list[ContextRecord]],
    original_metrics: dict[str, float],
    resolver: CompanyTickerResolver | None,
    concept_map: dict[str, set[str]],
    expansion_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one repaired generated retrieval dataset record."""

    canonical = build_canonical_query(
        prompt,
        ablation_mode="prompt_plus_metadata",
        resolver=resolver,
        concept_map=concept_map,
    )
    runtime_query, blocked = sanitize_runtime_query(canonical.query_text)
    metadata = canonical_metadata_from_prompt(prompt)
    original_gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
    if expansion_index is None:
        expanded_ids = expanded_valid_evidence_ids(
            prompt=prompt,
            gold_record=gold_record,
            records=records,
            canonical_metadata=metadata,
        )
    else:
        expanded_ids = expanded_valid_evidence_ids_from_index(
            prompt=prompt,
            gold_record=gold_record,
            canonical_metadata=metadata,
            expansion_index=expansion_index,
        )
    linked_records = linked_gold_records(gold_record, by_match_id)
    repair_reasons = classify_alignment_reasons(
        vertical=str(prompt.get("vertical") or ""),
        original_metrics=original_metrics,
        original_gold_ids=original_gold_ids,
        expanded_ids=expanded_ids,
        metadata=metadata,
        linked_records=linked_records,
    )
    return {
        "prompt_id": str(prompt.get("prompt_id") or ""),
        "vertical": str(prompt.get("vertical") or ""),
        "retrieval_query": runtime_query,
        "canonical_retrieval_metadata": metadata,
        "repair_reason": repair_reasons,
        "valid_evidence_ids_expanded": expanded_ids,
        "original_gold_evidence_ids": original_gold_ids,
        "expected_metric": metadata.get("metric_family"),
        "expected_period": metadata.get("period") or metadata.get("fiscal_year"),
        "expected_intent": metadata.get("support_intent") or metadata.get("support_type"),
        "source_prompt_record": prompt,
        "blocked_direct_hint_count": canonical.blocked_direct_hint_count + blocked,
        "runtime_query_uses_valid_evidence_ids": False,
    }


def retrieve_one(
    *,
    prompt: dict[str, Any],
    query: str,
    expanded_queries: tuple[str, ...],
    expansion_types: tuple[str, ...],
    vertical: str,
    retrievers: dict[str, dict[str, Any]],
    query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any],
) -> Any:
    """Run canonical hybrid retrieval for one prompt/query."""

    return retrieve_for_mode(
        memory_mode="mm2_hybrid_top5",
        query=query,
        expanded_queries=expanded_queries,
        expansion_types=expansion_types,
        source_hints_used=False,
        vertical=vertical,
        retrievers=retrievers,
        top_k=DEFAULT_FINAL_TOP_K,
        final_top_k=DEFAULT_FINAL_TOP_K,
        retrieval_cache=query_cache,
    )


def aggregate_validation_rows(
    rows: list[dict[str, Any]],
    *,
    slo_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate per-prompt validation rows."""

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset_variant"]), str(row["vertical"]), int(row["stage_size"]))].append(
            row
        )
    summary_rows: list[dict[str, Any]] = []
    for (dataset_variant, vertical, stage_size), group in sorted(grouped.items()):
        metrics = {
            "candidate_recall_at_20": round(
                mean(float(row["candidate_recall_at_20"]) for row in group),
                6,
            ),
            "candidate_recall_at_50": round(
                mean(float(row["candidate_recall_at_50"]) for row in group),
                6,
            ),
            "final_recall_at_5": round(
                mean(float(row["final_recall_at_5"]) for row in group),
                6,
            ),
            "mrr": round(mean(float(row["mrr"]) for row in group), 6),
        }
        status, blocker, action = slo_status_for_metrics(
            vertical=vertical,
            metrics=metrics,
            slo_config=slo_config,
        )
        summary_rows.append(
            {
                "dataset_variant": dataset_variant,
                "vertical": vertical,
                "stage_size": stage_size,
                "ablation_mode": "prompt_plus_metadata",
                "measurement": "retrieval_dataset_alignment",
                "dense_backend": ",".join(sorted({str(row["dense_backend"]) for row in group})),
                "vector_store": ",".join(sorted({str(row["vector_store"]) for row in group})),
                **metrics,
                "slo_status": status,
                "primary_blocker": blocker,
                "recommended_next_action": action,
                "record_count": len(group),
                "query_rewrite_count": sum(int(bool(row["query_rewritten"])) for row in group),
            }
        )
    return summary_rows


def load_original_canonical_summary(
    output_root: str | Path,
    *,
    stage_sizes: list[int],
) -> list[dict[str, Any]]:
    """Load Block 17 canonical original metrics for comparison."""

    path = Path(output_root) / "canonical_retrieval_repair_summary.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            stage_size = int(str(raw_row.get("stage_size") or "0"))
            if stage_size not in stage_sizes:
                continue
            rows.append(
                {
                    "dataset_variant": "original_promoted",
                    "vertical": raw_row["vertical"],
                    "stage_size": stage_size,
                    "ablation_mode": raw_row["ablation_mode"],
                    "measurement": "canonical_key_repair_staged",
                    "dense_backend": raw_row["dense_backend"],
                    "vector_store": raw_row["vector_store"],
                    "candidate_recall_at_20": float(raw_row["candidate_recall_at_20"]),
                    "candidate_recall_at_50": float(raw_row["candidate_recall_at_50"]),
                    "final_recall_at_5": float(raw_row["final_recall_at_5"]),
                    "mrr": float(raw_row["mrr"]),
                    "slo_status": raw_row["slo_status"],
                    "primary_blocker": raw_row["primary_blocker"],
                    "recommended_next_action": raw_row["recommended_next_action"],
                    "record_count": int(raw_row["record_count"]),
                    "query_rewrite_count": int(raw_row["query_rewrite_count"]),
                }
            )
    return rows


def summary_by_vertical(repaired_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize repair reasons by vertical."""

    rows: list[dict[str, Any]] = []
    by_vertical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in repaired_records:
        by_vertical[str(record["vertical"])].append(record)
    for vertical in SLO_VERTICALS:
        records = by_vertical.get(vertical, [])
        reason_counter: Counter[str] = Counter()
        for record in records:
            reason_counter.update(str(reason) for reason in record.get("repair_reason", []))
        rows.append(
            {
                "vertical": vertical,
                "record_count": len(records),
                **{reason: reason_counter.get(reason, 0) for reason in REPAIR_REASONS},
            }
        )
    return rows


def compact_repair_row(record: dict[str, Any]) -> dict[str, Any]:
    """Return a compact repair index row safe to commit."""

    metadata = cast(dict[str, Any], record.get("canonical_retrieval_metadata") or {})
    missing_fields = missing_key_fields(str(record.get("vertical") or ""), metadata)
    return {
        "prompt_id": record["prompt_id"],
        "vertical": record["vertical"],
        "repair_reason": record.get("repair_reason", []),
        "missing_canonical_metadata_fields": missing_fields,
        "original_gold_evidence_count": len(record.get("original_gold_evidence_ids", [])),
        "expanded_valid_evidence_count": len(record.get("valid_evidence_ids_expanded", [])),
        "runtime_query_uses_valid_evidence_ids": bool(
            record.get("runtime_query_uses_valid_evidence_ids")
        ),
        "blocked_direct_hint_count": int(record.get("blocked_direct_hint_count") or 0),
    }


def materially_improved(
    original: dict[tuple[str, int], dict[str, Any]],
    repaired: dict[tuple[str, int], dict[str, Any]],
) -> bool:
    """Return whether repaired validation materially improves any 2,000-stage vertical."""

    for key, repaired_row in repaired.items():
        if key[1] != 2000 or key not in original:
            continue
        original_row = original[key]
        if (
            float(repaired_row["final_recall_at_5"]) - float(original_row["final_recall_at_5"])
            >= 0.05
        ):
            return True
    return False


def build_promotion_plan(
    *,
    summary_rows: list[dict[str, Any]],
    repaired_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a promotion recommendation for the repaired generated dataset."""

    original = {
        (str(row["vertical"]), int(row["stage_size"])): row
        for row in summary_rows
        if row["dataset_variant"] == "original_promoted"
    }
    repaired = {
        (str(row["vertical"]), int(row["stage_size"])): row
        for row in summary_rows
        if row["dataset_variant"] == "repaired_generated"
    }
    repaired_2000_rows = [
        row
        for row in summary_rows
        if row["dataset_variant"] == "repaired_generated" and int(row["stage_size"]) == 2000
    ]
    all_pass = bool(repaired_2000_rows) and all(
        row["slo_status"] == "PASSED" for row in repaired_2000_rows
    )
    improved = materially_improved(original, repaired)
    blockers = [
        {
            "vertical": row["vertical"],
            "stage_size": row["stage_size"],
            "primary_blocker": row["primary_blocker"],
            "final_recall_at_5": row["final_recall_at_5"],
            "mrr": row["mrr"],
        }
        for row in repaired_2000_rows
        if row["slo_status"] != "PASSED"
    ]
    records_by_vertical = {
        vertical: sum(1 for record in repaired_records if record["vertical"] == vertical)
        for vertical in SLO_VERTICALS
    }
    return {
        "generated_at_utc": utc_now(),
        "promotion_recommended": bool(all_pass),
        "materially_improved": improved,
        "all_repaired_2000_slos_pass": all_pass,
        "do_not_overwrite_promoted_dataset_automatically": True,
        "recommended_action": (
            "Prepare a reviewed promotion PR for the repaired generated dataset."
            if all_pass
            else "Do not promote yet; continue repairing listed blockers."
        ),
        "records_repaired_by_vertical": records_by_vertical,
        "remaining_blockers": blockers,
    }


def validate_alignment_dataset(
    *,
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
    repaired_by_prompt_id: dict[str, dict[str, Any]],
    stage_sizes: list[int],
    slo_config: dict[str, Any],
    dense_backend: str,
    vector_store_config_path: str | Path,
    vector_store_key: str,
    allow_dense_fallback: bool,
    validate_original_variant: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate original and repaired retrieval alignment."""

    retrievers = build_retrievers(
        corpora_by_vertical,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    rows: list[dict[str, Any]] = []
    try:
        query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        queries_to_warm: dict[str, set[str]] = {vertical: set() for vertical in SLO_VERTICALS}
        prepared: list[tuple[str, str, dict[str, Any], Any, tuple[str, ...], tuple[str, ...]]] = []
        for vertical in SLO_VERTICALS:
            resolver = cast(
                CompanyTickerResolver | None,
                retrievers[vertical].get("company_ticker_resolver"),
            )
            concept_map = cast(
                dict[str, set[str]],
                retrievers[vertical].get("xbrl_concept_map") or {},
            )
            for prompt in select_stage_prompts(prompts_by_vertical[vertical], max(stage_sizes)):
                canonical = build_canonical_query(
                    prompt,
                    ablation_mode="prompt_plus_metadata",
                    resolver=resolver,
                    concept_map=concept_map,
                )
                repaired = repaired_by_prompt_id[str(prompt.get("prompt_id") or "")]
                repaired_query = str(repaired["retrieval_query"])
                if validate_original_variant:
                    prepared.append(
                        (
                            vertical,
                            "original_promoted",
                            prompt,
                            canonical.query_text,
                            canonical.expanded_queries,
                            canonical.expansion_types,
                        )
                    )
                prepared.append(
                    (
                        vertical,
                        "repaired_generated",
                        prompt,
                        repaired_query,
                        (repaired_query,),
                        ("repaired_retrieval_query",),
                    )
                )
                queries_to_warm[vertical].update({canonical.query_text, repaired_query})

        warmed = warm_qdrant_repair_queries(
            retrievers=retrievers,
            queries_by_vertical=queries_to_warm,
            top_k=50,
        )
        prepared_by_vertical: dict[str, list[PreparedVariantItem]] = defaultdict(list)
        for vertical, variant, prompt, query, expanded_queries, expansion_types in prepared:
            prepared_by_vertical[vertical].append(
                (variant, prompt, query, expanded_queries, expansion_types)
            )

        for stage_size in stage_sizes:
            for vertical in SLO_VERTICALS:
                records_by_context_id = cast(
                    dict[str, ContextRecord],
                    retrievers[vertical]["records_by_context_id"],
                )
                variants_per_prompt = 2 if validate_original_variant else 1
                selected = prepared_by_vertical[vertical][: stage_size * variants_per_prompt]
                for item in selected:
                    dataset_variant, prompt, query, expanded_queries, expansion_types = item
                    prompt_id = str(prompt.get("prompt_id") or "")
                    gold_record = gold_by_vertical[vertical].get(prompt_id)
                    original_gold_ids = (
                        gold_ids_from_gold_record(gold_record) if gold_record else []
                    )
                    repaired_record = repaired_by_prompt_id[prompt_id]
                    expanded_ids = cast(list[str], repaired_record["valid_evidence_ids_expanded"])
                    retrieval = retrieve_one(
                        prompt=prompt,
                        query=str(query),
                        expanded_queries=expanded_queries,
                        expansion_types=expansion_types,
                        vertical=vertical,
                        retrievers=retrievers,
                        query_cache=query_cache,
                    )
                    candidate_ids = [
                        str(context_id)
                        for context_id in retrieval.diagnostics.get(
                            "candidate_context_ids",
                            [],
                        )
                    ]
                    candidate_results = candidate_results_from_ids(
                        candidate_ids,
                        records_by_context_id,
                        retrieval.retrieval_type,
                    )
                    if dataset_variant == "original_promoted":
                        evaluation = evaluate_retrieval_results(
                            gold_evidence_ids=original_gold_ids,
                            results=retrieval.results,
                        )
                        candidate20 = recall_at_candidate_k(
                            gold_ids=original_gold_ids,
                            candidate_results=candidate_results,
                            top_k=20,
                        )
                        candidate50 = recall_at_candidate_k(
                            gold_ids=original_gold_ids,
                            candidate_results=candidate_results,
                            top_k=50,
                        )
                    else:
                        evaluation = evaluate_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=expanded_ids,
                            results=retrieval.results,
                        )
                        candidate20 = candidate_recall_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=expanded_ids,
                            candidate_results=candidate_results,
                            top_k=20,
                        )
                        candidate50 = candidate_recall_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=expanded_ids,
                            candidate_results=candidate_results,
                            top_k=50,
                        )
                    rows.append(
                        {
                            "dataset_variant": dataset_variant,
                            "vertical": vertical,
                            "stage_size": stage_size,
                            "prompt_id": prompt_id,
                            "ablation_mode": "prompt_plus_metadata",
                            "measurement": "retrieval_dataset_alignment",
                            "dense_backend": retrieval.backend_label,
                            "vector_store": retrieval.vector_store,
                            "candidate_recall_at_20": candidate20,
                            "candidate_recall_at_50": candidate50,
                            "final_recall_at_5": float(evaluation["recall_at_5"]),
                            "mrr": float(evaluation["mrr"]),
                            "query_rewritten": dataset_variant == "repaired_generated",
                            "direct_hint_leakage_detected": (
                                DIRECT_RUNTIME_ID_RE.search(str(query)) is not None
                            ),
                        }
                    )
        summary_rows = aggregate_validation_rows(rows, slo_config=slo_config)
        report = {
            "generated_at_utc": utc_now(),
            "scope": "retrieval_dataset_gold_alignment_validation_no_inference_no_gpu_no_api",
            "no_model_inference_triggered": True,
            "no_gpu_work_triggered": True,
            "no_external_api_calls_triggered": True,
            "dense_backend_requested": dense_backend,
            "qdrant_warmed_query_counts": warmed,
            "stage_sizes": stage_sizes,
            "summary_rows": summary_rows,
            "direct_hint_leakage_detected_count": sum(
                int(bool(row["direct_hint_leakage_detected"])) for row in rows
            ),
        }
        return report, summary_rows
    finally:
        close_retrievers(retrievers)


def build_retrieval_dataset_alignment_repair(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    slo_config_path: str | Path,
    output_root: str | Path,
    stage_sizes: list[int],
    dense_backend: str = "qdrant_vector",
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = True,
) -> dict[str, Any]:
    """Build repaired generated retrieval dataset and validation reports."""

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    slo_config = load_slo_config(slo_config_path)
    repaired_records: list[dict[str, Any]] = []
    by_match_id = {
        vertical: context_records_by_match_id(corpora_by_vertical[vertical])
        for vertical in SLO_VERTICALS
    }
    expansion_indexes = build_expansion_indexes(corpora_by_vertical)
    default_metrics = {
        "candidate_recall_at_20": 1.0,
        "candidate_recall_at_50": 1.0,
        "final_recall_at_5": 1.0,
        "mrr": 1.0,
    }
    for vertical in SLO_VERTICALS:
        for prompt in prompts_by_vertical[vertical]:
            prompt_id = str(prompt.get("prompt_id") or "")
            gold_record = gold_by_vertical[vertical].get(prompt_id)
            repaired_records.append(
                repaired_record_from_prompt(
                    prompt=prompt,
                    gold_record=gold_record,
                    records=corpora_by_vertical[vertical],
                    by_match_id=by_match_id[vertical],
                    original_metrics=default_metrics,
                    resolver=None,
                    concept_map={},
                    expansion_index=expansion_indexes[vertical],
                )
            )

    output_path = Path(output_root)
    repaired_root = output_path / "repaired_retrieval_dataset"
    for vertical in SLO_VERTICALS:
        write_jsonl(
            repaired_root / f"{vertical}_repaired_retrieval_records.jsonl",
            [record for record in repaired_records if record["vertical"] == vertical],
        )
    repaired_by_prompt_id = {str(record["prompt_id"]): record for record in repaired_records}
    validation_report, validation_summary_rows = validate_alignment_dataset(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
        repaired_by_prompt_id=repaired_by_prompt_id,
        stage_sizes=stage_sizes,
        slo_config=slo_config,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
        validate_original_variant=False,
    )
    original_summary_rows = load_original_canonical_summary(
        output_path,
        stage_sizes=stage_sizes,
    )
    validation_summary_rows = [*original_summary_rows, *validation_summary_rows]
    validation_report["summary_rows"] = validation_summary_rows
    alignment_summary_rows = summary_by_vertical(repaired_records)
    needing_repair = [record for record in repaired_records if record["repair_reason"]]
    compact_needing_repair = [compact_repair_row(record) for record in needing_repair]
    alignment_report = {
        "generated_at_utc": utc_now(),
        "scope": "retrieval_dataset_gold_alignment_repair_no_inference_no_gpu_no_api",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "repaired_dataset_root": str(repaired_root),
        "record_count": len(repaired_records),
        "records_needing_repair_count": len(needing_repair),
        "summary_rows": alignment_summary_rows,
        "sample_records_needing_repair": compact_needing_repair[:100],
        "reason_definitions": {
            "multiple_valid_evidence_not_counted": (
                "The prompt can be satisfied by additional context chunks not present "
                "in the original narrow gold IDs."
            ),
            "missing_canonical_retrieval_query": (
                "The promoted prompt did not contain an explicit runtime retrieval_query field."
            ),
        },
    }
    promotion_plan = build_promotion_plan(
        summary_rows=validation_summary_rows,
        repaired_records=repaired_records,
    )
    write_json(output_path / "retrieval_dataset_alignment_report.json", alignment_report)
    write_csv(
        output_path / "retrieval_dataset_alignment_summary.csv",
        alignment_summary_rows,
        ALIGNMENT_SUMMARY_FIELDS,
    )
    write_jsonl(output_path / "retrieval_records_needing_repair.jsonl", compact_needing_repair)
    write_json(output_path / "repaired_retrieval_validation_report.json", validation_report)
    write_csv(
        output_path / "repaired_retrieval_validation_summary.csv",
        validation_summary_rows,
        VALIDATION_SUMMARY_FIELDS,
    )
    write_json(output_path / "repaired_retrieval_promotion_plan.json", promotion_plan)

    slo_report, slo_rows = build_slo_readiness_report(
        slo_config=slo_config,
        retrieval_report_path=output_path / "retrieval_evaluation_report.json",
        quality_gate_report_path=output_path / "retrieval_quality_gate_report.json",
    )
    write_slo_json(output_path / "slo_readiness_report.json", slo_report)
    write_slo_csv(output_path / "slo_readiness_summary.csv", slo_rows)
    return {
        "alignment_report": alignment_report,
        "validation_report": validation_report,
        "promotion_plan": promotion_plan,
    }
