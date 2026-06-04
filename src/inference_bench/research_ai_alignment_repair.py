"""Research AI retrieval dataset/gold alignment repair for Phase 3 Block 19.

This module repairs Research AI alignment only. Expanded valid evidence sets are
used for offline retrieval evaluation and promotion planning; they are never
rendered into runtime retrieval queries.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, cast

from inference_bench.canonical_queries import build_canonical_query
from inference_bench.context_schema import ContextRecord
from inference_bench.gold_evidence_audit import gold_ids_from_gold_record
from inference_bench.memory_workloads import (
    build_retrievers,
    close_retrievers,
    load_context_corpora,
    load_prompts_and_gold,
    retrieve_for_mode,
)
from inference_bench.retrieval import (
    DEFAULT_FINAL_TOP_K,
    CompanyTickerResolver,
    RetrievalResult,
    context_match_ids,
    normalize_identifier,
    tokenize,
)
from inference_bench.retrieval_dataset_alignment import (
    DIRECT_RUNTIME_ID_RE,
    VALIDATION_SUMMARY_FIELDS,
    aggregate_validation_rows,
    build_promotion_plan,
    canonical_metadata_from_prompt,
    context_records_by_match_id,
    expanded_valid_evidence_ids_from_index,
    load_original_canonical_summary,
    sanitize_runtime_query,
    write_csv,
    write_json,
    write_jsonl,
)
from inference_bench.slo import SLO_METRIC_FAMILIES, SLO_VERTICALS, load_slo_config
from inference_bench.vertical_retrieval_repair import (
    candidate_results_from_ids,
    select_stage_prompts,
    slo_status_for_metrics,
    warm_qdrant_repair_queries,
)

RESEARCH_VERTICAL = "research_ai"
FAILURE_CLASSES = (
    "paper_title_ambiguity",
    "section_type_ambiguity",
    "method_vs_result_confusion",
    "limitation_vs_discussion_confusion",
    "topic_overlap_across_papers",
    "narrow_gold_section",
    "multiple_valid_sections_not_counted",
    "candidate_absent_from_top50",
    "candidate_present_but_not_top5",
    "near_duplicate_section_confusion",
)
ALIGNMENT_FIELDS = [
    "stage_size",
    "dataset_variant",
    "record_count",
    "candidate_recall_at_20",
    "candidate_recall_at_50",
    "final_recall_at_5",
    "mrr",
    "slo_status",
    *FAILURE_CLASSES,
]
RESEARCH_SECTION_FAMILIES = {
    "abstract": "overview",
    "introduction": "overview",
    "background": "overview",
    "overview": "overview",
    "related_work": "overview",
    "relatedwork": "overview",
    "relatedworks": "overview",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "algorithm": "method",
    "training": "method",
    "data": "method",
    "dataset": "method",
    "datasets": "method",
    "experiments": "results",
    "experiment": "results",
    "experimental": "results",
    "experimentalsetup": "results",
    "mainresults": "results",
    "results": "results",
    "result": "results",
    "evaluation": "results",
    "analysis": "results",
    "ablation": "results",
    "limitations": "limitations",
    "limitation": "limitations",
    "discussion": "limitations",
    "conclusion": "limitations",
    "conclusions": "limitations",
}
FAMILY_ALIASES = {
    "overview": {
        "abstract",
        "introduction",
        "background",
        "overview",
        "related_work",
        "relatedwork",
        "relatedworks",
    },
    "method": {
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "algorithm",
        "training",
        "data",
        "dataset",
        "datasets",
    },
    "results": {
        "results",
        "result",
        "experiments",
        "experiment",
        "evaluation",
        "analysis",
        "ablation",
    },
    "limitations": {
        "limitations",
        "limitation",
        "discussion",
        "conclusion",
        "conclusions",
    },
}
PAPER_TITLE_MISSING_RE = re.compile(r"\bwhich cited section supports\b", re.I)
BLOCK18_RESEARCH_AI_BASELINE = {
    500: {
        "candidate_recall_at_20": 0.738,
        "candidate_recall_at_50": 0.9745,
        "final_recall_at_5": 0.6175,
        "mrr": 1.0,
    },
    2000: {
        "candidate_recall_at_20": 0.742017,
        "candidate_recall_at_50": 0.939806,
        "final_recall_at_5": 0.596498,
        "mrr": 0.882217,
    },
}


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def normalize_section(value: str | None) -> str:
    """Normalize a section label."""

    return normalize_identifier(str(value or "").replace("-", "_"))


def section_family(value: str | None) -> str:
    """Map a section label to a coarse Research AI family."""

    normalized = normalize_section(value)
    return RESEARCH_SECTION_FAMILIES.get(normalized, normalized or "other")


def record_section_family(record: ContextRecord) -> str:
    """Return the section family for a context record."""

    for field in ("section_type", "evidence_type", "section_title", "document_type"):
        value = record.metadata.get(field)
        if value is None:
            continue
        for token in tokenize(str(value).replace("-", "_")):
            family = section_family(token)
            if family != "other":
                return family
    if record.source_type == "paper_abstract":
        return "overview"
    return "other"


def target_section_families(metadata: dict[str, Any]) -> list[str]:
    """Return target section families from prompt-visible metadata."""

    sections = [str(item) for item in metadata.get("section_types", []) if str(item).strip()]
    if metadata.get("section_type"):
        sections.append(str(metadata["section_type"]))
    families = [section_family(section) for section in sections if section_family(section)]
    if bool(metadata.get("method_signal")) and "method" not in families:
        families.append("method")
    if bool(metadata.get("results_signal")) and "results" not in families:
        families.append("results")
    return list(dict.fromkeys(families or ["overview", "method", "results"]))


def paper_key_from_metadata(metadata: dict[str, Any]) -> str:
    """Return normalized paper title key from prompt metadata."""

    return normalize_identifier(str(metadata.get("paper_title") or ""))


def record_paper_key(record: ContextRecord) -> str:
    """Return normalized paper title key from a context record."""

    return normalize_identifier(
        str(record.metadata.get("paper_title") or record.title or record.parent_id)
    )


def record_topic_key(record: ContextRecord) -> str:
    """Return normalized topic key from a context record."""

    topic = record.metadata.get("topic") or record.metadata.get("topics") or ""
    if isinstance(topic, list):
        return normalize_identifier(" ".join(str(item) for item in topic))
    return normalize_identifier(str(topic))


def record_valid_ids(record: ContextRecord) -> list[str]:
    """Return stable IDs that can match this Research AI context."""

    ids = [
        record.context_id,
        record.chunk_id,
        str(record.metadata.get("original_doc_id") or ""),
        str(record.metadata.get("section_record_id") or ""),
        str(record.metadata.get("source_manifest_record_id") or ""),
        str(record.metadata.get("document_record_id") or ""),
    ]
    return [value for value in dict.fromkeys(ids) if value]


def expanded_research_ai_valid_ids(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    records: list[ContextRecord],
    metadata: dict[str, Any],
) -> list[str]:
    """Return Research AI valid evidence alternatives for offline evaluation only."""

    original = gold_ids_from_gold_record(gold_record) if gold_record else []
    paper_key = paper_key_from_metadata(metadata)
    topic_terms = {
        normalize_identifier(str(item))
        for item in metadata.get("topic_terms", [])
        if str(item).strip()
    }
    if metadata.get("topic"):
        topic_terms.add(normalize_identifier(str(metadata["topic"])))
    families = set(target_section_families(metadata))
    expanded: list[str] = []
    for record in records:
        same_paper = paper_key and paper_key == record_paper_key(record)
        same_topic = bool(topic_terms) and record_topic_key(record) in topic_terms
        family = record_section_family(record)
        family_match = family in families
        if same_paper:
            expanded.extend(record_valid_ids(record))
        elif not paper_key and same_topic and family_match:
            expanded.extend(record_valid_ids(record))
    return list(dict.fromkeys([*original, *expanded]))


def research_required_context_count(
    *,
    original_gold_ids: list[str],
    records_by_context_id: dict[str, ContextRecord],
) -> int:
    """Return required Research AI context count after gold ID alias deduplication."""

    if not original_gold_ids:
        return 1
    normalized_gold_ids = {
        candidate
        for gold_id in original_gold_ids
        for candidate in (gold_id, normalize_identifier(gold_id))
        if candidate
    }
    matched_context_ids = {
        record.context_id
        for record in records_by_context_id.values()
        if context_match_ids(record) & normalized_gold_ids
    }
    if matched_context_ids:
        return max(1, len(matched_context_ids))
    return max(1, len(dict.fromkeys(original_gold_ids)))


def evaluate_research_ai_with_expanded_valid_evidence(
    *,
    original_gold_ids: list[str],
    expanded_valid_ids: list[str],
    results: list[RetrievalResult],
    records_by_context_id: dict[str, ContextRecord],
    required_context_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate Research AI retrieval with gold ID aliases deduplicated by context."""

    required_count = required_context_count or research_required_context_count(
        original_gold_ids=original_gold_ids,
        records_by_context_id=records_by_context_id,
    )
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
        "required_context_count": required_count,
    }


def research_candidate_recall_with_expanded_valid_evidence(
    *,
    original_gold_ids: list[str],
    expanded_valid_ids: list[str],
    candidate_results: list[RetrievalResult],
    top_k: int,
    records_by_context_id: dict[str, ContextRecord],
    required_context_count: int | None = None,
) -> float:
    """Return Research AI candidate recall after gold ID alias deduplication."""

    return float(
        evaluate_research_ai_with_expanded_valid_evidence(
            original_gold_ids=original_gold_ids,
            expanded_valid_ids=expanded_valid_ids,
            results=candidate_results[:top_k],
            records_by_context_id=records_by_context_id,
            required_context_count=required_context_count,
        )["recall_at_5"]
    )


def render_research_ai_retrieval_query(
    *,
    prompt: dict[str, Any],
    metadata: dict[str, Any],
    canonical_query: str,
) -> tuple[str, int]:
    """Render a non-leaking Research AI runtime retrieval query."""

    families = target_section_families(metadata)
    section_aliases = sorted(
        {alias for family in families for alias in FAMILY_ALIASES.get(family, {family}) if alias}
    )
    parts = [
        canonical_query,
        f"Target paper title: {metadata.get('paper_title') or ''}",
        f"Topic terms: {' '.join(map(str, metadata.get('topic_terms') or []))}",
        f"Target section families: {' '.join(families)}",
        f"Section aliases: {' '.join(section_aliases[:8])}",
    ]
    return sanitize_runtime_query(" ".join(part for part in parts if part.strip()))


def research_ai_repaired_record_from_prompt(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    records: list[ContextRecord],
    by_match_id: dict[str, list[ContextRecord]],
    resolver: CompanyTickerResolver | None = None,
    concept_map: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Create one Research AI repaired generated retrieval record."""

    canonical = build_canonical_query(
        prompt,
        ablation_mode="prompt_plus_metadata",
        resolver=resolver,
        concept_map=concept_map or {},
    )
    metadata = canonical_metadata_from_prompt(prompt)
    runtime_query, blocked = render_research_ai_retrieval_query(
        prompt=prompt,
        metadata=metadata,
        canonical_query=canonical.query_text,
    )
    expanded_ids = expanded_research_ai_valid_ids(
        prompt=prompt,
        gold_record=gold_record,
        records=records,
        metadata=metadata,
    )
    original_gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
    linked = []
    for gold_id in original_gold_ids:
        linked.extend(by_match_id.get(gold_id, []))
        linked.extend(by_match_id.get(normalize_identifier(gold_id), []))
    return {
        "prompt_id": str(prompt.get("prompt_id") or ""),
        "vertical": RESEARCH_VERTICAL,
        "retrieval_query": runtime_query,
        "canonical_retrieval_metadata": {
            **metadata,
            "section_family_targets": target_section_families(metadata),
        },
        "paper_title_terms": metadata.get("paper_title_terms", []),
        "topic_terms": metadata.get("topic_terms", []),
        "section_type": metadata.get("section_type"),
        "method_signal": bool(metadata.get("method_signal")),
        "result_signal": bool(metadata.get("results_signal")),
        "limitation_signal": "limitations" in target_section_families(metadata),
        "valid_evidence_ids_expanded": expanded_ids,
        "original_gold_evidence_ids": original_gold_ids,
        "repair_reason": research_repair_reasons(
            metadata=metadata,
            original_gold_ids=original_gold_ids,
            expanded_ids=expanded_ids,
            linked_record_count=len({record.context_id for record in linked}),
        ),
        "repair_reason_detail": (
            "Expanded same-paper section-family alternatives for Research AI paper/section "
            "alignment; runtime query contains only prompt-visible title/topic/section terms."
        ),
        "source_prompt_record": prompt,
        "blocked_direct_hint_count": canonical.blocked_direct_hint_count + blocked,
        "runtime_query_uses_valid_evidence_ids": False,
    }


def research_repair_reasons(
    *,
    metadata: dict[str, Any],
    original_gold_ids: list[str],
    expanded_ids: list[str],
    linked_record_count: int,
) -> list[str]:
    """Return compact Research AI repair reasons."""

    reasons = ["multiple_valid_sections_not_counted"]
    if len(expanded_ids) > len(original_gold_ids):
        reasons.extend(["narrow_gold_section", "near_duplicate_section_confusion"])
    if not metadata.get("paper_title"):
        reasons.append("paper_title_ambiguity")
    if not metadata.get("section_type") and not metadata.get("section_types"):
        reasons.append("section_type_ambiguity")
    families = set(target_section_families(metadata))
    if {"method", "results"} <= families:
        reasons.append("method_vs_result_confusion")
    if "limitations" in families:
        reasons.append("limitation_vs_discussion_confusion")
    if linked_record_count == 0 and original_gold_ids:
        reasons.append("candidate_absent_from_top50")
    return [reason for reason in FAILURE_CLASSES if reason in set(reasons)]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file if present."""

    input_path = Path(path)
    if not input_path.exists():
        return []
    return [
        cast(dict[str, Any], json.loads(line))
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_repaired_research_dataset(
    *,
    output_root: Path,
    records: list[dict[str, Any]],
) -> Path:
    """Write local generated Research AI repaired records."""

    repaired_root = output_root / "repaired_retrieval_dataset"
    return write_jsonl(
        repaired_root / "research_ai_repaired_retrieval_records.jsonl",
        records,
    )


def build_all_repaired_records_for_promotion(
    *,
    output_root: Path,
    research_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load existing repaired records and replace only Research AI records."""

    repaired_root = output_root / "repaired_retrieval_dataset"
    all_records: list[dict[str, Any]] = []
    for vertical in SLO_VERTICALS:
        if vertical == RESEARCH_VERTICAL:
            all_records.extend(research_records)
            continue
        path = repaired_root / f"{vertical}_repaired_retrieval_records.jsonl"
        rows = load_jsonl(path)
        if rows:
            all_records.extend(rows)
    return all_records


def validate_research_ai_variants(
    *,
    prompts: list[dict[str, Any]],
    gold_by_prompt_id: dict[str, dict[str, Any]],
    records: list[ContextRecord],
    variant_records: dict[str, dict[str, dict[str, Any]]],
    stage_sizes: list[int],
    slo_config: dict[str, Any],
    dense_backend: str,
    vector_store_config_path: str | Path,
    vector_store_key: str,
    allow_dense_fallback: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate Research AI variants and return report, summary, rows, failures."""

    corpora_by_vertical = {RESEARCH_VERTICAL: records}
    retrievers = build_retrievers(
        corpora_by_vertical,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    try:
        query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        queries_to_warm: dict[str, set[str]] = {RESEARCH_VERTICAL: set()}
        selected_prompts = select_stage_prompts(prompts, max(stage_sizes))
        for prompt in selected_prompts:
            prompt_id = str(prompt.get("prompt_id") or "")
            for record in variant_records.values():
                payload = record.get(prompt_id)
                if payload:
                    queries_to_warm[RESEARCH_VERTICAL].add(str(payload["retrieval_query"]))
        warmed = warm_qdrant_repair_queries(
            retrievers=retrievers,
            queries_by_vertical=queries_to_warm,
            top_k=50,
        )
        records_by_context_id = cast(
            dict[str, ContextRecord],
            retrievers[RESEARCH_VERTICAL]["records_by_context_id"],
        )
        required_context_count_by_prompt_id: dict[str, int] = {}
        for prompt in selected_prompts:
            prompt_id = str(prompt.get("prompt_id") or "")
            gold_record = gold_by_prompt_id.get(prompt_id)
            original_gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
            required_context_count_by_prompt_id[prompt_id] = research_required_context_count(
                original_gold_ids=original_gold_ids,
                records_by_context_id=records_by_context_id,
            )
        for stage_size in stage_sizes:
            for prompt in selected_prompts[:stage_size]:
                prompt_id = str(prompt.get("prompt_id") or "")
                gold_record = gold_by_prompt_id.get(prompt_id)
                original_gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
                required_context_count = required_context_count_by_prompt_id[prompt_id]
                for variant_name, by_prompt_id in variant_records.items():
                    repaired = by_prompt_id[prompt_id]
                    query = str(repaired["retrieval_query"])
                    retrieval = retrieve_for_mode(
                        memory_mode="mm2_hybrid_top5",
                        query=query,
                        expanded_queries=(query,),
                        expansion_types=(f"{variant_name}_query",),
                        source_hints_used=False,
                        vertical=RESEARCH_VERTICAL,
                        retrievers=retrievers,
                        top_k=DEFAULT_FINAL_TOP_K,
                        final_top_k=DEFAULT_FINAL_TOP_K,
                        retrieval_cache=query_cache,
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
                    if variant_name == "original_promoted":
                        evaluation = evaluate_research_ai_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=original_gold_ids,
                            results=retrieval.results,
                            records_by_context_id=records_by_context_id,
                            required_context_count=required_context_count,
                        )
                        candidate20 = research_candidate_recall_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=original_gold_ids,
                            candidate_results=candidate_results,
                            top_k=20,
                            records_by_context_id=records_by_context_id,
                            required_context_count=required_context_count,
                        )
                        candidate50 = research_candidate_recall_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=original_gold_ids,
                            candidate_results=candidate_results,
                            top_k=50,
                            records_by_context_id=records_by_context_id,
                            required_context_count=required_context_count,
                        )
                        expanded_ids = original_gold_ids
                    else:
                        expanded_ids = cast(list[str], repaired["valid_evidence_ids_expanded"])
                        evaluation = evaluate_research_ai_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=expanded_ids,
                            results=retrieval.results,
                            records_by_context_id=records_by_context_id,
                            required_context_count=required_context_count,
                        )
                        candidate20 = research_candidate_recall_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=expanded_ids,
                            candidate_results=candidate_results,
                            top_k=20,
                            records_by_context_id=records_by_context_id,
                            required_context_count=required_context_count,
                        )
                        candidate50 = research_candidate_recall_with_expanded_valid_evidence(
                            original_gold_ids=original_gold_ids,
                            expanded_valid_ids=expanded_ids,
                            candidate_results=candidate_results,
                            top_k=50,
                            records_by_context_id=records_by_context_id,
                            required_context_count=required_context_count,
                        )
                    failure_classes = classify_failure(
                        prompt=prompt,
                        repaired_record=repaired,
                        candidate_results=candidate_results,
                        final_context_ids=[
                            result.context_record.context_id for result in retrieval.results
                        ],
                        original_gold_ids=original_gold_ids,
                        expanded_valid_ids=expanded_ids,
                        candidate20=float(candidate20),
                        candidate50=float(candidate50),
                        final_recall=float(evaluation["recall_at_5"]),
                        records_by_context_id=records_by_context_id,
                    )
                    row = {
                        "dataset_variant": variant_name,
                        "vertical": RESEARCH_VERTICAL,
                        "stage_size": stage_size,
                        "prompt_id": prompt_id,
                        "ablation_mode": "prompt_plus_metadata",
                        "measurement": "research_ai_alignment_repair",
                        "dense_backend": retrieval.backend_label,
                        "vector_store": retrieval.vector_store,
                        "candidate_recall_at_20": float(candidate20),
                        "candidate_recall_at_50": float(candidate50),
                        "final_recall_at_5": float(evaluation["recall_at_5"]),
                        "mrr": float(evaluation["mrr"]),
                        "query_rewritten": variant_name != "original_promoted",
                        "direct_hint_leakage_detected": (
                            DIRECT_RUNTIME_ID_RE.search(query) is not None
                        ),
                        "failure_classes": failure_classes,
                    }
                    rows.append(row)
                    if failure_classes and (stage_size == max(stage_sizes)):
                        failure_rows.append(
                            compact_failure_example(
                                prompt=prompt,
                                row=row,
                                repaired_record=repaired,
                                candidate_results=candidate_results[:10],
                                final_context_ids=[
                                    result.context_record.context_id for result in retrieval.results
                                ],
                            )
                        )
        summary_rows = aggregate_validation_rows(rows, slo_config=slo_config)
        report = {
            "generated_at_utc": utc_now(),
            "scope": "research_ai_retrieval_alignment_repair_no_inference_no_gpu_no_api",
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
        alignment_summary = research_alignment_summary(rows, slo_config=slo_config)
        return report, summary_rows, alignment_summary, failure_rows
    finally:
        close_retrievers(retrievers)


def classify_failure(
    *,
    prompt: dict[str, Any],
    repaired_record: dict[str, Any],
    candidate_results: list[Any],
    final_context_ids: list[str],
    original_gold_ids: list[str],
    expanded_valid_ids: list[str],
    candidate20: float,
    candidate50: float,
    final_recall: float,
    records_by_context_id: dict[str, ContextRecord],
) -> list[str]:
    """Classify one Research AI retrieval failure."""

    classes: list[str] = []
    metadata = cast(dict[str, Any], repaired_record.get("canonical_retrieval_metadata") or {})
    families = set(target_section_families(metadata))
    if not metadata.get("paper_title") or PAPER_TITLE_MISSING_RE.search(
        str(prompt.get("question") or "")
    ):
        classes.append("paper_title_ambiguity")
    if not metadata.get("section_types"):
        classes.append("section_type_ambiguity")
    final_families = {
        record_section_family(records_by_context_id[context_id])
        for context_id in final_context_ids
        if context_id in records_by_context_id
    }
    if "method" in families and "results" in final_families and "method" not in final_families:
        classes.append("method_vs_result_confusion")
    if "results" in families and "method" in final_families and "results" not in final_families:
        classes.append("method_vs_result_confusion")
    if "limitations" in families and not (final_families & {"limitations"}):
        classes.append("limitation_vs_discussion_confusion")
    top_papers = [
        record_paper_key(result.context_record)
        for result in candidate_results[:20]
        if record_paper_key(result.context_record)
    ]
    if len(set(top_papers)) > 4:
        classes.append("topic_overlap_across_papers")
    if len(expanded_valid_ids) > len(original_gold_ids):
        classes.extend(["narrow_gold_section", "multiple_valid_sections_not_counted"])
    if candidate50 < 1.0:
        classes.append("candidate_absent_from_top50")
    elif final_recall < 1.0:
        classes.append("candidate_present_but_not_top5")
    family_counts = Counter(
        (
            record_paper_key(result.context_record),
            record_section_family(result.context_record),
        )
        for result in candidate_results[:20]
    )
    if any(count >= 4 for count in family_counts.values()):
        classes.append("near_duplicate_section_confusion")
    return [item for item in FAILURE_CLASSES if item in set(classes)]


def compact_failure_example(
    *,
    prompt: dict[str, Any],
    row: dict[str, Any],
    repaired_record: dict[str, Any],
    candidate_results: list[Any],
    final_context_ids: list[str],
) -> dict[str, Any]:
    """Return a compact failure example safe to commit."""

    metadata = cast(dict[str, Any], repaired_record.get("canonical_retrieval_metadata") or {})
    return {
        "prompt_id": prompt.get("prompt_id"),
        "dataset_variant": row["dataset_variant"],
        "stage_size": row["stage_size"],
        "failure_classes": row["failure_classes"],
        "question": prompt.get("question"),
        "paper_title_terms": metadata.get("paper_title_terms", []),
        "target_section_families": metadata.get("section_family_targets", []),
        "candidate_recall_at_20": row["candidate_recall_at_20"],
        "candidate_recall_at_50": row["candidate_recall_at_50"],
        "final_recall_at_5": row["final_recall_at_5"],
        "mrr": row["mrr"],
        "final_context_ids": final_context_ids,
        "top_candidate_context_ids": [
            result.context_record.context_id for result in candidate_results
        ],
    }


def research_alignment_summary(
    rows: list[dict[str, Any]],
    *,
    slo_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate Research AI failure classes by variant and stage."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset_variant"]), int(row["stage_size"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (variant, stage_size), group in sorted(grouped.items()):
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
        status, _blocker, _action = slo_status_for_metrics(
            vertical=RESEARCH_VERTICAL,
            metrics=metrics,
            slo_config=slo_config,
        )
        counter: Counter[str] = Counter()
        for row in group:
            counter.update(str(item) for item in row.get("failure_classes", []))
        summary.append(
            {
                "stage_size": stage_size,
                "dataset_variant": variant,
                "record_count": len(group),
                **metrics,
                "slo_status": status,
                **{
                    failure_class: counter.get(failure_class, 0)
                    for failure_class in FAILURE_CLASSES
                },
            }
        )
    return summary


def block18_research_ai_baseline_rows(
    *,
    stage_sizes: list[int],
    slo_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return recorded Block 18 Research AI baseline rows for before/after reporting."""

    rows: list[dict[str, Any]] = []
    for stage_size in stage_sizes:
        if stage_size not in BLOCK18_RESEARCH_AI_BASELINE:
            continue
        metrics = BLOCK18_RESEARCH_AI_BASELINE[stage_size]
        status, _blocker, _action = slo_status_for_metrics(
            vertical=RESEARCH_VERTICAL,
            metrics=metrics,
            slo_config=slo_config,
        )
        rows.append(
            {
                "stage_size": stage_size,
                "dataset_variant": "block18_repaired_baseline",
                "record_count": stage_size,
                **metrics,
                "slo_status": status,
                **{failure_class: "" for failure_class in FAILURE_CLASSES},
            }
        )
    return rows


def load_previous_research_records(output_root: Path) -> list[dict[str, Any]]:
    """Load pre-existing Research AI repaired records if present."""

    path = (
        output_root / "repaired_retrieval_dataset" / "research_ai_repaired_retrieval_records.jsonl"
    )
    return load_jsonl(path)


def original_promoted_variant_records(
    *,
    prompts: list[dict[str, Any]],
    gold_by_prompt_id: dict[str, dict[str, Any]],
    records: list[ContextRecord],
) -> dict[str, dict[str, Any]]:
    """Build non-mutating original query records for comparison."""

    index = build_research_expansion_index(records)
    output: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id") or "")
        gold_record = gold_by_prompt_id.get(prompt_id)
        canonical = build_canonical_query(prompt, ablation_mode="prompt_plus_metadata")
        query, blocked = sanitize_runtime_query(canonical.query_text)
        metadata = canonical_metadata_from_prompt(prompt)
        output[prompt_id] = {
            "prompt_id": prompt_id,
            "vertical": RESEARCH_VERTICAL,
            "retrieval_query": query,
            "canonical_retrieval_metadata": metadata,
            "valid_evidence_ids_expanded": expanded_valid_evidence_ids_from_index(
                prompt=prompt,
                gold_record=gold_record,
                canonical_metadata=metadata,
                expansion_index=index,
            ),
            "original_gold_evidence_ids": gold_ids_from_gold_record(gold_record)
            if gold_record is not None
            else [],
            "blocked_direct_hint_count": blocked + canonical.blocked_direct_hint_count,
            "runtime_query_uses_valid_evidence_ids": False,
        }
    return output


def build_research_expansion_index(records: list[ContextRecord]) -> dict[str, Any]:
    """Build the previous Block 18-style Research AI expansion index."""

    by_paper_section: dict[tuple[str, str], list[str]] = defaultdict(list)
    by_paper: dict[str, list[str]] = defaultdict(list)
    by_topic_section: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        ids = record_valid_ids(record)
        paper = record_paper_key(record)
        topic = record_topic_key(record)
        section = normalize_section(
            str(record.metadata.get("section_type") or record.metadata.get("evidence_type") or "")
        )
        by_paper[paper].extend(ids)
        by_paper_section[(paper, section)].extend(ids)
        by_topic_section[(topic, section)].extend(ids)
    return {
        "by_paper_section": dict(by_paper_section),
        "by_paper": dict(by_paper),
        "by_topic_section": dict(by_topic_section),
    }


def update_combined_validation_outputs(
    *,
    output_root: Path,
    research_summary_rows: list[dict[str, Any]],
    all_repaired_records: list[dict[str, Any]],
    stage_sizes: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Update combined repaired validation and promotion outputs."""

    previous_summary_path = output_root / "repaired_retrieval_validation_summary.csv"
    combined_rows: list[dict[str, Any]] = []
    if previous_summary_path.exists():
        with previous_summary_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if (
                    row.get("dataset_variant") == "repaired_generated"
                    and row.get("vertical") == RESEARCH_VERTICAL
                    and int(str(row.get("stage_size") or "0")) in stage_sizes
                ):
                    continue
                combined_rows.append(cast(dict[str, Any], row))
    else:
        combined_rows.extend(load_original_canonical_summary(output_root, stage_sizes=stage_sizes))
    for row in research_summary_rows:
        if row["dataset_variant"] != "repaired_generated":
            continue
        combined_rows.append(row)
    combined_rows = sorted(
        combined_rows,
        key=lambda row: (
            str(row.get("dataset_variant")),
            str(row.get("vertical")),
            int(str(row.get("stage_size") or "0")),
        ),
    )
    validation_report = {
        "generated_at_utc": utc_now(),
        "scope": "repaired_retrieval_validation_after_research_ai_alignment_repair",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "stage_sizes": stage_sizes,
        "summary_rows": combined_rows,
    }
    promotion_plan = build_promotion_plan(
        summary_rows=combined_rows,
        repaired_records=all_repaired_records,
    )
    write_json(output_root / "repaired_retrieval_validation_report.json", validation_report)
    write_csv(
        output_root / "repaired_retrieval_validation_summary.csv",
        combined_rows,
        VALIDATION_SUMMARY_FIELDS,
    )
    write_json(output_root / "repaired_retrieval_promotion_plan.json", promotion_plan)
    return validation_report, combined_rows, promotion_plan


def build_repaired_slo_readiness(
    *,
    output_root: Path,
    combined_rows: list[dict[str, Any]],
    slo_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an SLO readiness report from repaired retrieval validation rows."""

    rows_2000 = [
        row
        for row in combined_rows
        if row.get("dataset_variant") == "repaired_generated"
        and int(str(row.get("stage_size") or "0")) == 2000
    ]
    summary_rows: list[dict[str, Any]] = []
    for row in rows_2000:
        vertical = str(row["vertical"])
        targets = cast(dict[str, Any], slo_config["verticals"][vertical]["retrieval_slo"])
        observed = {
            "candidate_recall_at_20_min": float(row["candidate_recall_at_20"]),
            "candidate_recall_at_50_min": float(row["candidate_recall_at_50"]),
            "final_recall_at_5_min": float(row["final_recall_at_5"]),
            "mrr_min": float(row["mrr"]),
        }
        for metric_name, value in observed.items():
            target = float(targets[metric_name])
            status = "PASS" if value >= target else "BLOCKED"
            summary_rows.append(
                {
                    "vertical": vertical,
                    "metric_family": "retrieval_slo",
                    "metric_name": metric_name,
                    "target": target,
                    "observed": value,
                    "status": status,
                    "gap": round(value - target, 6),
                    "recommended_action": "No action required."
                    if status == "PASS"
                    else "Repair retrieval before inference scaling or final benchmark claims.",
                }
            )
        for family in SLO_METRIC_FAMILIES:
            if family == "retrieval_slo":
                continue
            summary_rows.append(
                {
                    "vertical": vertical,
                    "metric_family": family,
                    "metric_name": "not_measured_yet",
                    "target": "",
                    "observed": "",
                    "status": "NOT_AVAILABLE",
                    "gap": "",
                    "recommended_action": f"Run an experiment/report that measures {family}.",
                }
            )
    blocked_retrieval = [
        row
        for row in summary_rows
        if row["metric_family"] == "retrieval_slo" and row["status"] == "BLOCKED"
    ]
    status_counts = Counter(str(row["status"]) for row in summary_rows)
    report = {
        "generated_at_utc": utc_now(),
        "scope": "production_slo_readiness_from_repaired_retrieval_validation",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "retrieval_validation_report_path": str(
            output_root / "repaired_retrieval_validation_report.json"
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "retrieval_slo_blocked_count": len(blocked_retrieval),
        "inference_scaling_blocked_by_retrieval_slos": bool(blocked_retrieval),
        "blocked_retrieval_metrics": blocked_retrieval,
        "summary": {
            "overall_status": "BLOCKED" if blocked_retrieval else "READY_WITH_GAPS",
            "retrieval_slos_currently_measured": True,
            "future_inference_cost_resource_metrics_available": False,
        },
        "results": summary_rows,
    }
    write_json(output_root / "slo_readiness_report.json", report)
    from inference_bench.slo import write_csv as write_slo_csv

    write_slo_csv(output_root / "slo_readiness_summary.csv", summary_rows)
    return report, summary_rows


def build_research_ai_alignment_repair(
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
    """Build Research AI alignment repair outputs and update promotion plan."""

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    prompts = prompts_by_vertical[RESEARCH_VERTICAL]
    gold_by_prompt_id = gold_by_vertical[RESEARCH_VERTICAL]
    records = corpora_by_vertical[RESEARCH_VERTICAL]
    slo_config = load_slo_config(slo_config_path)
    output_path = Path(output_root)
    by_match_id = context_records_by_match_id(records)
    previous_records = load_previous_research_records(output_path)
    repaired_records = [
        research_ai_repaired_record_from_prompt(
            prompt=prompt,
            gold_record=gold_by_prompt_id.get(str(prompt.get("prompt_id") or "")),
            records=records,
            by_match_id=by_match_id,
        )
        for prompt in prompts
    ]
    write_repaired_research_dataset(output_root=output_path, records=repaired_records)
    repaired_by_prompt_id = {str(record["prompt_id"]): record for record in repaired_records}
    previous_by_prompt_id = {
        str(record["prompt_id"]): record for record in previous_records
    } or original_promoted_variant_records(
        prompts=prompts,
        gold_by_prompt_id=gold_by_prompt_id,
        records=records,
    )
    variant_records = {
        "previous_repaired_generated": previous_by_prompt_id,
        "repaired_generated": repaired_by_prompt_id,
    }
    validation_report, research_summary_rows, alignment_summary_rows, failure_examples = (
        validate_research_ai_variants(
            prompts=prompts,
            gold_by_prompt_id=gold_by_prompt_id,
            records=records,
            variant_records=variant_records,
            stage_sizes=stage_sizes,
            slo_config=slo_config,
            dense_backend=dense_backend,
            vector_store_config_path=vector_store_config_path,
            vector_store_key=vector_store_key,
            allow_dense_fallback=allow_dense_fallback,
        )
    )
    validation_report["summary_rows"] = research_summary_rows
    alignment_summary_rows = [
        *block18_research_ai_baseline_rows(
            stage_sizes=stage_sizes,
            slo_config=slo_config,
        ),
        *alignment_summary_rows,
    ]
    alignment_report = {
        "generated_at_utc": utc_now(),
        "scope": "research_ai_retrieval_dataset_gold_alignment_repair_no_inference_no_gpu_no_api",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "repair_scope": "research_ai_only",
        "runtime_query_uses_gold_or_source_ids": False,
        "expanded_valid_evidence_used_only_for_offline_evaluation": True,
        "block18_repaired_baseline_metrics": BLOCK18_RESEARCH_AI_BASELINE,
        "record_count": len(repaired_records),
        "summary_rows": alignment_summary_rows,
        "failure_class_definitions": {
            "paper_title_ambiguity": "Paper-title cues are absent or hard to distinguish.",
            "section_type_ambiguity": "Prompt lacks a specific paper-section family.",
            "method_vs_result_confusion": "Method and result sections compete in top candidates.",
            "limitation_vs_discussion_confusion": (
                "Limitation, discussion, and conclusion sections are interchangeable."
            ),
            "topic_overlap_across_papers": "Many candidate papers share the same topic terms.",
            "narrow_gold_section": "Original gold IDs undercount same-paper section alternatives.",
            "multiple_valid_sections_not_counted": (
                "Multiple same-paper sections can ground the prompt."
            ),
            "candidate_absent_from_top50": "Expanded valid evidence is absent from top 50.",
            "candidate_present_but_not_top5": (
                "Expanded valid evidence is in candidates but not top 5."
            ),
            "near_duplicate_section_confusion": (
                "Several same-paper/section chunks compete for limited top-5 slots."
            ),
        },
        "sample_repaired_records": [
            {
                "prompt_id": record["prompt_id"],
                "paper_title_terms": record["paper_title_terms"],
                "topic_terms": record["topic_terms"],
                "section_type": record["section_type"],
                "method_signal": record["method_signal"],
                "result_signal": record["result_signal"],
                "limitation_signal": record["limitation_signal"],
                "expanded_valid_evidence_count": len(record["valid_evidence_ids_expanded"]),
                "repair_reason": record["repair_reason"],
            }
            for record in repaired_records[:25]
        ],
    }
    all_repaired_records = build_all_repaired_records_for_promotion(
        output_root=output_path,
        research_records=repaired_records,
    )
    _combined_report, combined_rows, promotion_plan = update_combined_validation_outputs(
        output_root=output_path,
        research_summary_rows=research_summary_rows,
        all_repaired_records=all_repaired_records,
        stage_sizes=stage_sizes,
    )
    slo_report, _slo_rows = build_repaired_slo_readiness(
        output_root=output_path,
        combined_rows=combined_rows,
        slo_config=slo_config,
    )
    write_json(output_path / "research_ai_alignment_report.json", alignment_report)
    write_csv(
        output_path / "research_ai_alignment_summary.csv",
        alignment_summary_rows,
        ALIGNMENT_FIELDS,
    )
    write_jsonl(
        output_path / "research_ai_failure_examples.jsonl",
        failure_examples[:100],
    )
    return {
        "alignment_report": alignment_report,
        "validation_report": validation_report,
        "promotion_plan": promotion_plan,
        "slo_readiness_report": slo_report,
        "research_summary_rows": research_summary_rows,
    }
