"""Retail and Finance retrieval recovery diagnostics for Block 16A.

This module runs retrieval-only staged validation for Retail and Finance. It
does not run model inference, GPU work, external APIs, or use gold/source IDs as
retrieval query terms.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, cast

from inference_bench.context_schema import ContextRecord
from inference_bench.gold_evidence_audit import gold_ids_from_gold_record
from inference_bench.memory_workloads import (
    build_retrievers,
    close_retrievers,
    load_context_corpora,
    load_prompts_and_gold,
    prompt_query_text,
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
    retail_evidence_kind,
    tokenize,
)
from inference_bench.vertical_retrieval_repair import (
    DIRECT_HINT_RE,
    EnrichmentResult,
    build_audit_report,
    repaired_query,
    select_stage_prompts,
)

RECOVERY_SCOPE = "block16a_retail_finance_recovery_no_inference_no_gpu_no_api"
TARGET_VERTICALS = ("retail", "finance")
VALIDATION_FIELDS = [
    "vertical",
    "stage_size",
    "measurement",
    "ablation_mode",
    "dense_backend",
    "vector_store",
    "candidate_recall_at_20",
    "candidate_recall_at_50",
    "final_recall_at_5",
    "mrr",
    "record_count",
    "query_rewrite_count",
    "direct_hint_leakage_count",
]
RETAIL_FAILURE_FIELDS = [
    "stage_size",
    "failed_query_count",
    "candidate_failure_count",
    "candidate_failure_pct",
    "reranker_failure_count",
    "reranker_failure_pct",
    "metadata_failure_count",
    "metadata_failure_pct",
    "chunking_failure_count",
    "product_title_mismatch_count",
    "category_mismatch_count",
    "review_issue_mismatch_count",
    "policy_mismatch_count",
]
FINANCE_FLOW_FIELDS = [
    "prompt_id",
    "query_text",
    "derived_period",
    "derived_metric",
    "derived_filing",
    "derived_section",
    "materialized_query",
    "direct_hint_leakage_detected",
]
FINANCE_FLOW_SUMMARY_FIELDS = [
    "total_prompts",
    "filing_materialized_count",
    "metric_materialized_count",
    "period_materialized_count",
    "section_materialized_count",
    "direct_hint_leakage_detected_count",
]


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def candidate_results_from_ids(
    context_ids: list[str],
    records_by_context_id: dict[str, ContextRecord],
    retrieval_mode: str,
) -> list[RetrievalResult]:
    """Rebuild candidate result objects from diagnostic context IDs."""

    return [
        RetrievalResult(
            context_record=records_by_context_id[context_id],
            score=0.0,
            rank=index,
            retrieval_mode=retrieval_mode,
            component_scores={},
        )
        for index, context_id in enumerate(context_ids, start=1)
        if context_id in records_by_context_id
    ]


def known_match_ids(records: list[ContextRecord]) -> set[str]:
    """Return all gold-compatible match IDs for a corpus."""

    return {match_id for record in records for match_id in context_match_ids(record)}


def compact_query_text(value: str, max_chars: int = 1000) -> str:
    """Return a bounded query excerpt for generated diagnostics."""

    normalized = " ".join(value.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def retail_issue_terms(prompt: dict[str, Any]) -> set[str]:
    """Return Retail issue terms from visible prompt metadata."""

    terms = set(tokenize(str(prompt.get("issue_type") or "")))
    metadata = prompt.get("metadata")
    if isinstance(metadata, dict):
        terms.update(tokenize(str(metadata.get("prompt_category") or "")))
    terms.update(tokenize(str(prompt.get("question") or "")))
    return terms


def retail_failure_reasons(
    *,
    prompt: dict[str, Any],
    gold_ids: list[str],
    candidate_results: list[RetrievalResult],
    final_results: list[RetrievalResult],
    recall_at_5: float,
    known_ids: set[str],
) -> list[str]:
    """Classify one failed Retail query."""

    if recall_at_5 >= 1.0:
        return []
    candidate_recall_50 = recall_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=50,
    )
    reasons: list[str] = []
    if any(
        gold_id not in known_ids and normalize_identifier(gold_id) not in known_ids
        for gold_id in gold_ids
    ):
        reasons.append("chunking_failure")
    if candidate_recall_50 < 1.0:
        reasons.append("candidate_retrieval_failure")
    else:
        reasons.append("reranker_failure")

    product_title_tokens = set(tokenize(str(prompt.get("product_title") or "")))
    category_tokens = set(tokenize(str(prompt.get("category") or "")))
    issue_terms = retail_issue_terms(prompt)
    top_records = [result.context_record for result in final_results]
    candidate_records = [result.context_record for result in candidate_results[:50]]

    if product_title_tokens and not any(
        product_title_tokens
        & set(tokenize(str(record.metadata.get("product_title") or record.title)))
        for record in top_records
    ):
        reasons.append("product_title_mismatch")
    if category_tokens and not any(
        category_tokens & set(tokenize(str(record.metadata.get("category") or "")))
        for record in top_records
    ):
        reasons.append("category_mismatch")
    if issue_terms and not any(
        issue_terms
        & set(tokenize(" ".join(str(value) for value in record.metadata.get("issue_terms", []))))
        for record in [*top_records, *candidate_records[:10]]
    ):
        reasons.append("review_issue_mismatch")
    policy_needed = bool(
        issue_terms & {"return_refund", "policy_reasoning", "return", "refund", "policy"}
    )
    if policy_needed and not any(
        retail_evidence_kind(record) == "policy" for record in top_records
    ):
        reasons.append("policy_mismatch")
    return list(dict.fromkeys(reasons))


def aggregate_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-prompt validation rows."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["vertical"]), int(row["stage_size"]), str(row["measurement"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (vertical, stage_size, measurement), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "vertical": vertical,
                "stage_size": stage_size,
                "measurement": measurement,
                "ablation_mode": "prompt_plus_metadata",
                "dense_backend": ",".join(sorted({str(row["dense_backend"]) for row in group})),
                "vector_store": ",".join(sorted({str(row["vector_store"]) for row in group})),
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
                "record_count": len(group),
                "query_rewrite_count": sum(int(bool(row["query_rewritten"])) for row in group),
                "direct_hint_leakage_count": sum(
                    int(bool(row["direct_hint_leakage_detected"])) for row in group
                ),
            }
        )
    return summary_rows


def retail_failure_summary(
    *,
    failure_rows: list[dict[str, Any]],
    stage_size: int,
) -> dict[str, Any]:
    """Summarize Retail failures."""

    reason_counter: Counter[str] = Counter()
    for row in failure_rows:
        reason_counter.update(row["failure_reasons"])
    failed_count = len(failure_rows)

    def pct(count: int) -> float:
        return round(count / failed_count, 6) if failed_count else 0.0

    candidate_count = reason_counter["candidate_retrieval_failure"]
    reranker_count = reason_counter["reranker_failure"]
    metadata_count = sum(
        reason_counter[key]
        for key in (
            "product_title_mismatch",
            "category_mismatch",
            "review_issue_mismatch",
            "policy_mismatch",
        )
    )
    return {
        "stage_size": stage_size,
        "failed_query_count": failed_count,
        "candidate_failure_count": candidate_count,
        "candidate_failure_pct": pct(candidate_count),
        "reranker_failure_count": reranker_count,
        "reranker_failure_pct": pct(reranker_count),
        "metadata_failure_count": metadata_count,
        "metadata_failure_pct": pct(metadata_count),
        "chunking_failure_count": reason_counter["chunking_failure"],
        "product_title_mismatch_count": reason_counter["product_title_mismatch"],
        "category_mismatch_count": reason_counter["category_mismatch"],
        "review_issue_mismatch_count": reason_counter["review_issue_mismatch"],
        "policy_mismatch_count": reason_counter["policy_mismatch"],
    }


def build_finance_metadata_flow_rows(
    *,
    prompts: list[dict[str, Any]],
    enrichments: dict[str, EnrichmentResult],
    resolver: CompanyTickerResolver | None,
    concept_map: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Build Finance metadata-flow rows for all prompts."""

    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id") or "")
        enrichment = enrichments[prompt_id]
        base_query = prompt_query_text(
            prompt,
            "prompt_plus_metadata",
            company_ticker_resolver=resolver,
            xbrl_concept_map=concept_map,
        )
        materialized, _expanded, _types, _blocked = repaired_query(
            prompt=prompt,
            enrichment=enrichment,
            resolver=resolver,
            concept_map=concept_map,
            ablation_mode="prompt_plus_metadata",
        )
        rows.append(
            {
                "prompt_id": prompt_id,
                "query_text": compact_query_text(base_query.query_text),
                "derived_period": enrichment.fields.get("period"),
                "derived_metric": enrichment.fields.get("metric_family"),
                "derived_filing": enrichment.fields.get("filing_type"),
                "derived_section": enrichment.fields.get("section_type"),
                "materialized_query": compact_query_text(materialized),
                "direct_hint_leakage_detected": DIRECT_HINT_RE.search(materialized) is not None,
            }
        )
    return rows


def run_retail_finance_validation(
    *,
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
    enrichments: dict[str, dict[str, EnrichmentResult]],
    stage_sizes: list[int],
    dense_backend: str,
    vector_store_config_path: str | Path,
    vector_store_key: str,
    allow_dense_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run staged Retail and Finance before/after retrieval validation."""

    selected_corpora = {vertical: corpora_by_vertical[vertical] for vertical in TARGET_VERTICALS}
    retrievers = build_retrievers(
        selected_corpora,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    rows: list[dict[str, Any]] = []
    retail_failures: list[dict[str, Any]] = []
    finance_flow_rows: list[dict[str, Any]] = []
    try:
        records_by_context_id = {
            vertical: cast(dict[str, ContextRecord], retrievers[vertical]["records_by_context_id"])
            for vertical in TARGET_VERTICALS
        }
        known_ids_by_vertical = {
            vertical: known_match_ids(corpora_by_vertical[vertical])
            for vertical in TARGET_VERTICALS
        }
        finance_retrievers = retrievers["finance"]
        finance_resolver = cast(
            CompanyTickerResolver | None,
            finance_retrievers.get("company_ticker_resolver"),
        )
        finance_concept_map = cast(
            dict[str, set[str]], finance_retrievers.get("xbrl_concept_map") or {}
        )
        finance_flow_rows = build_finance_metadata_flow_rows(
            prompts=prompts_by_vertical["finance"],
            enrichments=enrichments["finance"],
            resolver=finance_resolver,
            concept_map=finance_concept_map,
        )
        query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        max_stage_size = max(stage_sizes)
        prepared: dict[
            str,
            list[
                tuple[
                    dict[str, Any],
                    tuple[str, tuple[str, ...], tuple[str, ...], int],
                    tuple[str, tuple[str, ...], tuple[str, ...], int],
                ]
            ],
        ] = {vertical: [] for vertical in TARGET_VERTICALS}
        for vertical in TARGET_VERTICALS:
            resolver = cast(
                CompanyTickerResolver | None,
                retrievers[vertical].get("company_ticker_resolver"),
            )
            concept_map = cast(
                dict[str, set[str]], retrievers[vertical].get("xbrl_concept_map") or {}
            )
            for prompt in select_stage_prompts(prompts_by_vertical[vertical], max_stage_size):
                base = prompt_query_text(
                    prompt,
                    "prompt_plus_metadata",
                    company_ticker_resolver=resolver,
                    xbrl_concept_map=concept_map,
                )
                before_query = (
                    base.query_text,
                    base.expanded_queries,
                    base.expansion_types,
                    base.blocked_direct_hint_count,
                )
                enrichment = enrichments[vertical][str(prompt.get("prompt_id") or "")]
                if vertical == "finance" and not any(
                    enrichment.fields.get(field_name)
                    for field_name in ("metric_family", "period", "section_type")
                ):
                    after_query = before_query
                else:
                    after_query = repaired_query(
                        prompt=prompt,
                        enrichment=enrichment,
                        resolver=resolver,
                        concept_map=concept_map,
                        ablation_mode="prompt_plus_metadata",
                    )
                prepared[vertical].append((prompt, before_query, after_query))

        for stage_size in stage_sizes:
            for vertical in TARGET_VERTICALS:
                for prompt, before_query, after_query in prepared[vertical][:stage_size]:
                    prompt_id = str(prompt.get("prompt_id") or "")
                    gold_record = gold_by_vertical[vertical].get(prompt_id)
                    gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
                    for measurement, query_tuple in (
                        ("before_recovery", before_query),
                        ("after_recovery", after_query),
                    ):
                        query_text, expanded_queries, expansion_types, blocked_count = query_tuple
                        retrieval = retrieve_for_mode(
                            memory_mode="mm2_hybrid_top5",
                            query=query_text,
                            expanded_queries=expanded_queries,
                            expansion_types=expansion_types,
                            source_hints_used=False,
                            vertical=vertical,
                            retrievers=retrievers,
                            top_k=DEFAULT_FINAL_TOP_K,
                            final_top_k=DEFAULT_FINAL_TOP_K,
                            retrieval_cache=query_cache,
                        )
                        candidate_ids = [
                            str(context_id)
                            for context_id in retrieval.diagnostics.get("candidate_context_ids", [])
                        ]
                        candidate_results = candidate_results_from_ids(
                            candidate_ids,
                            records_by_context_id[vertical],
                            retrieval.retrieval_type,
                        )
                        evaluation = evaluate_retrieval_results(
                            gold_evidence_ids=gold_ids,
                            results=retrieval.results,
                        )
                        row = {
                            "vertical": vertical,
                            "stage_size": stage_size,
                            "measurement": measurement,
                            "prompt_id": prompt_id,
                            "dense_backend": retrieval.backend_label,
                            "vector_store": retrieval.vector_store,
                            "candidate_recall_at_20": recall_at_candidate_k(
                                gold_ids=gold_ids,
                                candidate_results=candidate_results,
                                top_k=20,
                            ),
                            "candidate_recall_at_50": recall_at_candidate_k(
                                gold_ids=gold_ids,
                                candidate_results=candidate_results,
                                top_k=50,
                            ),
                            "final_recall_at_5": evaluation["recall_at_5"],
                            "mrr": evaluation["mrr"],
                            "query_rewritten": measurement == "after_recovery",
                            "blocked_direct_hint_count": blocked_count,
                            "direct_hint_leakage_detected": DIRECT_HINT_RE.search(query_text)
                            is not None,
                        }
                        rows.append(row)
                        if (
                            vertical == "retail"
                            and stage_size == max_stage_size
                            and measurement == "before_recovery"
                            and float(row["final_recall_at_5"]) < 1.0
                        ):
                            reasons = retail_failure_reasons(
                                prompt=prompt,
                                gold_ids=gold_ids,
                                candidate_results=candidate_results,
                                final_results=retrieval.results,
                                recall_at_5=float(row["final_recall_at_5"]),
                                known_ids=known_ids_by_vertical["retail"],
                            )
                            retail_failures.append(
                                {
                                    "prompt_id": prompt_id,
                                    "stage_size": stage_size,
                                    "issue_type": prompt.get("issue_type"),
                                    "product_title": prompt.get("product_title"),
                                    "category": prompt.get("category"),
                                    "candidate_recall_at_20": row["candidate_recall_at_20"],
                                    "candidate_recall_at_50": row["candidate_recall_at_50"],
                                    "final_recall_at_5": row["final_recall_at_5"],
                                    "mrr": row["mrr"],
                                    "failure_reasons": reasons,
                                    "retrieved_evidence_kinds": [
                                        retail_evidence_kind(result.context_record)
                                        for result in retrieval.results
                                    ],
                                    "candidate_evidence_kinds_top10": [
                                        retail_evidence_kind(result.context_record)
                                        for result in candidate_results[:10]
                                    ],
                                }
                            )
        return rows, retail_failures, finance_flow_rows
    finally:
        close_retrievers(retrievers)


def build_retail_finance_recovery(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    output_root: str | Path,
    stage_sizes: list[int],
    dense_backend: str = "qdrant_vector",
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = True,
) -> dict[str, Any]:
    """Build Block 16A Retail/Finance recovery reports."""

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    _audit_report, enrichments = build_audit_report(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
    )
    rows, retail_failures, finance_flow_rows = run_retail_finance_validation(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
        enrichments=enrichments,
        stage_sizes=stage_sizes,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    summary_rows = aggregate_validation_rows(rows)
    max_stage_size = max(stage_sizes)
    retail_summary_row = retail_failure_summary(
        failure_rows=retail_failures,
        stage_size=max_stage_size,
    )
    output_path = Path(output_root)
    retail_report = {
        "generated_at_utc": utc_now(),
        "scope": RECOVERY_SCOPE,
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "stage_size": max_stage_size,
        "summary": retail_summary_row,
        "failure_rows": retail_failures,
    }
    finance_flow_summary = {
        "total_prompts": len(finance_flow_rows),
        "period_materialized_count": sum(
            int(bool(row["derived_period"])) for row in finance_flow_rows
        ),
        "metric_materialized_count": sum(
            int(bool(row["derived_metric"])) for row in finance_flow_rows
        ),
        "filing_materialized_count": sum(
            int(bool(row["derived_filing"])) for row in finance_flow_rows
        ),
        "section_materialized_count": sum(
            int(bool(row["derived_section"])) for row in finance_flow_rows
        ),
        "direct_hint_leakage_detected_count": sum(
            int(bool(row["direct_hint_leakage_detected"])) for row in finance_flow_rows
        ),
    }
    finance_flow_report = {
        "generated_at_utc": utc_now(),
        "scope": RECOVERY_SCOPE,
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "row_count": len(finance_flow_rows),
        "summary": finance_flow_summary,
        "rows": finance_flow_rows,
    }
    recovery_report = {
        "generated_at_utc": utc_now(),
        "scope": RECOVERY_SCOPE,
        "known_blocking_baseline_from_block15": {
            "retail": {"recall_at_5": 0.249, "mrr": 0.245},
            "finance": {"recall_at_5": 0.216, "mrr": 0.119467},
        },
        "stage_sizes": stage_sizes,
        "summary_rows": summary_rows,
        "direct_hint_leakage_count": sum(
            int(bool(row["direct_hint_leakage_detected"])) for row in rows
        ),
        "retail_failure_summary": retail_summary_row,
        "finance_metadata_flow_summary": finance_flow_summary,
    }
    write_json(output_path / "retail_failure_report.json", retail_report)
    write_csv(
        output_path / "retail_failure_summary.csv",
        [retail_summary_row],
        RETAIL_FAILURE_FIELDS,
    )
    write_json(output_path / "finance_metadata_flow_report.json", finance_flow_report)
    write_csv(
        output_path / "finance_metadata_flow_summary.csv",
        [finance_flow_summary],
        FINANCE_FLOW_SUMMARY_FIELDS,
    )
    write_json(output_path / "retail_finance_recovery_report.json", recovery_report)
    write_csv(
        output_path / "retail_finance_recovery_summary.csv",
        summary_rows,
        VALIDATION_FIELDS,
    )
    return recovery_report
