"""Block 16B Airline/Healthcare enrichment and Research AI scale validation.

This module runs retrieval-only validation. It does not run model inference,
GPU work, external APIs, or use gold/source IDs as query terms.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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
    evaluate_retrieval_results,
)
from inference_bench.slo import load_slo_config
from inference_bench.vertical_retrieval_repair import (
    DEFAULT_ABLATION_MODE,
    DIRECT_HINT_RE,
    EnrichmentResult,
    build_audit_report,
    candidate_results_from_ids,
    repaired_query,
    select_stage_prompts,
    slo_status_for_metrics,
    warm_qdrant_repair_queries,
)

BLOCK16B_SCOPE = "block16b_airline_healthcare_research_validation_no_inference_no_gpu_no_api"
TARGET_VERTICALS = ("airline", "healthcare_admin")
RESEARCH_VERTICAL = "research_ai"
ENRICHMENT_FIELDS = [
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
    "slo_status",
    "primary_blocker",
    "recommended_next_action",
]
RESEARCH_FIELDS = [
    "vertical",
    "stage_size",
    "ablation_mode",
    "dense_backend",
    "vector_store",
    "candidate_recall_at_20",
    "candidate_recall_at_50",
    "final_recall_at_5",
    "mrr",
    "record_count",
    "direct_hint_leakage_count",
    "drift_vs_500_final_recall_at_5",
    "drift_vs_500_mrr",
    "performance_drift",
    "candidate_degradation",
    "reranking_degradation",
]
KNOWN_BLOCK15_BASELINE = {
    "airline": {
        "stage_size": 500,
        "candidate_recall_at_20": 0.744,
        "candidate_recall_at_50": 0.786333,
        "final_recall_at_5": 0.759,
        "mrr": 0.901467,
    },
    "healthcare_admin": {
        "stage_size": 500,
        "candidate_recall_at_20": 0.84,
        "candidate_recall_at_50": 0.868333,
        "final_recall_at_5": 0.84,
        "mrr": 0.949667,
    },
    "research_ai": {
        "stage_size": 500,
        "candidate_recall_at_20": 1.0,
        "candidate_recall_at_50": 1.0,
        "final_recall_at_5": 0.989333,
        "mrr": 1.0,
    },
}


PreparedQuery = tuple[str, tuple[str, ...], tuple[str, ...], int]


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


def aggregate_enrichment_rows(
    rows: list[dict[str, Any]],
    *,
    slo_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate Airline/Healthcare per-prompt rows."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["vertical"]), int(row["stage_size"]), str(row["measurement"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (vertical, stage_size, measurement), group in sorted(grouped.items()):
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
                "vertical": vertical,
                "stage_size": stage_size,
                "measurement": measurement,
                "ablation_mode": DEFAULT_ABLATION_MODE,
                "dense_backend": ",".join(sorted({str(row["dense_backend"]) for row in group})),
                "vector_store": ",".join(sorted({str(row["vector_store"]) for row in group})),
                **metrics,
                "record_count": len(group),
                "query_rewrite_count": sum(int(bool(row["query_rewritten"])) for row in group),
                "direct_hint_leakage_count": sum(
                    int(bool(row["direct_hint_leakage_detected"])) for row in group
                ),
                "slo_status": status,
                "primary_blocker": blocker,
                "recommended_next_action": action,
            }
        )
    return summary_rows


def aggregate_research_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Research AI stage rows and calculate scale drift."""

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["stage_size"])].append(row)

    raw_summary: list[dict[str, Any]] = []
    for stage_size, group in sorted(grouped.items()):
        raw_summary.append(
            {
                "vertical": RESEARCH_VERTICAL,
                "stage_size": stage_size,
                "ablation_mode": DEFAULT_ABLATION_MODE,
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
                "direct_hint_leakage_count": sum(
                    int(bool(row["direct_hint_leakage_detected"])) for row in group
                ),
            }
        )
    baseline_500 = next((row for row in raw_summary if int(row["stage_size"]) == 500), None)
    baseline_recall = float(baseline_500["final_recall_at_5"]) if baseline_500 else 0.0
    baseline_mrr = float(baseline_500["mrr"]) if baseline_500 else 0.0
    baseline_c20 = float(baseline_500["candidate_recall_at_20"]) if baseline_500 else 0.0
    enriched: list[dict[str, Any]] = []
    for row in raw_summary:
        recall_drift = round(float(row["final_recall_at_5"]) - baseline_recall, 6)
        mrr_drift = round(float(row["mrr"]) - baseline_mrr, 6)
        candidate_drift = round(float(row["candidate_recall_at_20"]) - baseline_c20, 6)
        enriched.append(
            {
                **row,
                "drift_vs_500_final_recall_at_5": recall_drift,
                "drift_vs_500_mrr": mrr_drift,
                "performance_drift": abs(recall_drift) > 0.02 or abs(mrr_drift) > 0.02,
                "candidate_degradation": candidate_drift < -0.02,
                "reranking_degradation": recall_drift < -0.02 and candidate_drift >= -0.02,
            }
        )
    return enriched


def run_enrichment_validation(
    *,
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
    enrichments: dict[str, dict[str, EnrichmentResult]],
    slo_config: dict[str, Any],
    stage_sizes: list[int],
    dense_backend: str,
    vector_store_config_path: str | Path,
    vector_store_key: str,
    allow_dense_fallback: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run Airline/Healthcare before/after staged validation."""

    selected_corpora = {vertical: corpora_by_vertical[vertical] for vertical in TARGET_VERTICALS}
    retrievers = build_retrievers(
        selected_corpora,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    rows: list[dict[str, Any]] = []
    try:
        query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        queries_to_warm: dict[str, set[str]] = {vertical: set() for vertical in TARGET_VERTICALS}
        prepared: dict[
            str,
            list[tuple[dict[str, Any], PreparedQuery, PreparedQuery]],
        ] = {vertical: [] for vertical in TARGET_VERTICALS}
        max_stage_size = max(stage_sizes)
        for vertical in TARGET_VERTICALS:
            vertical_retrievers = retrievers[vertical]
            resolver = cast(
                CompanyTickerResolver | None,
                vertical_retrievers.get("company_ticker_resolver"),
            )
            concept_map = cast(
                dict[str, set[str]],
                vertical_retrievers.get("xbrl_concept_map") or {},
            )
            for prompt in select_stage_prompts(prompts_by_vertical[vertical], max_stage_size):
                base = prompt_query_text(
                    prompt,
                    DEFAULT_ABLATION_MODE,
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
                after_query = repaired_query(
                    prompt=prompt,
                    enrichment=enrichment,
                    resolver=resolver,
                    concept_map=concept_map,
                    ablation_mode=DEFAULT_ABLATION_MODE,
                )
                queries_to_warm[vertical].update({before_query[0], after_query[0]})
                prepared[vertical].append((prompt, before_query, after_query))

        warmed = warm_qdrant_repair_queries(
            retrievers=retrievers,
            queries_by_vertical=queries_to_warm,
            top_k=50,
        )

        for stage_size in stage_sizes:
            for vertical in TARGET_VERTICALS:
                records_by_context_id = cast(
                    dict[str, ContextRecord],
                    retrievers[vertical]["records_by_context_id"],
                )
                for prompt, before_query, after_query in prepared[vertical][:stage_size]:
                    prompt_id = str(prompt.get("prompt_id") or "")
                    gold_record = gold_by_vertical[vertical].get(prompt_id)
                    gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
                    for measurement, query_tuple in (
                        ("before_current_query", before_query),
                        ("after_enrichment", after_query),
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
                            records_by_context_id,
                            retrieval.retrieval_type,
                        )
                        evaluation = evaluate_retrieval_results(
                            gold_evidence_ids=gold_ids,
                            results=retrieval.results,
                        )
                        rows.append(
                            {
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
                                "query_rewritten": measurement == "after_enrichment",
                                "blocked_direct_hint_count": blocked_count,
                                "direct_hint_leakage_detected": DIRECT_HINT_RE.search(query_text)
                                is not None,
                            }
                        )
        summary_rows = aggregate_enrichment_rows(rows, slo_config=slo_config)
        report = {
            "generated_at_utc": utc_now(),
            "scope": BLOCK16B_SCOPE,
            "no_model_inference_triggered": True,
            "no_gpu_work_triggered": True,
            "no_external_api_calls_triggered": True,
            "gold_ids_used_as_query_terms": False,
            "source_ids_used_as_query_terms": False,
            "dense_backend_requested": dense_backend,
            "qdrant_warmed_query_counts": warmed,
            "stage_sizes": stage_sizes,
            "known_block15_baseline": {
                vertical: KNOWN_BLOCK15_BASELINE[vertical] for vertical in TARGET_VERTICALS
            },
            "summary_rows": summary_rows,
            "direct_hint_leakage_detected_count": sum(
                int(bool(row["direct_hint_leakage_detected"])) for row in rows
            ),
        }
        return report, summary_rows
    finally:
        close_retrievers(retrievers)


def run_research_scale_validation(
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run Research AI 500/2000 retrieval-only scale validation."""

    selected_corpora = {RESEARCH_VERTICAL: corpora_by_vertical[RESEARCH_VERTICAL]}
    retrievers = build_retrievers(
        selected_corpora,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    rows: list[dict[str, Any]] = []
    try:
        query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        vertical_retrievers = retrievers[RESEARCH_VERTICAL]
        resolver = cast(
            CompanyTickerResolver | None,
            vertical_retrievers.get("company_ticker_resolver"),
        )
        concept_map = cast(dict[str, set[str]], vertical_retrievers.get("xbrl_concept_map") or {})
        max_stage_size = max(stage_sizes)
        prepared: list[tuple[dict[str, Any], PreparedQuery]] = []
        queries_to_warm: dict[str, set[str]] = {RESEARCH_VERTICAL: set()}
        for prompt in select_stage_prompts(
            prompts_by_vertical[RESEARCH_VERTICAL],
            max_stage_size,
        ):
            enrichment = enrichments[RESEARCH_VERTICAL][str(prompt.get("prompt_id") or "")]
            query_tuple = repaired_query(
                prompt=prompt,
                enrichment=enrichment,
                resolver=resolver,
                concept_map=concept_map,
                ablation_mode=DEFAULT_ABLATION_MODE,
            )
            queries_to_warm[RESEARCH_VERTICAL].add(query_tuple[0])
            prepared.append((prompt, query_tuple))

        warmed = warm_qdrant_repair_queries(
            retrievers=retrievers,
            queries_by_vertical=queries_to_warm,
            top_k=50,
        )
        records_by_context_id = cast(
            dict[str, ContextRecord],
            vertical_retrievers["records_by_context_id"],
        )
        for stage_size in stage_sizes:
            for prompt, query_tuple in prepared[:stage_size]:
                prompt_id = str(prompt.get("prompt_id") or "")
                gold_record = gold_by_vertical[RESEARCH_VERTICAL].get(prompt_id)
                gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
                query_text, expanded_queries, expansion_types, blocked_count = query_tuple
                retrieval = retrieve_for_mode(
                    memory_mode="mm2_hybrid_top5",
                    query=query_text,
                    expanded_queries=expanded_queries,
                    expansion_types=expansion_types,
                    source_hints_used=False,
                    vertical=RESEARCH_VERTICAL,
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
                    records_by_context_id,
                    retrieval.retrieval_type,
                )
                evaluation = evaluate_retrieval_results(
                    gold_evidence_ids=gold_ids,
                    results=retrieval.results,
                )
                rows.append(
                    {
                        "vertical": RESEARCH_VERTICAL,
                        "stage_size": stage_size,
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
                        "blocked_direct_hint_count": blocked_count,
                        "direct_hint_leakage_detected": DIRECT_HINT_RE.search(query_text)
                        is not None,
                    }
                )
        summary_rows = aggregate_research_rows(rows)
        report = {
            "generated_at_utc": utc_now(),
            "scope": BLOCK16B_SCOPE,
            "no_model_inference_triggered": True,
            "no_gpu_work_triggered": True,
            "no_external_api_calls_triggered": True,
            "gold_ids_used_as_query_terms": False,
            "source_ids_used_as_query_terms": False,
            "dense_backend_requested": dense_backend,
            "qdrant_warmed_query_counts": warmed,
            "stage_sizes": stage_sizes,
            "known_block15_baseline": KNOWN_BLOCK15_BASELINE[RESEARCH_VERTICAL],
            "summary_rows": summary_rows,
            "direct_hint_leakage_detected_count": sum(
                int(bool(row["direct_hint_leakage_detected"])) for row in rows
            ),
        }
        return report, summary_rows
    finally:
        close_retrievers(retrievers)


def build_airline_healthcare_research_validation(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    output_root: str | Path,
    slo_config_path: str | Path,
    stage_sizes: list[int],
    research_stage_sizes: list[int],
    dense_backend: str = "qdrant_vector",
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = True,
) -> dict[str, Any]:
    """Build and write Block 16B reports."""

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    slo_config = load_slo_config(slo_config_path)
    _audit_report, enrichments = build_audit_report(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
    )
    enrichment_report, enrichment_summary = run_enrichment_validation(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
        enrichments=enrichments,
        slo_config=slo_config,
        stage_sizes=stage_sizes,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    research_report, research_summary = run_research_scale_validation(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
        enrichments=enrichments,
        stage_sizes=research_stage_sizes,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    output_path = Path(output_root)
    write_json(output_path / "airline_healthcare_enrichment_report.json", enrichment_report)
    write_csv(
        output_path / "airline_healthcare_enrichment_summary.csv",
        enrichment_summary,
        ENRICHMENT_FIELDS,
    )
    write_json(output_path / "research_ai_scale_validation_report.json", research_report)
    write_csv(
        output_path / "research_ai_scale_validation_summary.csv",
        research_summary,
        RESEARCH_FIELDS,
    )
    return {
        "airline_healthcare_enrichment": enrichment_report,
        "research_ai_scale_validation": research_report,
    }
