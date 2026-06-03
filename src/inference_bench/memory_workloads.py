"""Build Phase 3 memory-mode workload records.

This module turns promoted prompts, gold/eval rows, and normalized context
corpora into model-ready workload records. It does not run inference.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, cast

from inference_bench.config import load_memory_modes_config
from inference_bench.context_corpora import VERTICALS, benchmark_paths, read_jsonl
from inference_bench.context_schema import ContextRecord, WorkloadRecord
from inference_bench.evidence_contract import (
    EVIDENCE_SELECTION_SUMMARY_FIELDS,
    build_evidence_selection_report,
    evidence_contracts_from_results,
    validate_evidence_contract,
)
from inference_bench.gold_evidence_audit import (
    GOLD_EVIDENCE_AUDIT_SUMMARY_FIELDS,
    build_gold_evidence_audit_report,
)
from inference_bench.reranker_calibration import (
    RERANKER_CALIBRATION_SUMMARY_FIELDS,
    build_reranker_calibration_report,
)
from inference_bench.retrieval import (
    DEFAULT_CANDIDATE_TOP_K_DENSE,
    DEFAULT_CANDIDATE_TOP_K_LEXICAL,
    DEFAULT_FINAL_TOP_K,
    BM25Retriever,
    CompanyTickerResolver,
    DenseRetrieverProtocol,
    HybridRetriever,
    LocalFallbackDenseRetriever,
    QdrantDenseRetriever,
    RetrievalResult,
    TimedRetrieval,
    build_xbrl_concept_map,
    compress_retrieval_results,
    enrich_query_text,
    evaluate_retrieval_results,
    rerank_candidate_results,
    retrieval_record_payload,
    tokenize,
)
from inference_bench.retrieval_quality_gate import (
    QUALITY_GATE_SUMMARY_FIELDS,
    build_retrieval_quality_gate_report,
)
from inference_bench.run_safety_audit import build_run_safety_audit
from inference_bench.vector_store import QdrantVectorSearcher

CONTEXT_REGEN_COMMAND = (
    "python scripts/phase3/build_context_corpora.py --dataset-root data/scaleup_2000_full "
    "--output-root data/generated/context_engineering"
)

SUPPORTED_MEMORY_MODES = {
    "mm0_no_context",
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
}

SUPPORTED_DENSE_BACKENDS = {"local_fallback", "qdrant_vector"}
SUPPORTED_ABLATION_MODES = {
    "prompt_text_only",
    "prompt_plus_metadata",
    "prompt_plus_source_hints",
}


@dataclass(frozen=True)
class SplitPlan:
    """Prompt-count plan for generated workload splits."""

    smoke_per_vertical: int = 100
    controlled_total: int = 2000
    final_expected_total: int = 10000


DEFAULT_SPLIT_PLAN = SplitPlan()


@dataclass(frozen=True)
class WorkloadBuildResult:
    """Generated workload and report payloads."""

    workload_build_report: dict[str, Any]
    workload_build_summary_rows: list[dict[str, Any]]
    retrieval_evaluation_report: dict[str, Any]
    retrieval_evaluation_summary_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class QueryBuildResult:
    """Ablation-aware retrieval query text and audit labels."""

    query_text: str
    ablation_mode: str
    uses_metadata: bool
    uses_source_hints: bool
    uses_gold_ids: bool = False
    query_enrichment_used: bool = True
    leakage_guard_applied: bool = True
    blocked_direct_hint_count: int = 0
    enrichment_terms: tuple[str, ...] = ()
    expanded_queries: tuple[str, ...] = ()
    expansion_types: tuple[str, ...] = ()


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write a CSV file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_workload_jsonl(path: str | Path, records: list[WorkloadRecord]) -> Path:
    """Write workload records as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(asdict(record), ensure_ascii=True, sort_keys=True) for record in records
    )
    output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return output_path


def write_workload_jsonl_line(file: Any, record: WorkloadRecord) -> None:
    """Write one workload record to an open JSONL file."""

    file.write(json.dumps(asdict(record), ensure_ascii=True, sort_keys=True))
    file.write("\n")


def load_prompts_and_gold(
    dataset_root: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """Load promoted prompts and gold/eval rows by vertical."""

    prompts_by_vertical: dict[str, list[dict[str, Any]]] = {}
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]] = {}
    for vertical in VERTICALS:
        paths = benchmark_paths(dataset_root, vertical)
        prompts = sorted(read_jsonl(paths["prompts"]), key=lambda row: str(row.get("prompt_id")))
        gold_rows = read_jsonl(paths["gold"])
        prompts_by_vertical[vertical] = prompts
        gold_by_vertical[vertical] = {
            str(row.get("prompt_id")): row for row in gold_rows if row.get("prompt_id")
        }
    return prompts_by_vertical, gold_by_vertical


def load_context_corpora(context_root: str | Path) -> dict[str, list[ContextRecord]]:
    """Load generated normalized context corpora."""

    root = Path(context_root)
    corpora: dict[str, list[ContextRecord]] = {}
    missing_paths: list[Path] = []
    for vertical in VERTICALS:
        path = root / "corpora" / f"{vertical}_context_corpus.jsonl"
        if not path.exists():
            missing_paths.append(path)
            continue
        corpora[vertical] = [ContextRecord(**row) for row in read_jsonl(path)]

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        msg = (
            "Missing context corpora required for memory-mode workload generation:\n"
            f"{missing}\nRegenerate them with:\n{CONTEXT_REGEN_COMMAND}"
        )
        raise RuntimeError(msg)
    return corpora


def gold_evidence_ids(gold_record: dict[str, Any] | None) -> list[str]:
    """Return unique evidence IDs from a gold/eval row."""

    if not gold_record:
        return []
    ids: list[str] = []
    for field_name in ("required_doc_ids", "required_evidence_ids", "required_chunk_ids"):
        value = gold_record.get(field_name)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return list(dict.fromkeys(ids))


def validate_ablation_mode(ablation_mode: str) -> None:
    """Validate a retrieval ablation mode."""

    if ablation_mode not in SUPPORTED_ABLATION_MODES:
        msg = f"Unknown ablation mode '{ablation_mode}'"
        raise ValueError(msg)


def prompt_query_text(
    prompt: dict[str, Any],
    ablation_mode: str = "prompt_plus_source_hints",
    *,
    company_ticker_resolver: CompanyTickerResolver | None = None,
    xbrl_concept_map: dict[str, set[str]] | None = None,
) -> QueryBuildResult:
    """Build retrieval query text under one strict ablation policy."""

    validate_ablation_mode(ablation_mode)
    raw_metadata = prompt.get("metadata")
    metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
    vertical = str(prompt.get("vertical") or "")
    parts: list[str] = []

    def append_value(value: Any, repeat: int = 1) -> None:
        if isinstance(value, list):
            for item in value:
                append_value(item, repeat=repeat)
            return
        if value:
            parts.extend([str(value)] * repeat)

    for field_name in ("question", "issue"):
        append_value(prompt.get(field_name))

    uses_metadata = ablation_mode in {"prompt_plus_metadata", "prompt_plus_source_hints"}
    uses_source_hints = ablation_mode == "prompt_plus_source_hints"

    if not uses_metadata:
        enriched = enrich_query_text(
            " ".join(parts),
            vertical=vertical,
            allow_direct_identifiers=False,
            resolver=company_ticker_resolver,
            concept_map=xbrl_concept_map,
        )
        return QueryBuildResult(
            query_text=enriched.query_text,
            ablation_mode=ablation_mode,
            uses_metadata=False,
            uses_source_hints=False,
            blocked_direct_hint_count=enriched.blocked_direct_hint_count,
            enrichment_terms=enriched.enrichment_terms,
            expanded_queries=enriched.expanded_queries,
            expansion_types=enriched.expansion_types,
        )

    for field_name in (
        "vertical",
        "task_type",
        "expected_output_format",
        "expected_status",
        "category",
        "product_title",
        "ticker",
        "company",
        "filing_form",
        "topic",
        "support_type",
        "department",
        "product_id",
    ):
        append_value(prompt.get(field_name))
    for metadata_field in (
        "prompt_category",
        "evidence_type",
        "topics",
        "required_section_types",
    ):
        append_value(metadata.get(metadata_field), repeat=2)

    allow_direct_identifiers = False
    if uses_source_hints:
        allow_direct_identifiers = True
        for hint_field in (
            "required_doc_ids",
            "required_evidence_ids",
            "required_chunk_ids",
            "required_policy_ids",
            "required_paper_ids",
            "source_paper_ids",
            "source_parent_asins",
            "source_product_ids",
        ):
            append_value(prompt.get(hint_field), repeat=4)
        for metadata_field in (
            "source_titles",
            "source_parent_asins",
            "required_paper_ids",
        ):
            append_value(metadata.get(metadata_field), repeat=2)

    enriched = enrich_query_text(
        " ".join(parts),
        vertical=vertical,
        allow_direct_identifiers=allow_direct_identifiers,
        resolver=company_ticker_resolver,
        concept_map=xbrl_concept_map,
        metadata_terms=set(tokenize(" ".join(parts))) if uses_metadata else None,
    )
    return QueryBuildResult(
        query_text=enriched.query_text,
        ablation_mode=ablation_mode,
        uses_metadata=uses_metadata,
        uses_source_hints=uses_source_hints,
        leakage_guard_applied=not allow_direct_identifiers,
        blocked_direct_hint_count=enriched.blocked_direct_hint_count,
        enrichment_terms=enriched.enrichment_terms,
        expanded_queries=enriched.expanded_queries,
        expansion_types=enriched.expansion_types,
    )


def expected_output_format(prompt: dict[str, Any], gold_record: dict[str, Any] | None) -> str:
    """Return the expected output format."""

    if prompt.get("expected_output_format"):
        return str(prompt["expected_output_format"])
    if gold_record:
        metadata = gold_record.get("metadata")
        if isinstance(metadata, dict) and metadata.get("expected_output_format"):
            return str(metadata["expected_output_format"])
    return "text"


def stratified_select(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Select records deterministically across task/status/output-format groups."""

    sorted_records = sorted(records, key=lambda row: str(row.get("prompt_id")))
    if len(sorted_records) <= count:
        return sorted_records

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted_records:
        key = (
            str(row.get("task_type") or "missing"),
            str(row.get("expected_output_format") or "missing"),
            str(row.get("expected_status") or "missing"),
        )
        groups[key].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    total = len(sorted_records)
    for key in sorted(groups):
        group = groups[key]
        allocation = max(1, round(count * len(group) / total))
        for row in group[:allocation]:
            if len(selected) >= count:
                break
            selected.append(row)
            selected_ids.add(str(row.get("prompt_id")))

    if len(selected) < count:
        for row in sorted_records:
            prompt_id = str(row.get("prompt_id"))
            if prompt_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(prompt_id)
            if len(selected) >= count:
                break

    return sorted(selected[:count], key=lambda row: str(row.get("prompt_id")))


def select_prompts_for_split(
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    split: str,
    split_plan: SplitPlan | None = None,
) -> list[dict[str, Any]]:
    """Select prompt rows for one workload split."""

    active_split_plan = split_plan or DEFAULT_SPLIT_PLAN
    if split == "smoke_500":
        selected: list[dict[str, Any]] = []
        for vertical in VERTICALS:
            prompts = prompts_by_vertical.get(vertical, [])
            if len(prompts) < active_split_plan.smoke_per_vertical:
                msg = (
                    "smoke_500 cannot produce "
                    f"{active_split_plan.smoke_per_vertical} records for {vertical}"
                )
                raise RuntimeError(msg)
            selected.extend(prompts[: active_split_plan.smoke_per_vertical])
        return selected

    if split == "controlled_2000":
        if active_split_plan.controlled_total % len(VERTICALS) != 0:
            msg = "controlled_total must divide evenly across verticals"
            raise ValueError(msg)
        per_vertical = active_split_plan.controlled_total // len(VERTICALS)
        selected = []
        for vertical in VERTICALS:
            prompts = prompts_by_vertical.get(vertical, [])
            if len(prompts) < per_vertical:
                msg = f"controlled_2000 cannot produce {per_vertical} records for {vertical}"
                raise RuntimeError(msg)
            selected.extend(stratified_select(prompts, per_vertical))
        return selected

    if split == "final_10000":
        selected = [row for vertical in VERTICALS for row in prompts_by_vertical.get(vertical, [])]
        if len(selected) != active_split_plan.final_expected_total:
            msg = (
                f"final_10000 expected {active_split_plan.final_expected_total} promoted prompts, "
                f"found {len(selected)}"
            )
            raise RuntimeError(msg)
        return sorted(selected, key=lambda row: str(row.get("prompt_id")))

    msg = f"Unknown dataset split '{split}'"
    raise ValueError(msg)


def build_retrievers(
    corpora_by_vertical: dict[str, list[ContextRecord]],
    *,
    dense_backend: str = "local_fallback",
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build lexical, dense, and hybrid retrievers per vertical."""

    if dense_backend not in SUPPORTED_DENSE_BACKENDS:
        msg = f"Unknown dense backend '{dense_backend}'"
        raise ValueError(msg)

    retrievers: dict[str, dict[str, Any]] = {}
    qdrant_searchers: dict[str, QdrantVectorSearcher] = {}
    qdrant_client: Any | None = None
    qdrant_embedding_provider: Any | None = None
    qdrant_vector_config: Any | None = None
    if dense_backend == "qdrant_vector":
        try:
            from inference_bench.config import resolve_vector_store
            from inference_bench.vector_store import build_embedding_provider, build_qdrant_client

            qdrant_vector_config = resolve_vector_store(vector_store_key, vector_store_config_path)
            qdrant_embedding_provider = build_embedding_provider(qdrant_vector_config)
            qdrant_client = build_qdrant_client(qdrant_vector_config)
        except RuntimeError:
            if not allow_dense_fallback:
                raise
            qdrant_client = None
            qdrant_embedding_provider = None
            qdrant_vector_config = None
    for vertical, records in corpora_by_vertical.items():
        lexical = BM25Retriever(records)
        dense: DenseRetrieverProtocol
        if (
            dense_backend == "qdrant_vector"
            and qdrant_client is not None
            and qdrant_embedding_provider is not None
            and qdrant_vector_config is not None
        ):
            try:
                searcher = QdrantVectorSearcher(
                    config=qdrant_vector_config,
                    vertical=vertical,
                    embedding_provider=qdrant_embedding_provider,
                    client=qdrant_client,
                    records_by_id={record.context_id: record for record in records},
                )
                qdrant_searchers[vertical] = searcher
                dense = QdrantDenseRetriever(searcher)
            except RuntimeError:
                if not allow_dense_fallback:
                    if qdrant_client is not None:
                        qdrant_client.close()
                    raise
                dense = LocalFallbackDenseRetriever(records)
        else:
            dense = LocalFallbackDenseRetriever(records)
        hybrid = HybridRetriever(lexical, dense)
        retrievers[vertical] = {
            "lexical": lexical,
            "dense": dense,
            "hybrid": hybrid,
            "records_by_context_id": {record.context_id: record for record in records},
            "company_ticker_resolver": CompanyTickerResolver.from_records(records)
            if vertical == "finance"
            else None,
            "xbrl_concept_map": build_xbrl_concept_map(records) if vertical == "finance" else {},
        }
    if qdrant_searchers:
        retrievers["_qdrant_searchers"] = qdrant_searchers
    if qdrant_client is not None:
        retrievers["_qdrant_client"] = {"client": qdrant_client}
    return retrievers


def close_retrievers(retrievers: dict[str, dict[str, Any]]) -> None:
    """Close local vector-store resources held by retrievers."""

    qdrant_client_payload = retrievers.get("_qdrant_client", {})
    qdrant_client = qdrant_client_payload.get("client") if qdrant_client_payload else None
    if qdrant_client is not None:
        qdrant_client.close()


def warm_qdrant_query_embeddings(
    *,
    retrievers: dict[str, dict[str, Any]],
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    splits: list[str],
    ablation_modes: list[str],
    split_plan: SplitPlan,
) -> dict[str, int]:
    """Batch-encode Qdrant query embeddings before large workload generation."""

    qdrant_searchers = retrievers.get("_qdrant_searchers", {})
    if not qdrant_searchers:
        return {}

    queries_by_vertical: dict[str, set[str]] = {vertical: set() for vertical in VERTICALS}
    for split in splits:
        for prompt in select_prompts_for_split(prompts_by_vertical, split, split_plan):
            vertical = str(prompt.get("vertical") or "")
            if vertical not in queries_by_vertical:
                continue
            resolver = cast(Any, retrievers[vertical]).get("company_ticker_resolver")
            concept_map = cast(Any, retrievers[vertical]).get("xbrl_concept_map") or {}
            for ablation_mode in ablation_modes:
                query = prompt_query_text(
                    prompt,
                    ablation_mode,
                    company_ticker_resolver=resolver,
                    xbrl_concept_map=concept_map,
                )
                queries_by_vertical[vertical].add(query.query_text)

    warmed_counts: dict[str, int] = {}
    for vertical, queries in queries_by_vertical.items():
        searcher = qdrant_searchers.get(vertical)
        if searcher is None:
            continue
        cast(Any, searcher).warm_snapshot_search_results(
            sorted(queries),
            top_k=max(DEFAULT_CANDIDATE_TOP_K_DENSE, 120),
        )
        warmed_counts[vertical] = len(queries)
    return warmed_counts


def no_context_retrieval() -> TimedRetrieval:
    """Return no-context retrieval metadata."""

    return TimedRetrieval(
        results=[],
        latency_ms=0.0,
        backend_label="unavailable",
        retrieval_type="none",
        vector_store="none",
    )


def retrieve_for_mode(
    *,
    memory_mode: str,
    query: str,
    expanded_queries: tuple[str, ...] = (),
    expansion_types: tuple[str, ...] = (),
    source_hints_used: bool = False,
    vertical: str,
    retrievers: dict[str, dict[str, Any]],
    top_k: int,
    candidate_top_k_dense: int = DEFAULT_CANDIDATE_TOP_K_DENSE,
    candidate_top_k_lexical: int = DEFAULT_CANDIDATE_TOP_K_LEXICAL,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    retrieval_cache: dict[tuple[str, str, str, tuple[str, ...], int], TimedRetrieval] | None = None,
) -> TimedRetrieval:
    """Run retrieval for one memory mode."""

    if memory_mode == "mm0_no_context":
        return no_context_retrieval()
    cache_mode = "mm2_hybrid_top5" if memory_mode == "mm3_compressed_hybrid_top5" else memory_mode
    active_expanded_queries = expanded_queries or (query,)
    cache_key = (cache_mode, vertical, query, active_expanded_queries, top_k)
    if retrieval_cache is not None and cache_key in retrieval_cache:
        return retrieval_cache[cache_key]
    if memory_mode == "mm1_dense_top5":
        started = time.perf_counter()
        candidate_results: list[RetrievalResult] = []
        dense_retriever = cast(Any, retrievers[vertical]["dense"])
        candidate_results.extend(dense_retriever.retrieve(query, candidate_top_k_dense).results)
        retrieval = rerank_candidate_results(
            query=query,
            candidate_results=candidate_results,
            final_top_k=final_top_k,
            retrieval_mode="dense",
            lexical_weight=0.0,
            dense_weight=1.0,
            source_hints_used=source_hints_used,
            candidate_top_k_dense=candidate_top_k_dense,
            candidate_top_k_lexical=0,
            expanded_query_count=len(active_expanded_queries),
            expansion_types=expansion_types,
            started=started,
            backend_label=dense_retriever.backend_label,
            vector_store=dense_retriever.vector_store,
            boost_features_by_context_id=cast(
                HybridRetriever, retrievers[vertical]["hybrid"]
            ).boost_features,
        )
    elif memory_mode in {"mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}:
        retrieval = cast(HybridRetriever, retrievers[vertical]["hybrid"]).retrieve(
            query,
            final_top_k,
            expanded_queries=active_expanded_queries,
            candidate_top_k_dense=candidate_top_k_dense,
            candidate_top_k_lexical=candidate_top_k_lexical,
            source_hints_used=source_hints_used,
            expansion_types=expansion_types,
        )
    else:
        msg = f"Unknown memory mode '{memory_mode}'"
        raise ValueError(msg)
    if retrieval_cache is not None:
        retrieval_cache[cache_key] = retrieval
    return retrieval


def assemble_messages(
    *,
    question: str,
    context_records: list[ContextRecord],
    memory_mode: str,
) -> list[dict[str, str]]:
    """Assemble chat messages for a future model runner."""

    if not context_records:
        return [
            {"role": "system", "content": "Answer the user request."},
            {"role": "user", "content": question},
        ]

    context_lines: list[str] = []
    for index, record in enumerate(context_records, start=1):
        context_lines.append(
            "\n".join(
                [
                    f"[{index}] {record.title}",
                    f"Context ID: {record.context_id}",
                    f"Provenance: {record.provenance}",
                    record.text,
                ]
            )
        )
    user_content = (
        "Use the provided context records when answering. Cite context IDs when relevant.\n\n"
        f"Memory mode: {memory_mode}\n\n"
        f"Context:\n\n{chr(10).join(context_lines)}\n\nQuestion:\n{question}"
    )
    return [
        {
            "role": "system",
            "content": "Answer using only the supplied context when context is provided.",
        },
        {"role": "user", "content": user_content},
    ]


def retrieval_metadata_payload(
    *,
    retrieval: TimedRetrieval,
    selected_results: list[RetrievalResult],
    evaluation: dict[str, Any],
    configured_top_k: int,
    query_build: QueryBuildResult,
    compression_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build per-workload retrieval metadata."""

    payload: dict[str, Any] = {
        "retrieval_type": retrieval.retrieval_type,
        "retrieval_backend_label": retrieval.backend_label,
        "dense_backend": retrieval.backend_label,
        "vector_store": retrieval.vector_store,
        "ablation_mode": query_build.ablation_mode,
        "source_hints_used": query_build.uses_source_hints,
        "metadata_used": query_build.uses_metadata,
        "gold_ids_used_in_query": query_build.uses_gold_ids,
        "query_enrichment_used": query_build.query_enrichment_used,
        "leakage_guard_applied": query_build.leakage_guard_applied,
        "blocked_direct_hint_count": query_build.blocked_direct_hint_count,
        "enrichment_terms": list(query_build.enrichment_terms),
        "expanded_queries": list(query_build.expanded_queries),
        "expanded_query_count": retrieval.diagnostics.get("expanded_query_count", 0),
        "expansion_types": retrieval.diagnostics.get("expansion_types", []),
        "reranking_used": bool(retrieval.diagnostics.get("reranked"))
        or retrieval.retrieval_type == "hybrid",
        "reranker_enabled": retrieval.diagnostics.get("reranker_enabled", False),
        "retrieval_latency_ms": round(retrieval.latency_ms, 6),
        "configured_top_k": configured_top_k,
        "candidate_top_k_dense": retrieval.diagnostics.get("candidate_top_k_dense", 0),
        "candidate_top_k_lexical": retrieval.diagnostics.get("candidate_top_k_lexical", 0),
        "final_top_k": retrieval.diagnostics.get("final_top_k", configured_top_k),
        "candidates_before_dedupe": retrieval.diagnostics.get("candidates_before_dedupe", 0),
        "candidates_after_dedupe": retrieval.diagnostics.get("candidates_after_dedupe", 0),
        "retrieved_count": len(retrieval.results),
        "selected_context_ids": [result.context_record.context_id for result in selected_results],
        "candidate_context_ids_sample": retrieval.diagnostics.get("candidate_context_ids", [])[:10],
        "pre_rerank_top_context_ids": retrieval.diagnostics.get(
            "pre_rerank_top_context_ids",
            [],
        ),
        "ranked_results": [retrieval_record_payload(result) for result in retrieval.results],
        "recall_at_5": evaluation["recall_at_5"],
        "mrr": evaluation["mrr"],
        "gold_evidence_included": evaluation["gold_evidence_included"],
        "missing_gold_evidence_count": evaluation["missing_gold_evidence_count"],
    }
    if compression_metadata is not None:
        payload["compression"] = compression_metadata
    return payload


def context_records_from_results(results: list[RetrievalResult]) -> list[ContextRecord]:
    """Return context records in retrieval order."""

    return [result.context_record for result in results]


def recall_at_candidate_k(
    *,
    gold_ids: list[str],
    candidate_results: list[RetrievalResult],
    top_k: int,
) -> float:
    """Return recall against the candidate list truncated to top-k."""

    return float(
        evaluate_retrieval_results(
            gold_evidence_ids=gold_ids,
            results=candidate_results[:top_k],
        )["recall_at_5"]
    )


def mrr_at_candidate_k(
    *,
    gold_ids: list[str],
    candidate_results: list[RetrievalResult],
    top_k: int,
) -> float:
    """Return MRR against the candidate list truncated to top-k."""

    return float(
        evaluate_retrieval_results(
            gold_evidence_ids=gold_ids,
            results=candidate_results[:top_k],
        )["mrr"]
    )


def build_one_workload_record(
    *,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    dataset_split: str,
    memory_mode: str,
    ablation_mode: str,
    retrievers: dict[str, dict[str, Any]],
    mode_configs: dict[str, Any],
    retrieval_cache: dict[tuple[str, str, str, tuple[str, ...], int], TimedRetrieval] | None = None,
) -> tuple[WorkloadRecord, dict[str, Any]]:
    """Build one workload record and its evaluation row."""

    mode_config = mode_configs[memory_mode]
    vertical = str(prompt.get("vertical") or "")
    prompt_id = str(prompt.get("prompt_id") or "")
    vertical_retrievers = retrievers[vertical]
    query_build = prompt_query_text(
        prompt,
        ablation_mode,
        company_ticker_resolver=cast(
            CompanyTickerResolver | None,
            vertical_retrievers.get("company_ticker_resolver"),
        ),
        xbrl_concept_map=cast(
            dict[str, set[str]], vertical_retrievers.get("xbrl_concept_map") or {}
        ),
    )
    query = query_build.query_text
    gold_ids = gold_evidence_ids(gold_record)
    retrieval = retrieve_for_mode(
        memory_mode=memory_mode,
        query=query,
        expanded_queries=query_build.expanded_queries,
        expansion_types=query_build.expansion_types,
        source_hints_used=query_build.uses_source_hints,
        vertical=vertical,
        retrievers=retrievers,
        top_k=mode_config.top_k,
        final_top_k=mode_config.top_k,
        retrieval_cache=retrieval_cache,
    )

    selected_results = retrieval.results
    compression_metadata: dict[str, Any] | None = None
    pre_compression_evaluation = evaluate_retrieval_results(
        gold_evidence_ids=gold_ids,
        results=retrieval.results,
    )
    records_by_context_id = cast(
        dict[str, ContextRecord], vertical_retrievers["records_by_context_id"]
    )
    candidate_context_ids = [
        str(context_id) for context_id in retrieval.diagnostics.get("candidate_context_ids", [])
    ]
    candidate_results = [
        RetrievalResult(
            context_record=records_by_context_id[context_id],
            score=0.0,
            rank=index,
            retrieval_mode=retrieval.retrieval_type,
            component_scores={},
        )
        for index, context_id in enumerate(candidate_context_ids, start=1)
        if context_id in records_by_context_id
    ]
    candidate_evaluation = evaluate_retrieval_results(
        gold_evidence_ids=gold_ids,
        results=candidate_results,
    )
    candidate_recall_at_10 = recall_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=10,
    )
    candidate_recall_at_20 = recall_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=20,
    )
    candidate_recall_at_50 = recall_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=50,
    )
    candidate_recall_at_100 = recall_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=100,
    )
    candidate_recall_at_200 = recall_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=200,
    )
    candidate_mrr_at_100 = mrr_at_candidate_k(
        gold_ids=gold_ids,
        candidate_results=candidate_results,
        top_k=100,
    )
    pre_rerank_context_ids = [
        str(context_id)
        for context_id in retrieval.diagnostics.get("pre_rerank_top_context_ids", [])
    ]
    pre_rerank_results = [
        RetrievalResult(
            context_record=records_by_context_id[context_id],
            score=0.0,
            rank=index,
            retrieval_mode=retrieval.retrieval_type,
            component_scores={},
        )
        for index, context_id in enumerate(pre_rerank_context_ids, start=1)
        if context_id in records_by_context_id
    ]
    pre_rerank_evaluation = evaluate_retrieval_results(
        gold_evidence_ids=gold_ids,
        results=pre_rerank_results,
    )
    if memory_mode == "mm3_compressed_hybrid_top5":
        compressed = compress_retrieval_results(
            retrieval.results,
            max_context_tokens=mode_config.max_context_tokens,
        )
        selected_results = compressed.results
        post_compression_evaluation = evaluate_retrieval_results(
            gold_evidence_ids=gold_ids,
            results=selected_results,
        )
        compression_metadata = {
            "compression_type": "deterministic_score_dedupe_budget",
            "original_context_tokens": compressed.original_token_count,
            "compressed_context_tokens": compressed.compressed_token_count,
            "token_reduction": compressed.token_reduction,
            "token_reduction_pct": round(1 - compressed.compression_ratio, 6),
            "compression_ratio": compressed.compression_ratio,
            "dropped_context_ids": compressed.dropped_context_ids,
            "recall_before_compression": pre_compression_evaluation["recall_at_5"],
            "recall_after_compression": post_compression_evaluation["recall_at_5"],
            "recall_loss": round(
                max(
                    0.0,
                    float(pre_compression_evaluation["recall_at_5"])
                    - float(post_compression_evaluation["recall_at_5"]),
                ),
                6,
            ),
            "gold_evidence_retained_after_compression": post_compression_evaluation[
                "gold_evidence_included"
            ],
        }

    selected_context_records = context_records_from_results(selected_results)
    evaluation = evaluate_retrieval_results(
        gold_evidence_ids=gold_ids,
        results=selected_results,
    )
    selection_reasons_by_context_id = {
        str(context_id): str(reason)
        for context_id, reason in (
            retrieval.diagnostics.get("selection_reasons_by_context_id") or {}
        ).items()
    }
    evidence_contract_valid = True
    evidence_contract_count = 0
    for contract in evidence_contracts_from_results(
        selected_results[:1],
        selection_reasons_by_context_id=selection_reasons_by_context_id,
    ):
        validate_evidence_contract(contract)
        evidence_contract_count += 1
    metadata = retrieval_metadata_payload(
        retrieval=retrieval,
        selected_results=selected_results,
        evaluation=evaluation,
        configured_top_k=mode_config.top_k,
        query_build=query_build,
        compression_metadata=compression_metadata,
    )
    metadata["reranker_backend"] = retrieval.diagnostics.get("reranker_backend", "heuristic")
    metadata["calibrated_reranker_enabled"] = retrieval.diagnostics.get(
        "calibrated_reranker_enabled",
        False,
    )
    metadata["evidence_selector_strategy"] = retrieval.diagnostics.get(
        "evidence_selector_strategy",
        "unavailable",
    )
    metadata["selection_reasons_by_context_id"] = selection_reasons_by_context_id
    metadata["evidence_contract_schema"] = "phase3_block12"
    metadata["selected_evidence_contract_count_validated"] = evidence_contract_count
    metadata["evidence_contract_valid"] = evidence_contract_valid
    question = str(prompt.get("question") or prompt.get("issue") or "")
    workload_record = WorkloadRecord(
        workload_id=f"{dataset_split}:{ablation_mode}:{memory_mode}:{prompt_id}",
        prompt_id=prompt_id,
        vertical=vertical,
        memory_mode=memory_mode,
        messages=assemble_messages(
            question=question,
            context_records=selected_context_records,
            memory_mode=memory_mode,
        ),
        context_records=list(selected_context_records),
        context_token_estimate=sum(record.token_estimate for record in selected_context_records),
        retrieval_metadata=metadata,
        expected_output_format=expected_output_format(prompt, gold_record),
        gold_evidence_ids=gold_ids,
        dataset_split=dataset_split,
        source_prompt_record=prompt,
    )
    eval_row = {
        "split": dataset_split,
        "ablation_mode": ablation_mode,
        "memory_mode": memory_mode,
        "vertical": vertical,
        "prompt_id": prompt_id,
        "recall_at_5": evaluation["recall_at_5"],
        "mrr": evaluation["mrr"],
        "gold_evidence_included": evaluation["gold_evidence_included"],
        "missing_gold_evidence_count": evaluation["missing_gold_evidence_count"],
        "retrieval_latency_ms": retrieval.latency_ms,
        "context_token_count": workload_record.context_token_estimate,
        "context_rows_selected": len(selected_context_records),
        "distinct_context_ids": [result.context_record.context_id for result in selected_results],
        "retrieval_backend_label": retrieval.backend_label,
        "dense_backend": retrieval.backend_label,
        "vector_store": retrieval.vector_store,
        "source_hints_used": query_build.uses_source_hints,
        "metadata_used": query_build.uses_metadata,
        "gold_ids_used_in_query": query_build.uses_gold_ids,
        "query_enrichment_used": query_build.query_enrichment_used,
        "leakage_guard_applied": query_build.leakage_guard_applied,
        "blocked_direct_hint_count": query_build.blocked_direct_hint_count,
        "enrichment_terms": list(query_build.enrichment_terms),
        "reranking_used": bool(retrieval.diagnostics.get("reranked"))
        or retrieval.retrieval_type == "hybrid",
        "compression_ratio": compression_metadata["compression_ratio"]
        if compression_metadata
        else None,
        "token_reduction_pct": compression_metadata["token_reduction_pct"]
        if compression_metadata
        else None,
        "recall_before_compression": compression_metadata["recall_before_compression"]
        if compression_metadata
        else None,
        "recall_after_compression": compression_metadata["recall_after_compression"]
        if compression_metadata
        else None,
        "recall_loss": compression_metadata["recall_loss"] if compression_metadata else None,
        "gold_evidence_retained_after_compression": compression_metadata[
            "gold_evidence_retained_after_compression"
        ]
        if compression_metadata
        else None,
        "token_reduction": compression_metadata["token_reduction"] if compression_metadata else 0,
        "query_text": query,
        "gold_evidence_ids": gold_ids,
        "matched_gold_evidence_ids": evaluation["matched_gold_evidence_ids"],
        "retrieved_context_ids": [result.context_record.context_id for result in retrieval.results],
        "candidate_context_ids": candidate_context_ids,
        "pre_rerank_top_context_ids": pre_rerank_context_ids,
        "gold_in_candidate_pool": candidate_evaluation["gold_evidence_included"],
        "candidate_recall_at_10": candidate_recall_at_10,
        "candidate_recall_at_20": candidate_recall_at_20,
        "candidate_recall_at_50": candidate_recall_at_50,
        "candidate_recall_at_100": candidate_recall_at_100,
        "candidate_recall_at_200": candidate_recall_at_200,
        "candidate_mrr_at_100": candidate_mrr_at_100,
        "candidate_diagnostic_max_k_available": len(candidate_results),
        "candidate_recall_at_100_feasible": len(candidate_results) >= 100,
        "candidate_recall_at_200_feasible": len(candidate_results) >= 200,
        "pre_rerank_recall_at_5": pre_rerank_evaluation["recall_at_5"],
        "gold_in_top50_but_not_top5": (
            candidate_recall_at_50 > 0 and float(evaluation["recall_at_5"]) <= 0
        ),
        "gold_in_top100_but_not_top5": (
            candidate_recall_at_100 > 0 and float(evaluation["recall_at_5"]) <= 0
        ),
        "gold_absent_from_top100": candidate_recall_at_100 <= 0 and bool(gold_ids),
        "reranker_rescued_gold": (
            bool(evaluation["gold_evidence_included"])
            and not bool(pre_rerank_evaluation["gold_evidence_included"])
            and bool(candidate_evaluation["gold_evidence_included"])
        ),
        "reranker_backend": retrieval.diagnostics.get("reranker_backend", "heuristic"),
        "calibrated_reranker_enabled": retrieval.diagnostics.get(
            "calibrated_reranker_enabled",
            False,
        ),
        "evidence_selector_strategy": retrieval.diagnostics.get(
            "evidence_selector_strategy",
            "unavailable",
        ),
        "selection_reasons": [
            selection_reasons_by_context_id.get(result.context_record.context_id, "")
            for result in selected_results
        ],
        "evidence_contract_valid": evidence_contract_valid,
        "duplicate_avoidance_applied": True,
        "candidate_top_k_dense": retrieval.diagnostics.get(
            "candidate_top_k_dense",
            0,
        ),
        "candidate_top_k_lexical": retrieval.diagnostics.get(
            "candidate_top_k_lexical",
            0,
        ),
        "final_top_k": retrieval.diagnostics.get("final_top_k", mode_config.top_k),
        "reranker_enabled": retrieval.diagnostics.get("reranker_enabled", False),
        "candidates_before_dedupe": retrieval.diagnostics.get("candidates_before_dedupe", 0),
        "candidates_after_dedupe": retrieval.diagnostics.get("candidates_after_dedupe", 0),
        "expanded_query_count": retrieval.diagnostics.get("expanded_query_count", 0),
        "expansion_types": retrieval.diagnostics.get("expansion_types", []),
    }
    return workload_record, eval_row


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval evaluation rows."""

    if not rows:
        return {
            "record_count": 0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "avg_retrieval_latency_ms": 0.0,
            "avg_context_tokens": 0.0,
            "gold_evidence_included_count": 0,
            "missing_gold_evidence_count": 0,
            "total_context_rows_selected": 0,
            "distinct_context_rows_used": 0,
            "retrieval_backend_label": "unavailable",
            "dense_backend": "unavailable",
            "vector_store": "none",
            "compression_ratio": None,
            "token_reduction_pct": None,
            "recall_loss": None,
            "token_reduction": 0,
            "query_enrichment_used": False,
            "reranking_used": False,
            "reranker_enabled": False,
            "source_hints_used": False,
            "leakage_guard_applied": False,
            "blocked_direct_hint_count": 0,
            "candidate_top_k_dense": 0,
            "candidate_top_k_lexical": 0,
            "final_top_k": 0,
            "avg_candidates_before_dedupe": 0.0,
            "avg_candidates_after_dedupe": 0.0,
            "avg_expanded_query_count": 0.0,
            "gold_in_candidate_pool_count": 0,
            "candidate_recall_at_10": 0.0,
            "candidate_recall_at_20": 0.0,
            "candidate_recall_at_50": 0.0,
            "candidate_recall_at_100": 0.0,
            "candidate_recall_at_200": 0.0,
            "candidate_mrr_at_100": 0.0,
            "candidate_diagnostic_max_k_available": 0,
            "candidate_recall_at_100_feasible": False,
            "candidate_recall_at_200_feasible": False,
            "pre_rerank_recall_at_5": 0.0,
            "reranker_rescued_gold_count": 0,
            "gold_in_top50_but_not_top5_rate": 0.0,
            "gold_in_top100_but_not_top5_rate": 0.0,
            "gold_absent_from_top100_rate": 0.0,
            "reranker_backend": "unavailable",
            "calibrated_reranker_enabled": False,
            "evidence_selector_strategy": "unavailable",
        }
    distinct_context_ids: set[str] = set()
    compression_values = [
        float(row["compression_ratio"]) for row in rows if row.get("compression_ratio") is not None
    ]
    token_reduction_pct_values = [
        float(row["token_reduction_pct"])
        for row in rows
        if row.get("token_reduction_pct") is not None
    ]
    recall_loss_values = [
        float(row["recall_loss"]) for row in rows if row.get("recall_loss") is not None
    ]
    for row in rows:
        distinct_context_ids.update(str(context_id) for context_id in row["distinct_context_ids"])
    backend_labels = sorted(set(str(row["retrieval_backend_label"]) for row in rows))
    dense_backends = sorted(set(str(row["dense_backend"]) for row in rows))
    vector_stores = sorted(set(str(row["vector_store"]) for row in rows))
    return {
        "record_count": len(rows),
        "recall_at_5": round(mean(float(row["recall_at_5"]) for row in rows), 6),
        "mrr": round(mean(float(row["mrr"]) for row in rows), 6),
        "avg_retrieval_latency_ms": round(
            mean(float(row["retrieval_latency_ms"]) for row in rows), 6
        ),
        "avg_context_tokens": round(mean(float(row["context_token_count"]) for row in rows), 6),
        "gold_evidence_included_count": sum(
            1 for row in rows if bool(row["gold_evidence_included"])
        ),
        "missing_gold_evidence_count": sum(int(row["missing_gold_evidence_count"]) for row in rows),
        "total_context_rows_selected": sum(int(row["context_rows_selected"]) for row in rows),
        "distinct_context_rows_used": len(distinct_context_ids),
        "retrieval_backend_label": ",".join(backend_labels),
        "dense_backend": ",".join(dense_backends),
        "vector_store": ",".join(vector_stores),
        "compression_ratio": round(mean(compression_values), 6) if compression_values else None,
        "token_reduction_pct": round(mean(token_reduction_pct_values), 6)
        if token_reduction_pct_values
        else None,
        "recall_loss": round(mean(recall_loss_values), 6) if recall_loss_values else None,
        "token_reduction": sum(int(row["token_reduction"]) for row in rows),
        "query_enrichment_used": any(bool(row.get("query_enrichment_used")) for row in rows),
        "reranking_used": any(bool(row.get("reranking_used")) for row in rows),
        "reranker_enabled": any(bool(row.get("reranker_enabled")) for row in rows),
        "source_hints_used": any(bool(row.get("source_hints_used")) for row in rows),
        "leakage_guard_applied": any(bool(row.get("leakage_guard_applied")) for row in rows),
        "blocked_direct_hint_count": sum(
            int(row.get("blocked_direct_hint_count") or 0) for row in rows
        ),
        "candidate_top_k_dense": max(int(row.get("candidate_top_k_dense") or 0) for row in rows),
        "candidate_top_k_lexical": max(
            int(row.get("candidate_top_k_lexical") or 0) for row in rows
        ),
        "final_top_k": max(int(row.get("final_top_k") or 0) for row in rows),
        "avg_candidates_before_dedupe": round(
            mean(float(row.get("candidates_before_dedupe") or 0) for row in rows),
            6,
        ),
        "avg_candidates_after_dedupe": round(
            mean(float(row.get("candidates_after_dedupe") or 0) for row in rows),
            6,
        ),
        "avg_expanded_query_count": round(
            mean(float(row.get("expanded_query_count") or 0) for row in rows),
            6,
        ),
        "gold_in_candidate_pool_count": sum(
            1 for row in rows if bool(row.get("gold_in_candidate_pool"))
        ),
        "candidate_recall_at_10": round(
            mean(float(row.get("candidate_recall_at_10") or 0.0) for row in rows),
            6,
        ),
        "candidate_recall_at_20": round(
            mean(float(row.get("candidate_recall_at_20") or 0.0) for row in rows),
            6,
        ),
        "candidate_recall_at_50": round(
            mean(float(row.get("candidate_recall_at_50") or 0.0) for row in rows),
            6,
        ),
        "candidate_recall_at_100": round(
            mean(float(row.get("candidate_recall_at_100") or 0.0) for row in rows),
            6,
        ),
        "candidate_recall_at_200": round(
            mean(float(row.get("candidate_recall_at_200") or 0.0) for row in rows),
            6,
        ),
        "candidate_mrr_at_100": round(
            mean(float(row.get("candidate_mrr_at_100") or 0.0) for row in rows),
            6,
        ),
        "candidate_diagnostic_max_k_available": max(
            int(row.get("candidate_diagnostic_max_k_available") or 0) for row in rows
        ),
        "candidate_recall_at_100_feasible": all(
            bool(row.get("candidate_recall_at_100_feasible")) for row in rows
        ),
        "candidate_recall_at_200_feasible": all(
            bool(row.get("candidate_recall_at_200_feasible")) for row in rows
        ),
        "pre_rerank_recall_at_5": round(
            mean(float(row.get("pre_rerank_recall_at_5") or 0.0) for row in rows),
            6,
        ),
        "reranker_rescued_gold_count": sum(
            1 for row in rows if bool(row.get("reranker_rescued_gold"))
        ),
        "gold_in_top50_but_not_top5_rate": round(
            mean(1.0 if row.get("gold_in_top50_but_not_top5") else 0.0 for row in rows),
            6,
        ),
        "gold_in_top100_but_not_top5_rate": round(
            mean(1.0 if row.get("gold_in_top100_but_not_top5") else 0.0 for row in rows),
            6,
        ),
        "gold_absent_from_top100_rate": round(
            mean(1.0 if row.get("gold_absent_from_top100") else 0.0 for row in rows),
            6,
        ),
        "reranker_backend": ",".join(
            sorted({str(row.get("reranker_backend") or "unavailable") for row in rows})
        ),
        "calibrated_reranker_enabled": any(
            bool(row.get("calibrated_reranker_enabled")) for row in rows
        ),
        "evidence_selector_strategy": ",".join(
            sorted({str(row.get("evidence_selector_strategy") or "unavailable") for row in rows})
        ),
    }


def build_evaluation_report(
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build retrieval evaluation report and summary rows."""

    summary_rows: list[dict[str, Any]] = []
    by_split: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        grouped[
            (
                str(row["split"]),
                str(row["ablation_mode"]),
                str(row["memory_mode"]),
                str(row["vertical"]),
            )
        ].append(row)

    for (split, ablation_mode, memory_mode, vertical), rows in sorted(grouped.items()):
        metrics = aggregate_eval_rows(rows)
        by_split.setdefault(split, {}).setdefault(ablation_mode, {}).setdefault(memory_mode, {})[
            vertical
        ] = metrics
        summary_rows.append(
            {
                "split": split,
                "ablation_mode": ablation_mode,
                "memory_mode": memory_mode,
                "vertical": vertical,
                "context_token_count": metrics["avg_context_tokens"],
                **metrics,
            }
        )

    overall_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        overall_grouped[
            (str(row["split"]), str(row["ablation_mode"]), str(row["memory_mode"]))
        ].append(row)
    split_names = sorted({str(row["split"]) for row in evaluation_rows})
    overall_by_split_ablation_mode = {
        split: {
            ablation: {
                mode: aggregate_eval_rows(rows)
                for (row_split, row_ablation, mode), rows in overall_grouped.items()
                if row_split == split and row_ablation == ablation
            }
            for ablation in sorted(
                {
                    row_ablation
                    for row_split, row_ablation, _mode in overall_grouped
                    if row_split == split
                }
            )
        }
        for split in split_names
    }
    overall_by_split_mode = {
        split: ablation_payload.get("prompt_plus_source_hints", {})
        for split, ablation_payload in overall_by_split_ablation_mode.items()
    }
    dense_statuses = sorted(
        {
            str(row["dense_backend"])
            for row in evaluation_rows
            if row["memory_mode"] == "mm1_dense_top5"
        }
    )
    hybrid_statuses = sorted(
        {
            str(row["dense_backend"])
            for row in evaluation_rows
            if row["memory_mode"] in {"mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}
        }
    )
    vector_stores = sorted({str(row["vector_store"]) for row in evaluation_rows})
    ablation_modes = sorted({str(row["ablation_mode"]) for row in evaluation_rows})
    query_enrichment_used = any(bool(row.get("query_enrichment_used")) for row in evaluation_rows)
    reranking_used = any(bool(row.get("reranking_used")) for row in evaluation_rows)

    return (
        {
            "generated_at_utc": utc_now(),
            "no_model_inference_triggered": True,
            "dense_retrieval_status": ",".join(dense_statuses) or "unavailable",
            "hybrid_dense_component_status": ",".join(hybrid_statuses) or "unavailable",
            "vector_stores": vector_stores,
            "ablation_modes": ablation_modes,
            "qdrant_used": "qdrant_vector" in dense_statuses or "qdrant_vector" in hybrid_statuses,
            "source_hint_modes_are_hint_assisted": ["prompt_plus_source_hints"],
            "query_enrichment_used": query_enrichment_used,
            "reranking_used": reranking_used,
            "candidate_expansion": {
                "candidate_top_k_dense_default": DEFAULT_CANDIDATE_TOP_K_DENSE,
                "candidate_top_k_lexical_default": DEFAULT_CANDIDATE_TOP_K_LEXICAL,
                "final_top_k_default": DEFAULT_FINAL_TOP_K,
                "separates_candidate_generation_reranking_and_final_selection": True,
            },
            "strict_modes_block_direct_source_hints": True,
            "by_split": by_split,
            "overall_by_split_ablation_mode": overall_by_split_ablation_mode,
            "overall_by_split_mode": overall_by_split_mode,
        },
        summary_rows,
    )


def corpus_match_ids_by_vertical(
    corpora_by_vertical: dict[str, list[ContextRecord]],
) -> dict[str, set[str]]:
    """Return all known match IDs per vertical."""

    from inference_bench.retrieval import context_match_ids

    return {
        vertical: {match_id for record in records for match_id in context_match_ids(record)}
        for vertical, records in corpora_by_vertical.items()
    }


def retrieval_failure_reasons(
    row: dict[str, Any],
    corpus_match_ids: dict[str, set[str]],
) -> list[str]:
    """Classify retrieval misses for diagnostics."""

    if float(row["recall_at_5"]) >= 1.0 or row["memory_mode"] == "mm0_no_context":
        return []
    vertical = str(row["vertical"])
    query_text = str(row.get("query_text") or "")
    query_lower = query_text.lower()
    gold_ids = [str(item) for item in row.get("gold_evidence_ids", [])]
    matched_ids = set(str(item) for item in row.get("matched_gold_evidence_ids", []))
    known_ids = corpus_match_ids.get(vertical, set())
    missing_ids = [evidence_id for evidence_id in gold_ids if evidence_id not in matched_ids]
    reasons: list[str] = []

    if any(evidence_id not in known_ids for evidence_id in missing_ids):
        reasons.append("missing_gold_mapping")
    if missing_ids and not reasons:
        reasons.append("poor_scoring")
    if row.get("gold_in_candidate_pool") and missing_ids:
        reasons.append("gold_in_top50_not_top5")
    if not row.get("gold_in_candidate_pool") and missing_ids:
        reasons.append("gold_not_in_candidate_pool")
    if str(row["memory_mode"]) == "mm1_dense_top5" and row.get("dense_backend") == "local_fallback":
        reasons.append("dense_fallback_limitation")
    if vertical == "finance":
        has_finance_identifier = any(token in query_lower for token in ("10-k", "10-q", "8-k"))
        has_finance_identifier = has_finance_identifier or any(
            token in query_text.upper()
            for token in ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD")
        )
        if not has_finance_identifier:
            reasons.append("bad_query_terms")
        if not any(
            token in query_lower
            for token in (
                "revenue",
                "sales",
                "income",
                "cash",
                "flow",
                "risk",
                "margin",
                "capex",
                "selected financial metric",
            )
        ):
            reasons.append("missed_finance_metric")
        if not any(re.fullmatch(r"20\d{2}", token) for token in tokenize(query_lower)):
            reasons.append("missed_period")
    if float(row.get("context_token_count") or 0) > 900:
        reasons.append("chunk_too_broad")
    if int(row.get("context_rows_selected") or 0) < 2 and missing_ids:
        reasons.append("chunk_too_narrow")
    if not reasons and missing_ids:
        reasons.append("evaluation_mismatch")
    return list(dict.fromkeys(reasons))


def build_retrieval_diagnostic_report(
    evaluation_rows: list[dict[str, Any]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build retrieval diagnostics from workload evaluation rows."""

    known_match_ids = corpus_match_ids_by_vertical(corpora_by_vertical)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    failure_examples: list[dict[str, Any]] = []
    finance_failure_examples_by_ablation: dict[str, list[dict[str, Any]]] = {
        "prompt_text_only": [],
        "prompt_plus_metadata": [],
    }
    reason_counts_by_vertical: dict[str, Counter[str]] = defaultdict(Counter)

    for row in evaluation_rows:
        split = str(row["split"])
        ablation_mode = str(row.get("ablation_mode") or "prompt_plus_source_hints")
        memory_mode = str(row["memory_mode"])
        vertical = str(row["vertical"])
        grouped[(split, ablation_mode, memory_mode, vertical)].append(row)
        reasons = retrieval_failure_reasons(row, known_match_ids)
        if reasons:
            reason_counts_by_vertical[vertical].update(reasons)
        if (
            split == "final_10000"
            and ablation_mode in {"prompt_text_only", "prompt_plus_metadata"}
            and memory_mode == "mm2_hybrid_top5"
            and vertical == "finance"
            and float(row["recall_at_5"]) < 1.0
            and len(finance_failure_examples_by_ablation[ablation_mode]) < 30
        ):
            finance_failure_examples_by_ablation[ablation_mode].append(
                {
                    "prompt_id": row["prompt_id"],
                    "ablation_mode": ablation_mode,
                    "recall_at_5": row["recall_at_5"],
                    "mrr": row["mrr"],
                    "gold_evidence_ids": row.get("gold_evidence_ids", []),
                    "matched_gold_evidence_ids": row.get("matched_gold_evidence_ids", []),
                    "retrieved_context_ids": row.get("retrieved_context_ids", []),
                    "candidate_context_ids": row.get("candidate_context_ids", [])[:50],
                    "gold_in_candidate_pool": row.get("gold_in_candidate_pool"),
                    "candidate_recall_at_10": row.get("candidate_recall_at_10"),
                    "candidate_recall_at_20": row.get("candidate_recall_at_20"),
                    "candidate_recall_at_50": row.get("candidate_recall_at_50"),
                    "candidate_recall_at_100": row.get("candidate_recall_at_100"),
                    "candidate_recall_at_200": row.get("candidate_recall_at_200"),
                    "pre_rerank_top_context_ids": row.get("pre_rerank_top_context_ids", []),
                    "pre_rerank_recall_at_5": row.get("pre_rerank_recall_at_5"),
                    "reranker_rescued_gold": row.get("reranker_rescued_gold"),
                    "failure_reasons": reasons,
                    "query_enrichment_used": row.get("query_enrichment_used"),
                    "reranking_used": row.get("reranking_used"),
                    "source_hints_used": row.get("source_hints_used"),
                    "expanded_query_count": row.get("expanded_query_count"),
                    "expansion_types": row.get("expansion_types", []),
                    "query_excerpt": str(row.get("query_text") or "")[:500],
                }
            )
        if (
            split == "final_10000"
            and memory_mode in {"mm1_dense_top5", "mm2_hybrid_top5"}
            and float(row["recall_at_5"]) < 1.0
            and len(failure_examples) < 40
        ):
            failure_examples.append(
                {
                    "prompt_id": row["prompt_id"],
                    "vertical": vertical,
                    "ablation_mode": ablation_mode,
                    "memory_mode": memory_mode,
                    "recall_at_5": row["recall_at_5"],
                    "failure_reasons": reasons,
                }
            )

    summary_rows: list[dict[str, Any]] = []
    by_split: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (split, ablation_mode, memory_mode, vertical), rows in sorted(grouped.items()):
        if rows:
            reason_counter: Counter[str] = Counter()
            for row in rows:
                reason_counter.update(retrieval_failure_reasons(row, known_match_ids))
            payload = {
                "record_count": len(rows),
                "recall_at_5": round(mean(float(row["recall_at_5"]) for row in rows), 6),
                "mrr": round(mean(float(row["mrr"]) for row in rows), 6),
                "failure_count": sum(1 for row in rows if float(row["recall_at_5"]) < 1.0),
                "candidate_recall_at_50": round(
                    mean(float(row.get("candidate_recall_at_50") or 0.0) for row in rows),
                    6,
                ),
                "candidate_recall_at_100": round(
                    mean(float(row.get("candidate_recall_at_100") or 0.0) for row in rows),
                    6,
                ),
                "candidate_recall_at_200": round(
                    mean(float(row.get("candidate_recall_at_200") or 0.0) for row in rows),
                    6,
                ),
                "candidate_diagnostic_max_k_available": max(
                    int(row.get("candidate_diagnostic_max_k_available") or 0) for row in rows
                ),
                "gold_in_top50_but_not_top5_rate": round(
                    mean(1.0 if row.get("gold_in_top50_but_not_top5") else 0.0 for row in rows),
                    6,
                ),
                "gold_in_top100_but_not_top5_rate": round(
                    mean(1.0 if row.get("gold_in_top100_but_not_top5") else 0.0 for row in rows),
                    6,
                ),
                "gold_absent_from_top100_rate": round(
                    mean(1.0 if row.get("gold_absent_from_top100") else 0.0 for row in rows),
                    6,
                ),
                "gold_in_candidate_pool_count": sum(
                    1 for row in rows if bool(row.get("gold_in_candidate_pool"))
                ),
                "reranker_rescued_gold_count": sum(
                    1 for row in rows if bool(row.get("reranker_rescued_gold"))
                ),
                "top_failure_reasons": dict(reason_counter.most_common(8)),
            }
        else:
            payload = {
                "record_count": 0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "failure_count": 0,
                "candidate_recall_at_50": 0.0,
                "candidate_recall_at_100": 0.0,
                "candidate_recall_at_200": 0.0,
                "candidate_diagnostic_max_k_available": 0,
                "gold_in_top50_but_not_top5_rate": 0.0,
                "gold_in_top100_but_not_top5_rate": 0.0,
                "gold_absent_from_top100_rate": 0.0,
                "gold_in_candidate_pool_count": 0,
                "reranker_rescued_gold_count": 0,
                "top_failure_reasons": {},
            }
        by_split.setdefault(split, {}).setdefault(ablation_mode, {}).setdefault(memory_mode, {})[
            vertical
        ] = payload
        summary_rows.append(
            {
                "split": split,
                "ablation_mode": ablation_mode,
                "memory_mode": memory_mode,
                "vertical": vertical,
                "record_count": payload["record_count"],
                "recall_at_5": payload["recall_at_5"],
                "mrr": payload["mrr"],
                "failure_count": payload["failure_count"],
                "candidate_recall_at_50": payload["candidate_recall_at_50"],
                "candidate_recall_at_100": payload["candidate_recall_at_100"],
                "candidate_recall_at_200": payload["candidate_recall_at_200"],
                "candidate_diagnostic_max_k_available": payload[
                    "candidate_diagnostic_max_k_available"
                ],
                "gold_in_top50_but_not_top5_rate": payload["gold_in_top50_but_not_top5_rate"],
                "gold_in_top100_but_not_top5_rate": payload["gold_in_top100_but_not_top5_rate"],
                "gold_absent_from_top100_rate": payload["gold_absent_from_top100_rate"],
                "gold_in_candidate_pool_count": payload["gold_in_candidate_pool_count"],
                "reranker_rescued_gold_count": payload["reranker_rescued_gold_count"],
                "query_enrichment_used": any(
                    bool(row.get("query_enrichment_used")) for row in rows
                ),
                "reranking_used": any(bool(row.get("reranking_used")) for row in rows),
                "source_hints_used": any(bool(row.get("source_hints_used")) for row in rows),
                "top_failure_reasons": json.dumps(payload["top_failure_reasons"], sort_keys=True),
            }
        )

    dense_statuses = sorted(
        {
            str(row.get("dense_backend") or row.get("retrieval_backend_label") or "unavailable")
            for row in evaluation_rows
        }
    )
    report = {
        "generated_at_utc": utc_now(),
        "no_model_inference_triggered": True,
        "diagnostic_scope": "retrieval_only_no_gpu_no_api",
        "match_logic": (
            "Evaluation matches gold evidence IDs against context IDs, source IDs, parent IDs, "
            "chunk IDs, and flattened context metadata values using exact and normalized forms."
        ),
        "query_logic": (
            "Retrieval query text is built according to ablation mode. prompt_text_only uses "
            "only user-visible prompt text after generated evidence IDs are scrubbed; "
            "prompt_plus_metadata adds realistic prompt metadata but still blocks direct source "
            "identifiers; prompt_plus_source_hints is explicitly hint-assisted. Gold/eval rows "
            "are not used for retrieval."
        ),
        "strict_leakage_guard": {
            "prompt_text_only_blocks_direct_evidence_ids": True,
            "prompt_plus_metadata_blocks_direct_source_ids": True,
            "source_hint_mode_is_assisted_upper_bound": True,
        },
        "query_enrichment": {
            "enabled": True,
            "allowed_sources": [
                "visible prompt text",
                "realistic prompt metadata for prompt_plus_metadata",
            ],
            "blocked_sources": [
                "gold evidence IDs",
                "direct source IDs",
                "direct parent IDs",
                "answer-side evidence hints",
            ],
        },
        "reranking": {
            "enabled_for_hybrid": True,
            "candidate_top_k_dense": DEFAULT_CANDIDATE_TOP_K_DENSE,
            "candidate_top_k_lexical": DEFAULT_CANDIDATE_TOP_K_LEXICAL,
            "final_top_k": DEFAULT_FINAL_TOP_K,
            "signals": [
                "lexical overlap",
                "metadata overlap",
                "finance ticker/company/form/metric/period matches",
                "section/title matches",
                "BM25 score",
                "Qdrant score",
            ],
        },
        "dense_status": ",".join(dense_statuses) or "unavailable",
        "ablation_modes": sorted(
            {str(row.get("ablation_mode") or "prompt_plus_source_hints") for row in evaluation_rows}
        ),
        "by_split": by_split,
        "top_failure_reasons_by_vertical": {
            vertical: dict(counter.most_common(8))
            for vertical, counter in sorted(reason_counts_by_vertical.items())
        },
        "finance_specific": {
            "failure_examples_by_ablation": finance_failure_examples_by_ablation,
            "uses_prompt_fields": [
                "ticker",
                "company",
                "filing_form",
            ],
            "strict_modes_block_direct_source_hints": True,
            "failure_reason_labels": [
                "missing_gold_mapping",
                "bad_query_terms",
                "missing_metadata",
                "poor_scoring",
                "chunk_too_broad",
                "chunk_too_narrow",
                "dense_fallback_limitation",
                "evaluation_mismatch",
                "gold_in_top50_not_top5",
                "gold_not_in_candidate_pool",
                "missed_finance_metric",
                "missed_period",
            ],
        },
        "sample_failure_examples": failure_examples,
    }
    return report, summary_rows


def build_compression_diagnostic_report(
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build compression diagnostics for mm3."""

    compression_rows = [
        row for row in evaluation_rows if row["memory_mode"] == "mm3_compressed_hybrid_top5"
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compression_rows:
        grouped[
            (
                str(row["split"]),
                str(row.get("ablation_mode") or "prompt_plus_source_hints"),
                str(row["vertical"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    by_split: dict[str, dict[str, Any]] = defaultdict(dict)
    for (split, ablation_mode, vertical), rows in sorted(grouped.items()):
        original_tokens = sum(
            int(row["context_token_count"]) + int(row["token_reduction"]) for row in rows
        )
        compressed_tokens = sum(int(row["context_token_count"]) for row in rows)
        token_reduction = sum(int(row["token_reduction"]) for row in rows)
        token_reduction_pct = (
            round(token_reduction / original_tokens, 6) if original_tokens else 0.0
        )
        recall_loss_values = [
            float(row["recall_loss"]) for row in rows if row.get("recall_loss") is not None
        ]
        recall_loss = round(mean(recall_loss_values), 6) if recall_loss_values else 0.0
        gold_retained_count = sum(
            1 for row in rows if row.get("gold_evidence_retained_after_compression") is True
        )
        payload = {
            "record_count": len(rows),
            "original_context_tokens": original_tokens,
            "compressed_context_tokens": compressed_tokens,
            "token_reduction": token_reduction,
            "token_reduction_pct": token_reduction_pct,
            "recall_before_compression": round(
                mean(float(row["recall_before_compression"]) for row in rows), 6
            ),
            "recall_after_compression": round(
                mean(float(row["recall_after_compression"]) for row in rows), 6
            ),
            "recall_loss": recall_loss,
            "gold_evidence_retained_count": gold_retained_count,
        }
        by_split.setdefault(split, {}).setdefault(ablation_mode, {})[vertical] = payload
        summary_rows.append(
            {
                "split": split,
                "ablation_mode": ablation_mode,
                "vertical": vertical,
                **payload,
            }
        )

    overall_by_split_ablation_mode: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in compression_rows}):
        overall_by_split_ablation_mode[split] = {}
        for ablation_mode in sorted(
            {
                str(row.get("ablation_mode") or "prompt_plus_source_hints")
                for row in compression_rows
                if row["split"] == split
            }
        ):
            rows = [
                row
                for row in compression_rows
                if row["split"] == split
                and str(row.get("ablation_mode") or "prompt_plus_source_hints") == ablation_mode
            ]
            original_tokens = sum(
                int(row["context_token_count"]) + int(row["token_reduction"]) for row in rows
            )
            token_reduction = sum(int(row["token_reduction"]) for row in rows)
            overall_by_split_ablation_mode[split][ablation_mode] = {
                "record_count": len(rows),
                "token_reduction_pct": round(token_reduction / original_tokens, 6)
                if original_tokens
                else 0.0,
                "recall_loss": round(
                    mean(
                        float(row["recall_loss"]) for row in rows if row["recall_loss"] is not None
                    ),
                    6,
                )
                if rows
                else 0.0,
            }
    overall_by_split = {
        split: ablation_payload.get("prompt_plus_source_hints", {})
        for split, ablation_payload in overall_by_split_ablation_mode.items()
    }

    report = {
        "generated_at_utc": utc_now(),
        "no_model_inference_triggered": True,
        "compression_type": "deterministic_score_dedupe_budget_and_extractive_truncation",
        "ablation_modes": sorted(
            {
                str(row.get("ablation_mode") or "prompt_plus_source_hints")
                for row in compression_rows
            }
        ),
        "by_split": by_split,
        "overall_by_split_ablation_mode": overall_by_split_ablation_mode,
        "overall_by_split": overall_by_split,
        "safety_notes": [
            "Compression preserves context IDs, provenance, and metadata.",
            "Compression never removes all context when retrieval found evidence.",
            "Recall before and after compression is measured for every mm3 workload row.",
        ],
    }
    return report, summary_rows


def write_workload_reports(
    *,
    output_report_root: Path,
    workload_build_report: dict[str, Any],
    workload_build_summary_rows: list[dict[str, Any]],
    retrieval_report: dict[str, Any],
    retrieval_summary_rows: list[dict[str, Any]],
    retrieval_diagnostic_report: dict[str, Any],
    retrieval_diagnostic_rows: list[dict[str, Any]],
    compression_diagnostic_report: dict[str, Any],
    compression_diagnostic_rows: list[dict[str, Any]],
    run_safety_report: dict[str, Any],
    run_safety_rows: list[dict[str, Any]],
) -> None:
    """Write workload and retrieval report files."""

    write_json(output_report_root / "workload_build_report.json", workload_build_report)
    write_json(output_report_root / "retrieval_evaluation_report.json", retrieval_report)
    write_json(output_report_root / "retrieval_diagnostic_report.json", retrieval_diagnostic_report)
    write_json(
        output_report_root / "compression_diagnostic_report.json",
        compression_diagnostic_report,
    )
    write_json(output_report_root / "run_safety_audit_report.json", run_safety_report)
    write_csv(
        output_report_root / "workload_build_summary.csv",
        workload_build_summary_rows,
        [
            "split",
            "ablation_mode",
            "memory_mode",
            "record_count",
            "airline",
            "healthcare_admin",
            "retail",
            "finance",
            "research_ai",
            "output_path",
            "validated",
        ],
    )
    write_csv(
        output_report_root / "retrieval_evaluation_summary.csv",
        retrieval_summary_rows,
        [
            "split",
            "ablation_mode",
            "memory_mode",
            "vertical",
            "record_count",
            "recall_at_5",
            "mrr",
            "dense_backend",
            "vector_store",
            "context_token_count",
            "avg_retrieval_latency_ms",
            "avg_context_tokens",
            "gold_evidence_included_count",
            "missing_gold_evidence_count",
            "total_context_rows_selected",
            "distinct_context_rows_used",
            "retrieval_backend_label",
            "source_hints_used",
            "query_enrichment_used",
            "reranking_used",
            "reranker_enabled",
            "leakage_guard_applied",
            "blocked_direct_hint_count",
            "candidate_top_k_dense",
            "candidate_top_k_lexical",
            "final_top_k",
            "avg_candidates_before_dedupe",
            "avg_candidates_after_dedupe",
            "avg_expanded_query_count",
            "gold_in_candidate_pool_count",
            "candidate_recall_at_10",
            "candidate_recall_at_20",
            "candidate_recall_at_50",
            "candidate_recall_at_100",
            "candidate_recall_at_200",
            "candidate_mrr_at_100",
            "candidate_diagnostic_max_k_available",
            "candidate_recall_at_100_feasible",
            "candidate_recall_at_200_feasible",
            "pre_rerank_recall_at_5",
            "reranker_rescued_gold_count",
            "gold_in_top50_but_not_top5_rate",
            "gold_in_top100_but_not_top5_rate",
            "gold_absent_from_top100_rate",
            "reranker_backend",
            "calibrated_reranker_enabled",
            "evidence_selector_strategy",
            "compression_ratio",
            "token_reduction_pct",
            "recall_loss",
            "token_reduction",
        ],
    )
    write_csv(
        output_report_root / "retrieval_diagnostic_summary.csv",
        retrieval_diagnostic_rows,
        [
            "split",
            "ablation_mode",
            "memory_mode",
            "vertical",
            "record_count",
            "recall_at_5",
            "mrr",
            "failure_count",
            "candidate_recall_at_50",
            "candidate_recall_at_100",
            "candidate_recall_at_200",
            "candidate_diagnostic_max_k_available",
            "gold_in_top50_but_not_top5_rate",
            "gold_in_top100_but_not_top5_rate",
            "gold_absent_from_top100_rate",
            "gold_in_candidate_pool_count",
            "reranker_rescued_gold_count",
            "query_enrichment_used",
            "reranking_used",
            "source_hints_used",
            "top_failure_reasons",
        ],
    )
    write_csv(
        output_report_root / "compression_diagnostic_summary.csv",
        compression_diagnostic_rows,
        [
            "split",
            "ablation_mode",
            "vertical",
            "record_count",
            "original_context_tokens",
            "compressed_context_tokens",
            "token_reduction",
            "token_reduction_pct",
            "recall_before_compression",
            "recall_after_compression",
            "recall_loss",
            "gold_evidence_retained_count",
        ],
    )
    write_csv(
        output_report_root / "run_safety_audit_summary.csv",
        run_safety_rows,
        [
            "area",
            "artifact",
            "current_capability",
            "reusable",
            "phase4_gap",
            "phase5_gap",
            "priority",
        ],
    )


def build_memory_mode_workloads(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    memory_modes: list[str],
    dense_backend: str = "local_fallback",
    ablation_modes: list[str] | None = None,
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = False,
    split_plan: SplitPlan | None = None,
) -> WorkloadBuildResult:
    """Build memory-mode workload JSONL files and reports."""

    active_split_plan = split_plan or DEFAULT_SPLIT_PLAN
    configured_modes = load_memory_modes_config()
    for memory_mode in memory_modes:
        if memory_mode not in configured_modes:
            msg = f"Unknown memory mode '{memory_mode}'"
            raise ValueError(msg)
        if memory_mode not in SUPPORTED_MEMORY_MODES:
            msg = f"Memory mode '{memory_mode}' is not implemented in Phase 3 Block 3"
            raise ValueError(msg)
    if dense_backend not in SUPPORTED_DENSE_BACKENDS:
        msg = f"Unknown dense backend '{dense_backend}'"
        raise ValueError(msg)
    active_ablation_modes = ablation_modes or ["prompt_plus_source_hints"]
    for ablation_mode in active_ablation_modes:
        validate_ablation_mode(ablation_mode)

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    retrievers = build_retrievers(
        corpora_by_vertical,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    qdrant_query_embedding_warmup = {}
    if dense_backend == "qdrant_vector" and any(
        mode in {"mm1_dense_top5", "mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}
        for mode in memory_modes
    ):
        qdrant_query_embedding_warmup = warm_qdrant_query_embeddings(
            retrievers=retrievers,
            prompts_by_vertical=prompts_by_vertical,
            splits=splits,
            ablation_modes=active_ablation_modes,
            split_plan=active_split_plan,
        )
    output_path = Path(output_root)
    report_root = Path(context_root)
    workload_report: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "context_root": str(context_root),
        "output_root": str(output_root),
        "memory_modes": memory_modes,
        "dense_backend_requested": dense_backend,
        "vector_store": vector_store_key if dense_backend == "qdrant_vector" else "none",
        "ablation_modes": active_ablation_modes,
        "splits": splits,
        "candidate_expansion": {
            "candidate_top_k_dense": DEFAULT_CANDIDATE_TOP_K_DENSE,
            "candidate_top_k_lexical": DEFAULT_CANDIDATE_TOP_K_LEXICAL,
            "final_top_k": DEFAULT_FINAL_TOP_K,
            "candidate_generation_then_reranking": True,
        },
        "qdrant_query_embedding_warmup": qdrant_query_embedding_warmup,
        "no_model_inference_triggered": True,
        "all_workload_records_validated": True,
        "by_split": {},
    }
    workload_summary_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    retrieval_cache: dict[tuple[str, str, str, tuple[str, ...], int], TimedRetrieval] = {}
    use_legacy_output_paths = active_ablation_modes == ["prompt_plus_source_hints"]

    try:
        for split in splits:
            selected_prompts = select_prompts_for_split(
                prompts_by_vertical, split, active_split_plan
            )
            split_payload: dict[str, Any] = {}
            for ablation_mode in active_ablation_modes:
                ablation_payload: dict[str, Any] = {}
                for memory_mode in memory_modes:
                    eval_rows_for_file: list[dict[str, Any]] = []
                    by_vertical_counts: dict[str, int] = {vertical: 0 for vertical in VERTICALS}
                    if use_legacy_output_paths:
                        output_file = output_path / split / f"{memory_mode}.jsonl"
                    else:
                        output_file = output_path / split / ablation_mode / f"{memory_mode}.jsonl"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    record_count = 0

                    with output_file.open("w", encoding="utf-8") as file:
                        for prompt in selected_prompts:
                            vertical = str(prompt.get("vertical"))
                            prompt_id = str(prompt.get("prompt_id"))
                            gold_record = gold_by_vertical.get(vertical, {}).get(prompt_id)
                            workload_record, eval_row = build_one_workload_record(
                                prompt=prompt,
                                gold_record=gold_record,
                                dataset_split=split,
                                memory_mode=memory_mode,
                                ablation_mode=ablation_mode,
                                retrievers=retrievers,
                                mode_configs=configured_modes,
                                retrieval_cache=retrieval_cache,
                            )
                            write_workload_jsonl_line(file, workload_record)
                            eval_rows_for_file.append(eval_row)
                            by_vertical_counts[vertical] += 1
                            record_count += 1

                    evaluation_rows.extend(eval_rows_for_file)
                    ablation_payload[memory_mode] = {
                        "output_path": str(output_file),
                        "record_count": record_count,
                        "by_vertical": by_vertical_counts,
                        "validated": True,
                    }
                    workload_summary_rows.append(
                        {
                            "split": split,
                            "ablation_mode": ablation_mode,
                            "memory_mode": memory_mode,
                            "record_count": record_count,
                            **by_vertical_counts,
                            "output_path": str(output_file),
                            "validated": True,
                        }
                    )
                split_payload[ablation_mode] = ablation_payload
            workload_report["by_split"][split] = split_payload
    finally:
        close_retrievers(retrievers)

    retrieval_report, retrieval_summary_rows = build_evaluation_report(evaluation_rows)
    retrieval_diagnostic_report, retrieval_diagnostic_rows = build_retrieval_diagnostic_report(
        evaluation_rows,
        corpora_by_vertical,
    )
    compression_diagnostic_report, compression_diagnostic_rows = (
        build_compression_diagnostic_report(evaluation_rows)
    )
    quality_gate_report, quality_gate_rows = build_retrieval_quality_gate_report(
        retrieval_summary_rows,
        compression_diagnostic_rows,
    )
    gold_evidence_audit_report, gold_evidence_audit_rows = build_gold_evidence_audit_report(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
        evaluation_rows=evaluation_rows,
    )
    reranker_calibration_report, reranker_calibration_rows = build_reranker_calibration_report(
        evaluation_rows
    )
    evidence_selection_report, evidence_selection_rows = build_evidence_selection_report(
        evaluation_rows
    )
    workload_report["quality_gate_status"] = quality_gate_report["quality_gate_status"]
    workload_report["quality_gate_passed"] = quality_gate_report["passed"]
    run_safety_report, run_safety_rows = build_run_safety_audit()
    write_workload_reports(
        output_report_root=report_root,
        workload_build_report=workload_report,
        workload_build_summary_rows=workload_summary_rows,
        retrieval_report=retrieval_report,
        retrieval_summary_rows=retrieval_summary_rows,
        retrieval_diagnostic_report=retrieval_diagnostic_report,
        retrieval_diagnostic_rows=retrieval_diagnostic_rows,
        compression_diagnostic_report=compression_diagnostic_report,
        compression_diagnostic_rows=compression_diagnostic_rows,
        run_safety_report=run_safety_report,
        run_safety_rows=run_safety_rows,
    )
    write_json(report_root / "retrieval_quality_gate_report.json", quality_gate_report)
    write_csv(
        report_root / "retrieval_quality_gate_summary.csv",
        quality_gate_rows,
        QUALITY_GATE_SUMMARY_FIELDS,
    )
    write_json(report_root / "gold_evidence_audit_report.json", gold_evidence_audit_report)
    write_csv(
        report_root / "gold_evidence_audit_summary.csv",
        gold_evidence_audit_rows,
        GOLD_EVIDENCE_AUDIT_SUMMARY_FIELDS,
    )
    write_json(report_root / "reranker_calibration_report.json", reranker_calibration_report)
    write_csv(
        report_root / "reranker_calibration_summary.csv",
        reranker_calibration_rows,
        RERANKER_CALIBRATION_SUMMARY_FIELDS,
    )
    write_json(report_root / "evidence_selection_report.json", evidence_selection_report)
    write_csv(
        report_root / "evidence_selection_summary.csv",
        evidence_selection_rows,
        EVIDENCE_SELECTION_SUMMARY_FIELDS,
    )
    return WorkloadBuildResult(
        workload_build_report=workload_report,
        workload_build_summary_rows=workload_summary_rows,
        retrieval_evaluation_report=retrieval_report,
        retrieval_evaluation_summary_rows=retrieval_summary_rows,
    )
