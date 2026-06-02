from inference_bench.context_schema import ContextRecord
from inference_bench.memory_workloads import (
    build_evaluation_report,
    prompt_query_text,
)
from inference_bench.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalFallbackDenseRetriever,
    compress_retrieval_results,
    enrich_query_text,
    rerank_boost_score,
)
from inference_bench.vector_store import vector_text


def context_record(
    *,
    context_id: str,
    text: str,
    title: str = "AAPL 10-K Revenue",
    metadata: dict[str, object] | None = None,
    vertical: str = "finance",
) -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical=vertical,
        source_id="finance_sec_edgar_xbrl" if vertical == "finance" else f"{vertical}_source",
        parent_id="finance_doc_AAPL_10K_2024" if vertical == "finance" else context_id,
        chunk_id=context_id,
        chunk_strategy="finance_filing_section_sentence_window"
        if vertical == "finance"
        else "paper_section",
        source_type="sec_filing_section" if vertical == "finance" else "paper_section",
        title=title,
        text=text,
        metadata=metadata
        or {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "filing_date": "2024-11-01",
            "report_date": "2024-09-28",
            "concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
            "section_type": "management_discussion_and_analysis",
            "original_doc_id": "finance_kb_sec_AAPL_10K_revenue",
        },
        token_estimate=len(text.split()),
        provenance="finance_sec_edgar_xbrl",
        is_gold_linked=True,
    )


def test_prompt_text_only_does_not_use_direct_source_or_gold_ids() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "question": (
            "Answer using records "
            "finance_kb_sec_MSFT_10K_000095017024087843_management_discussion_and_analysis."
        ),
        "issue": "Do not rely on required_doc_ids.",
        "required_doc_ids": ["finance_kb_sec_MSFT_10K_000095017024087843"],
        "required_evidence_ids": ["finance_kb_sec_MSFT_10K_000095017024087843"],
        "vertical": "finance",
    }

    query = prompt_query_text(prompt, "prompt_text_only")

    assert query.uses_source_hints is False
    assert query.leakage_guard_applied is True
    assert query.blocked_direct_hint_count >= 1
    assert "finance_kb_sec" not in query.query_text
    assert "required_doc_ids" not in query.query_text


def test_prompt_plus_metadata_does_not_use_source_ids_but_keeps_realistic_metadata() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "question": "What did Apple report about revenue?",
        "vertical": "finance",
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "filing_form": "10-K",
        "required_doc_ids": ["finance_kb_sec_AAPL_10K_revenue"],
        "metadata": {"source_titles": ["Hidden source title"]},
    }

    query = prompt_query_text(prompt, "prompt_plus_metadata")

    assert "AAPL" in query.query_text
    assert "10-K" in query.query_text
    assert "finance_kb_sec" not in query.query_text
    assert "Hidden source title" not in query.query_text
    assert query.uses_metadata is True
    assert query.uses_source_hints is False


def test_query_enrichment_expands_finance_metric_synonyms() -> None:
    enriched = enrich_query_text(
        "Compare R&D, capex, cash flow, and net income for Apple in fiscal year 2024.",
        vertical="finance",
    )

    assert "research and development" in enriched.query_text
    assert "capital expenditure" in enriched.query_text
    assert "operating cash flow" in enriched.query_text
    assert "earnings" in enriched.query_text
    assert "10-k" in enriched.query_text


def test_ticker_and_company_extraction_from_visible_prompt_text() -> None:
    enriched = enrich_query_text(
        "Using only visible prompt text, summarize Apple Inc. (AAPL) revenue.",
        vertical="finance",
    )

    assert "aapl" in enriched.query_text.lower()
    assert "apple inc" in enriched.query_text.lower()
    assert "net sales" in enriched.query_text.lower()


def test_qdrant_indexed_text_includes_title_text_and_selected_metadata() -> None:
    record = context_record(
        context_id="target",
        text="Revenue increased for the annual period.",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "section_type": "management_discussion_and_analysis",
            "concept": "Revenue",
        },
    )

    indexed_text = vector_text(record)

    assert "title: AAPL 10-K Revenue" in indexed_text
    assert "text: Revenue increased" in indexed_text
    assert "AAPL" in indexed_text
    assert "Apple Inc." in indexed_text
    assert "management_discussion_and_analysis" in indexed_text


def test_finance_reranking_changes_candidate_order_when_metadata_matches() -> None:
    target = context_record(
        context_id="target",
        text="Revenue evidence for the annual report.",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "fiscal_year": "2024",
            "concept": "Revenue",
            "section_type": "management_discussion_and_analysis",
        },
    )
    distractor = context_record(
        context_id="distractor",
        text="Revenue revenue revenue evidence for the annual report.",
        metadata={
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "form": "10-K",
            "fiscal_year": "2024",
            "concept": "Revenue",
            "section_type": "management_discussion_and_analysis",
        },
    )
    query = "Apple Inc. AAPL 10-K 2024 revenue"

    assert rerank_boost_score(query, target) > rerank_boost_score(query, distractor)
    result = HybridRetriever(
        BM25Retriever([distractor, target]),
        LocalFallbackDenseRetriever([distractor, target]),
    ).retrieve(query, top_k=1)

    assert result.results[0].context_record.context_id == "target"
    assert result.results[0].component_scores["rerank_boost"] > 0


def test_retrieval_report_includes_ablation_enrichment_and_reranking_fields() -> None:
    report, summary_rows = build_evaluation_report(
        [
            {
                "split": "final_10000",
                "ablation_mode": "prompt_text_only",
                "memory_mode": "mm2_hybrid_top5",
                "vertical": "finance",
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "retrieval_latency_ms": 1.0,
                "context_token_count": 10,
                "context_rows_selected": 1,
                "distinct_context_ids": ["target"],
                "retrieval_backend_label": "qdrant_vector",
                "dense_backend": "qdrant_vector",
                "vector_store": "qdrant_local",
                "source_hints_used": False,
                "metadata_used": False,
                "gold_ids_used_in_query": False,
                "query_enrichment_used": True,
                "leakage_guard_applied": True,
                "blocked_direct_hint_count": 0,
                "reranking_used": True,
                "compression_ratio": None,
                "token_reduction_pct": None,
                "recall_loss": None,
                "token_reduction": 0,
                "gold_evidence_included": True,
                "missing_gold_evidence_count": 0,
            }
        ]
    )

    assert report["query_enrichment_used"] is True
    assert report["reranking_used"] is True
    assert report["strict_modes_block_direct_source_hints"] is True
    assert report["no_model_inference_triggered"] is True
    assert summary_rows[0]["query_enrichment_used"] is True
    assert summary_rows[0]["reranking_used"] is True


def test_compression_still_reduces_tokens_after_retrieval_changes() -> None:
    record = context_record(
        context_id="target",
        text=" ".join(
            [
                "Revenue evidence for the annual report gives the deterministic compressor room"
                " to reduce tokens while preserving provenance metadata."
            ]
            * 8
        ),
    )
    retrieval = HybridRetriever(
        BM25Retriever([record]),
        LocalFallbackDenseRetriever([record]),
    ).retrieve("AAPL 10-K revenue", top_k=1)

    compressed = compress_retrieval_results(retrieval.results, max_context_tokens=4096)

    assert compressed.results
    assert compressed.compressed_token_count < compressed.original_token_count
