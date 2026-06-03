from typing import Any

from inference_bench.context_schema import ContextRecord
from inference_bench.memory_workloads import (
    build_retrieval_diagnostic_report,
    prompt_query_text,
    retrieve_for_mode,
)
from inference_bench.retrieval import (
    DEFAULT_CANDIDATE_TOP_K_DENSE,
    DEFAULT_CANDIDATE_TOP_K_LEXICAL,
    DEFAULT_FINAL_TOP_K,
    BM25Retriever,
    CompanyTickerResolver,
    HybridRetriever,
    LocalFallbackDenseRetriever,
    build_xbrl_concept_map,
    compress_retrieval_results,
    extract_period_terms,
    finance_metric_expansion_terms,
)


def context_record(
    *,
    context_id: str,
    text: str,
    ticker: str = "AAPL",
    company_name: str = "Apple Inc.",
    form: str = "10-K",
    concept: str | None = "RevenueFromContractWithCustomerExcludingAssessedTax",
    section_type: str = "management_discussion_and_analysis",
) -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical="finance",
        source_id="finance_sec_edgar_xbrl",
        parent_id=f"finance_doc_{ticker}_{form}_2024",
        chunk_id=context_id,
        chunk_strategy="finance_filing_section_sentence_window",
        source_type="sec_filing_section",
        title=f"{ticker} {form} {section_type}",
        text=text,
        metadata={
            "ticker": ticker,
            "company_name": company_name,
            "form": form,
            "filing_date": "2024-11-01",
            "report_date": "2024-09-28",
            "fiscal_year": "2024",
            "concept": concept,
            "concepts": [concept] if concept else [],
            "section_type": section_type,
            "original_doc_id": f"finance_kb_sec_{ticker}_{form}_target",
        },
        token_estimate=len(text.split()),
        provenance="finance_sec_edgar_xbrl",
        is_gold_linked=True,
    )


def test_candidate_generation_returns_more_than_final_top_k() -> None:
    target = context_record(
        context_id="target",
        text="Apple revenue net sales management discussion evidence for fiscal year 2024.",
    )
    records = [
        target,
        *[
            context_record(
                context_id=f"distractor_{index}",
                text=f"Apple revenue decoy {index} fiscal year 2024.",
            )
            for index in range(12)
        ],
    ]
    retrieval = HybridRetriever(
        BM25Retriever(records),
        LocalFallbackDenseRetriever(records),
    ).retrieve(
        "Apple AAPL revenue fiscal year 2024",
        top_k=DEFAULT_FINAL_TOP_K,
        expanded_queries=("Apple revenue", "AAPL net sales 2024"),
        candidate_top_k_dense=DEFAULT_CANDIDATE_TOP_K_DENSE,
        candidate_top_k_lexical=DEFAULT_CANDIDATE_TOP_K_LEXICAL,
    )

    assert retrieval.diagnostics["candidates_before_dedupe"] > DEFAULT_FINAL_TOP_K
    assert retrieval.diagnostics["candidates_after_dedupe"] > DEFAULT_FINAL_TOP_K
    assert len(retrieval.results) == DEFAULT_FINAL_TOP_K


def test_prompt_text_only_and_metadata_do_not_use_direct_evidence_ids() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "question": (
            "Use records finance_kb_sec_AAPL_10K_000032019324000123_"
            "management_discussion_and_analysis for Apple revenue."
        ),
        "company": "Apple Inc. (AAPL) 10-K",
        "ticker": "AAPL",
        "filing_form": "10-K",
        "required_doc_ids": ["finance_kb_sec_AAPL_10K_000032019324000123"],
        "vertical": "finance",
    }

    text_only = prompt_query_text(prompt, "prompt_text_only")
    metadata = prompt_query_text(prompt, "prompt_plus_metadata")

    assert "finance_kb_sec" not in text_only.query_text
    assert "finance_kb_sec" not in metadata.query_text
    assert text_only.uses_source_hints is False
    assert metadata.uses_source_hints is False
    assert metadata.uses_metadata is True


def test_source_hint_features_only_enabled_for_source_hint_mode() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "question": "What does Apple report?",
        "required_doc_ids": ["finance_kb_sec_AAPL_10K_target"],
        "vertical": "finance",
    }

    strict = prompt_query_text(prompt, "prompt_plus_metadata")
    assisted = prompt_query_text(prompt, "prompt_plus_source_hints")

    assert strict.uses_source_hints is False
    assert assisted.uses_source_hints is True
    assert "finance_kb_sec_AAPL_10K_target" in assisted.query_text
    assert "finance_kb_sec_AAPL_10K_target" not in strict.query_text


def test_company_ticker_resolver_uses_corpus_metadata() -> None:
    resolver = CompanyTickerResolver.from_records(
        [
            context_record(
                context_id="target",
                text="Apple revenue evidence.",
                ticker="AAPL",
                company_name="Apple Inc.",
            )
        ]
    )

    terms = resolver.resolve_terms("Summarize Apple revenue.")

    assert "aapl" in terms
    assert "apple" in terms
    assert "msft" not in terms


def test_finance_metric_synonym_mapper_and_period_extraction_work() -> None:
    metric_terms = finance_metric_expansion_terms(
        "Compare capex, R&D, cash flow, cloud revenue, and net income in FY 2024 Q3."
    )
    period_terms = extract_period_terms("Latest quarter Q3 fiscal year 2024 annual filing.")

    assert "capital expenditure" in metric_terms
    assert "research and development" in metric_terms
    assert "operating cash flow" in metric_terms
    assert "net sales" in metric_terms
    assert "q3" in period_terms
    assert "2024" in period_terms
    assert "fiscal year" in period_terms


def test_xbrl_concept_map_uses_only_corpus_concepts() -> None:
    records = [
        context_record(
            context_id="revenue",
            text="Revenue fact.",
            concept="RevenueFromContractWithCustomerExcludingAssessedTax",
        ),
        context_record(
            context_id="rnd",
            text="Research expense fact.",
            concept="ResearchAndDevelopmentExpense",
        ),
    ]

    concept_map = build_xbrl_concept_map(records)
    flattened = {concept for concepts in concept_map.values() for concept in concepts}

    assert "RevenueFromContractWithCustomerExcludingAssessedTax" in flattened
    assert "ResearchAndDevelopmentExpense" in flattened
    assert "FabricatedConcept" not in flattened


def test_reranker_changes_ordering_when_finance_features_match() -> None:
    target = context_record(
        context_id="target",
        text="Apple revenue net sales evidence for fiscal year 2024.",
        ticker="AAPL",
        company_name="Apple Inc.",
        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
    )
    distractor = context_record(
        context_id="distractor",
        text="Revenue revenue revenue revenue evidence for fiscal year 2024.",
        ticker="MSFT",
        company_name="Microsoft Corporation",
        concept="RevenueFromContractWithCustomerExcludingAssessedTax",
    )

    result = HybridRetriever(
        BM25Retriever([distractor, target]),
        LocalFallbackDenseRetriever([distractor, target]),
    ).retrieve("Apple AAPL revenue 2024", top_k=1)

    assert result.results[0].context_record.context_id == "target"
    assert result.results[0].component_scores["company_ticker_match"] > 0


def test_diagnostic_report_includes_gold_in_top_50() -> None:
    row = {
        "split": "final_10000",
        "ablation_mode": "prompt_text_only",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "finance",
        "prompt_id": "finance_prompt_001",
        "recall_at_5": 0.0,
        "mrr": 0.0,
        "gold_evidence_ids": ["finance_kb_sec_AAPL_10K_target"],
        "matched_gold_evidence_ids": [],
        "retrieved_context_ids": ["distractor"],
        "candidate_context_ids": ["target", "distractor"],
        "gold_in_candidate_pool": True,
        "candidate_recall_at_50": 1.0,
        "pre_rerank_top_context_ids": ["distractor"],
        "pre_rerank_recall_at_5": 0.0,
        "reranker_rescued_gold": False,
        "query_text": "Apple revenue",
        "query_enrichment_used": True,
        "reranking_used": True,
        "source_hints_used": False,
        "expanded_query_count": 2,
        "expansion_types": ["normalized_original", "synonym_expanded"],
        "context_token_count": 20,
        "context_rows_selected": 1,
    }

    report, summary = build_retrieval_diagnostic_report(
        [row],
        {
            "finance": [
                context_record(
                    context_id="target",
                    text="Apple revenue evidence.",
                    ticker="AAPL",
                )
            ]
        },
    )

    assert summary[0]["candidate_recall_at_50"] == 1.0
    assert report["finance_specific"]["failure_examples_by_ablation"]["prompt_text_only"]
    assert "gold_in_top50_not_top5" in report["top_failure_reasons_by_vertical"]["finance"]


def test_compression_still_reduces_tokens() -> None:
    record = context_record(
        context_id="target",
        text=" ".join(["Apple revenue evidence preserves provenance metadata."] * 20),
    )
    retrieval = HybridRetriever(
        BM25Retriever([record]),
        LocalFallbackDenseRetriever([record]),
    ).retrieve("Apple AAPL revenue 2024", top_k=1)

    compressed = compress_retrieval_results(retrieval.results, max_context_tokens=4096)

    assert compressed.results
    assert compressed.compressed_token_count < compressed.original_token_count


def test_no_model_api_or_gpu_call_is_triggered() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "question": "What did Apple report about revenue in 2024?",
        "vertical": "finance",
    }
    record = context_record(context_id="target", text="Apple revenue evidence.")
    retrievers: dict[str, dict[str, Any]] = {
        "finance": {
            "dense": LocalFallbackDenseRetriever([record]),
            "hybrid": HybridRetriever(
                BM25Retriever([record]),
                LocalFallbackDenseRetriever([record]),
            ),
            "records_by_context_id": {record.context_id: record},
            "company_ticker_resolver": CompanyTickerResolver.from_records([record]),
            "xbrl_concept_map": build_xbrl_concept_map([record]),
        }
    }
    query = prompt_query_text(
        prompt,
        "prompt_text_only",
        company_ticker_resolver=retrievers["finance"]["company_ticker_resolver"],
        xbrl_concept_map=retrievers["finance"]["xbrl_concept_map"],
    )

    retrieval = retrieve_for_mode(
        memory_mode="mm2_hybrid_top5",
        query=query.query_text,
        expanded_queries=query.expanded_queries,
        expansion_types=query.expansion_types,
        source_hints_used=query.uses_source_hints,
        vertical="finance",
        retrievers=retrievers,
        top_k=5,
    )

    assert retrieval.results
    assert retrieval.diagnostics["reranker_enabled"] is True
    assert retrieval.backend_label == "local_fallback"
