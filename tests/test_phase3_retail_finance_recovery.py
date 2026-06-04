from __future__ import annotations

from inference_bench.context_schema import ContextRecord
from inference_bench.retail_finance_recovery import (
    build_finance_metadata_flow_rows,
    known_match_ids,
    retail_failure_reasons,
)
from inference_bench.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalFallbackDenseRetriever,
    RetrievalResult,
    evaluate_retrieval_results,
)
from inference_bench.vertical_retrieval_repair import EnrichmentResult


def retail_context(
    context_id: str,
    *,
    document_type: str,
    evidence_type: str | None,
    text: str,
    issue_terms: list[str] | None = None,
) -> ContextRecord:
    return ContextRecord(
        context_id=f"retail:{context_id}",
        vertical="retail",
        source_id="retail_fixture",
        parent_id="B00TEST",
        chunk_id=context_id,
        chunk_strategy="retail_parent_child_product_review",
        source_type=document_type,
        title="Test Product - Review Evidence",
        text=text,
        metadata={
            "category": "All_Beauty",
            "document_type": document_type,
            "evidence_type": evidence_type,
            "issue_terms": issue_terms or [],
            "original_doc_id": context_id,
            "parent_asin": "B00TEST",
            "product_title": "Test Product",
            "tags": ["retail", document_type],
        },
        token_estimate=len(text.split()),
        provenance="retail_fixture",
        is_gold_linked=False,
    )


def test_retail_reranker_prefers_review_evidence_for_review_intent() -> None:
    multicategory = retail_context(
        "retail_seed_expand_0001_B00TEST",
        document_type="retail_multicategory_review_evidence",
        evidence_type=None,
        text="Test Product support evidence issue identification selected record.",
        issue_terms=["issue_identification"],
    )
    review = retail_context(
        "retail_review_0001_target",
        document_type="review_evidence",
        evidence_type="review",
        text="Test Product review evidence reports a broken damaged quality defect.",
        issue_terms=["quality", "defect", "damaged"],
    )

    retrieval = HybridRetriever(
        BM25Retriever([multicategory, review]),
        LocalFallbackDenseRetriever([multicategory, review]),
    ).retrieve("quality_complaint Test Product broken damaged review", top_k=2)

    assert retrieval.results[0].context_record.context_id == "retail:retail_review_0001_target"
    assert retrieval.diagnostics["evidence_selector_strategy"] == "retail_balanced_top5"
    assert retrieval.results[0].component_scores["retail_kind_match"] > 0


def test_retail_failure_reasons_distinguish_reranker_and_metadata_failures() -> None:
    target = retail_context(
        "retail_review_0001_target",
        document_type="review_evidence",
        evidence_type="review",
        text="Test Product review says damaged.",
        issue_terms=["damaged"],
    )
    selected = retail_context(
        "retail_seed_expand_0001_B00TEST",
        document_type="retail_multicategory_review_evidence",
        evidence_type=None,
        text="Test Product generic support evidence.",
        issue_terms=["issue_identification"],
    )
    candidate_results = [
        RetrievalResult(target, 1.0, 1, "hybrid", {}),
        RetrievalResult(selected, 0.9, 2, "hybrid", {}),
    ]
    final_results = [RetrievalResult(selected, 0.9, 1, "hybrid", {})]
    prompt = {
        "prompt_id": "retail_fixture",
        "product_title": "Test Product",
        "category": "All_Beauty",
        "issue_type": "quality_complaint",
        "question": "Resolve a quality complaint for Test Product.",
    }

    reasons = retail_failure_reasons(
        prompt=prompt,
        gold_ids=["retail_review_0001_target"],
        candidate_results=candidate_results,
        final_results=final_results,
        recall_at_5=0.0,
        known_ids=known_match_ids([target, selected]),
    )

    assert "reranker_failure" in reasons
    assert "review_issue_mismatch" in reasons


def test_finance_metadata_flow_materializes_non_id_query_terms() -> None:
    prompt = {
        "prompt_id": "finance_fixture",
        "question": "Summarize the selected SEC/XBRL evidence for Apple Inc. (AAPL) 10-K.",
        "company": "Apple Inc. (AAPL) 10-K",
        "ticker": "AAPL",
        "filing_form": "10-K",
        "vertical": "finance",
        "metadata": {"evidence_type": "sec_xbrl_filing_evidence"},
        "required_doc_ids": ["finance_kb_sec_AAPL_10K_target"],
    }
    enrichment = EnrichmentResult(
        vertical="finance",
        prompt_id="finance_fixture",
        fields={
            "company": "Apple Inc. (AAPL) 10-K",
            "ticker": "AAPL",
            "metric_family": None,
            "period": None,
            "filing_type": "10-K",
            "section_type": None,
            "xbrl_concept": None,
        },
        missing_fields=["metric_family", "period", "section_type", "xbrl_concept"],
        query_terms=[
            "Apple Inc. (AAPL) 10-K",
            "AAPL",
            "10-K",
            "annual report",
            "financial statement facts",
            "net sales",
        ],
    )

    rows = build_finance_metadata_flow_rows(
        prompts=[prompt],
        enrichments={"finance_fixture": enrichment},
        resolver=None,
        concept_map={},
    )

    assert rows[0]["derived_filing"] == "10-K"
    assert "annual report" in rows[0]["materialized_query"]
    assert "net sales" in rows[0]["materialized_query"]
    assert "finance_kb_sec" not in rows[0]["materialized_query"]
    assert rows[0]["direct_hint_leakage_detected"] is False


def test_retail_recovery_path_does_not_require_model_outputs() -> None:
    target = retail_context(
        "retail_review_0001_target",
        document_type="review_evidence",
        evidence_type="review",
        text="Test Product review says damaged.",
        issue_terms=["damaged"],
    )
    retrieval = HybridRetriever(
        BM25Retriever([target]),
        LocalFallbackDenseRetriever([target]),
    ).retrieve("quality_complaint Test Product damaged", top_k=1)
    evaluation = evaluate_retrieval_results(
        gold_evidence_ids=["retail_review_0001_target"],
        results=retrieval.results,
    )

    assert evaluation["recall_at_5"] == 1.0
