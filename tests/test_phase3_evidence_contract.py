from inference_bench.context_schema import ContextRecord
from inference_bench.evidence_contract import (
    evidence_contracts_from_results,
    validate_evidence_contract,
)
from inference_bench.retrieval import BM25Retriever, HybridRetriever, LocalFallbackDenseRetriever


def context_record(context_id: str, text: str | None = None) -> ContextRecord:
    body = text or f"Apple AAPL revenue fiscal year 2024 evidence chunk {context_id}."
    return ContextRecord(
        context_id=context_id,
        vertical="finance",
        source_id="finance_sec_edgar_xbrl",
        parent_id=f"finance_doc_AAPL_10K_2024_{context_id}",
        chunk_id=context_id,
        chunk_strategy="finance_filing_section_sentence_window",
        source_type="sec_filing_section",
        title=f"AAPL 10-K Revenue {context_id}",
        text=body,
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "fiscal_year": "2024",
            "concept": "Revenue",
            "section_type": "management_discussion_and_analysis",
        },
        token_estimate=len(body.split()),
        provenance="finance_sec_edgar_xbrl",
        is_gold_linked=True,
    )


def test_final_evidence_selector_returns_exactly_5_items() -> None:
    records = [context_record(f"context_{index}") for index in range(8)]
    retrieval = HybridRetriever(
        BM25Retriever(records),
        LocalFallbackDenseRetriever(records),
    ).retrieve("Apple AAPL revenue fiscal year 2024", top_k=5)

    assert len(retrieval.results) == 5
    assert retrieval.diagnostics["evidence_selector_strategy"] == "finance_calibrated_top5"
    assert retrieval.diagnostics["calibrated_reranker_enabled"] is True


def test_evidence_selector_avoids_duplicate_chunks_where_possible() -> None:
    duplicate_text = "Apple AAPL revenue fiscal year 2024 duplicated evidence."
    records = [
        context_record("duplicate_a", duplicate_text),
        context_record("duplicate_b", duplicate_text),
        *[context_record(f"context_{index}") for index in range(5)],
    ]
    retrieval = HybridRetriever(
        BM25Retriever(records),
        LocalFallbackDenseRetriever(records),
    ).retrieve("Apple AAPL revenue fiscal year 2024", top_k=5)

    selected_ids = [result.context_record.context_id for result in retrieval.results]
    assert not {"duplicate_a", "duplicate_b"}.issubset(selected_ids)


def test_evidence_contract_validates_and_records_selection_reason() -> None:
    record = context_record("target")
    retrieval = HybridRetriever(
        BM25Retriever([record]),
        LocalFallbackDenseRetriever([record]),
    ).retrieve("Apple AAPL revenue fiscal year 2024", top_k=1)
    contracts = evidence_contracts_from_results(
        retrieval.results,
        selection_reasons_by_context_id={"target": "metric_or_concept_match"},
    )

    validate_evidence_contract(contracts[0])
    assert contracts[0]["evidence_id"] == "target"
    assert contracts[0]["selection_reason"] == "metric_or_concept_match"
    assert "gold_evidence_ids" not in contracts[0]


def test_evidence_contract_rejects_gold_labels() -> None:
    record = context_record("target")
    retrieval = HybridRetriever(
        BM25Retriever([record]),
        LocalFallbackDenseRetriever([record]),
    ).retrieve("Apple AAPL revenue fiscal year 2024", top_k=1)
    contract = evidence_contracts_from_results(retrieval.results)[0]
    contract["gold_evidence_ids"] = ["target"]

    try:
        validate_evidence_contract(contract)
    except ValueError as exc:
        assert "forbidden gold-label fields" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected gold label validation failure")
