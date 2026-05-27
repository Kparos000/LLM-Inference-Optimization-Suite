from pathlib import Path

from inference_bench.context_schema import ContextRecord
from inference_bench.memory_workloads import (
    build_compression_diagnostic_report,
    build_retrieval_diagnostic_report,
    write_csv,
    write_json,
)
from inference_bench.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalFallbackDenseRetriever,
    compress_retrieval_results,
    evaluate_retrieval_results,
)


def context_record(
    *,
    context_id: str,
    vertical: str = "finance",
    text: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ContextRecord:
    body = text or (
        "Revenue from contracts with customers was reported for the fiscal period. "
        "The filing section gives traceable numeric context and source metadata. "
        "This sentence adds enough content for deterministic compression testing."
    )
    return ContextRecord(
        context_id=context_id,
        vertical=vertical,
        source_id="finance_sec_edgar_xbrl" if vertical == "finance" else f"{vertical}_source",
        parent_id="finance_doc_AAPL_10K_2024" if vertical == "finance" else context_id,
        chunk_id=context_id,
        chunk_strategy="finance_filing_section_sentence_window"
        if vertical == "finance"
        else "policy_section",
        source_type="sec_filing_section" if vertical == "finance" else "policy",
        title="AAPL 10-K Revenue",
        text=body,
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
            "section_record_id": "finance_section_AAPL_10K_revenue_1",
        },
        token_estimate=len(body.split()),
        provenance="finance_sec_edgar_xbrl",
        is_gold_linked=True,
    )


def hybrid(records: list[ContextRecord]) -> HybridRetriever:
    return HybridRetriever(BM25Retriever(records), LocalFallbackDenseRetriever(records))


def test_finance_aware_retrieval_boosts_exact_ticker_matches() -> None:
    target = context_record(context_id="target")
    distractor = context_record(
        context_id="distractor",
        metadata={
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "form": "10-K",
            "concept": "Revenue",
            "original_doc_id": "finance_kb_sec_MSFT_10K_revenue",
        },
    )

    result = hybrid([distractor, target]).retrieve("AAPL 10-K revenue", top_k=1)

    assert result.results[0].context_record.context_id == "target"
    assert result.results[0].component_scores["metadata_boost"] > 0


def test_finance_aware_retrieval_boosts_concept_metric_matches() -> None:
    target = context_record(context_id="target")
    distractor = context_record(
        context_id="distractor",
        text="Equity balance evidence describes stockholders capital and share counts.",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "concept": "CommonStocksIncludingAdditionalPaidInCapital",
            "original_doc_id": "finance_kb_sec_AAPL_10K_equity",
        },
    )

    result = hybrid([distractor, target]).retrieve("AAPL revenue metric", top_k=1)

    assert result.results[0].context_record.context_id == "target"


def test_finance_aware_retrieval_boosts_filing_period_form_when_available() -> None:
    target = context_record(context_id="target")
    distractor = context_record(
        context_id="distractor",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "8-K",
            "filing_date": "2023-01-01",
            "report_date": "2022-12-31",
            "concept": "Revenue",
            "original_doc_id": "finance_kb_sec_AAPL_8K_revenue",
        },
    )

    result = hybrid([distractor, target]).retrieve("AAPL 10-K 2024 revenue", top_k=1)

    assert result.results[0].context_record.context_id == "target"


def test_hybrid_retrieval_still_works_for_all_verticals() -> None:
    for vertical in ["airline", "healthcare_admin", "retail", "finance", "research_ai"]:
        record = context_record(
            context_id=f"{vertical}_ctx",
            vertical=vertical,
            text=f"{vertical} policy evidence includes refund safety citation details.",
            metadata={"original_doc_id": f"{vertical}_doc_001"},
        )
        result = hybrid([record]).retrieve(f"{vertical} refund evidence", top_k=1)

        assert result.results[0].context_record.vertical == vertical


def test_duplicate_context_chunks_are_removed() -> None:
    duplicate_a = context_record(context_id="duplicate_a")
    duplicate_b = context_record(context_id="duplicate_b", text=duplicate_a.text)

    result = hybrid([duplicate_a, duplicate_b]).retrieve("AAPL 10-K revenue", top_k=5)

    assert len(result.results) == 1


def test_compression_reduces_token_count_and_preserves_context() -> None:
    retrieval = hybrid([context_record(context_id="target")]).retrieve("AAPL 10-K revenue", top_k=1)
    compressed = compress_retrieval_results(retrieval.results, max_context_tokens=4096)

    assert compressed.results
    assert compressed.compressed_token_count < compressed.original_token_count
    assert compressed.compression_ratio < 1.0


def test_compression_preserves_at_least_one_record_when_retrieval_succeeds() -> None:
    retrieval = hybrid([context_record(context_id="target")]).retrieve("AAPL 10-K revenue", top_k=1)
    compressed = compress_retrieval_results(retrieval.results, max_context_tokens=1)

    assert len(compressed.results) == 1


def test_compression_preserves_provenance_metadata() -> None:
    retrieval = hybrid([context_record(context_id="target")]).retrieve("AAPL 10-K revenue", top_k=1)
    compressed = compress_retrieval_results(retrieval.results, max_context_tokens=4096)
    record = compressed.results[0].context_record

    assert record.provenance == "finance_sec_edgar_xbrl"
    assert record.metadata["ticker"] == "AAPL"
    assert record.metadata["compression"]["type"] == "deterministic_extractive_truncation"


def test_compressed_recall_loss_is_computed() -> None:
    retrieval = hybrid([context_record(context_id="target")]).retrieve("AAPL 10-K revenue", top_k=1)
    compressed = compress_retrieval_results(retrieval.results, max_context_tokens=4096)
    before = evaluate_retrieval_results(
        gold_evidence_ids=["finance_kb_sec_AAPL_10K_revenue"],
        results=retrieval.results,
    )
    after = evaluate_retrieval_results(
        gold_evidence_ids=["finance_kb_sec_AAPL_10K_revenue"],
        results=compressed.results,
    )
    recall_loss = before["recall_at_5"] - after["recall_at_5"]

    assert recall_loss == 0.0


def test_retrieval_diagnostics_report_is_generated(tmp_path: Path) -> None:
    rows = [
        {
            "split": "final_10000",
            "memory_mode": "mm2_hybrid_top5",
            "vertical": "finance",
            "prompt_id": "finance_prompt_001",
            "recall_at_5": 1.0,
            "mrr": 1.0,
            "gold_evidence_ids": ["finance_kb_sec_AAPL_10K_revenue"],
            "matched_gold_evidence_ids": ["finance_kb_sec_AAPL_10K_revenue"],
            "retrieved_context_ids": ["target"],
            "query_text": "AAPL 10-K revenue",
            "context_token_count": 32,
            "context_rows_selected": 1,
        }
    ]
    report, summary_rows = build_retrieval_diagnostic_report(
        rows,
        {"finance": [context_record(context_id="target")]},
    )

    write_json(tmp_path / "retrieval_diagnostic_report.json", report)
    write_csv(
        tmp_path / "retrieval_diagnostic_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )

    assert (tmp_path / "retrieval_diagnostic_report.json").exists()
    assert report["no_model_inference_triggered"] is True


def test_compression_diagnostics_report_is_generated(tmp_path: Path) -> None:
    rows = [
        {
            "split": "final_10000",
            "memory_mode": "mm3_compressed_hybrid_top5",
            "vertical": "finance",
            "context_token_count": 70,
            "token_reduction": 30,
            "recall_before_compression": 1.0,
            "recall_after_compression": 1.0,
            "recall_loss": 0.0,
            "gold_evidence_retained_after_compression": True,
        }
    ]
    report, summary_rows = build_compression_diagnostic_report(rows)

    write_json(tmp_path / "compression_diagnostic_report.json", report)
    write_csv(
        tmp_path / "compression_diagnostic_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )

    assert (tmp_path / "compression_diagnostic_report.json").exists()
    assert report["overall_by_split"]["final_10000"]["token_reduction_pct"] == 0.3


def test_no_model_inference_or_gpu_api_calls_are_triggered() -> None:
    report, _ = build_compression_diagnostic_report([])

    assert report["no_model_inference_triggered"] is True
