from inference_bench.context_schema import ContextRecord
from inference_bench.finance_retrieval_repair import (
    audit_finance_gold,
    audit_finance_prompts,
    build_finance_metadata_enrichment_report,
    context_records_by_match_id,
    derive_finance_enrichment,
    measure_finance_retrieval_repair,
    rewritten_retrieval_query,
)
from inference_bench.retrieval import CompanyTickerResolver, build_xbrl_concept_map


def finance_context(
    context_id: str = "finance_ctx_target",
    *,
    text: str = (
        "Apple Inc. reported 2024 revenue and net sales growth in the Form 10-K "
        "management discussion section."
    ),
    metadata: dict[str, object] | None = None,
) -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical="finance",
        source_id="finance_sec_edgar_xbrl",
        parent_id="finance_doc_AAPL_10K_2024",
        chunk_id=context_id,
        chunk_strategy="finance_filing_section_sentence_window",
        source_type="sec_filing_section",
        title="Apple 2024 Form 10-K Revenue",
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
            "original_doc_id": "finance_kb_sec_AAPL_10K_revenue_2024",
        },
        token_estimate=len(text.split()),
        provenance="finance_sec_edgar_xbrl",
        is_gold_linked=True,
    )


def finance_prompt() -> dict[str, object]:
    return {
        "prompt_id": "finance_prompt_001",
        "vertical": "finance",
        "ticker": "AAPL",
        "company": "Apple Inc. (AAPL)",
        "filing_form": "",
        "task_type": "answer_grounded",
        "expected_output_format": "text",
        "expected_status": "answer",
        "question": (
            "Using only cited SEC filing evidence, summarize the finance-relevant "
            "point for Apple Inc. (AAPL)."
        ),
    }


def finance_gold() -> dict[str, object]:
    return {
        "prompt_id": "finance_prompt_001",
        "vertical": "finance",
        "task_type": "answer_grounded",
        "expected_status": "answer",
        "reference_answer": (
            "The answer should cite the Form 10-K filing section and avoid investment advice."
        ),
        "must_include": ["AAPL", "finance_kb_sec_AAPL_10K_revenue_2024"],
        "must_not_include": ["price target"],
        "required_doc_ids": ["finance_kb_sec_AAPL_10K_revenue_2024"],
        "required_evidence_ids": ["finance_kb_sec_AAPL_10K_revenue_2024"],
    }


def test_finance_prompt_quality_audit_counts_missing_fields() -> None:
    prompt_a = finance_prompt()
    prompt_b = {
        **finance_prompt(),
        "prompt_id": "finance_prompt_002",
        "filing_form": "10-K",
        "question": "What was Apple revenue in fiscal year 2024 in the MD&A section?",
    }

    _report, summary = audit_finance_prompts([prompt_a, prompt_b])

    assert summary["total_prompts"] == 2
    assert summary["metric_missing_count"] == 1
    assert summary["period_missing_count"] == 1
    assert summary["filing_type_missing_count"] == 1
    assert summary["section_missing_count"] == 1


def test_finance_gold_quality_audit_recovers_linked_context_metadata() -> None:
    target = finance_context()
    by_match_id = context_records_by_match_id([target])

    _report, summary = audit_finance_gold({"finance_prompt_001": finance_gold()}, by_match_id)

    assert summary["total_gold_records"] == 1
    assert summary["metric_recoverable_from_linked_context_count"] == 1
    assert summary["period_recoverable_from_linked_context_count"] == 1
    assert summary["filing_recoverable_from_linked_context_count"] == 1
    assert summary["section_recoverable_from_linked_context_count"] == 1


def test_finance_metadata_enrichment_derives_metric_period_and_filing() -> None:
    prompt = finance_prompt()
    gold = finance_gold()
    target = finance_context()

    enrichment = derive_finance_enrichment(prompt, gold, [target])

    assert enrichment.ticker == "AAPL"
    assert enrichment.filing_type == "10-K"
    assert enrichment.fiscal_year == "2024"
    assert enrichment.metric_family == "revenue"
    assert enrichment.filing_section == "management_discussion_and_analysis"


def test_rewritten_finance_query_adds_context_without_direct_id_leakage() -> None:
    prompt = {
        **finance_prompt(),
        "question": ("Use finance_kb_sec_AAPL_10K_revenue_2024 to summarize Apple Inc. (AAPL)."),
    }
    target = finance_context()
    enrichment = derive_finance_enrichment(prompt, finance_gold(), [target])
    resolver = CompanyTickerResolver.from_records([target])
    concept_map = build_xbrl_concept_map([target])

    query, expanded_queries, _expansion_types, blocked_count = rewritten_retrieval_query(
        prompt,
        enrichment,
        ablation_mode="prompt_text_only",
        resolver=resolver,
        concept_map=concept_map,
    )

    assert blocked_count >= 1
    assert "finance_kb_sec" not in query
    assert "required_doc_ids" not in query
    assert "fiscal year 2024" in query
    assert "form 10-k" in query.lower()
    assert "revenue" in query
    assert expanded_queries


def test_metadata_enrichment_report_marks_no_inference() -> None:
    target = finance_context()
    by_match_id = context_records_by_match_id([target])

    report, enrichments = build_finance_metadata_enrichment_report(
        [finance_prompt()],
        {"finance_prompt_001": finance_gold()},
        by_match_id,
    )

    assert report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert report["summary"]["period_enriched_count"] == 1
    assert enrichments["finance_prompt_001"].xbrl_concept


def test_finance_retrieval_repair_measurement_improves_or_preserves_fixture() -> None:
    target = finance_context()
    distractor = finance_context(
        "finance_ctx_distractor",
        text="Apple Inc. discussed general risk factors and operations in a filing.",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "filing_date": "2024-11-01",
            "report_date": "2024-09-28",
            "concept": "RiskFactors",
            "section_type": "risk_factors",
            "original_doc_id": "finance_kb_sec_AAPL_10K_risk_2024",
        },
    )
    prompt = finance_prompt()
    gold = finance_gold()
    enrichment = derive_finance_enrichment(prompt, gold, [target])

    report, summary_rows = measure_finance_retrieval_repair(
        prompts=[prompt],
        gold_by_prompt_id={"finance_prompt_001": gold},
        finance_records=[distractor, target],
        enrichments={"finance_prompt_001": enrichment},
        dense_backend="local_fallback",
        ablation_modes=("prompt_text_only",),
    )

    before = report["impact_by_ablation"]["prompt_text_only"]["before"]
    after = report["impact_by_ablation"]["prompt_text_only"]["after_metadata_repair"]
    assert report["no_model_inference_triggered"] is True
    assert report["leakage_guard"]["direct_id_leakage_detected_count"] == 0
    assert after["candidate_recall_at_20"] >= before["candidate_recall_at_20"]
    assert after["final_recall_at_5"] >= before["final_recall_at_5"]
    assert summary_rows[0]["measurement"] == "after_metadata_repair"
