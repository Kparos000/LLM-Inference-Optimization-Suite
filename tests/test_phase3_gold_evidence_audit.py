from inference_bench.context_schema import ContextRecord
from inference_bench.gold_evidence_audit import build_gold_evidence_audit_report


def finance_context(context_id: str = "finance_ctx_AAPL_revenue") -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical="finance",
        source_id="finance_sec_edgar_xbrl",
        parent_id="finance_doc_AAPL_10K_2024",
        chunk_id=context_id,
        chunk_strategy="finance_filing_section_sentence_window",
        source_type="sec_filing_section",
        title="AAPL 10-K Revenue",
        text="Apple revenue evidence for fiscal year 2024.",
        metadata={
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "fiscal_year": "2024",
            "original_doc_id": "finance_gold_AAPL_revenue"
            if context_id == "finance_ctx_AAPL_revenue"
            else context_id,
        },
        token_estimate=7,
        provenance="finance_sec_edgar_xbrl",
        is_gold_linked=True,
    )


def test_gold_evidence_audit_detects_missing_gold_ids() -> None:
    report, rows = build_gold_evidence_audit_report(
        prompts_by_vertical={
            "finance": [
                {
                    "prompt_id": "finance_001",
                    "question": "What was the selected financial metric?",
                    "vertical": "finance",
                }
            ]
        },
        gold_by_vertical={
            "finance": {
                "finance_001": {
                    "prompt_id": "finance_001",
                    "required_doc_ids": ["missing_gold_id"],
                }
            }
        },
        corpora_by_vertical={
            "finance": [
                *[finance_context(f"distractor_{index}") for index in range(5)],
                finance_context(),
            ]
        },
        evaluation_rows=[],
    )

    assert rows[0]["gold_not_in_corpus_count"] == 1
    assert report["examples"]["finance"][0]["audit_reason"] == "gold_id_not_found_in_context_corpus"
    assert report["no_model_inference_triggered"] is True


def test_gold_evidence_audit_detects_prompt_missing_entity_metric_period() -> None:
    _, rows = build_gold_evidence_audit_report(
        prompts_by_vertical={
            "finance": [
                {
                    "prompt_id": "finance_001",
                    "question": "What was the selected figure?",
                    "vertical": "finance",
                }
            ]
        },
        gold_by_vertical={
            "finance": {
                "finance_001": {
                    "prompt_id": "finance_001",
                    "required_doc_ids": ["finance_gold_AAPL_revenue"],
                }
            }
        },
        corpora_by_vertical={
            "finance": [
                *[finance_context(f"distractor_{index}") for index in range(5)],
                finance_context(),
            ]
        },
        evaluation_rows=[],
    )

    assert rows[0]["prompt_missing_entity_count"] == 1
    assert rows[0]["prompt_missing_metric_count"] == 1
    assert rows[0]["prompt_missing_period_count"] == 1


def test_gold_evidence_audit_counts_candidate_top50_not_top5() -> None:
    _, rows = build_gold_evidence_audit_report(
        prompts_by_vertical={
            "finance": [
                {
                    "prompt_id": "finance_001",
                    "question": "Apple revenue 2024.",
                    "vertical": "finance",
                }
            ]
        },
        gold_by_vertical={
            "finance": {
                "finance_001": {
                    "prompt_id": "finance_001",
                    "required_doc_ids": ["finance_gold_AAPL_revenue"],
                }
            }
        },
        corpora_by_vertical={
            "finance": [
                *[finance_context(f"distractor_{index}") for index in range(5)],
                finance_context(),
            ]
        },
        evaluation_rows=[
            {
                "split": "final_10000",
                "ablation_mode": "prompt_plus_metadata",
                "memory_mode": "mm2_hybrid_top5",
                "vertical": "finance",
                "gold_evidence_ids": ["finance_gold_AAPL_revenue"],
                "candidate_context_ids": [
                    *[f"distractor_{index}" for index in range(5)],
                    "finance_ctx_AAPL_revenue",
                ],
            }
        ],
    )

    assert rows[0]["gold_in_top50_but_not_top5_count"] == 1
