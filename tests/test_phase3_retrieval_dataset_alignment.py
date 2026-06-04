from typing import Any

from inference_bench.context_schema import ContextRecord
from inference_bench.retrieval_dataset_alignment import (
    DIRECT_RUNTIME_ID_RE,
    build_promotion_plan,
    canonical_metadata_from_prompt,
    expanded_valid_evidence_ids,
    repaired_record_from_prompt,
    sanitize_runtime_query,
    summary_by_vertical,
    validate_alignment_dataset,
)
from inference_bench.slo import load_slo_config


def context_record(
    vertical: str,
    context_id: str,
    text: str,
    metadata: dict[str, object],
) -> ContextRecord:
    return ContextRecord(
        context_id=context_id,
        vertical=vertical,
        source_id=f"{vertical}_fixture_source",
        parent_id=f"{vertical}_fixture_parent",
        chunk_id=context_id,
        chunk_strategy=f"{vertical}_fixture_chunking",
        source_type=f"{vertical}_fixture_source_type",
        title=str(metadata.get("title") or context_id),
        text=text,
        metadata=metadata,
        token_estimate=len(text.split()),
        provenance=f"{vertical}_fixture",
        is_gold_linked=True,
    )


def fixture_data() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[ContextRecord]],
]:
    prompts: dict[str, list[dict[str, Any]]] = {
        "airline": [
            {
                "prompt_id": "airline_001",
                "vertical": "airline",
                "question": "A traveler needs cancellation refund help on route YYZ-YVR.",
                "support_type": "cancellation_refund",
                "route": "YYZ-YVR",
            }
        ],
        "healthcare_admin": [
            {
                "prompt_id": "healthcare_001",
                "vertical": "healthcare_admin",
                "question": "A patient asks about appointment booking and privacy.",
                "support_type": "appointment_booking",
                "department": "scheduling",
                "safety_boundary": "administrative_only",
                "privacy_sensitive": True,
            }
        ],
        "retail": [
            {
                "prompt_id": "retail_001",
                "vertical": "retail",
                "question": "Summarize support signal for Compact Bluetooth Speaker.",
                "category": "Electronics",
                "product_title": "Compact Bluetooth Speaker",
                "issue_type": "review_summary",
            }
        ],
        "finance": [
            {
                "prompt_id": "finance_001",
                "vertical": "finance",
                "question": "What did Apple report about revenue in fiscal year 2024?",
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "filing_form": "10-K",
            }
        ],
        "research_ai": [
            {
                "prompt_id": "research_001",
                "vertical": "research_ai",
                "question": "What method is described in the FastKV paper?",
                "topic": "llm inference",
                "metadata": {
                    "source_titles": ["FastKV: Efficient Inference for Long Context Models"],
                    "evidence_type": ["method"],
                    "topics": ["llm_serving_inference_optimization"],
                },
            }
        ],
    }
    gold = {
        vertical: {
            str(rows[0]["prompt_id"]): {
                "prompt_id": rows[0]["prompt_id"],
                "vertical": vertical,
                "required_evidence_ids": [f"{vertical}_doc_001"],
            }
        }
        for vertical, rows in prompts.items()
    }
    corpora = {
        "airline": [
            context_record(
                "airline",
                "airline_ctx_001",
                "Cancellation refund policy applies to route disruption.",
                {"original_doc_id": "airline_doc_001", "policy_tags": ["cancellation", "refund"]},
            )
        ],
        "healthcare_admin": [
            context_record(
                "healthcare_admin",
                "healthcare_ctx_001",
                "Appointment booking procedure includes privacy and identity checks.",
                {"original_doc_id": "healthcare_admin_doc_001", "tags": ["appointment"]},
            )
        ],
        "retail": [
            context_record(
                "retail",
                "retail_ctx_001",
                "Compact Bluetooth Speaker review mentions support signal.",
                {
                    "original_doc_id": "retail_doc_001",
                    "product_title": "Compact Bluetooth Speaker",
                    "category": "Electronics",
                    "evidence_type": "review",
                },
            )
        ],
        "finance": [
            context_record(
                "finance",
                "finance_ctx_001",
                "Apple Inc. Form 10-K reported fiscal year 2024 revenue and net sales.",
                {
                    "original_doc_id": "finance_doc_001",
                    "ticker": "AAPL",
                    "company_name": "Apple Inc.",
                    "form": "10-K",
                    "concept": "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "report_date": "2024-09-28",
                },
            )
        ],
        "research_ai": [
            context_record(
                "research_ai",
                "research_ctx_001",
                "The method section describes long context KV compression.",
                {
                    "original_doc_id": "research_ai_doc_001",
                    "paper_title": "FastKV: Efficient Inference for Long Context Models",
                    "section_type": "method",
                    "topic": "llm inference",
                },
            )
        ],
    }
    return prompts, gold, corpora


def test_repaired_record_preserves_prompt_id_and_excludes_runtime_ids() -> None:
    prompt = {
        "prompt_id": "retail_001",
        "vertical": "retail",
        "question": "Use retail_review_0001_hidden for Compact Bluetooth Speaker B00SOURCE1.",
        "category": "Electronics",
        "product_title": "Compact Bluetooth Speaker",
        "issue_type": "review_summary",
    }
    gold = {"required_evidence_ids": ["retail_doc_001"]}
    records = [
        context_record(
            "retail",
            "retail_ctx_001",
            "Compact Bluetooth Speaker review.",
            {"original_doc_id": "retail_doc_001", "product_title": "Compact Bluetooth Speaker"},
        )
    ]

    repaired = repaired_record_from_prompt(
        prompt=prompt,
        gold_record=gold,
        records=records,
        by_match_id={"retail_doc_001": records},
        original_metrics={
            "candidate_recall_at_20": 0.0,
            "candidate_recall_at_50": 0.0,
            "final_recall_at_5": 0.0,
            "mrr": 0.0,
        },
        resolver=None,
        concept_map={},
    )

    assert repaired["prompt_id"] == "retail_001"
    assert DIRECT_RUNTIME_ID_RE.search(str(repaired["retrieval_query"])) is None
    assert repaired["runtime_query_uses_valid_evidence_ids"] is False


def test_valid_evidence_ids_expanded_is_not_used_as_runtime_query_input() -> None:
    query, blocked = sanitize_runtime_query("Please use CA-POL-001 and retail_doc_001.")

    assert "CA-POL" not in query
    assert "retail_doc_001" not in query
    assert blocked >= 1


def test_finance_repair_adds_metric_period_and_filing_fields() -> None:
    metadata = canonical_metadata_from_prompt(
        {
            "prompt_id": "finance_001",
            "vertical": "finance",
            "question": "What did Apple report about revenue in fiscal year 2024?",
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "filing_form": "10-K",
        }
    )

    assert metadata["ticker"] == "AAPL"
    assert metadata["metric_family"] == "revenue"
    assert metadata["fiscal_year"] == "2024"
    assert metadata["filing_type"] == "10-K"


def test_retail_repair_adds_support_intent_and_product_fields() -> None:
    metadata = canonical_metadata_from_prompt(
        {
            "prompt_id": "retail_001",
            "vertical": "retail",
            "question": "Summarize support signal for Compact Bluetooth Speaker.",
            "category": "Electronics",
            "product_title": "Compact Bluetooth Speaker",
            "issue_type": "review_summary",
        }
    )

    assert metadata["product_title"] == "Compact Bluetooth Speaker"
    assert metadata["product_category"] == "Electronics"
    assert metadata["support_intent"] == "review_summary"


def test_research_repair_adds_paper_and_section_fields() -> None:
    metadata = canonical_metadata_from_prompt(
        {
            "prompt_id": "research_001",
            "vertical": "research_ai",
            "question": "What method is described in the FastKV paper?",
            "topic": "llm inference",
            "metadata": {
                "source_titles": ["FastKV: Efficient Inference for Long Context Models"],
                "evidence_type": ["method"],
            },
        }
    )

    assert metadata["paper_title"] == "FastKV: Efficient Inference for Long Context Models"
    assert metadata["section_type"] == "method"
    assert "method_result_limitation_cue" in metadata


def test_ambiguous_records_are_classified_and_expanded() -> None:
    prompts, gold, corpora = fixture_data()
    prompt = prompts["retail"][0]
    expanded = expanded_valid_evidence_ids(
        prompt=prompt,
        gold_record=gold["retail"]["retail_001"],
        records=corpora["retail"],
        canonical_metadata=canonical_metadata_from_prompt(prompt),
    )

    assert "retail_doc_001" in expanded
    rows = summary_by_vertical(
        [
            {
                "vertical": "retail",
                "repair_reason": [
                    "missing_canonical_retrieval_query",
                    "multiple_valid_evidence_not_counted",
                ],
            }
        ]
    )
    retail_row = next(row for row in rows if row["vertical"] == "retail")
    assert retail_row["missing_canonical_retrieval_query"] == 1


def test_validation_report_and_promotion_plan_are_produced() -> None:
    prompts, gold, corpora = fixture_data()
    repaired_records = []
    for vertical, rows in prompts.items():
        prompt = rows[0]
        prompt_id = str(prompt["prompt_id"])
        repaired_records.append(
            repaired_record_from_prompt(
                prompt=prompt,
                gold_record=gold[vertical][prompt_id],
                records=corpora[vertical],
                by_match_id={f"{vertical}_doc_001": corpora[vertical]},
                original_metrics={
                    "candidate_recall_at_20": 0.0,
                    "candidate_recall_at_50": 0.0,
                    "final_recall_at_5": 0.0,
                    "mrr": 0.0,
                },
                resolver=None,
                concept_map={},
            )
        )
    repaired_by_prompt_id = {str(record["prompt_id"]): record for record in repaired_records}

    report, summary_rows = validate_alignment_dataset(
        prompts_by_vertical=prompts,
        gold_by_vertical=gold,
        corpora_by_vertical=corpora,
        repaired_by_prompt_id=repaired_by_prompt_id,
        stage_sizes=[1],
        slo_config=load_slo_config("configs/slo_targets.yaml"),
        dense_backend="local_fallback",
        vector_store_config_path="configs/vector_stores.yaml",
        vector_store_key="qdrant_local",
        allow_dense_fallback=True,
    )
    plan = build_promotion_plan(summary_rows=summary_rows, repaired_records=repaired_records)

    assert report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert report["no_external_api_calls_triggered"] is True
    assert len(summary_rows) == 10
    assert "promotion_recommended" in plan
