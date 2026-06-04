from typing import Any, cast

from inference_bench.context_schema import ContextRecord
from inference_bench.slo import load_slo_config
from inference_bench.vertical_retrieval_repair import (
    DIRECT_HINT_RE,
    build_audit_report,
    enrich_prompt_metadata,
    repair_profiles,
    validate_stages,
)


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
                "support_type": "cancellation",
                "issue": "Scenario asks for policy evidence.",
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
                "question": "Summarize negative reviews about broken product quality and refund.",
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
                "question": "What method and results are described in the paper?",
                "topic": "llm inference",
                "metadata": {
                    "source_titles": ["FastKV: Efficient Inference for Long Context Models"],
                    "evidence_type": ["method"],
                    "topics": ["llm_serving_inference_optimization"],
                },
            }
        ],
    }
    gold: dict[str, dict[str, dict[str, Any]]] = {
        vertical: {
            str(rows[0]["prompt_id"]): {
                "prompt_id": rows[0]["prompt_id"],
                "vertical": vertical,
                "required_doc_ids": [f"{vertical}_doc_001"],
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
                "Cancellation refund policy applies to route disruption and booking help.",
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
                "Compact Bluetooth Speaker reviews mention broken sound quality and refund.",
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
                "The paper method improves long context inference and reports benchmark results.",
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


def test_each_vertical_has_a_repair_profile() -> None:
    profiles = repair_profiles()

    assert set(profiles) == {"airline", "healthcare_admin", "retail", "finance", "research_ai"}
    assert all(profile.enrichment_fields for profile in profiles.values())


def test_metadata_enrichment_excludes_gold_and_source_ids() -> None:
    prompt = {
        "prompt_id": "airline_001",
        "vertical": "airline",
        "question": "Use CA-POL-001 and source_id to answer refund policy.",
        "support_type": "refund",
    }

    enrichment = enrich_prompt_metadata(vertical="airline", prompt=prompt)

    joined_terms = " ".join(enrichment.query_terms)
    assert "CA-POL" not in joined_terms
    assert "source_id" not in joined_terms
    assert DIRECT_HINT_RE.search(joined_terms) is None


def test_airline_repair_extracts_policy_issue_fields() -> None:
    enrichment = enrich_prompt_metadata(
        vertical="airline",
        prompt={
            "prompt_id": "airline_001",
            "vertical": "airline",
            "question": "A traveler needs baggage refund help on route YYZ-YVR.",
            "support_type": "baggage",
            "route": "YYZ-YVR",
        },
    )

    assert enrichment.fields["policy_type"] == "baggage"
    assert enrichment.fields["travel_issue"] in {"refund", "baggage"}
    assert enrichment.fields["route_region"] == "YYZ-YVR"


def test_healthcare_repair_extracts_admin_safety_fields() -> None:
    enrichment = enrich_prompt_metadata(
        vertical="healthcare_admin",
        prompt={
            "prompt_id": "healthcare_001",
            "vertical": "healthcare_admin",
            "question": "A patient asks about appointment booking and privacy.",
            "support_type": "appointment_booking",
            "department": "scheduling",
            "safety_boundary": "administrative_only",
            "privacy_sensitive": True,
        },
    )

    assert enrichment.fields["admin_task_type"] == "appointment_booking"
    assert enrichment.fields["safety_boundary"] == "administrative_only"
    assert enrichment.fields["privacy_sensitive"] is True


def test_retail_repair_extracts_product_review_fields() -> None:
    enrichment = enrich_prompt_metadata(
        vertical="retail",
        prompt={
            "prompt_id": "retail_001",
            "vertical": "retail",
            "question": "Summarize broken product reviews and refund issues.",
            "category": "Electronics",
            "product_title": "Compact Bluetooth Speaker",
            "issue_type": "review_summary",
        },
    )

    assert enrichment.fields["product_category"] == "Electronics"
    title_values = cast(list[str], enrichment.fields["product_title_terms"])
    assert "compact" in title_values
    assert enrichment.fields["review_issue_type"] == "review_summary"


def test_finance_repair_extracts_company_metric_period_fields() -> None:
    enrichment = enrich_prompt_metadata(
        vertical="finance",
        prompt={
            "prompt_id": "finance_001",
            "vertical": "finance",
            "question": "What did Apple report about revenue in fiscal year 2024?",
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "filing_form": "10-K",
        },
    )

    assert enrichment.fields["company"] == "Apple Inc."
    assert enrichment.fields["ticker"] == "AAPL"
    assert enrichment.fields["metric_family"] == "revenue"
    assert enrichment.fields["fiscal_year"] == "2024"


def test_research_ai_repair_extracts_paper_section_fields() -> None:
    enrichment = enrich_prompt_metadata(
        vertical="research_ai",
        prompt={
            "prompt_id": "research_001",
            "vertical": "research_ai",
            "question": "What method and results are described in the paper?",
            "topic": "llm inference",
            "metadata": {
                "source_titles": ["FastKV: Efficient Inference for Long Context Models"],
                "evidence_type": ["method"],
                "topics": ["llm_serving_inference_optimization"],
            },
        },
    )

    assert (
        enrichment.fields["paper_id_public"]
        == "FastKV: Efficient Inference for Long Context Models"
    )
    assert enrichment.fields["section_type"] == "method"
    assert enrichment.fields["method_signal"] is True


def test_staged_validation_can_run_on_fixture_without_inference() -> None:
    prompts, gold, corpora = fixture_data()
    slo_config = load_slo_config("configs/slo_targets.yaml")
    audit_report, enrichments = build_audit_report(
        prompts_by_vertical=prompts,
        gold_by_vertical=gold,
        corpora_by_vertical=corpora,
    )
    report, summary_rows, examples = validate_stages(
        prompts_by_vertical=prompts,
        gold_by_vertical=gold,
        corpora_by_vertical=corpora,
        enrichments=enrichments,
        slo_config=slo_config,
        stage_sizes=[1],
        dense_backend="local_fallback",
        vector_store_config_path="configs/vector_stores.yaml",
        vector_store_key="qdrant_local",
        allow_dense_fallback=True,
    )

    assert audit_report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert report["direct_hint_leakage_detected_count"] == 0
    assert len(summary_rows) == 5
    assert all(row["record_count"] == 1 for row in summary_rows)
    assert isinstance(examples, list)
