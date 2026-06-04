from typing import Any

from inference_bench.canonical_queries import build_canonical_query
from inference_bench.context_schema import ContextRecord
from inference_bench.retrieval_keys import derive_retrieval_keys, retrieval_key_terms
from inference_bench.slo import load_slo_config
from inference_bench.vertical_final_selectors import (
    RankedCandidate,
    select_canonical_final_candidates,
)
from inference_bench.vertical_retrieval_repair import enrich_prompt_metadata, validate_stages


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
                "Compact Bluetooth Speaker reviews mention support signal and refund.",
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


def test_canonical_keys_exclude_gold_and_source_identifiers() -> None:
    prompt = {
        "prompt_id": "airline_001",
        "vertical": "airline",
        "question": "Use CA-POL-001 to answer refund policy.",
        "support_type": "refund",
        "required_evidence_ids": ["CA-POL-001"],
        "source_id": "airline_policy_source_001",
        "parent_id": "hidden_parent",
    }

    keys = derive_retrieval_keys(prompt, ablation_mode="prompt_plus_metadata")
    terms = " ".join(retrieval_key_terms(keys))

    assert "CA-POL" not in terms
    assert "source_id" not in terms
    assert "hidden_parent" not in terms
    assert "required_evidence_ids" not in keys.values


def test_canonical_query_uses_allowed_keys_without_direct_ids() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "vertical": "finance",
        "question": "Summarize Apple revenue in fiscal year 2024 from the 10-K.",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "filing_form": "10-K",
        "required_doc_ids": ["finance_kb_sec_AAPL_10K_hidden"],
    }

    query = build_canonical_query(prompt, ablation_mode="prompt_plus_metadata")

    assert "Apple" in query.qdrant_query
    assert "revenue" in query.qdrant_query.lower()
    assert "2024" in query.qdrant_query
    assert "finance_kb_sec" not in query.qdrant_query
    assert "required_doc_ids" not in query.qdrant_query


def test_finance_keys_materialize_company_metric_period_filing() -> None:
    prompt = {
        "prompt_id": "finance_001",
        "vertical": "finance",
        "question": "What did Apple report about revenue in fiscal year 2024?",
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "filing_form": "10-K",
    }

    keys = derive_retrieval_keys(prompt, ablation_mode="prompt_plus_metadata")

    assert keys.values["company"] == "Apple Inc."
    assert keys.values["ticker"] == "AAPL"
    assert keys.values["filing_type"] == "10-K"
    assert keys.values["metric_family"] == "revenue"
    assert keys.values["fiscal_year"] == "2024"


def test_retail_selector_prioritizes_matching_product_review_intent() -> None:
    records = {
        "wrong": context_record(
            "retail",
            "wrong",
            "Different headphones policy information.",
            {
                "product_title": "Wireless Headphones",
                "category": "Electronics",
                "evidence_type": "policy",
            },
        ),
        "right": context_record(
            "retail",
            "right",
            "Compact Bluetooth Speaker review mentions poor battery support signal.",
            {
                "product_title": "Compact Bluetooth Speaker",
                "category": "Electronics",
                "evidence_type": "review",
            },
        ),
    }
    ranked: list[RankedCandidate] = [("wrong", 10.0, {}), ("right", 9.0, {})]

    selected = select_canonical_final_candidates(
        ranked=ranked,
        records_by_id=records,
        query_tokens={"compact", "bluetooth", "speaker", "review", "support"},
        query_text="Compact Bluetooth Speaker review support signal",
        final_top_k=1,
    )

    assert selected[0][0] == "right"


def test_research_selector_prioritizes_target_section_without_forced_paper_skip() -> None:
    records = {
        "same_paper_generic": context_record(
            "research_ai",
            "same_paper_generic",
            "Generic introduction about efficient inference.",
            {"paper_title": "FastKV", "paper_id": "paper_a", "section_type": "introduction"},
        ),
        "method": context_record(
            "research_ai",
            "method",
            "Method section describes long context KV compression.",
            {"paper_title": "FastKV", "paper_id": "paper_a", "section_type": "method"},
        ),
        "other": context_record(
            "research_ai",
            "other",
            "Another paper results section.",
            {"paper_title": "OtherKV", "paper_id": "paper_b", "section_type": "results"},
        ),
    }
    ranked: list[RankedCandidate] = [
        ("same_paper_generic", 10.0, {}),
        ("method", 9.9, {}),
        ("other", 9.0, {}),
    ]

    selected = select_canonical_final_candidates(
        ranked=ranked,
        records_by_id=records,
        query_tokens={"fastkv", "method", "long", "context"},
        query_text="FastKV method long context",
        final_top_k=2,
    )

    assert selected[0][0] == "method"
    assert selected[1][0] in {"same_paper_generic", "other"}


def test_airline_selector_improves_policy_issue_mrr_ordering() -> None:
    records = {
        "generic": context_record(
            "airline",
            "generic",
            "General booking policy.",
            {"policy_tags": ["booking"]},
        ),
        "refund": context_record(
            "airline",
            "refund",
            "Cancellation refund policy with travel credit rules.",
            {"policy_tags": ["cancellation", "refund"]},
        ),
    }
    ranked: list[RankedCandidate] = [("generic", 10.0, {}), ("refund", 9.8, {})]

    selected = select_canonical_final_candidates(
        ranked=ranked,
        records_by_id=records,
        query_tokens={"cancellation", "refund", "policy"},
        query_text="cancellation refund policy",
        final_top_k=1,
    )

    assert selected[0][0] == "refund"


def test_healthcare_selector_keeps_good_admin_path_conservative() -> None:
    records = {
        "appointment": context_record(
            "healthcare_admin",
            "appointment",
            "Appointment booking procedure with administrative privacy checks.",
            {"support_type": "appointment_booking", "department": "scheduling"},
        ),
        "clinical": context_record(
            "healthcare_admin",
            "clinical",
            "Clinical triage boundary and urgent symptoms.",
            {"support_type": "clinical_boundary", "department": "clinical_staff_review"},
        ),
    }
    ranked: list[RankedCandidate] = [("appointment", 10.0, {}), ("clinical", 9.8, {})]

    selected = select_canonical_final_candidates(
        ranked=ranked,
        records_by_id=records,
        query_tokens={"appointment", "booking", "privacy"},
        query_text="appointment booking privacy administrative procedure",
        final_top_k=1,
    )

    assert selected[0][0] == "appointment"


def test_canonical_staged_validation_updates_report_shape_without_inference() -> None:
    prompts, gold, corpora = fixture_data()
    slo_config = load_slo_config("configs/slo_targets.yaml")
    enrichments = {
        vertical: {
            str(rows[0]["prompt_id"]): enrich_prompt_metadata(
                vertical=vertical,
                prompt=rows[0],
            )
        }
        for vertical, rows in prompts.items()
    }

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
        use_canonical_retrieval_keys=True,
    )

    assert report["use_canonical_retrieval_keys"] is True
    assert report["no_model_inference_triggered"] is True
    assert report["no_external_api_calls_triggered"] is True
    assert report["direct_hint_leakage_detected_count"] == 0
    assert all(row["measurement"] == "canonical_key_repair_staged" for row in summary_rows)
    assert isinstance(examples, list)
