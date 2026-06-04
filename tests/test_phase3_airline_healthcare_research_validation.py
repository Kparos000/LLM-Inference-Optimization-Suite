from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from inference_bench.airline_healthcare_research_validation import (
    build_airline_healthcare_research_validation,
)
from inference_bench.context_schema import ContextRecord
from inference_bench.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalFallbackDenseRetriever,
)
from inference_bench.vertical_retrieval_repair import (
    DIRECT_HINT_RE,
    enrich_prompt_metadata,
    repaired_query,
)

VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")


def context_record(
    vertical: str,
    context_id: str,
    text: str,
    metadata: dict[str, object],
    title: str | None = None,
) -> ContextRecord:
    return ContextRecord(
        context_id=f"{vertical}:{context_id}",
        vertical=vertical,
        source_id=f"{vertical}_fixture_source",
        parent_id=f"{vertical}_fixture_parent",
        chunk_id=context_id,
        chunk_strategy=f"{vertical}_fixture_chunking",
        source_type=str(metadata.get("document_type") or "fixture_policy"),
        title=title or context_id,
        text=text,
        metadata=metadata,
        token_estimate=len(text.split()),
        provenance=f"{vertical}_fixture",
        is_gold_linked=True,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_airline_enrichment_expands_policy_terms_without_direct_ids() -> None:
    prompt = {
        "prompt_id": "airline_fixture",
        "vertical": "airline",
        "question": "A traveler has a baggage damage issue on Canada Air records CA-POL-008.",
        "support_type": "baggage_damage",
        "route": "YYZ-YVR",
        "travel_type": "domestic",
    }
    enrichment = enrich_prompt_metadata(vertical="airline", prompt=prompt)
    query, _expanded, _types, blocked_count = repaired_query(
        prompt=prompt,
        enrichment=enrichment,
        resolver=None,
        concept_map={},
        ablation_mode="prompt_plus_metadata",
    )

    assert "damaged bag" in " ".join(enrichment.query_terms)
    assert "CA-POL" not in query
    assert blocked_count > 0
    assert DIRECT_HINT_RE.search(query) is None


def test_airline_reranker_prefers_matching_policy_family() -> None:
    target = context_record(
        "airline",
        "airline_baggage_damage",
        "Baggage damage claims require a damage report and checked bag documentation.",
        {
            "document_type": "policy",
            "original_doc_id": "airline_baggage_damage",
            "policy_tags": ["airline", "baggage", "damage"],
        },
        title="Baggage Damage Policy",
    )
    distractor = context_record(
        "airline",
        "airline_passport",
        "International passport and visa documentation guidance for entry documents.",
        {
            "document_type": "compliance_note",
            "original_doc_id": "airline_passport",
            "policy_tags": ["airline", "visa", "passport", "documentation"],
        },
        title="Passport and Visa Responsibility Policy",
    )

    retrieval = HybridRetriever(
        BM25Retriever([target, distractor]),
        LocalFallbackDenseRetriever([target, distractor]),
    ).retrieve("baggage_damage damaged bag claim checked bag", top_k=2)

    assert retrieval.results[0].context_record.context_id == target.context_id
    assert retrieval.results[0].component_scores["airline_primary_family_match"] == 1.0


def test_healthcare_enrichment_expands_admin_boundary_terms_without_ids() -> None:
    prompt = {
        "prompt_id": "healthcare_fixture",
        "vertical": "healthcare_admin",
        "question": "A patient asks about medical records request using MCH-POL-008.",
        "support_type": "medical_records_request",
        "department": "records",
        "safety_boundary": "privacy_sensitive",
        "privacy_sensitive": True,
    }
    enrichment = enrich_prompt_metadata(vertical="healthcare_admin", prompt=prompt)
    query, _expanded, _types, blocked_count = repaired_query(
        prompt=prompt,
        enrichment=enrichment,
        resolver=None,
        concept_map={},
        ablation_mode="prompt_plus_metadata",
    )

    joined_terms = " ".join(enrichment.query_terms)
    assert "identity verification" in joined_terms
    assert "privacy review" in joined_terms
    assert "MCH-POL" not in query
    assert blocked_count > 0
    assert DIRECT_HINT_RE.search(query) is None


def test_healthcare_reranker_prefers_matching_admin_procedure() -> None:
    target = context_record(
        "healthcare_admin",
        "healthcare_records",
        (
            "Medical records release requires authorization, identity verification, "
            "and privacy review."
        ),
        {
            "document_type": "healthcare_admin_policy",
            "original_doc_id": "healthcare_records",
            "tags": ["healthcare-admin", "medical-records", "privacy"],
        },
        title="Medical Records Request Policy",
    )
    distractor = context_record(
        "healthcare_admin",
        "healthcare_appointment",
        "Appointment booking requests capture preferred clinic and date range.",
        {
            "document_type": "healthcare_admin_policy",
            "original_doc_id": "healthcare_appointment",
            "tags": ["healthcare-admin", "appointment", "scheduling"],
        },
        title="Appointment Booking Policy",
    )

    retrieval = HybridRetriever(
        BM25Retriever([target, distractor]),
        LocalFallbackDenseRetriever([target, distractor]),
    ).retrieve("medical_records_request records release privacy review", top_k=2)

    assert retrieval.results[0].context_record.context_id == target.context_id
    assert retrieval.results[0].component_scores["healthcare_primary_family_match"] == 1.0


def test_block16b_report_generation_on_fixture(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    context_root = tmp_path / "context"
    output_root = tmp_path / "reports"

    prompt_rows: dict[str, dict[str, Any]] = {
        "airline": {
            "prompt_id": "airline_001",
            "vertical": "airline",
            "question": "A traveler asks about baggage damage.",
            "support_type": "baggage_damage",
            "route": "YYZ-YVR",
            "travel_type": "domestic",
        },
        "healthcare_admin": {
            "prompt_id": "healthcare_001",
            "vertical": "healthcare_admin",
            "question": "A patient asks about medical records request.",
            "support_type": "medical_records_request",
            "department": "records",
            "safety_boundary": "privacy_sensitive",
            "privacy_sensitive": True,
        },
        "retail": {
            "prompt_id": "retail_001",
            "vertical": "retail",
            "question": "Summarize product reviews.",
            "category": "Electronics",
            "product_title": "Test Speaker",
            "issue_type": "review_summary",
        },
        "finance": {
            "prompt_id": "finance_001",
            "vertical": "finance",
            "question": "What did Apple report about revenue in fiscal year 2024?",
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "filing_form": "10-K",
        },
        "research_ai": {
            "prompt_id": "research_001",
            "vertical": "research_ai",
            "question": "What method does the paper describe?",
            "topic": "llm inference",
            "metadata": {"source_titles": ["Fixture Paper"], "evidence_type": ["method"]},
        },
    }
    context_rows: dict[str, ContextRecord] = {
        "airline": context_record(
            "airline",
            "airline_baggage_damage",
            "Baggage damage claims require a damage report.",
            {"original_doc_id": "airline_baggage_damage", "policy_tags": ["baggage", "damage"]},
            title="Baggage Damage Policy",
        ),
        "healthcare_admin": context_record(
            "healthcare_admin",
            "healthcare_records",
            "Medical records release requires authorization and identity verification.",
            {"original_doc_id": "healthcare_records", "tags": ["medical-records", "privacy"]},
            title="Medical Records Request Policy",
        ),
        "retail": context_record(
            "retail",
            "retail_reviews",
            "Test Speaker reviews summarize rating signals.",
            {"original_doc_id": "retail_reviews", "product_title": "Test Speaker"},
            title="Test Speaker Review Summary",
        ),
        "finance": context_record(
            "finance",
            "finance_revenue",
            "Apple Inc. Form 10-K reported fiscal year 2024 revenue.",
            {
                "original_doc_id": "finance_revenue",
                "ticker": "AAPL",
                "company_name": "Apple Inc.",
                "form": "10-K",
                "concept": "Revenue",
            },
            title="Apple Revenue Filing Fact",
        ),
        "research_ai": context_record(
            "research_ai",
            "research_method",
            "The paper method improves long context inference.",
            {
                "original_doc_id": "research_method",
                "paper_title": "Fixture Paper",
                "section_type": "method",
            },
            title="Fixture Paper Method",
        ),
    }

    for vertical in VERTICALS:
        prompt = prompt_rows[vertical]
        evidence_id = str(context_rows[vertical].metadata["original_doc_id"])
        write_jsonl(
            dataset_root / vertical / f"{vertical}_prompts_2000.jsonl",
            [prompt],
        )
        write_jsonl(
            dataset_root / vertical / f"{vertical}_gold_2000.jsonl",
            [
                {
                    "prompt_id": prompt["prompt_id"],
                    "vertical": vertical,
                    "required_doc_ids": [evidence_id],
                    "required_evidence_ids": [evidence_id],
                }
            ],
        )
        write_jsonl(
            context_root / "corpora" / f"{vertical}_context_corpus.jsonl",
            [asdict(context_rows[vertical])],
        )

    report = build_airline_healthcare_research_validation(
        dataset_root=dataset_root,
        context_root=context_root,
        output_root=output_root,
        slo_config_path="configs/slo_targets.yaml",
        stage_sizes=[1],
        research_stage_sizes=[1],
        dense_backend="local_fallback",
        allow_dense_fallback=True,
    )

    assert report["airline_healthcare_enrichment"]["no_model_inference_triggered"] is True
    assert report["research_ai_scale_validation"]["direct_hint_leakage_detected_count"] == 0
    assert (output_root / "airline_healthcare_enrichment_report.json").exists()
    assert (output_root / "research_ai_scale_validation_summary.csv").exists()
