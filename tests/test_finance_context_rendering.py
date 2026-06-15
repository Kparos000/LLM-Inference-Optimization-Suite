from __future__ import annotations

import json

from inference_bench.context_alignment_repair import (
    AlignmentSelection,
    runner_item_from_alignment,
)
from inference_bench.context_schema import ContextRecord, WorkloadRecord


def _finance_context() -> ContextRecord:
    return ContextRecord(
        context_id="finance:finance_kb_secret",
        vertical="finance",
        source_id="finance_source_secret",
        parent_id="finance_doc_secret",
        chunk_id="finance_kb_secret",
        chunk_strategy="finance_atomic",
        source_type="xbrl_fact_evidence",
        title="AAPL revenue evidence finance_kb_secret",
        text=(
            "Apple revenue was 100 USD for fiscal year 2025 from accession 0000320193-25-000079."
        ),
        metadata={
            "original_doc_id": "finance_kb_secret",
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "form": "10-K",
            "fiscal_year": "2025",
            "concept": "Revenue",
            "section_title": "Financial Statements",
            "section_type": "financial_statements",
        },
        token_estimate=20,
        provenance="fixture",
        is_gold_linked=True,
    )


def _workload(context: ContextRecord) -> WorkloadRecord:
    return WorkloadRecord(
        workload_id="fixture:finance",
        prompt_id="finance-1",
        vertical="finance",
        memory_mode="mm2_hybrid_top5",
        messages=[{"role": "user", "content": "Question"}],
        context_records=[context],
        context_token_estimate=20,
        retrieval_metadata={},
        expected_output_format="text",
        gold_evidence_ids=["finance_kb_secret"],
        dataset_split="test_fixture",
        source_prompt_record={},
    )


def test_finance_rendering_includes_safe_metadata_and_hides_source_ids() -> None:
    context = _finance_context()
    selection = AlignmentSelection(
        contexts=(context,),
        private_alias_map={"E1": ["finance_kb_secret", "finance:finance_kb_secret"]},
        expected_ids=("finance_kb_secret",),
        represented_ids=("finance_kb_secret",),
        missing_ids=(),
        family_alias_bindings={},
        status="all",
        changed=False,
    )

    item = runner_item_from_alignment(
        source_workload=_workload(context),
        source_prompt={
            "question": "Summarize finance_kb_secret for Apple.",
            "ticker": "AAPL",
        },
        selection=selection,
        retrieval_metadata={},
    )

    assert "ticker: AAPL" in item.prompt
    assert "company: Apple Inc." in item.prompt
    assert "filing_form: 10-K" in item.prompt
    assert "period: 2025" in item.prompt
    assert "metric: Revenue" in item.prompt
    assert "section_title: Financial Statements" in item.prompt
    assert "finance_kb_secret" not in item.prompt
    assert "0000320193-25-000079" not in item.prompt
    assert "citation_aliases:" not in item.prompt
    aliases = json.loads(item.metadata["citation_id_aliases"])
    assert aliases["E1"][0] == "finance_kb_secret"
    assert item.metadata["canonical_ids_exposed_to_model"] == "false"
