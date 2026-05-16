import importlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast


def test_phase2_data_source_validation_matrix_doc_contains_required_terms() -> None:
    doc_path = Path("docs/30_phase2_data_source_validation_matrix.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Phase 2 Data Source Validation Matrix",
        "Finance Document QA",
        "Airline Customer Support",
        "Retail / E-commerce Support",
        "AI Research Assistant",
        "Healthcare Administrative Support",
        "Rejected or Deferred Sources",
        "Phase 2A Completion Gate",
        "Stack Overflow",
        "BigQuery",
    ]

    for term in required_terms:
        assert term in content


def test_data_readme_contains_storage_policy_terms() -> None:
    doc_path = Path("data/README.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Data Directory Policy",
        "What May Be Committed",
        "What Must Not Be Committed",
        "Three Data Assets Per Vertical",
        "data/raw/finance/sec/",
        "data/generated/airline/",
        "data/raw/research_ai/",
    ]

    for term in required_terms:
        assert term in content


def test_source_registry_exists_contains_sources_and_parses_when_yaml_available() -> None:
    registry_path = Path("data/sources/source_registry.yaml")

    assert registry_path.exists()

    content = registry_path.read_text(encoding="utf-8")
    required_terms = [
        "sources:",
        "finance_sec_edgar_xbrl",
        "airline_canada_air_synthetic",
        "retail_amazon_reviews_2023",
        "research_ai_papers",
        "healthcare_admin_synthetic",
    ]

    for term in required_terms:
        assert term in content

    if importlib.util.find_spec("yaml") is None:
        return

    yaml_module = importlib.import_module("yaml")
    safe_load = cast(Callable[[str], object], yaml_module.__dict__["safe_load"])
    parsed_yaml = safe_load(content)

    assert isinstance(parsed_yaml, dict)
    sources = parsed_yaml.get("sources")
    assert isinstance(sources, list)
    source_ids = {source.get("source_id") for source in sources if isinstance(source, dict)}
    assert source_ids == {
        "finance_sec_edgar_xbrl",
        "airline_canada_air_synthetic",
        "retail_amazon_reviews_2023",
        "research_ai_papers",
        "healthcare_admin_synthetic",
    }


def test_kb_document_schema_exists_parses_and_contains_required_terms() -> None:
    schema_path = Path("data/kb/schema/kb_document_schema.json")

    assert schema_path.exists()

    content = schema_path.read_text(encoding="utf-8")
    json.loads(content)

    required_terms = [
        "doc_id",
        "vertical",
        "document_type",
        "research_paper_section",
        "healthcare_admin_policy",
        "sec_filing_section",
        "xbrl_fact_table",
    ]

    for term in required_terms:
        assert term in content


def test_gold_record_schema_exists_parses_and_contains_required_terms() -> None:
    schema_path = Path("data/eval/schema/gold_record_schema.json")

    assert schema_path.exists()

    content = schema_path.read_text(encoding="utf-8")
    json.loads(content)

    required_terms = [
        "prompt_id",
        "task_type",
        "required_doc_ids",
        "required_chunk_ids",
        "expected_status",
        "calculation_answer",
        "boundary_response",
        "compare_papers",
        "policy_lookup",
    ]

    for term in required_terms:
        assert term in content
