import json
from pathlib import Path

APPROVED_VERTICALS = (
    "finance",
    "airline",
    "retail",
    "research_ai",
    "healthcare_admin",
)

SOURCE_SAMPLE_PATHS = {
    "finance": Path("data/real_world_samples/finance_sample.jsonl"),
    "airline": Path("data/real_world_samples/airline_sample.jsonl"),
    "retail": Path("data/real_world_samples/retail_sample.jsonl"),
    "research_ai": Path("data/real_world_samples/research_ai_sample.jsonl"),
    "healthcare_admin": Path("data/real_world_samples/healthcare_admin_sample.jsonl"),
}

KB_SAMPLE_PATHS = {
    vertical: Path(f"data/kb/{vertical}/kb_sample.jsonl") for vertical in APPROVED_VERTICALS
}

GOLD_SAMPLE_PATHS = {
    vertical: Path(f"data/eval/gold/{vertical}_gold_sample.jsonl")
    for vertical in APPROVED_VERTICALS
}

SCHEMA_PATHS = (
    Path("data/schemas/finance_prompt_record_schema.json"),
    Path("data/schemas/airline_ticket_record_schema.json"),
    Path("data/schemas/retail_support_record_schema.json"),
    Path("data/schemas/research_ai_prompt_record_schema.json"),
    Path("data/schemas/healthcare_admin_ticket_schema.json"),
)

SHARED_KB_REQUIRED_FIELDS = {
    "doc_id",
    "vertical",
    "title",
    "document_type",
    "source_type",
    "body",
    "version",
    "tags",
}

SHARED_GOLD_REQUIRED_FIELDS = {
    "prompt_id",
    "vertical",
    "task_type",
    "expected_status",
    "must_include",
    "must_not_include",
}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    assert path.exists(), f"Missing JSONL file: {path}"

    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed_record = json.loads(line)
        assert isinstance(parsed_record, dict), f"Expected object in {path}"
        records.append(parsed_record)

    assert records, f"Expected at least one record in {path}"
    return records


def test_vertical_contract_doc_exists() -> None:
    doc_path = Path("docs/31_phase2_vertical_data_contracts.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Phase 2 Vertical Data Contracts",
        "Finance Document QA",
        "Airline Customer Support",
        "Retail / E-commerce Support",
        "AI Research Assistant",
        "Healthcare Administrative Support",
        "source/prompt records",
        "KB/policy/context records",
        "gold/eval records",
    ]

    for term in required_terms:
        assert term in content


def test_vertical_schema_files_exist_and_parse() -> None:
    for schema_path in SCHEMA_PATHS:
        assert schema_path.exists(), f"Missing schema file: {schema_path}"
        json.loads(schema_path.read_text(encoding="utf-8"))


def test_required_schema_terms() -> None:
    schema_terms = {
        Path("data/schemas/finance_prompt_record_schema.json"): (
            "ticker",
            "source_doc_ids",
            "calculation_answer",
        ),
        Path("data/schemas/airline_ticket_record_schema.json"): (
            "Canada Air",
            "required_policy_ids",
            "spam_or_fraud",
        ),
        Path("data/schemas/retail_support_record_schema.json"): (
            "product_id",
            "issue_type",
            "expected_action",
        ),
        Path("data/schemas/research_ai_prompt_record_schema.json"): (
            "required_paper_ids",
            "required_chunk_ids",
            "compare_papers",
        ),
        Path("data/schemas/healthcare_admin_ticket_schema.json"): (
            "expected_queue",
            "safety_boundary",
            "privacy_sensitive",
        ),
    }

    for schema_path, terms in schema_terms.items():
        content = schema_path.read_text(encoding="utf-8")
        for term in terms:
            assert term in content


def test_jsonl_samples_parse() -> None:
    for path in SOURCE_SAMPLE_PATHS.values():
        for record in _read_jsonl(path):
            assert isinstance(record.get("prompt_id"), str)

    for path in KB_SAMPLE_PATHS.values():
        for record in _read_jsonl(path):
            assert isinstance(record.get("doc_id"), str)

    for path in GOLD_SAMPLE_PATHS.values():
        for record in _read_jsonl(path):
            assert isinstance(record.get("prompt_id"), str)


def test_every_vertical_has_three_assets() -> None:
    for vertical in APPROVED_VERTICALS:
        assert SOURCE_SAMPLE_PATHS[vertical].exists()
        assert KB_SAMPLE_PATHS[vertical].exists()
        assert GOLD_SAMPLE_PATHS[vertical].exists()


def test_kb_samples_have_shared_required_fields() -> None:
    for vertical, path in KB_SAMPLE_PATHS.items():
        for record in _read_jsonl(path):
            assert record.get("vertical") == vertical
            assert SHARED_KB_REQUIRED_FIELDS.issubset(record)


def test_gold_samples_have_shared_required_fields() -> None:
    for vertical, path in GOLD_SAMPLE_PATHS.items():
        for record in _read_jsonl(path):
            assert record.get("vertical") == vertical
            assert SHARED_GOLD_REQUIRED_FIELDS.issubset(record)
