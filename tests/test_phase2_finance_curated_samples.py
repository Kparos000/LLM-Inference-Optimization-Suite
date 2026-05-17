import json
from collections import Counter
from pathlib import Path
from typing import Any

PROMPT_PATH = Path("data/real_world_samples/finance_sample.jsonl")
KB_PATH = Path("data/kb/finance/kb_sample.jsonl")
GOLD_PATH = Path("data/eval/gold/finance_gold_sample.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def test_finance_curated_sample_files_exist() -> None:
    assert PROMPT_PATH.exists()
    assert KB_PATH.exists()
    assert GOLD_PATH.exists()


def test_finance_prompt_records_parse_and_count() -> None:
    records = _read_jsonl(PROMPT_PATH)
    required_fields = {
        "prompt_id",
        "vertical",
        "company",
        "ticker",
        "source_doc_ids",
        "task_type",
        "question",
        "expected_output_format",
        "expected_status",
    }

    assert len(records) == 40
    assert len({record["prompt_id"] for record in records}) == 40
    for record in records:
        assert required_fields.issubset(record)
        assert record["vertical"] == "finance"
        assert isinstance(record["source_doc_ids"], list)
        assert record["source_doc_ids"]
        assert record["metadata"]["prompt_category"]


def test_finance_prompt_distribution() -> None:
    records = _read_jsonl(PROMPT_PATH)
    task_counts = Counter(record["task_type"] for record in records)
    category_counts = Counter(record["metadata"]["prompt_category"] for record in records)

    assert task_counts["answer_short"] == 8
    assert task_counts["answer_grounded"] >= 8
    assert task_counts["extract_structured"] == 6
    assert task_counts["trend_summary"] == 5
    assert task_counts["compare_companies"] == 4
    assert task_counts["calculation_answer"] == 3
    assert task_counts["escalation_response"] == 2

    assert category_counts == {
        "direct_numeric_fact": 8,
        "single_document_grounded_qa": 6,
        "structured_json_extraction": 6,
        "trend_analysis": 5,
        "cross_company_comparison": 4,
        "summarization_risk_mda": 4,
        "calculation": 3,
        "evidence_citation_lookup": 2,
        "escalation_insufficient_evidence": 2,
    }


def test_finance_kb_records_parse_and_minimum_count() -> None:
    records = _read_jsonl(KB_PATH)
    required_fields = {
        "doc_id",
        "vertical",
        "title",
        "document_type",
        "source_type",
        "body",
        "version",
        "tags",
    }

    assert len(records) >= 25
    assert len({record["doc_id"] for record in records}) == len(records)
    for record in records:
        assert required_fields.issubset(record)
        assert record["vertical"] == "finance"
        assert record["body"]
        assert isinstance(record["tags"], list)

    document_types = {record["document_type"] for record in records}
    assert "sec_filing_section" in document_types
    assert "xbrl_fact_table" in document_types


def test_finance_gold_records_parse_and_align_with_prompts() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    gold_records = _read_jsonl(GOLD_PATH)
    prompt_ids = {record["prompt_id"] for record in prompts}
    gold_prompt_ids = [record["prompt_id"] for record in gold_records]
    required_fields = {
        "prompt_id",
        "vertical",
        "task_type",
        "expected_status",
        "must_include",
        "must_not_include",
    }

    assert len(gold_records) == 40
    assert set(gold_prompt_ids) == prompt_ids
    assert len(gold_prompt_ids) == len(set(gold_prompt_ids))
    for record in gold_records:
        assert required_fields.issubset(record)
        assert record["vertical"] == "finance"
        assert isinstance(record["must_include"], list)
        assert isinstance(record["must_not_include"], list)


def test_finance_gold_escalation_ratio() -> None:
    gold_records = _read_jsonl(GOLD_PATH)
    escalation_records = [
        record
        for record in gold_records
        if record.get("expected_status") in {"escalate", "insufficient_evidence"}
        or record.get("expected_escalation") is True
    ]

    assert len(escalation_records) == 2
    assert len(gold_records) - len(escalation_records) == 38


def test_finance_gold_numeric_records_have_numeric_answers() -> None:
    gold_records = _read_jsonl(GOLD_PATH)
    for record in gold_records:
        category = record["metadata"]["prompt_category"]
        if category == "direct_numeric_fact":
            assert "numeric_answer" in record
            assert "tolerance" in record
        if category == "calculation":
            assert "numeric_answer" in record
            assert "formula" in record
            assert "tolerance" in record


def test_finance_structured_json_prompts_have_json_output() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    gold_by_prompt = {record["prompt_id"]: record for record in _read_jsonl(GOLD_PATH)}
    expected_json_keys = {"ticker", "filing_form", "filing_date", "section_type"}

    for prompt in prompts:
        if prompt["metadata"]["prompt_category"] != "structured_json_extraction":
            continue
        assert prompt["expected_output_format"] == "json"
        gold = gold_by_prompt[prompt["prompt_id"]]
        assert expected_json_keys & set(gold["must_include"])


def test_finance_curated_samples_do_not_reference_private_paths() -> None:
    forbidden_terms = [
        "C:\\Users",
        "/home/",
        "akpoogaga",
        "kparo",
        "API key",
        "token",
    ]
    combined_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (PROMPT_PATH, KB_PATH, GOLD_PATH)
    )

    for term in forbidden_terms:
        assert term not in combined_text
