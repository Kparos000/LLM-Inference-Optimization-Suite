import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "data/real_world_samples/retail_sample.jsonl"
KB_PATH = ROOT / "data/kb/retail/kb_sample.jsonl"
GOLD_PATH = ROOT / "data/eval/gold/retail_gold_sample.jsonl"
SCRIPT_PATH = ROOT / "scripts/phase2/curate_retail_seed.py"
DOC_PATH = ROOT / "docs/37_phase2_retail_curated_seed.md"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def _load_curation_module() -> Any:
    spec = importlib.util.spec_from_file_location("curate_retail_seed", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retail_curated_files_exist() -> None:
    assert PROMPT_PATH.exists()
    assert KB_PATH.exists()
    assert GOLD_PATH.exists()


def test_retail_prompt_count_and_required_fields() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    required = {
        "prompt_id",
        "vertical",
        "task_type",
        "question",
        "expected_output_format",
        "expected_status",
        "required_evidence_ids",
        "source_parent_asins",
    }

    assert len(prompts) == 40
    assert len({record["prompt_id"] for record in prompts}) == 40
    for record in prompts:
        assert required.issubset(record)
        assert record["vertical"] == "retail"


def test_retail_prompt_distribution() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    counts = Counter(record["metadata"]["prompt_category"] for record in prompts)

    assert counts == {
        "review_summary": 6,
        "issue_identification": 7,
        "compare_products": 5,
        "structured_extraction": 6,
        "support_policy_reasoning": 5,
        "evidence_citation_lookup": 4,
        "spam_or_low_quality_review": 3,
        "insufficient_evidence_or_escalation": 3,
        "out_of_scope": 1,
    }


def test_retail_status_distribution() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    status_counts = Counter(record["expected_status"] for record in prompts)
    category_statuses: dict[str, Counter[str]] = {}
    for record in prompts:
        category = record["metadata"]["prompt_category"]
        category_statuses.setdefault(category, Counter())[record["expected_status"]] += 1

    assert status_counts["answer"] == 33
    assert sum(category_statuses["spam_or_low_quality_review"].values()) == 3
    assert set(category_statuses["spam_or_low_quality_review"]).issubset(
        {"spam_or_low_quality", "escalate"}
    )
    assert sum(category_statuses["insufficient_evidence_or_escalation"].values()) == 3
    assert set(category_statuses["insufficient_evidence_or_escalation"]).issubset(
        {"insufficient_evidence", "escalate"}
    )
    assert status_counts["out_of_scope"] == 1


def test_retail_kb_minimum_and_required_fields() -> None:
    records = _read_jsonl(KB_PATH)
    required = {
        "doc_id",
        "vertical",
        "title",
        "document_type",
        "source_type",
        "body",
        "version",
        "tags",
    }
    document_types = {record["document_type"] for record in records}

    assert len(records) >= 40
    assert len({record["doc_id"] for record in records}) == len(records)
    for record in records:
        assert required.issubset(record)
        assert record["vertical"] == "retail"
    assert "product_metadata" in document_types
    assert "review_evidence" in document_types
    assert "support_policy" in document_types


def test_retail_gold_alignment() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    gold = _read_jsonl(GOLD_PATH)
    prompt_ids = {record["prompt_id"] for record in prompts}
    required = {
        "prompt_id",
        "vertical",
        "task_type",
        "expected_status",
        "must_include",
        "must_not_include",
    }

    assert len(gold) == 40
    assert len({record["prompt_id"] for record in gold}) == 40
    assert {record["prompt_id"] for record in gold} == prompt_ids
    for record in gold:
        assert required.issubset(record)


def test_retail_answerable_gold_has_evidence() -> None:
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        if record["expected_status"] == "answer":
            assert record.get("required_doc_ids")
            assert record.get("required_citations") or record.get("required_chunk_ids")
            assert record.get("must_include")


def test_retail_structured_prompts_have_json_output() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    gold_by_id = {record["prompt_id"]: record for record in _read_jsonl(GOLD_PATH)}
    required_keys = {
        "product_id",
        "product_title",
        "issue_type",
        "rating",
        "evidence_summary",
        "recommended_action",
        "evidence_id",
    }

    for prompt in prompts:
        if prompt["metadata"]["prompt_category"] != "structured_extraction":
            continue
        assert prompt["expected_output_format"] == "json"
        gold = gold_by_id[prompt["prompt_id"]]
        assert required_keys.issubset(set(gold["must_include"]))
        parsed = json.loads(gold["reference_answer"])
        assert required_keys.issubset(parsed)


def test_retail_negative_status_records() -> None:
    gold = _read_jsonl(GOLD_PATH)
    negative = [
        record
        for record in gold
        if record["expected_status"]
        in {"out_of_scope", "insufficient_evidence", "escalate", "spam_or_low_quality"}
    ]

    assert any(record["expected_status"] == "out_of_scope" for record in negative)
    assert any(record["expected_status"] == "insufficient_evidence" for record in negative)
    assert any(record["expected_status"] == "spam_or_low_quality" for record in negative)
    for record in negative:
        answer = str(record.get("reference_answer") or "").lower()
        if record["expected_status"] == "out_of_scope":
            assert "outside the retail support corpus" in answer
            assert "general model memory" in " ".join(record["must_not_include"]).lower()
        if record["expected_status"] in {"insufficient_evidence", "escalate"}:
            assert "insufficient" in answer or "escalate" in answer or "not enough" in answer
        if record["expected_status"] == "spam_or_low_quality":
            assert "low-quality" in answer


def test_retail_no_raw_user_ids_or_private_paths() -> None:
    forbidden = ["user_id", "C:\\Users", "/home/", "akpoogaga", "kparo", "token", "API key"]
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in (PROMPT_PATH, KB_PATH, GOLD_PATH)
    )
    lowered = content.lower()

    for term in forbidden:
        assert term.lower() not in lowered
    assert not re.search(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", content)
    assert not re.search(r"\b(?:\+?\d[\s().-]*){7,}\b", content)


def test_retail_policy_records_marked_synthetic() -> None:
    policies = [
        record for record in _read_jsonl(KB_PATH) if record["document_type"] == "support_policy"
    ]

    assert policies
    for record in policies:
        body = record["body"]
        assert "synthetic benchmark policy" in body
        assert "not Amazon policy" in body
        assert record["metadata"]["synthetic_benchmark_policy"] is True
        assert record["metadata"]["not_amazon_policy"] is True


def test_retail_curation_report_shape() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    kb_records = _read_jsonl(KB_PATH)
    gold_records = _read_jsonl(GOLD_PATH)
    report = _load_curation_module().build_curation_report(
        prompts,
        kb_records,
        gold_records,
        1000,
        1000,
    )

    assert report["prompt_record_count"] == 40
    assert report["kb_record_count"] >= 40
    assert report["gold_record_count"] == 40
    assert report["next_step"]


def test_docs_include_retail_curated_seed() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")
    lowered = docs.lower()

    assert "Phase 2A-6C Retail Curated Seed" in docs
    assert "support policy" in lowered
    assert "spam_or_low_quality_review" in docs
    assert "rag" in lowered
