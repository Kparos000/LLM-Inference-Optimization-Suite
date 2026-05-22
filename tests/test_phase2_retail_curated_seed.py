import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


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


def test_build_product_metadata_index_uses_parent_asin() -> None:
    module = _load_curation_module()
    row = {"parent_asin": "B000TEST1", "title": "Rosemary Mint Hair Oil"}

    index = module.build_product_metadata_index([row])

    assert index["B000TEST1"] == row


def test_resolve_product_title_prefers_metadata_title() -> None:
    module = _load_curation_module()
    index = module.build_product_metadata_index(
        [{"parent_asin": "B000TEST1", "title": "Rosemary Mint Hair Oil"}]
    )

    title, resolution = module.resolve_product_title(
        "B000TEST1",
        None,
        index,
        "All_Beauty",
    )

    assert title == "Rosemary Mint Hair Oil"
    assert resolution["title_resolution"] == "metadata_title"
    assert resolution["metadata_found"] is True


def test_resolve_product_title_tracks_generic_fallback() -> None:
    module = _load_curation_module()

    title, resolution = module.resolve_product_title("B000TEST1", None, {}, "All_Beauty")

    assert "B000TEST1" in title
    assert resolution["title_resolution"] == "generic_fallback"
    assert resolution["metadata_found"] is False


def test_is_generic_retail_title() -> None:
    module = _load_curation_module()

    assert module.is_generic_retail_title("All_Beauty product B081TJ8YS3", "All_Beauty")
    assert module.is_generic_retail_title("Retail product B000TEST", "All_Beauty")
    assert not module.is_generic_retail_title("Rosemary Mint Hair Oil", "All_Beauty")


def test_extract_selected_parent_asins_from_seed(tmp_path: Path) -> None:
    module = _load_curation_module()
    prompt_path = tmp_path / "prompts.jsonl"
    kb_path = tmp_path / "kb.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(
        prompt_path,
        [{"source_parent_asins": ["B000TEST1"], "source_product_ids": ["B000TEST2"]}],
    )
    _write_jsonl(kb_path, [{"metadata": {"parent_asin": "B000TEST3", "asin": "B000TEST4"}}])
    _write_jsonl(gold_path, [{"metadata": {"required_parent_asins": ["B000TEST5"]}}])

    selected = module.extract_selected_parent_asins_from_seed(prompt_path, kb_path, gold_path)

    assert selected == {"B000TEST1", "B000TEST2", "B000TEST3", "B000TEST4", "B000TEST5"}


def test_targeted_metadata_merge_takes_precedence() -> None:
    module = _load_curation_module()
    base_rows = [{"parent_asin": "B000TEST1", "title": ""}]
    targeted_rows = [{"parent_asin": "B000TEST1", "title": "Rosemary Mint Hair Oil"}]

    merged = module.merge_metadata_rows(base_rows, targeted_rows)
    index = module.build_product_metadata_index(merged)
    title, resolution = module.resolve_product_title(
        "B000TEST1",
        None,
        index,
        "All_Beauty",
    )

    assert title == "Rosemary Mint Hair Oil"
    assert resolution["title_resolution"] == "metadata_title"


def test_enrich_selected_metadata_matches_local_rows(tmp_path: Path) -> None:
    module = _load_curation_module()
    prompt_path = tmp_path / "prompts.jsonl"
    kb_path = tmp_path / "kb.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    targeted_output = tmp_path / "targeted.jsonl"
    selected_output = tmp_path / "selected.txt"
    report_path = tmp_path / "report.json"
    _write_jsonl(prompt_path, [{"source_parent_asins": ["B000TEST1"]}])
    _write_jsonl(kb_path, [])
    _write_jsonl(gold_path, [])
    _write_jsonl(metadata_path, [{"parent_asin": "B000TEST1", "title": "Rosemary Mint Hair Oil"}])

    summary = module.enrich_selected_metadata(
        SimpleNamespace(
            retail_prompts=prompt_path,
            retail_kb=kb_path,
            retail_gold=gold_path,
            metadata_input=metadata_path,
            targeted_metadata_output=targeted_output,
            selected_parent_asins_output=selected_output,
            targeted_report=report_path,
            metadata_source_path=None,
            source_scan_batch_limit=None,
        )
    )

    rows = _read_jsonl(targeted_output)
    assert summary["matched_metadata_count"] == 1
    assert rows[0]["title"] == "Rosemary Mint Hair Oil"


def test_enrich_selected_metadata_reports_unmatched(tmp_path: Path) -> None:
    module = _load_curation_module()
    prompt_path = tmp_path / "prompts.jsonl"
    kb_path = tmp_path / "kb.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    metadata_path = tmp_path / "metadata.jsonl"
    targeted_output = tmp_path / "targeted.jsonl"
    selected_output = tmp_path / "selected.txt"
    report_path = tmp_path / "report.json"
    _write_jsonl(prompt_path, [{"source_parent_asins": ["B000TEST1"]}])
    _write_jsonl(kb_path, [])
    _write_jsonl(gold_path, [])
    _write_jsonl(metadata_path, [])

    module.enrich_selected_metadata(
        SimpleNamespace(
            retail_prompts=prompt_path,
            retail_kb=kb_path,
            retail_gold=gold_path,
            metadata_input=metadata_path,
            targeted_metadata_output=targeted_output,
            selected_parent_asins_output=selected_output,
            targeted_report=report_path,
            metadata_source_path=tmp_path / "missing.parquet",
            source_scan_batch_limit=None,
        )
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["unmatched_parent_asins"] == ["B000TEST1"]


def test_build_curated_samples_uses_targeted_metadata_when_present() -> None:
    module = _load_curation_module()
    base_rows = [{"parent_asin": "B000TEST1", "title": ""}]
    targeted_rows = [{"parent_asin": "B000TEST1", "title": "Rosemary Mint Hair Oil"}]
    metadata = module.merge_metadata_rows(base_rows, targeted_rows)
    index = module.build_product_metadata_index(metadata)

    title, resolution = module.resolve_product_title("B000TEST1", None, index, "All_Beauty")

    assert title == "Rosemary Mint Hair Oil"
    assert resolution["title_resolution"] == "metadata_title"


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
    assert "product_title_resolution_counts" in report
    assert "generic_product_title_count" in report
    assert "product_metadata_join_rate" in report
    assert "targeted_metadata_used" in report
    assert "targeted_metadata_record_count" in report
    assert "unmatched_parent_asins" in report
    assert report["next_step"]


def test_retail_curation_report_includes_title_quality() -> None:
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

    assert isinstance(report["product_title_resolution_counts"], dict)
    assert report["generic_product_title_count"] >= 0
    assert 0 <= report["product_metadata_join_rate"] <= 1


def test_retail_committed_prompts_have_title_resolution_metadata() -> None:
    for prompt in _read_jsonl(PROMPT_PATH):
        metadata = prompt["metadata"]
        assert metadata.get("source_titles")
        assert "title_resolution" in metadata


def test_retail_generic_title_warning_when_present() -> None:
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

    if report["generic_product_title_count"] > 0:
        warnings = " ".join(report["warnings"]).lower()
        assert "richer metadata" in warnings or "targeted metadata retrieval" in warnings


def test_docs_include_retail_curated_seed() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")
    lowered = docs.lower()

    assert "Phase 2A-6C Retail Curated Seed" in docs
    assert "support policy" in lowered
    assert "spam_or_low_quality_review" in docs
    assert "rag" in lowered
