import json
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "data/real_world_samples/research_ai_sample.jsonl"
KB_PATH = ROOT / "data/kb/research_ai/kb_sample.jsonl"
GOLD_PATH = ROOT / "data/eval/gold/research_ai_gold_sample.jsonl"
REPORT_PATH = ROOT / "data/generated/research_ai/research_ai_curation_report.json"
SCRIPT_PATH = ROOT / "scripts/phase2/curate_research_ai_seed.py"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def _load_curation_module() -> Any:
    spec = spec_from_file_location("curate_research_ai_seed", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_research_ai_curated_files_exist() -> None:
    assert PROMPT_PATH.exists()
    assert KB_PATH.exists()
    assert GOLD_PATH.exists()


def test_research_ai_prompt_count_and_required_fields() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    required = {
        "prompt_id",
        "vertical",
        "task_type",
        "question",
        "expected_output_format",
        "expected_status",
        "source_paper_ids",
        "required_evidence_ids",
    }

    assert len(prompts) == 40
    assert len({record["prompt_id"] for record in prompts}) == 40
    for record in prompts:
        assert required.issubset(record)
        assert record["vertical"] == "research_ai"


def test_research_ai_prompt_distribution() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    counts = Counter(record["metadata"]["prompt_category"] for record in prompts)

    assert counts == {
        "concept_explanation": 6,
        "paper_method": 7,
        "results_evaluation": 6,
        "structured_extraction": 6,
        "compare_papers": 5,
        "literature_table": 4,
        "evidence_citation_lookup": 3,
        "insufficient_evidence_or_escalation": 2,
        "out_of_scope": 1,
    }


def test_research_ai_status_distribution() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    statuses = Counter(record["expected_status"] for record in prompts)

    assert statuses["answer"] == 37
    assert statuses["insufficient_evidence"] + statuses["escalate"] == 2
    assert statuses["out_of_scope"] == 1


def test_research_ai_kb_minimum_and_required_fields() -> None:
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
    section_types = {
        record.get("metadata", {}).get("section_type")
        for record in records
        if isinstance(record.get("metadata"), dict)
    }

    assert len(records) >= 30
    assert len({record["doc_id"] for record in records}) == len(records)
    for record in records:
        assert required.issubset(record)
        assert record["vertical"] == "research_ai"
    assert any(record["document_type"] == "paper_abstract" for record in records)
    assert any(record["document_type"] == "paper_section" for record in records)
    assert section_types & {"method", "results", "evaluation", "limitations", "conclusion"}


def test_research_ai_gold_alignment() -> None:
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


def test_research_ai_answerable_gold_has_evidence() -> None:
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        if record["expected_status"] == "answer":
            assert record.get("required_doc_ids")
            assert record.get("required_citations")
            assert record.get("must_include")


def test_research_ai_out_of_scope_record() -> None:
    gold = _read_jsonl(GOLD_PATH)
    out_of_scope = [record for record in gold if record["expected_status"] == "out_of_scope"]

    assert len(out_of_scope) == 1
    assert "general model memory" in out_of_scope[0]["must_not_include"]
    assert "outside the Research AI corpus" in out_of_scope[0]["reference_answer"]


def test_research_ai_structured_prompts_have_json_output() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    gold_by_id = {record["prompt_id"]: record for record in _read_jsonl(GOLD_PATH)}

    for prompt in prompts:
        if prompt["metadata"]["prompt_category"] == "structured_extraction":
            assert prompt["expected_output_format"] == "json"
            gold = gold_by_id[prompt["prompt_id"]]
            assert {"paper_title", "method", "evidence_id"}.issubset(set(gold["must_include"]))


def test_research_ai_no_private_paths() -> None:
    forbidden = ["C:\\Users", "/home/", "akpoogaga", "kparo", "token", "API key"]
    content = "\n".join(
        path.read_text(encoding="utf-8") for path in (PROMPT_PATH, KB_PATH, GOLD_PATH)
    )
    lowered_content = content.lower()

    for term in forbidden:
        assert term.lower() not in lowered_content


def test_research_ai_curation_report_shape() -> None:
    prompts = _read_jsonl(PROMPT_PATH)
    kb_records = _read_jsonl(KB_PATH)
    gold_records = _read_jsonl(GOLD_PATH)
    source_paper_count = len(
        {
            str(paper_id)
            for prompt in prompts
            for paper_id in prompt.get("source_paper_ids", [])
            if paper_id
        }
    )
    report = _load_curation_module().build_curation_report(
        prompts,
        kb_records,
        gold_records,
        [],
        source_paper_count,
    )

    assert report["prompt_record_count"] == 40
    assert report["kb_record_count"] >= 30
    assert report["gold_record_count"] == 40
    assert report["next_step"]


def test_research_ai_reference_answers_not_mechanical() -> None:
    mechanical_phrases = [
        "addresses its stated research problem by focusing on",
        "using the cited paper evidence",
        "This paper is about",
    ]
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        answer = record.get("reference_answer", "")
        for phrase in mechanical_phrases:
            assert phrase not in answer


def test_research_ai_reference_answers_are_substantive() -> None:
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        if record["expected_status"] != "answer":
            continue
        prompt_category = record["metadata"]["prompt_category"]
        answer = str(record.get("reference_answer") or "")
        if prompt_category == "structured_extraction":
            continue
        answer_words = answer.split()
        assert len(answer_words) >= 20

        must_include_terms = [
            str(term).lower()
            for term in record.get("must_include", [])
            if len(str(term).strip()) >= 4
        ]
        source_title_terms: list[str] = []
        for title in record["metadata"].get("source_titles", []):
            source_title_terms.extend(
                word.lower()
                for word in str(title).replace(":", " ").replace("-", " ").split()
                if len(word) >= 5
            )
        lowered_answer = answer.lower()
        assert any(term in lowered_answer for term in [*must_include_terms, *source_title_terms])


def test_research_ai_structured_reference_answers_parse_or_have_json_shape() -> None:
    gold = _read_jsonl(GOLD_PATH)
    required_keys = {
        "paper_title",
        "method_or_system",
        "evidence_summary",
        "evidence_id",
    }

    for record in gold:
        if record["metadata"]["prompt_category"] != "structured_extraction":
            continue
        answer = str(record.get("reference_answer") or "")
        try:
            parsed = json.loads(answer)
        except json.JSONDecodeError:
            for key in required_keys:
                assert key in answer
        else:
            assert isinstance(parsed, dict)
            assert required_keys.issubset(parsed)


def test_research_ai_negative_status_answers_are_safe() -> None:
    gold = _read_jsonl(GOLD_PATH)
    negative_records = [
        record
        for record in gold
        if record["expected_status"] in {"insufficient_evidence", "escalate", "out_of_scope"}
    ]

    assert len(negative_records) == 3
    for record in negative_records:
        answer = str(record.get("reference_answer") or "")
        lowered_answer = answer.lower()
        assert "will outperform all future systems" not in lowered_answer
        assert "guess" in " ".join(record.get("must_not_include", [])).lower() or (
            "general model memory" in " ".join(record.get("must_not_include", [])).lower()
        )
        if record["expected_status"] == "out_of_scope":
            assert (
                "outside the research ai corpus" in lowered_answer
                or "outside selected corpus" in lowered_answer
            )
            assert "general model memory" in " ".join(record["must_not_include"])
        else:
            assert "insufficient" in lowered_answer or "expert review" in lowered_answer


def test_research_ai_reference_answers_have_no_spacing_artifacts() -> None:
    forbidden_fragments = [
        "calledAgent",
        "proposeDELTA",
        "calledDELTA",
        "  ",
    ]
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        answer = str(record.get("reference_answer") or "")
        for fragment in forbidden_fragments:
            assert fragment not in answer


def test_research_ai_reference_answers_avoid_internal_plumbing_style() -> None:
    forbidden_phrases = [
        "Its contribution is grounded in",
        "The paper frames the problem as follows",
        "with traceable paper evidence rather than a generic summary",
    ]
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        if record["expected_status"] != "answer":
            continue
        if record["metadata"]["prompt_category"] == "structured_extraction":
            continue
        answer = str(record.get("reference_answer") or "")
        for phrase in forbidden_phrases:
            assert phrase not in answer


def test_research_ai_reference_answers_do_not_end_with_orphan_fragments() -> None:
    orphan_endings = ("and", "or", "the", "their", "of", "to", "on their")
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        if record["expected_status"] != "answer":
            continue
        if record["metadata"]["prompt_category"] == "structured_extraction":
            continue
        answer = str(record.get("reference_answer") or "").strip().lower()
        normalized = answer.rstrip(" .|")
        assert not normalized.endswith(orphan_endings)


def test_research_ai_reference_answers_still_grounded() -> None:
    gold = _read_jsonl(GOLD_PATH)

    for record in gold:
        if record["expected_status"] != "answer":
            continue
        assert record.get("required_doc_ids")
        assert record.get("required_citations")
        assert record.get("must_include")
        answer = str(record.get("reference_answer") or "").lower()
        source_titles = record.get("metadata", {}).get("source_titles", [])
        leading_tokens = [
            str(title).replace(":", " ").split()[0].lower()
            for title in source_titles
            if str(title).split()
        ]
        must_include_terms = [
            str(term).lower()
            for term in record.get("must_include", [])
            if len(str(term).strip()) >= 4
        ]
        assert any(term in answer for term in [*leading_tokens, *must_include_terms])
