import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUERY_PLAN_PATH = ROOT / "data/sources/research_ai_query_plan.json"
GOLD_SCHEMA_PATH = ROOT / "data/eval/schema/gold_record_schema.json"
RESEARCH_SCHEMA_PATH = ROOT / "data/schemas/research_ai_prompt_record_schema.json"
SCRIPT_PATH = ROOT / "scripts/phase2/discover_research_ai_papers.py"
SAMPLE_PATH = ROOT / "data/sources/research_ai_candidate_papers_sample.jsonl"
DOC_PATH = ROOT / "docs/34_phase2_research_ai_paper_discovery.md"


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _load_discovery_module() -> Any:
    spec = importlib.util.spec_from_file_location("discover_research_ai_papers", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def test_query_plan_exists_and_parses() -> None:
    query_plan = _load_json(QUERY_PLAN_PATH)

    assert len(query_plan["query_groups"]) == 7
    assert query_plan["target_prompt_seed_count"] == 40
    assert set(query_plan["status_mix_for_seed"]) == {
        "answer",
        "escalation_or_insufficient_evidence",
        "out_of_scope",
    }


def test_status_schemas_include_out_of_scope() -> None:
    gold_schema = _load_json(GOLD_SCHEMA_PATH)
    research_schema = _load_json(RESEARCH_SCHEMA_PATH)

    assert "out_of_scope" in gold_schema["properties"]["expected_status"]["enum"]
    assert "out_of_scope" in research_schema["properties"]["expected_status"]["enum"]


def test_parse_arxiv_atom() -> None:
    module = _load_discovery_module()
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
        <updated>2024-02-03T00:00:00Z</updated>
        <published>2024-01-02T00:00:00Z</published>
        <title>Efficient LLM Inference Serving</title>
        <summary>We study batching and latency for large language model inference.</summary>
        <author><name>Researcher One</name></author>
        <author><name>Researcher Two</name></author>
        <arxiv:primary_category term="cs.CL" />
        <category term="cs.CL" />
        <category term="cs.LG" />
        <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
        <link title="pdf" href="http://arxiv.org/pdf/2401.12345v2"
              rel="related" type="application/pdf"/>
      </entry>
    </feed>
    """

    records = module.parse_arxiv_atom(xml_text, "query_1", "Test Topic")

    assert len(records) == 1
    record = records[0]
    assert record["arxiv_id"] == "2401.12345"
    assert record["title"] == "Efficient LLM Inference Serving"
    assert "batching" in record["abstract"]
    assert record["authors"] == ["Researcher One", "Researcher Two"]
    assert record["categories"] == ["cs.CL", "cs.LG"]
    assert record["abstract_url"] == "http://arxiv.org/abs/2401.12345v2"
    assert record["pdf_url"] == "http://arxiv.org/pdf/2401.12345v2"
    assert record["source"] == "arXiv"


def test_normalize_arxiv_id() -> None:
    module = _load_discovery_module()

    assert module.normalize_arxiv_id("http://arxiv.org/abs/2401.12345v2") == (
        "2401.12345",
        "2401.12345v2",
    )
    assert module.normalize_arxiv_id("https://arxiv.org/abs/1706.03762") == (
        "1706.03762",
        "1706.03762",
    )


def test_score_candidate() -> None:
    module = _load_discovery_module()
    candidate = {
        "title": "LLM inference serving with continuous batching",
        "abstract": "This paper studies latency, throughput, batching, and GPU memory.",
        "primary_category": "cs.CL",
        "categories": ["cs.CL", "cs.LG"],
        "published": "2024-01-01T00:00:00Z",
    }
    query_group = {
        "required_keywords": ["inference"],
        "preferred_keywords": ["serving", "latency", "throughput", "batching"],
        "categories_allowed": ["cs.CL", "cs.LG"],
    }

    assert module.score_candidate(candidate, query_group) > 0


def test_dedupe_candidates() -> None:
    module = _load_discovery_module()
    candidates = [
        {
            "arxiv_id": "2401.12345",
            "arxiv_id_version": "2401.12345v1",
            "query_ids": ["query_a"],
            "topics": ["Topic A"],
            "matched_keywords": ["inference"],
            "score": 5,
            "updated": "2024-01-01T00:00:00Z",
        },
        {
            "arxiv_id": "2401.12345",
            "arxiv_id_version": "2401.12345v2",
            "query_ids": ["query_b"],
            "topics": ["Topic B"],
            "matched_keywords": ["serving"],
            "score": 9,
            "updated": "2024-02-01T00:00:00Z",
        },
    ]

    deduped = module.dedupe_candidates(candidates)

    assert len(deduped) == 1
    assert deduped[0]["query_ids"] == ["query_a", "query_b"]
    assert deduped[0]["topics"] == ["Topic A", "Topic B"]
    assert deduped[0]["matched_keywords"] == ["inference", "serving"]
    assert deduped[0]["score"] == 9
    assert deduped[0]["arxiv_id_version"] == "2401.12345v2"


def test_rank_candidates() -> None:
    module = _load_discovery_module()
    candidates = [
        {"score": 3, "updated": "2024-01-01", "published": "2024-01-01", "title": "B"},
        {"score": 10, "updated": "2023-01-01", "published": "2023-01-01", "title": "A"},
    ]

    ranked = module.rank_candidates(candidates)

    assert ranked[0]["score"] == 10


def test_dry_run_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "dry_run"
    assert summary["planned_query_count"] == 7


def test_sample_file_exists_after_generation_or_static_fixture() -> None:
    assert SAMPLE_PATH.exists()
    records = _read_jsonl(SAMPLE_PATH)

    assert len(records) >= 1
    required_fields = {
        "paper_id",
        "arxiv_id",
        "title",
        "abstract",
        "authors",
        "categories",
        "abstract_url",
        "pdf_url",
        "source",
    }
    assert required_fields.issubset(records[0])


def test_docs_include_status_taxonomy() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "out_of_scope" in text
    assert "World Cup" in text
    assert "Phase 2A-5B" in text
    assert "RAG/context engineering remains deferred" in text
