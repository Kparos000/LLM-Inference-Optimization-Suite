import csv
import importlib.util
import io
import json
import subprocess
import sys
import urllib.error
from argparse import Namespace
from email.message import Message
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
QUERY_PLAN_PATH = ROOT / "data/sources/research_ai_query_plan.json"
GOLD_SCHEMA_PATH = ROOT / "data/eval/schema/gold_record_schema.json"
RESEARCH_SCHEMA_PATH = ROOT / "data/schemas/research_ai_prompt_record_schema.json"
SCRIPT_PATH = ROOT / "scripts/phase2/discover_research_ai_papers.py"
SAMPLE_PATH = ROOT / "data/sources/research_ai_candidate_papers_sample.jsonl"
DOC_PATH = ROOT / "docs/34_phase2_research_ai_paper_discovery.md"
MANUAL_TEMPLATE_PATH = ROOT / "data/sources/research_ai_manual_registry_template.csv"
APPROVED_PAPERS_EXAMPLE_PATH = ROOT / "data/sources/research_ai_approved_papers.example.jsonl"
APPROVED_REGISTRY_PATH = ROOT / "data/sources/research_ai_approved_papers.jsonl"
APPROVED_1000_SCALE_PAPERS_PATH = ROOT / "data/sources/research_ai_1000_scale_approved_papers.jsonl"


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


def test_query_plan_has_paper_window() -> None:
    query_plan = _load_json(QUERY_PLAN_PATH)
    paper_window = query_plan["paper_window"]

    assert paper_window["start_date"] == "2024-01-01"
    assert paper_window["end_date"] == "2026-05-30"
    assert paper_window["arxiv_submitted_date_start"] == "202401010000"
    assert paper_window["arxiv_submitted_date_end"] == "202605302359"


def test_query_plan_has_discovery_sources() -> None:
    query_plan = _load_json(QUERY_PLAN_PATH)
    sources = {source["source_id"]: source for source in query_plan["discovery_sources"]}

    assert {"iclr_2025_virtual", "huggingface_papers", "arxiv_api"}.issubset(sources)
    assert sources["arxiv_api"]["enabled"] is False


def test_status_schemas_include_out_of_scope() -> None:
    gold_schema = _load_json(GOLD_SCHEMA_PATH)
    research_schema = _load_json(RESEARCH_SCHEMA_PATH)

    assert "out_of_scope" in gold_schema["properties"]["expected_status"]["enum"]
    assert "out_of_scope" in research_schema["properties"]["expected_status"]["enum"]


def test_run_log_writer(tmp_path: Path) -> None:
    module = _load_discovery_module()
    run_log_path = tmp_path / "run_log.jsonl"

    module.write_run_log_event(run_log_path, {"mode": "test", "event_type": "run_started"})
    module.write_run_log_event(
        run_log_path,
        {"mode": "test", "event_type": "run_completed", "message": "done"},
    )

    rows = _read_jsonl(run_log_path)
    assert len(rows) == 2
    assert rows[0]["phase"] == "2A-5A"
    assert rows[0]["event_type"] == "run_started"


def test_query_id_filter() -> None:
    module = _load_discovery_module()
    query_plan = _load_json(QUERY_PLAN_PATH)

    selected = module.select_query_groups(query_plan, "llm_serving_inference_optimization")

    assert len(selected) == 1
    assert selected[0]["query_id"] == "llm_serving_inference_optimization"
    try:
        module.select_query_groups(query_plan, "unknown_query")
    except RuntimeError as exc:
        assert "Unknown query_id" in str(exc)
    else:
        raise AssertionError("Expected unknown query_id to raise RuntimeError")


def test_format_arxiv_submitted_date() -> None:
    module = _load_discovery_module()

    assert module.format_arxiv_submitted_date("2024-01-01") == "202401010000"
    assert module.format_arxiv_submitted_date("2026-05-30", end_of_day=True) == "202605302359"


def test_apply_arxiv_date_filter_simple_query() -> None:
    module = _load_discovery_module()

    query = module.apply_arxiv_date_filter(
        'all:"LLM inference"',
        "2024-01-01",
        "2026-05-30",
    )

    assert "submittedDate:[202401010000 TO 202605302359]" in query
    assert query.startswith('all:"LLM inference" AND')


def test_apply_arxiv_date_filter_compound_query() -> None:
    module = _load_discovery_module()

    query = module.apply_arxiv_date_filter(
        'all:"LLM inference" OR all:"LLM serving"',
        "2024-01-01",
        "2026-05-30",
    )

    assert query.startswith("(")
    assert "AND submittedDate:[202401010000 TO 202605302359]" in query


def test_build_huggingface_search_urls() -> None:
    module = _load_discovery_module()

    urls = module.build_huggingface_search_urls(
        "https://huggingface.co/papers",
        ["LLM inference", "RAG"],
    )

    assert len(urls) == 2
    assert urls[0]["url"].startswith("https://huggingface.co/papers?q=")
    assert "LLM+inference" in urls[0]["url"]


def test_parse_iclr_2025_papers_html() -> None:
    module = _load_discovery_module()
    html_text = """
    <html><body>
      <a href="/virtual/2025/poster/123">Efficient LLM Inference Serving</a>
      <a href="https://openreview.net/forum?id=abc">Speculative Decoding for Language Models</a>
    </body></html>
    """

    records = module.parse_iclr_2025_papers_html(
        html_text,
        "https://iclr.cc/virtual/2025/papers.html",
    )

    assert len(records) == 2
    assert records[0]["source"] == "ICLR"
    assert records[0]["venue"] == "ICLR 2025"
    assert records[0]["title"] == "Efficient LLM Inference Serving"
    assert records[0]["provenance_url"].startswith("https://iclr.cc/")


def test_parse_huggingface_papers_html() -> None:
    module = _load_discovery_module()
    html_text = """
    <html><body>
      <a href="/papers/2401.12345">RAGChecker: Evaluating Retrieval Augmented Generation</a>
      <a href="/papers/2502.00001">Automated Design of Agentic Systems</a>
    </body></html>
    """

    records = module.parse_huggingface_papers_html(
        html_text,
        "https://huggingface.co/papers?q=RAG",
        "RAG",
    )

    assert len(records) == 2
    assert records[0]["source"] == "Hugging Face Papers"
    assert records[0]["title"] == "RAGChecker: Evaluating Retrieval Augmented Generation"
    assert records[0]["provenance_url"].startswith("https://huggingface.co/papers/")


def test_infer_research_topics() -> None:
    module = _load_discovery_module()

    assert "speculative_decoding_kv_cache" in module.infer_research_topics(
        "Faster Cascades via Speculative Decoding"
    )
    assert "agentic_workflows_tool_use" in module.infer_research_topics(
        "Automated Design of Agentic Systems"
    )
    assert "rag_context_engineering" in module.infer_research_topics("RAGChecker")


def test_score_html_candidate_positive_for_relevant_title() -> None:
    module = _load_discovery_module()
    candidate = {
        "title": "Efficient LLM Inference Serving with Continuous Batching",
        "abstract": "",
        "source": "ICLR",
        "topics": ["llm_serving_inference_optimization"],
        "year": 2025,
    }

    assert module.score_html_candidate(candidate) > 0


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_fetch_url_with_retries_success() -> None:
    module = _load_discovery_module()
    xml_text = b"<feed><entry></entry></feed>"

    with mock.patch.object(
        module.urllib.request,
        "urlopen",
        return_value=_FakeResponse(xml_text),
    ):
        result = module.fetch_url_with_retries(
            url="https://export.arxiv.org/api/query",
            user_agent="test-agent",
            timeout_seconds=1,
            max_retries=3,
            backoff_seconds=0,
        )

    assert "<feed>" in result


def test_fetch_url_with_retries_429_then_success() -> None:
    module = _load_discovery_module()
    url = "https://export.arxiv.org/api/query"
    headers = Message()
    headers.add_header("Retry-After", "0")
    too_many_requests = urllib.error.HTTPError(
        url=url,
        code=429,
        msg="Too Many Requests",
        hdrs=headers,
        fp=None,
    )

    with (
        mock.patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=[too_many_requests, _FakeResponse(b"<feed>ok</feed>")],
        ) as urlopen_mock,
        mock.patch.object(module.time, "sleep") as sleep_mock,
    ):
        result = module.fetch_url_with_retries(
            url=url,
            user_agent="test-agent",
            timeout_seconds=1,
            max_retries=3,
            backoff_seconds=0,
        )

    assert result == "<feed>ok</feed>"
    assert urlopen_mock.call_count == 2
    sleep_mock.assert_called_once()


def test_fetch_url_with_retries_4xx_non_retry() -> None:
    module = _load_discovery_module()
    url = "https://export.arxiv.org/api/query"
    not_found = urllib.error.HTTPError(
        url=url,
        code=404,
        msg="Not Found",
        hdrs=Message(),
        fp=None,
    )

    with mock.patch.object(
        module.urllib.request,
        "urlopen",
        side_effect=not_found,
    ) as urlopen_mock:
        try:
            module.fetch_url_with_retries(
                url=url,
                user_agent="test-agent",
                timeout_seconds=1,
                max_retries=3,
                backoff_seconds=0,
            )
        except RuntimeError as exc:
            assert "HTTP 404" in str(exc)
        else:
            raise AssertionError("Expected HTTP 404 to raise RuntimeError")

    assert urlopen_mock.call_count == 1


def test_http_error_details_include_status_and_body_snippet() -> None:
    module = _load_discovery_module()
    url = "https://export.arxiv.org/api/query"
    headers = Message()
    headers.add_header("Retry-After", "2")
    too_many_requests = urllib.error.HTTPError(
        url=url,
        code=429,
        msg="Too Many Requests",
        hdrs=headers,
        fp=io.BytesIO(b"rate limited response body"),
    )

    with (
        mock.patch.object(module.urllib.request, "urlopen", side_effect=too_many_requests),
        mock.patch.object(module.time, "sleep"),
    ):
        try:
            module.fetch_url_with_retries(
                url=url,
                user_agent="test-agent",
                timeout_seconds=1,
                max_retries=0,
                backoff_seconds=0,
            )
        except module.ArxivFetchError as exc:
            details = exc.details
        else:
            raise AssertionError("Expected ArxivFetchError")

    assert details["status_code"] == 429
    assert details["response_body_snippet"] == "rate limited response body"
    assert details["retry_after"] == 2.0
    assert details["exception_type"] == "HTTPError"


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


def test_dedupe_candidates_without_arxiv_id_by_title() -> None:
    module = _load_discovery_module()
    candidates = [
        {
            "arxiv_id": "",
            "title": "Efficient LLM Inference Serving",
            "query_ids": ["iclr_2025_virtual"],
            "topics": ["llm_serving_inference_optimization"],
            "matched_keywords": ["inference"],
            "source_ids": ["iclr_2025_virtual"],
            "score": 5,
            "provenance_url": "https://iclr.cc/paper/1",
        },
        {
            "arxiv_id": "",
            "title": "Efficient LLM Inference Serving",
            "query_ids": ["huggingface_llm_inference"],
            "topics": ["llm_serving_inference_optimization"],
            "matched_keywords": ["serving"],
            "source_ids": ["huggingface_papers"],
            "score": 8,
            "provenance_url": "https://huggingface.co/papers/1",
        },
    ]

    deduped = module.dedupe_candidates(candidates)

    assert len(deduped) == 1
    assert deduped[0]["score"] == 8
    assert deduped[0]["query_ids"] == ["huggingface_llm_inference", "iclr_2025_virtual"]


def test_rank_candidates() -> None:
    module = _load_discovery_module()
    candidates = [
        {"score": 3, "updated": "2024-01-01", "published": "2024-01-01", "title": "B"},
        {"score": 10, "updated": "2023-01-01", "published": "2023-01-01", "title": "A"},
    ]

    ranked = module.rank_candidates(candidates)

    assert ranked[0]["score"] == 10


def test_partial_report_shape() -> None:
    module = _load_discovery_module()
    query_plan = _load_json(QUERY_PLAN_PATH)

    report = module.build_report(
        query_plan=query_plan,
        raw_candidate_count=1,
        ranked_candidates=[],
        sample_candidates=[],
        output_files={"discovery_report_json": "report.json"},
        discovery_status="partial",
        failed_query_groups=["query_b"],
        successful_query_groups=["query_a"],
        errors=[
            {
                "query_id": "query_b",
                "url": "https://export.arxiv.org/api/query",
                "error_message": "HTTP 429",
            }
        ],
        retry_policy={
            "max_retries": 3,
            "backoff_seconds": 10,
            "timeout_seconds": 30,
            "delay_seconds": 3,
        },
    )

    assert report["discovery_status"] == "partial"
    assert report["failed_query_groups"] == ["query_b"]
    assert report["successful_query_groups"] == ["query_a"]
    assert report["errors"][0]["error_message"] == "HTTP 429"
    assert report["retry_policy"]["max_retries"] == 3


def test_failure_report_written_when_all_queries_fail(tmp_path: Path) -> None:
    module = _load_discovery_module()
    report_path = tmp_path / "report.json"
    run_log_path = tmp_path / "run_log.jsonl"
    args = Namespace(
        query_plan=QUERY_PLAN_PATH,
        source="arxiv",
        query_id="llm_serving_inference_optimization",
        output_candidates=tmp_path / "candidate_papers.jsonl",
        output_review_csv=tmp_path / "candidate_papers_review.csv",
        output_sample=tmp_path / "candidate_papers_sample.jsonl",
        output_report=report_path,
        run_log_path=run_log_path,
        output_manual_template=tmp_path / "manual_template.csv",
        manual_validation_report=tmp_path / "manual_validation_report.json",
        max_results_per_query=1,
        sample_size=1,
        delay_seconds=0,
        timeout_seconds=1,
        max_retries=0,
        backoff_seconds=0,
        continue_on_error=False,
        allow_partial=False,
        simple_query_mode=True,
        start_date=None,
        end_date=None,
        disable_date_filter=False,
    )

    with mock.patch.object(
        module,
        "fetch_arxiv_atom",
        side_effect=module.ArxivFetchError(
            "mocked HTTP 429",
            {
                "query_id": "llm_serving_inference_optimization",
                "url": "https://export.arxiv.org/api/query",
                "attempt_number": 1,
                "status_code": 429,
                "exception_type": "HTTPError",
                "retry_after": None,
                "response_body_snippet": "rate limited",
                "error_message": "mocked HTTP 429",
            },
        ),
    ):
        try:
            module.discover(args)
        except RuntimeError as exc:
            assert "All selected arXiv query groups failed" in str(exc)
        else:
            raise AssertionError("Expected failed discovery to raise RuntimeError")

    report = _load_json(report_path)
    assert report["discovery_status"] == "failed"
    assert report["errors"]
    assert report["failed_query_groups"] == ["llm_serving_inference_optimization"]
    assert run_log_path.exists()


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
    assert summary["source"] == "all"
    assert summary["planned_source_count"] == 2


def test_dry_run_supports_query_id() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--source",
            "arxiv",
            "--query-id",
            "llm_serving_inference_optimization",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "dry_run"
    assert summary["planned_query_count"] == 1


def test_simple_query_mode_changes_dry_run_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--source",
            "arxiv",
            "--query-id",
            "llm_serving_inference_optimization",
            "--simple-query-mode",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    planned_url = summary["planned_queries"][0]["url"]
    assert "LLM+inference" in planned_url
    assert "large+language+model+inference" not in planned_url


def test_dry_run_url_contains_date_filter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--source",
            "arxiv",
            "--query-id",
            "llm_serving_inference_optimization",
            "--simple-query-mode",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    planned_url = summary["planned_queries"][0]["url"]
    assert "submittedDate" in planned_url


def test_disable_date_filter_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--source",
            "arxiv",
            "--query-id",
            "llm_serving_inference_optimization",
            "--simple-query-mode",
            "--disable-date-filter",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    planned_url = summary["planned_queries"][0]["url"]
    assert "submittedDate" not in planned_url


def test_dry_run_source_all() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run", "--source", "all"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    planned_sources = summary["planned_sources"]
    assert any(plan["source_id"] == "iclr_2025_virtual" for plan in planned_sources)
    assert any(plan["source_id"] == "huggingface_papers" for plan in planned_sources)
    assert all(plan["source_id"] != "arxiv_api" for plan in planned_sources)


def test_dry_run_source_arxiv_still_available() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--source",
            "arxiv",
            "--query-id",
            "llm_serving_inference_optimization",
            "--simple-query-mode",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["source"] == "arxiv"
    assert summary["planned_query_count"] == 1
    assert "export.arxiv.org/api/query" in summary["planned_queries"][0]["url"]


def _fake_candidate(
    rank: int,
    score: int,
    title: str,
    topics: str,
    paper_id: str | None = None,
) -> dict[str, str]:
    return {
        "rank": str(rank),
        "score": str(score),
        "paper_id": paper_id or f"research_ai_fake_{rank:03d}",
        "arxiv_id": "",
        "title": title,
        "source": "ICLR",
        "venue": "ICLR 2025",
        "year": "2025",
        "topics": topics,
        "authors": "",
        "published": "2025",
        "primary_category": "conference",
        "categories": "ICLR 2025",
        "provenance_url": f"https://iclr.cc/virtual/2025/poster/{rank}",
        "abstract_url": f"https://iclr.cc/virtual/2025/poster/{rank}",
        "pdf_url": "",
        "selection_status": "candidate",
        "review_notes": "",
    }


def test_build_approved_registry_from_fake_candidates() -> None:
    module = _load_discovery_module()
    candidates = [
        _fake_candidate(
            1,
            15,
            "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents",
            "agentic_workflows_tool_use",
        ),
        _fake_candidate(
            2, 14, "AgentTrek: Agent Trajectory Synthesis", "agentic_workflows_tool_use"
        ),
        _fake_candidate(
            3, 13, "Faster Cascades via Speculative Decoding", "speculative_decoding_kv_cache"
        ),
        _fake_candidate(
            4, 12, "Retrieval Head Explains Long-Context Factuality", "rag_context_engineering"
        ),
        _fake_candidate(
            5,
            11,
            "MiniPLM: Knowledge Distillation for Pre-training Language Models",
            "small_language_models_efficient_llms",
        ),
        _fake_candidate(
            6,
            10,
            "Inference Optimal VLMs Need Fewer Visual Tokens",
            "llm_serving_inference_optimization",
        ),
        _fake_candidate(7, 9, "CodeMMLU: A Benchmark for CodeLLMs", "research_ai_candidate"),
        _fake_candidate(
            8, 8, "Compute-Optimal LLMs Generalize Better with Scale", "research_ai_candidate"
        ),
    ]

    approved = module.build_approved_registry_from_candidates(candidates, approved_count=6)

    assert len(approved) == 6
    assert {record["selection_status"] for record in approved} == {"approved"}
    topics = {topic for record in approved for topic in record["topics"]}
    assert len(topics) >= 4


def test_approval_keyword_matching_does_not_match_substrings() -> None:
    module = _load_discovery_module()

    assert not module.is_project_relevant_candidate(
        _fake_candidate(
            1,
            1,
            "A Coefficient Makes SVRG Effective",
            "research_ai_candidate",
        )
    )


def test_approved_registry_records_allow_iclr_without_arxiv_fields() -> None:
    module = _load_discovery_module()
    record = {
        "paper_id": "research_ai_iclr_test",
        "title": "Faster Cascades via Speculative Decoding",
        "authors": [],
        "published": "2025",
        "year": 2025,
        "source": "ICLR",
        "source_id": "iclr_2025_virtual",
        "venue": "ICLR 2025",
        "primary_category": "conference",
        "categories": ["ICLR 2025"],
        "abstract_url": "https://iclr.cc/virtual/2025/poster/27888",
        "pdf_url": None,
        "provenance_url": "https://iclr.cc/virtual/2025/poster/27888",
        "topic": "speculative_decoding_kv_cache",
        "topics": ["speculative_decoding_kv_cache"],
        "reason_for_inclusion": "Included because it is relevant to speculative decoding.",
        "selection_status": "approved",
    }

    errors, warnings = module.validate_manual_registry_records(
        [record],
        "2024-01-01",
        "2026-05-30",
    )

    assert errors == []
    assert warnings


def test_approved_registry_validation_rejects_duplicate_paper_id() -> None:
    module = _load_discovery_module()
    record = {
        "paper_id": "research_ai_duplicate",
        "title": "Faster Cascades via Speculative Decoding",
        "published": "2025",
        "source": "ICLR",
        "source_id": "iclr_2025_virtual",
        "venue": "ICLR 2025",
        "primary_category": "conference",
        "categories": ["ICLR 2025"],
        "provenance_url": "https://iclr.cc/virtual/2025/poster/27888",
        "topic": "speculative_decoding_kv_cache",
        "topics": ["speculative_decoding_kv_cache"],
        "reason_for_inclusion": "Included for testing.",
        "selection_status": "approved",
    }

    errors, _warnings = module.validate_manual_registry_records(
        [record, dict(record)],
        "2024-01-01",
        "2026-05-30",
    )

    assert any(error["error_type"] == "duplicate_paper_id" for error in errors)


def test_approved_registry_validation_rejects_missing_provenance() -> None:
    module = _load_discovery_module()
    record = {
        "paper_id": "research_ai_missing_provenance",
        "title": "Faster Cascades via Speculative Decoding",
        "published": "2025",
        "source": "ICLR",
        "source_id": "iclr_2025_virtual",
        "venue": "ICLR 2025",
        "primary_category": "conference",
        "categories": ["ICLR 2025"],
        "provenance_url": "",
        "topic": "speculative_decoding_kv_cache",
        "topics": ["speculative_decoding_kv_cache"],
        "reason_for_inclusion": "Included for testing.",
        "selection_status": "approved",
    }

    errors, _warnings = module.validate_manual_registry_records(
        [record],
        "2024-01-01",
        "2026-05-30",
    )

    assert any(error["error_type"] == "missing_required_fields" for error in errors)
    assert any(error["error_type"] == "missing_provenance_url" for error in errors)


def test_approved_registry_validation_rejects_missing_categories() -> None:
    module = _load_discovery_module()
    record = {
        "paper_id": "research_ai_missing_categories",
        "title": "Faster Cascades via Speculative Decoding",
        "published": "2025",
        "source": "ICLR",
        "source_id": "iclr_2025_virtual",
        "venue": "ICLR 2025",
        "primary_category": "conference",
        "categories": [],
        "provenance_url": "https://iclr.cc/virtual/2025/poster/27888",
        "topic": "speculative_decoding_kv_cache",
        "topics": ["speculative_decoding_kv_cache"],
        "reason_for_inclusion": "Included for testing.",
        "selection_status": "approved",
    }

    errors, _warnings = module.validate_manual_registry_records(
        [record],
        "2024-01-01",
        "2026-05-30",
    )

    assert any(
        error["error_type"] == "missing_required_fields" and "categories" in error["fields"]
        for error in errors
    )


def test_build_approved_registry_report() -> None:
    module = _load_discovery_module()
    approved = [
        {
            "title": "Faster Cascades via Speculative Decoding",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "topics": ["speculative_decoding_kv_cache"],
            "authors": [],
            "pdf_url": None,
            "arxiv_id": None,
        }
    ]

    report = module.build_approved_registry_report(approved, Path("candidate_papers_review.csv"))

    assert report["approved_record_count"] == 1
    assert report["counts_by_topic"]["speculative_decoding_kv_cache"] == 1
    assert report["selected_titles"] == ["Faster Cascades via Speculative Decoding"]
    assert report["missing_pdf_url_count"] == 1
    assert "Phase 2A-5B" in report["next_step"]


def test_build_approved_registry_cli_missing_csv(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--build-approved-registry",
            "--candidate-review-csv",
            str(tmp_path / "missing.csv"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Candidate review CSV not found" in (result.stdout + result.stderr)


def test_write_manual_template_cli(tmp_path: Path) -> None:
    manual_template_path = tmp_path / "manual.csv"
    run_log_path = tmp_path / "manual_log.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--write-manual-template",
            "--output-manual-template",
            str(manual_template_path),
            "--run-log-path",
            str(run_log_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["mode"] == "write_manual_template"
    header = manual_template_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {
        "selection_status",
        "topic",
        "paper_id",
        "arxiv_id",
        "reason_for_inclusion",
    }.issubset(header)


def test_manual_registry_template_exists() -> None:
    assert MANUAL_TEMPLATE_PATH.exists()
    with MANUAL_TEMPLATE_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames is not None
        assert {
            "selection_status",
            "topic",
            "paper_id",
            "arxiv_id",
            "title",
            "authors",
            "published",
            "primary_category",
            "abstract_url",
            "pdf_url",
            "reason_for_inclusion",
            "notes",
        }.issubset(reader.fieldnames)


def test_approved_papers_example_exists() -> None:
    assert APPROVED_PAPERS_EXAMPLE_PATH.exists()
    records = _read_jsonl(APPROVED_PAPERS_EXAMPLE_PATH)

    assert records
    assert records[0]["selection_status"] == "example_not_approved"


def test_validate_manual_registry_with_tmp_file(tmp_path: Path) -> None:
    registry_path = tmp_path / "approved_papers.jsonl"
    report_path = tmp_path / "manual_registry_validation_report.json"
    registry_path.write_text(
        json.dumps(
            {
                "paper_id": "research_ai_manual_001",
                "arxiv_id": "2401.12345",
                "title": "Approved Test Paper",
                "authors": ["Researcher One"],
                "published": "2024-02-01",
                "source": "arXiv",
                "source_id": "arxiv_api",
                "venue": "arXiv",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "abstract_url": "https://arxiv.org/abs/2401.12345",
                "pdf_url": "https://arxiv.org/pdf/2401.12345",
                "topic": "LLM serving and inference optimization",
                "topics": ["llm_serving_inference_optimization"],
                "reason_for_inclusion": "Temporary validation fixture.",
                "selection_status": "approved",
                "provenance_url": "https://arxiv.org/abs/2401.12345",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--validate-manual-registry",
            "--manual-registry-path",
            str(registry_path),
            "--manual-validation-report",
            str(report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["validation_status"] == "passed"
    report = _load_json(report_path)
    assert report["validation_status"] == "passed"


def test_validate_manual_registry_rejects_out_of_window(tmp_path: Path) -> None:
    registry_path = tmp_path / "approved_papers.jsonl"
    report_path = tmp_path / "manual_registry_validation_report.json"
    registry_path.write_text(
        json.dumps(
            {
                "paper_id": "research_ai_manual_001",
                "arxiv_id": "2301.12345",
                "title": "Out of Window Test Paper",
                "authors": ["Researcher One"],
                "published": "2023-01-01",
                "source": "arXiv",
                "source_id": "arxiv_api",
                "venue": "arXiv",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "abstract_url": "https://arxiv.org/abs/2301.12345",
                "pdf_url": "https://arxiv.org/pdf/2301.12345",
                "topic": "LLM serving and inference optimization",
                "topics": ["llm_serving_inference_optimization"],
                "reason_for_inclusion": "Temporary validation fixture.",
                "selection_status": "approved",
                "provenance_url": "https://arxiv.org/abs/2301.12345",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--validate-manual-registry",
            "--manual-registry-path",
            str(registry_path),
            "--manual-validation-report",
            str(report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    report = _load_json(report_path)
    assert report["validation_status"] == "failed"
    assert report["errors"][0]["error_type"] == "published_date_out_of_window"


def test_failed_cli_mentions_report_and_log_paths(tmp_path: Path) -> None:
    query_plan = _load_json(QUERY_PLAN_PATH)
    query_plan["api_base_url"] = "file:///definitely_missing_arxiv_endpoint"
    query_plan_path = tmp_path / "query_plan.json"
    query_plan_path.write_text(json.dumps(query_plan), encoding="utf-8")
    report_path = tmp_path / "research_ai_discovery_report.json"
    run_log_path = tmp_path / "research_ai_discovery_run_log.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--discover",
            "--source",
            "arxiv",
            "--query-plan",
            str(query_plan_path),
            "--query-id",
            "llm_serving_inference_optimization",
            "--max-results-per-query",
            "1",
            "--max-retries",
            "0",
            "--timeout-seconds",
            "1",
            "--backoff-seconds",
            "0",
            "--output-report",
            str(report_path),
            "--run-log-path",
            str(run_log_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined_output = result.stdout + result.stderr
    assert "research_ai_discovery_report.json" in combined_output
    assert "research_ai_discovery_run_log.jsonl" in combined_output
    assert report_path.exists()
    assert run_log_path.exists()


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


def test_docs_include_rate_limit_guidance() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "HTTP 429" in text
    assert "retry/backoff" in text
    assert "--query-id" in text
    assert "--allow-partial" in text


def test_docs_include_observability_and_manual_fallback() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Discovery Observability" in text
    assert "run log" in text
    assert "Manual Paper Registry Fallback" in text
    assert "--simple-query-mode" in text
    assert "--write-manual-template" in text


def test_docs_include_paper_window_and_manual_registry() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "January 1, 2024" in text
    assert "May 30, 2026" in text
    assert "Manual Registry" in text
    assert "HTTP 429" in text
    assert "--validate-manual-registry" in text


def test_docs_include_multi_source_discovery() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Multi-source Discovery" in text
    assert "ICLR 2025" in text
    assert "Hugging Face Papers" in text
    assert "--source all" in text
    assert "arXiv API remains available" in text


def test_docs_include_approved_registry_section() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Approved Paper Registry" in text
    assert "--build-approved-registry" in text
    assert "validated approved registry" in text
    assert "Phase 2A-5B" in text


def test_committed_approved_registry_exists_after_generation() -> None:
    assert APPROVED_REGISTRY_PATH.exists()
    records = _read_jsonl(APPROVED_REGISTRY_PATH)

    assert len(records) >= 40
    assert len({record["paper_id"] for record in records}) == len(records)
    assert {record["selection_status"] for record in records} == {"approved"}
    assert all(record.get("provenance_url") for record in records)
    assert APPROVED_1000_SCALE_PAPERS_PATH.exists()
    expansion_records = _read_jsonl(APPROVED_1000_SCALE_PAPERS_PATH)
    assert len(expansion_records) == 20
    assert {record["approval_status"] for record in expansion_records} == {"approved"}
    assert all(record.get("source_url") for record in expansion_records)
