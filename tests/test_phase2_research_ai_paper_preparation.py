import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/prepare_research_ai_papers.py"
DOC_PATH = ROOT / "docs/34_phase2_research_ai_paper_discovery.md"
CURATED_DOC_PATH = ROOT / "docs/35_phase2_research_ai_curated_seed.md"
SCALEUP_PLAN_DOC_PATH = ROOT / "docs/44_phase2a_1000_scaleup_plan.md"


class _FakeBinaryResponse:
    def __init__(self, body: bytes, content_type: str = "application/pdf") -> None:
        self.body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> "_FakeBinaryResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _load_preparation_module() -> Any:
    spec = importlib.util.spec_from_file_location("prepare_research_ai_papers", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _approved_1000_candidate(index: int = 1) -> dict[str, Any]:
    return {
        "approval_status": "approved",
        "arxiv_url": f"https://arxiv.org/abs/2501.{index:05d}",
        "not_for_benchmark_claims": False,
        "paper_id": f"research_ai_extra_approved_{index:02d}",
        "pdf_url": f"https://arxiv.org/pdf/2501.{index:05d}",
        "publication_year": 2025,
        "source_url": f"https://openreview.net/forum?id=extra{index:02d}",
        "title": f"Additional Research AI Paper {index:02d}",
        "topic": "llm_serving_inference_optimization",
        "venue_or_source": "ICLR 2025",
    }


def _assert_runtime_error(callback: Any) -> str:
    try:
        callback()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("Expected RuntimeError")


def test_prepare_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_parse_iclr_poster_page_extracts_openreview_and_pdf_links() -> None:
    module = _load_preparation_module()
    html_text = """
    <html>
      <head><title>Test Paper | ICLR 2025</title></head>
      <body>
        <h1>Test Paper</h1>
        <a href="https://openreview.net/forum?id=test123">OpenReview</a>
        <a href="/virtual/2025/paper/test-paper.pdf">PDF</a>
      </body>
    </html>
    """

    parsed = module.parse_iclr_poster_page(
        html_text,
        "https://iclr.cc/virtual/2025/poster/12345",
    )

    assert parsed["openreview_url"] == "https://openreview.net/forum?id=test123"
    assert parsed["pdf_url"] == "https://iclr.cc/virtual/2025/paper/test-paper.pdf"


def test_parse_openreview_page_extracts_metadata() -> None:
    module = _load_preparation_module()
    html_text = """
    <html>
      <head>
        <meta name="citation_title" content="Efficient Agent Evaluation">
        <meta name="citation_author" content="Jane Doe">
        <meta name="citation_author" content="John Smith">
        <meta name="citation_abstract" content="We evaluate efficient bounded agents.">
      </head>
      <body>
        <a href="/pdf?id=test123">Download PDF</a>
      </body>
    </html>
    """

    parsed = module.parse_openreview_page(
        html_text,
        "https://openreview.net/forum?id=test123",
    )

    assert parsed["title"] == "Efficient Agent Evaluation"
    assert parsed["authors"] == ["Jane Doe", "John Smith"]
    assert parsed["abstract"] == "We evaluate efficient bounded agents."
    assert parsed["pdf_url"] == "https://openreview.net/pdf?id=test123"


def test_clean_iclr_abstract_removes_boilerplate() -> None:
    module = _load_preparation_module()
    raw_text = (
        "This paper studies efficient bounded agent evaluation. "
        "Show more Video ICLR uses cookies Useful links About ICLR"
    )

    cleaned = module.clean_iclr_abstract(raw_text)

    assert "This paper studies efficient bounded agent evaluation." in cleaned
    assert "Show more" not in cleaned
    assert "ICLR uses cookies" not in cleaned


def test_is_noisy_abstract() -> None:
    module = _load_preparation_module()

    assert module.is_noisy_abstract("This abstract is followed by ICLR uses cookies.")
    assert not module.is_noisy_abstract("This clean abstract studies model evaluation.")


def test_is_paper_specific_openreview_url() -> None:
    module = _load_preparation_module()

    assert not module.is_paper_specific_openreview_url("https://openreview.net/group?id=ICLR.cc")
    assert module.is_paper_specific_openreview_url("https://openreview.net/forum?id=abc123")
    assert module.is_paper_specific_openreview_url("https://openreview.net/pdf?id=abc123")


def test_classify_pdf_url() -> None:
    module = _load_preparation_module()

    assert module.classify_pdf_url("https://openreview.net/pdf?id=abc") == "openreview_pdf"
    assert module.classify_pdf_url("https://iclr.cc/virtual/2025/Slides/123.pdf") == "slides_pdf"
    assert module.classify_pdf_url("https://iclr.cc/poster.pdf") == "poster_pdf"
    assert module.classify_pdf_url("https://example.com/supplement.pdf") == "supplementary_pdf"
    assert module.classify_pdf_url("https://example.com/paper.pdf") == "unknown_pdf"
    assert module.classify_pdf_url(None) == "missing"


def test_select_preferred_pdf_url_prefers_openreview() -> None:
    module = _load_preparation_module()

    selected_url, pdf_link_type = module.select_preferred_pdf_url(
        [
            "https://iclr.cc/virtual/2025/Slides/123.pdf",
            "https://openreview.net/pdf?id=abc123",
        ]
    )

    assert selected_url == "https://openreview.net/pdf?id=abc123"
    assert pdf_link_type == "openreview_pdf"


def test_parse_iclr_poster_page_cleans_abstract_and_rejects_generic_openreview() -> None:
    module = _load_preparation_module()
    html_text = """
    <html>
      <body>
        <h2>Abstract</h2>
        This paper studies efficient bounded agent evaluation.
        Show more Video ICLR uses cookies Useful links
        <a href="https://openreview.net/group?id=ICLR.cc">OpenReview</a>
        <a href="/virtual/2025/Slides/123.pdf">Slides</a>
      </body>
    </html>
    """

    parsed = module.parse_iclr_poster_page(
        html_text,
        "https://iclr.cc/virtual/2025/poster/12345",
    )

    assert parsed["abstract_quality_status"] == "clean"
    assert "ICLR uses cookies" not in parsed["abstract"]
    assert parsed["openreview_url"] is None
    assert parsed["generic_openreview_rejected"] is True
    assert parsed["pdf_link_type"] == "slides_pdf"
    assert parsed["paper_body_available"] is False


def test_parse_iclr_poster_page_paper_pdf_ready_for_extraction() -> None:
    module = _load_preparation_module()
    html_text = """
    <html>
      <body>
        <a href="https://openreview.net/pdf?id=abc123">PDF</a>
      </body>
    </html>
    """

    parsed = module.parse_iclr_poster_page(
        html_text,
        "https://iclr.cc/virtual/2025/poster/12345",
    )

    assert parsed["pdf_link_type"] == "openreview_pdf"
    assert parsed["paper_body_available"] is True
    assert parsed["ready_for_text_extraction"] is True


def test_enrich_paper_record_preserves_original_metadata() -> None:
    module = _load_preparation_module()
    record: dict[str, Any] = {
        "paper_id": "research_ai_test",
        "title": "Original Title",
        "authors": [],
        "pdf_url": None,
        "provenance_url": "https://iclr.cc/virtual/2025/poster/12345",
    }
    fetched_pages: dict[str, dict[str, Any]] = {
        "iclr": {
            "source_url": "https://iclr.cc/virtual/2025/poster/12345",
            "openreview_url": "https://openreview.net/forum?id=test123",
        },
        "openreview": {
            "source_url": "https://openreview.net/forum?id=test123",
            "authors": ["Jane Doe"],
            "abstract": "A useful abstract.",
            "pdf_url": "https://openreview.net/pdf?id=test123",
        },
    }

    enriched = module.enrich_paper_record(record, fetched_pages)

    assert enriched["provenance_url"] == record["provenance_url"]
    assert enriched["authors_enriched"] == ["Jane Doe"]
    assert enriched["abstract_enriched"] == "A useful abstract."
    assert enriched["pdf_url_enriched"] == "https://openreview.net/pdf?id=test123"
    assert enriched["openreview_url"] == "https://openreview.net/forum?id=test123"
    assert enriched["enriched_metadata_status"] == "success"


def test_build_local_pdf_path() -> None:
    module = _load_preparation_module()
    path = module.build_local_pdf_path(
        {"paper_id": "research_ai_test"},
        Path("data/raw/research_ai/papers"),
    )

    assert path.as_posix().endswith("research_ai_test/paper.pdf")


def test_build_local_pdf_path_uses_short_filename_for_long_ids() -> None:
    module = _load_preparation_module()
    long_paper_id = "research_ai_iclr_2025_virtual_" + ("very_long_title_segment_" * 8)

    path = module.build_local_pdf_path(
        {"paper_id": long_paper_id},
        Path("data/raw/research_ai/papers"),
    )

    assert path.name == "paper.pdf"
    assert len(path.parent.name) <= module.MAX_LOCAL_PAPER_DIR_CHARS
    assert path.parent.name.endswith(module.stable_short_hash(long_paper_id))


def test_detect_research_paper_sections() -> None:
    module = _load_preparation_module()
    text = """
Abstract
This is the abstract.
Introduction
This is the introduction.
Method
This is the method.
Results
These are results.
Conclusion
This is the conclusion.
"""
    record = {
        "paper_id": "research_ai_test",
        "title": "Test Paper",
        "local_text_path": "data/processed/research_ai/paper_text/research_ai_test.txt",
        "extraction_method": "unit_test",
    }

    sections = module.detect_research_paper_sections(text, record)
    section_types = {section["section_type"] for section in sections}

    assert {"abstract", "introduction", "method", "results", "conclusion"}.issubset(section_types)
    assert all(section["paper_id"] == "research_ai_test" for section in sections)


def test_detect_research_paper_sections_numbered_headings() -> None:
    module = _load_preparation_module()
    text = """
ABSTRACT
Summary.
1 Introduction
Intro text.
2 Method
Method text.
3 Experiments
Experiment text.
4 Results
Result text.
5 Conclusion
Conclusion text.
References
Reference text.
"""

    sections = module.detect_research_paper_sections(text, {"paper_id": "paper_1"})
    section_types = {section["section_type"] for section in sections}

    assert {
        "abstract",
        "introduction",
        "method",
        "experiments",
        "results",
        "conclusion",
        "references",
    }.issubset(section_types)


def test_detect_research_paper_sections_title_case_headings() -> None:
    module = _load_preparation_module()
    text = """
Abstract
Summary.
Related Work
Prior work.
Approach
Approach text.
Evaluation
Evaluation text.
Limitations
Limitations text.
Conclusion
Conclusion text.
References
Reference text.
"""

    sections = module.detect_research_paper_sections(text, {"paper_id": "paper_1"})
    section_types = {section["section_type"] for section in sections}

    assert {
        "abstract",
        "related_work",
        "approach",
        "evaluation",
        "limitations",
        "conclusion",
        "references",
    }.issubset(section_types)


def test_section_detector_avoids_long_sentence_false_positive() -> None:
    module = _load_preparation_module()
    text = """
Abstract
This is the abstract.
This sentence describes a method that is useful for evaluating systems, but it is part of a
paragraph and should not be interpreted as a standalone method section heading.
References
Reference text.
"""

    sections = module.detect_research_paper_sections(text, {"paper_id": "paper_1"})
    section_types = {section["section_type"] for section in sections}

    assert "method" not in section_types


def test_section_quality_metrics_good() -> None:
    module = _load_preparation_module()
    section_rows = [
        {"paper_id": "paper_1", "title": "Paper", "section_type": "abstract", "word_count": 50},
        {
            "paper_id": "paper_1",
            "title": "Paper",
            "section_type": "introduction",
            "word_count": 200,
        },
        {"paper_id": "paper_1", "title": "Paper", "section_type": "method", "word_count": 500},
        {"paper_id": "paper_1", "title": "Paper", "section_type": "results", "word_count": 500},
        {
            "paper_id": "paper_1",
            "title": "Paper",
            "section_type": "conclusion",
            "word_count": 100,
        },
    ]

    metrics = module.aggregate_section_quality_metrics(section_rows)

    assert metrics["papers_with_good_sections_count"] == 1
    assert metrics["section_quality_records"][0]["section_quality_status"] == "good"


def test_section_quality_metrics_poor_only_abstract_references() -> None:
    module = _load_preparation_module()
    section_rows = [
        {"paper_id": "paper_1", "title": "Paper", "section_type": "abstract", "word_count": 50},
        {"paper_id": "paper_1", "title": "Paper", "section_type": "references", "word_count": 50},
    ]

    metrics = module.aggregate_section_quality_metrics(section_rows)

    assert metrics["papers_with_poor_sections_count"] == 1
    assert metrics["poor_section_quality_titles"] == ["Paper"]


def test_audit_sections_report_flags_large_abstract() -> None:
    module = _load_preparation_module()
    section_rows = [
        {
            "section_record_id": "section_1",
            "paper_id": "paper_1",
            "title": "Paper",
            "section_type": "abstract",
            "section_title": "Abstract",
            "word_count": 1501,
        }
    ]

    report = module.build_section_quality_audit_report([], section_rows)

    assert report["suspicious_large_sections"][0]["reason"] == "abstract_above_1500_words"


def test_audit_sections_cli(tmp_path: Path) -> None:
    module = _load_preparation_module()
    text_manifest_path = tmp_path / "text_manifest.jsonl"
    sections_manifest_path = tmp_path / "sections_manifest.jsonl"
    report_path = tmp_path / "section_quality_report.json"
    module.write_jsonl(
        text_manifest_path,
        [{"paper_id": "paper_1", "title": "Paper", "text_extraction_status": "extracted"}],
    )
    module.write_jsonl(
        sections_manifest_path,
        [
            {
                "paper_id": "paper_1",
                "title": "Paper",
                "section_type": "abstract",
                "word_count": 50,
            },
            {
                "paper_id": "paper_1",
                "title": "Paper",
                "section_type": "references",
                "word_count": 50,
            },
        ],
    )
    args = Namespace(
        text_manifest_path=text_manifest_path,
        sections_manifest_path=sections_manifest_path,
        section_quality_report_path=report_path,
    )

    summary = module.audit_sections(args)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert summary["total_sections"] == 2
    assert summary["sections_by_type"] == {"abstract": 1, "references": 1}
    assert summary["papers_with_poor_sections_count"] == 1
    assert report["recommendations"]


def test_extract_text_from_pdf_missing_dependency_is_graceful(tmp_path: Path) -> None:
    module = _load_preparation_module()
    pdf_path = tmp_path / "not_a_pdf.pdf"
    pdf_path.write_bytes(b"not actually a pdf")

    extracted_text, method = module.extract_text_from_pdf(pdf_path)

    assert extracted_text == ""
    assert method == "skipped_missing_dependency" or method.startswith("failed_")


def test_get_pdf_text_extraction_backend_returns_known_value() -> None:
    module = _load_preparation_module()

    backend = module.get_pdf_text_extraction_backend()

    assert backend in {"pypdf", "PyPDF2", "missing"}


def test_extract_text_from_pdf_missing_dependency_message(tmp_path: Path) -> None:
    module = _load_preparation_module()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

    with mock.patch.object(module.importlib, "import_module", side_effect=ImportError):
        extracted_text, method = module.extract_text_from_pdf(pdf_path)

    assert extracted_text == ""
    assert method == "skipped_missing_dependency"


def test_extract_text_from_pdf_with_mock_pypdf(tmp_path: Path) -> None:
    module = _load_preparation_module()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakePdfReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("Abstract\nFirst page text."), FakePage("Conclusion\nDone.")]

    fake_pypdf_module = type("FakePypdfModule", (), {"PdfReader": FakePdfReader})

    def fake_import_module(name: str) -> Any:
        if name == "pypdf":
            return fake_pypdf_module
        raise ImportError(name)

    with mock.patch.object(module.importlib, "import_module", side_effect=fake_import_module):
        extracted_text, method = module.extract_text_from_pdf(pdf_path)

    assert "First page text" in extracted_text
    assert "Conclusion" in extracted_text
    assert method == "pypdf"


def test_build_paper_preparation_report() -> None:
    module = _load_preparation_module()
    report = module.build_paper_preparation_report(
        approved_records=[{"paper_id": "paper_1", "source": "ICLR", "topics": ["agents"]}],
        enriched_records=[
            {
                "paper_id": "paper_1",
                "source": "ICLR",
                "topics": ["agents"],
                "enriched_metadata_status": "partial",
                "pdf_url_enriched": "https://openreview.net/pdf?id=test123",
            }
        ],
        text_manifest_rows=[{"text_extraction_status": "skipped_missing_dependency"}],
        section_rows=[],
        output_files={"report_path": "report.json"},
    )

    assert report["phase"] == "2A-5A-Text"
    assert report["approved_record_count"] == 1
    assert report["enriched_record_count"] == 1
    assert report["pdf_urls_found"] == 1
    assert report["text_extraction_skipped_count"] == 1
    assert report["pdf_text_backend"] in {"pypdf", "PyPDF2", "missing"}
    assert report["next_step"]


def test_extract_text_report_includes_backend() -> None:
    module = _load_preparation_module()
    report = module.build_paper_preparation_report(
        approved_records=[],
        enriched_records=[],
        text_manifest_rows=[{"paper_id": "paper_1", "text_extraction_status": "extracted"}],
        section_rows=[],
        output_files={},
        pdf_text_backend="pypdf",
    )

    assert report["pdf_text_backend"] == "pypdf"
    assert report["text_extracted_count"] == 1
    assert report["no_sections_detected_count"] == 1


def test_build_paper_preparation_report_includes_quality_counts() -> None:
    module = _load_preparation_module()
    report = module.build_paper_preparation_report(
        approved_records=[{"paper_id": "paper_1", "source": "ICLR", "topics": ["agents"]}],
        enriched_records=[
            {
                "paper_id": "paper_1",
                "source": "ICLR",
                "topics": ["agents"],
                "abstract_quality_status": "clean",
                "paper_body_available": True,
                "ready_for_text_extraction": True,
                "pdf_link_type": "openreview_pdf",
                "pdf_url_candidates": [
                    {
                        "url": "https://openreview.net/pdf?id=abc123",
                        "pdf_link_type": "openreview_pdf",
                    },
                    {
                        "url": "https://iclr.cc/virtual/2025/Slides/123.pdf",
                        "pdf_link_type": "slides_pdf",
                    },
                ],
            },
            {
                "paper_id": "paper_2",
                "source": "ICLR",
                "topics": ["agents"],
                "abstract_quality_status": "noisy",
                "paper_body_available": False,
                "ready_for_text_extraction": False,
                "pdf_link_type": "slides_pdf",
                "pdf_url_candidates": [
                    {
                        "url": "https://iclr.cc/virtual/2025/Slides/456.pdf",
                        "pdf_link_type": "slides_pdf",
                    }
                ],
            },
        ],
        text_manifest_rows=[],
        section_rows=[],
        output_files={},
    )

    assert report["clean_abstract_count"] == 1
    assert report["noisy_abstract_count"] == 1
    assert report["paper_body_available_count"] == 1
    assert report["slides_pdf_count"] == 2
    assert report["records_ready_for_text_extraction"] == 1


def test_download_pdfs_skips_non_paper_pdfs_by_default(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "slides_only",
            "title": "Slides Only",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/1",
            "pdf_url_enriched": "https://iclr.cc/virtual/2025/Slides/1.pdf",
            "pdf_link_type": "slides_pdf",
            "paper_body_available": False,
            "ready_for_text_extraction": False,
        },
        {
            "paper_id": "paper_body",
            "title": "Paper Body",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/2",
            "pdf_url_enriched": "https://openreview.net/pdf?id=abc123",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=tmp_path / "text_manifest.jsonl",
        sections_manifest_path=tmp_path / "sections_manifest.jsonl",
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=1,
        request_delay_seconds=0,
    )

    with mock.patch.object(
        module,
        "download_binary_with_retries",
        return_value={
            "download_status": "downloaded",
            "source_url": "https://openreview.net/pdf?id=abc123",
            "local_pdf_path": str(tmp_path / "paper.pdf"),
            "file_size_bytes": 10,
            "sha256": "abc",
            "error_message": "",
        },
    ) as download_mock:
        summary = module.download_pdfs(args)

    assert summary["pdfs_skipped_not_full_paper"] == 1
    assert summary["pdfs_downloaded_by_type"] == {"openreview_pdf": 1}
    assert download_mock.call_count == 1

    module.write_jsonl(enriched_path, records)
    args.include_non_paper_pdfs = True
    with mock.patch.object(
        module,
        "download_binary_with_retries",
        return_value={
            "download_status": "downloaded",
            "source_url": "https://example.com/paper.pdf",
            "local_pdf_path": str(tmp_path / "paper.pdf"),
            "file_size_bytes": 10,
            "sha256": "abc",
            "error_message": "",
        },
    ) as download_mock:
        module.download_pdfs(args)

    assert download_mock.call_count == 2


def test_download_binary_creates_parent_directories(tmp_path: Path) -> None:
    module = _load_preparation_module()
    destination = tmp_path / "nested" / "paper" / "paper.pdf"

    with mock.patch.object(
        module.urllib.request,
        "urlopen",
        return_value=_FakeBinaryResponse(b"%PDF-1.4 fake pdf"),
    ):
        result = module.download_binary(
            "https://openreview.net/pdf?id=abc123",
            destination,
            timeout_seconds=1,
            delay_seconds=0,
        )

    assert destination.exists()
    assert destination.parent.exists()
    assert result["status"] == "downloaded"
    assert result["bytes_written"] > 0
    assert result["sha256"]
    assert result["content_type"] == "application/pdf"


def test_download_binary_retries_429_then_success(tmp_path: Path) -> None:
    module = _load_preparation_module()
    destination = tmp_path / "paper.pdf"
    error = module.urllib.error.HTTPError(
        "https://openreview.net/pdf?id=abc123",
        429,
        "Too Many Requests",
        {},
        None,
    )

    with (
        mock.patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=[error, _FakeBinaryResponse(b"%PDF-1.4 fake pdf")],
        ) as urlopen_mock,
        mock.patch.object(module.time, "sleep") as sleep_mock,
    ):
        result = module.download_binary_with_retries(
            "https://openreview.net/pdf?id=abc123",
            destination,
            timeout_seconds=1,
            request_delay_seconds=0,
            max_retries=1,
            backoff_seconds=2,
        )

    assert destination.exists()
    assert result["status"] == "downloaded"
    assert result["attempts"] == 2
    assert result["sha256"]
    assert urlopen_mock.call_count == 2
    sleep_mock.assert_called_once_with(2)


def test_download_binary_honors_retry_after(tmp_path: Path) -> None:
    module = _load_preparation_module()
    destination = tmp_path / "paper.pdf"
    error = module.urllib.error.HTTPError(
        "https://openreview.net/pdf?id=abc123",
        429,
        "Too Many Requests",
        {"Retry-After": "7"},
        None,
    )

    with (
        mock.patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=[error, _FakeBinaryResponse(b"%PDF-1.4 fake pdf")],
        ),
        mock.patch.object(module.time, "sleep") as sleep_mock,
    ):
        result = module.download_binary_with_retries(
            "https://openreview.net/pdf?id=abc123",
            destination,
            timeout_seconds=1,
            request_delay_seconds=0,
            max_retries=1,
            backoff_seconds=30,
        )

    assert result["status"] == "downloaded"
    sleep_mock.assert_called_once_with(7.0)


def test_download_pdfs_continues_after_one_failure(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "paper_success",
            "title": "Paper Success",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/1",
            "pdf_url_enriched": "https://openreview.net/pdf?id=success",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
        {
            "paper_id": "paper_failure",
            "title": "Paper Failure",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/2",
            "pdf_url_enriched": "https://openreview.net/pdf?id=failure",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=tmp_path / "text_manifest.jsonl",
        sections_manifest_path=tmp_path / "sections_manifest.jsonl",
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=1,
        request_delay_seconds=0,
    )

    def fake_download(
        url: str,
        destination: Path,
        timeout_seconds: int,
        request_delay_seconds: float,
        max_retries: int,
        backoff_seconds: float,
    ) -> dict[str, Any]:
        _ = timeout_seconds, request_delay_seconds, max_retries, backoff_seconds
        if "failure" in url:
            return {
                "status": "failed",
                "download_status": "failed",
                "url": url,
                "source_url": url,
                "destination": str(destination),
                "local_pdf_path": str(destination),
                "bytes_written": 0,
                "file_size_bytes": 0,
                "sha256": None,
                "content_type": None,
                "downloaded_at_utc": "2026-05-18T00:00:00+00:00",
                "error_message": "simulated failure",
            }
        return {
            "status": "downloaded",
            "download_status": "downloaded",
            "url": url,
            "source_url": url,
            "destination": str(destination),
            "local_pdf_path": str(destination),
            "bytes_written": 10,
            "file_size_bytes": 10,
            "sha256": "abc",
            "content_type": "application/pdf",
            "downloaded_at_utc": "2026-05-18T00:00:00+00:00",
            "error_message": "",
        }

    with mock.patch.object(module, "download_binary_with_retries", side_effect=fake_download):
        summary = module.download_pdfs(args)

    assert summary["pdfs_downloaded"] == 1
    assert summary["pdf_download_failures"] == 1
    assert summary["pdf_download_failure_details"][0]["paper_id"] == "paper_failure"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pdf_download_failure_details"][0]["error_message"] == "simulated failure"


def test_download_pdfs_all_fail_returns_failure(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "paper_failure",
            "title": "Paper Failure",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/2",
            "pdf_url_enriched": "https://openreview.net/pdf?id=failure",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        }
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=tmp_path / "text_manifest.jsonl",
        sections_manifest_path=tmp_path / "sections_manifest.jsonl",
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=1,
        request_delay_seconds=0,
    )

    with mock.patch.object(
        module,
        "download_binary_with_retries",
        return_value={
            "status": "failed",
            "download_status": "failed",
            "url": "https://openreview.net/pdf?id=failure",
            "source_url": "https://openreview.net/pdf?id=failure",
            "destination": str(tmp_path / "paper.pdf"),
            "local_pdf_path": str(tmp_path / "paper.pdf"),
            "bytes_written": 0,
            "file_size_bytes": 0,
            "sha256": None,
            "content_type": None,
            "downloaded_at_utc": "2026-05-18T00:00:00+00:00",
            "error_message": "simulated failure",
        },
    ):
        try:
            module.download_pdfs(args)
        except RuntimeError as exc:
            assert "All attempted PDF downloads failed" in str(exc)
        else:
            raise AssertionError("Expected all failed downloads to raise RuntimeError")

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pdf_download_failures"] == 1


def test_download_pdfs_failed_only_filters_records(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "paper_success",
            "title": "Paper Success",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/1",
            "pdf_url_enriched": "https://openreview.net/pdf?id=success",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
        {
            "paper_id": "paper_failure",
            "title": "Paper Failure",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/2",
            "pdf_url_enriched": "https://openreview.net/pdf?id=failure",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    report_path.write_text(
        json.dumps(
            {
                "pdf_download_failure_details": [
                    {"paper_id": "paper_failure", "error_message": "HTTP Error 429"}
                ]
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=tmp_path / "text_manifest.jsonl",
        sections_manifest_path=tmp_path / "sections_manifest.jsonl",
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=1,
        request_delay_seconds=0,
        failed_only=True,
        paper_id="",
        download_max_retries=1,
        download_backoff_seconds=0,
    )

    with mock.patch.object(
        module,
        "download_binary_with_retries",
        return_value={
            "status": "downloaded",
            "download_status": "downloaded",
            "url": "https://openreview.net/pdf?id=failure",
            "source_url": "https://openreview.net/pdf?id=failure",
            "local_pdf_path": str(tmp_path / "paper.pdf"),
            "file_size_bytes": 10,
            "sha256": "abc",
            "attempts": 1,
            "status_code": None,
            "error_message": "",
        },
    ) as download_mock:
        summary = module.download_pdfs(args)

    assert summary["records_attempted"] == 1
    assert summary["failed_only"] is True
    assert download_mock.call_count == 1
    assert download_mock.call_args.args[0] == "https://openreview.net/pdf?id=failure"


def test_download_pdfs_paper_id_filter(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "paper_a",
            "title": "Paper A",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/1",
            "pdf_url_enriched": "https://openreview.net/pdf?id=a",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
        {
            "paper_id": "paper_b",
            "title": "Paper B",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/2",
            "pdf_url_enriched": "https://openreview.net/pdf?id=b",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        },
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=tmp_path / "text_manifest.jsonl",
        sections_manifest_path=tmp_path / "sections_manifest.jsonl",
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=1,
        request_delay_seconds=0,
        failed_only=False,
        paper_id="paper_b",
        download_max_retries=1,
        download_backoff_seconds=0,
    )

    with mock.patch.object(
        module,
        "download_binary_with_retries",
        return_value={
            "status": "downloaded",
            "download_status": "downloaded",
            "url": "https://openreview.net/pdf?id=b",
            "source_url": "https://openreview.net/pdf?id=b",
            "local_pdf_path": str(tmp_path / "paper.pdf"),
            "file_size_bytes": 10,
            "sha256": "abc",
            "attempts": 1,
            "status_code": None,
            "error_message": "",
        },
    ) as download_mock:
        summary = module.download_pdfs(args)

    assert summary["records_attempted"] == 1
    assert summary["paper_id_filter"] == "paper_b"
    assert download_mock.call_count == 1
    assert download_mock.call_args.args[0] == "https://openreview.net/pdf?id=b"


def test_download_report_includes_retry_policy(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "paper_retry",
            "title": "Paper Retry",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/1",
            "pdf_url_enriched": "https://openreview.net/pdf?id=retry",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        }
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=tmp_path / "text_manifest.jsonl",
        sections_manifest_path=tmp_path / "sections_manifest.jsonl",
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=5,
        request_delay_seconds=12,
        failed_only=False,
        paper_id="paper_retry",
        download_max_retries=4,
        download_backoff_seconds=60,
    )

    with mock.patch.object(
        module,
        "download_binary_with_retries",
        return_value={
            "status": "downloaded",
            "download_status": "downloaded",
            "url": "https://openreview.net/pdf?id=retry",
            "source_url": "https://openreview.net/pdf?id=retry",
            "local_pdf_path": str(tmp_path / "paper.pdf"),
            "file_size_bytes": 10,
            "sha256": "abc",
            "attempts": 1,
            "status_code": None,
            "error_message": "",
        },
    ):
        summary = module.download_pdfs(args)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["download_max_retries"] == 4
    assert summary["download_backoff_seconds"] == 60.0
    assert summary["request_delay_seconds"] == 12.0
    assert summary["failed_only"] is False
    assert summary["paper_id_filter"] == "paper_retry"
    assert report["download_max_retries"] == 4
    assert report["download_backoff_seconds"] == 60.0
    assert report["request_delay_seconds"] == 12.0
    assert report["failed_only"] is False
    assert report["paper_id_filter"] == "paper_retry"


def test_extract_text_missing_pdf_warning(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    enriched_path = tmp_path / "enriched.jsonl"
    text_manifest_path = tmp_path / "text_manifest.jsonl"
    sections_manifest_path = tmp_path / "sections_manifest.jsonl"
    report_path = tmp_path / "report.json"
    records = [
        {
            "paper_id": "missing_pdf",
            "title": "Missing PDF",
            "source": "ICLR",
            "venue": "ICLR 2025",
            "provenance_url": "https://iclr.cc/virtual/2025/poster/1",
            "pdf_url_enriched": "https://openreview.net/pdf?id=missing",
            "pdf_link_type": "openreview_pdf",
            "paper_body_available": True,
            "ready_for_text_extraction": True,
        }
    ]
    module.write_jsonl(approved_path, records)
    module.write_jsonl(enriched_path, records)
    args = Namespace(
        approved_registry_path=approved_path,
        enriched_registry_path=enriched_path,
        raw_paper_dir=tmp_path / "papers",
        paper_text_dir=tmp_path / "text",
        text_manifest_path=text_manifest_path,
        sections_manifest_path=sections_manifest_path,
        report_path=report_path,
        limit=0,
        skip_existing=False,
        include_non_paper_pdfs=False,
        timeout_seconds=1,
        request_delay_seconds=0,
    )

    module.extract_text(args)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_warning = (
        "No local PDFs were found. Run --download-pdfs --skip-existing before --extract-text."
    )
    assert any(warning == expected_warning for warning in report["warnings"])


def test_research_ai_40_paper_expansion_report_shape(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    sections_path = tmp_path / "sections.jsonl"
    expansion_report_path = tmp_path / "expansion_report.json"
    review_csv_path = tmp_path / "review.csv"
    quality_report_path = tmp_path / "quality.json"
    template_path = tmp_path / "candidate_template.jsonl"
    module.write_jsonl(
        approved_path,
        [
            {
                "paper_id": "paper_1",
                "title": "Paper 1",
                "selection_status": "approved",
            }
        ],
    )
    module.write_jsonl(
        sections_path,
        [
            {"paper_id": "paper_1", "section_type": "abstract"},
            {"paper_id": "paper_1", "section_type": "method"},
        ],
    )
    args = Namespace(
        approved_registry_path=approved_path,
        sections_manifest_path=sections_path,
        expansion_report_path=expansion_report_path,
        expansion_review_csv_path=review_csv_path,
        expanded_section_quality_report_path=quality_report_path,
        candidate_template_path=template_path,
    )

    summary = module.build_research_ai_40_paper_expansion(args)
    report = json.loads(expansion_report_path.read_text(encoding="utf-8"))

    assert summary["phase"] == "2A-12C"
    assert report["phase"] == "2A-12C"
    assert report["current_approved_paper_count"] == 1
    assert report["target_approved_paper_count"] == 40
    assert report["additional_papers_needed"] == 39
    assert report["current_section_count"] == 2
    assert report["target_section_count_range"] == {"min": 800, "max": 1200}
    assert report["expansion_ready_for_1000"] is False
    assert template_path.exists()
    assert review_csv_path.exists()
    assert quality_report_path.exists()


def test_research_ai_expansion_does_not_count_placeholders_as_approved(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    sections_path = tmp_path / "sections.jsonl"
    module.write_jsonl(
        approved_path,
        [
            {
                "paper_id": "approved_paper",
                "title": "Approved Paper",
                "selection_status": "approved",
            },
            {
                "approval_status": "needs_review",
                "missing_pdf_or_section_text": True,
                "not_for_benchmark_claims": True,
                "paper_id": "research_ai_1000_candidate_slot_01",
                "selection_status": "needs_review",
                "title": "Needs review Research AI paper slot 01",
            },
        ],
    )
    module.write_jsonl(sections_path, [])
    args = Namespace(
        approved_registry_path=approved_path,
        sections_manifest_path=sections_path,
        expansion_report_path=tmp_path / "expansion_report.json",
        expansion_review_csv_path=tmp_path / "review.csv",
        expanded_section_quality_report_path=tmp_path / "quality.json",
        candidate_template_path=tmp_path / "candidate_template.jsonl",
    )

    summary = module.build_research_ai_40_paper_expansion(args)

    assert summary["current_approved_paper_count"] == 1
    assert summary["additional_papers_needed"] == 39


def test_validate_candidate_papers_does_not_count_placeholders(tmp_path: Path) -> None:
    module = _load_preparation_module()
    candidate_path = tmp_path / "candidate_template.jsonl"
    validation_report = tmp_path / "validation_report.json"
    module.write_jsonl(candidate_path, module.build_research_ai_candidate_slots(2))
    args = Namespace(candidate_papers=candidate_path, validation_report=validation_report)

    summary = module.validate_1000_scale_candidate_papers(args)
    report = json.loads(validation_report.read_text(encoding="utf-8"))

    assert summary["validation_passed"] is True
    assert summary["placeholder_record_count"] == 2
    assert summary["approved_candidate_count"] == 0
    assert summary["placeholders_counted_as_approved"] is False
    assert report["ready_for_ingest"] is False


def test_validate_candidate_papers_accepts_real_approved_records(tmp_path: Path) -> None:
    module = _load_preparation_module()
    candidate_path = tmp_path / "candidate_papers.jsonl"
    validation_report = tmp_path / "validation_report.json"
    module.write_jsonl(candidate_path, [_approved_1000_candidate(1), _approved_1000_candidate(2)])
    args = Namespace(candidate_papers=candidate_path, validation_report=validation_report)

    summary = module.validate_1000_scale_candidate_papers(args)
    report = json.loads(validation_report.read_text(encoding="utf-8"))

    assert summary["validation_passed"] is True
    assert summary["approved_candidate_count"] == 2
    assert summary["ready_for_ingest"] is True
    assert report["invalid_record_count"] == 0


def test_ingest_rejects_placeholders(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_input = tmp_path / "approved_input.jsonl"
    approved_registry = tmp_path / "approved_registry.jsonl"
    ingest_report = tmp_path / "ingest_report.json"
    module.write_jsonl(approved_input, module.build_research_ai_candidate_slots(1))
    module.write_jsonl(approved_registry, [{"paper_id": "existing", "title": "Existing"}])
    args = Namespace(
        approved_papers_input=approved_input,
        approved_registry=approved_registry,
        approved_registry_path=approved_registry,
        ingest_report=ingest_report,
    )

    message = _assert_runtime_error(lambda: module.ingest_approved_1000_scale_papers(args))
    report = json.loads(ingest_report.read_text(encoding="utf-8"))
    registry_rows = module.read_jsonl(approved_registry)

    assert "failed validation" in message
    assert report["ingest_passed"] is False
    assert report["registry_updated"] is False
    assert report["placeholder_record_count"] == 1
    assert registry_rows == [{"paper_id": "existing", "title": "Existing"}]


def test_ingest_rejects_duplicate_papers(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_input = tmp_path / "approved_input.jsonl"
    approved_registry = tmp_path / "approved_registry.jsonl"
    ingest_report = tmp_path / "ingest_report.json"
    duplicate = _approved_1000_candidate(1)
    module.write_jsonl(approved_input, [duplicate])
    module.write_jsonl(
        approved_registry,
        [{"paper_id": duplicate["paper_id"], "title": "Previously Approved Paper"}],
    )
    args = Namespace(
        approved_papers_input=approved_input,
        approved_registry=approved_registry,
        approved_registry_path=approved_registry,
        ingest_report=ingest_report,
    )

    _assert_runtime_error(lambda: module.ingest_approved_1000_scale_papers(args))
    report = json.loads(ingest_report.read_text(encoding="utf-8"))
    registry_rows = module.read_jsonl(approved_registry)

    assert report["ingest_passed"] is False
    assert report["duplicate_existing_issue_count"] == 1
    assert report["registry_updated"] is False
    assert len(registry_rows) == 1


def test_ingest_merges_real_approved_records(tmp_path: Path) -> None:
    module = _load_preparation_module()
    approved_input = tmp_path / "approved_input.jsonl"
    approved_registry = tmp_path / "approved_registry.jsonl"
    ingest_report = tmp_path / "ingest_report.json"
    module.write_jsonl(approved_input, [_approved_1000_candidate(1), _approved_1000_candidate(2)])
    module.write_jsonl(
        approved_registry,
        [{"paper_id": "existing", "selection_status": "approved", "title": "Existing"}],
    )
    args = Namespace(
        approved_papers_input=approved_input,
        approved_registry=approved_registry,
        approved_registry_path=approved_registry,
        ingest_report=ingest_report,
    )

    summary = module.ingest_approved_1000_scale_papers(args)
    report = json.loads(ingest_report.read_text(encoding="utf-8"))
    registry_rows = module.read_jsonl(approved_registry)

    assert summary["approved_records_ingested"] == 2
    assert report["registry_updated"] is True
    assert len(registry_rows) == 3
    assert registry_rows[-1]["selection_status"] == "approved"
    assert registry_rows[-1]["metadata"]["requires_pdf_download_or_text_extraction"] is True


def test_build_40_paper_expansion_detects_40_approved_but_missing_sections(
    tmp_path: Path,
) -> None:
    module = _load_preparation_module()
    approved_path = tmp_path / "approved.jsonl"
    sections_path = tmp_path / "sections.jsonl"
    module.write_jsonl(
        approved_path,
        [
            {
                **_approved_1000_candidate(index),
                "missing_pdf_or_section_text": False,
                "selection_status": "approved",
            }
            for index in range(1, 41)
        ],
    )
    module.write_jsonl(sections_path, [{"paper_id": "research_ai_extra_approved_01"}])
    args = Namespace(
        approved_registry_path=approved_path,
        sections_manifest_path=sections_path,
        expansion_report_path=tmp_path / "expansion_report.json",
        expansion_review_csv_path=tmp_path / "review.csv",
        expanded_section_quality_report_path=tmp_path / "quality.json",
        candidate_template_path=tmp_path / "candidate_template.jsonl",
    )

    summary = module.build_research_ai_40_paper_expansion(args)

    assert summary["current_approved_paper_count"] == 40
    assert summary["additional_papers_needed"] == 0
    assert summary["expansion_ready_for_1000"] is False
    assert "section_extraction_required_for_approved_papers" in summary["missing_requirements"]


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
    assert summary["phase"] == "2A-5A-Text"
    assert summary["will_download_pdfs"] is False


def test_docs_include_paper_detail_acquisition() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-5A-Text" in text
    assert "Paper Detail Acquisition" in text
    assert "--enrich-metadata" in text
    assert "--download-pdfs" in text
    assert "--extract-text" in text


def test_docs_include_text_qa_quality_gate() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-5A-Text-QA" in text
    assert "Metadata Quality Gate" in text
    assert "paper_body_available" in text
    assert "slides_pdf" in text
    assert "OpenReview" in text


def test_docs_include_pdf_text_dependency_section() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "PDF Text Extraction Dependency" in text
    assert "pypdf" in text
    assert "--extract-text" in text


def test_docs_include_section_qa() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-5A-Text-Section-QA" in text
    assert "--audit-sections" in text
    assert "section quality" in text


def test_docs_include_research_ai_ingest_commands() -> None:
    curated_docs = CURATED_DOC_PATH.read_text(encoding="utf-8")
    plan_docs = SCALEUP_PLAN_DOC_PATH.read_text(encoding="utf-8")

    validate_command = (
        "python scripts/phase2/prepare_research_ai_papers.py --validate-1000-scale-candidate-papers"
    )
    ingest_command = (
        "python scripts/phase2/prepare_research_ai_papers.py --ingest-approved-1000-scale-papers"
    )
    expansion_command = (
        "python scripts/phase2/prepare_research_ai_papers.py --build-40-paper-expansion"
    )
    assert validate_command in curated_docs
    assert ingest_command in curated_docs
    assert expansion_command in curated_docs
    assert validate_command in plan_docs
    assert ingest_command in plan_docs
    assert "placeholders are not benchmark evidence" in curated_docs
    assert "no LLM calls" in curated_docs
    assert "section quality" in curated_docs
