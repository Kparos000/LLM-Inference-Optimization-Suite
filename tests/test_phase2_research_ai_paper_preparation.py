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

    assert path.as_posix().endswith("research_ai_test/research_ai_test.pdf")


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


def test_extract_text_from_pdf_missing_dependency_is_graceful(tmp_path: Path) -> None:
    module = _load_preparation_module()
    pdf_path = tmp_path / "not_a_pdf.pdf"
    pdf_path.write_bytes(b"not actually a pdf")

    extracted_text, method = module.extract_text_from_pdf(pdf_path)

    assert extracted_text == ""
    assert method == "skipped_missing_dependency" or method.startswith("failed_")


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
    assert report["next_step"]


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
        "download_binary",
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
        "download_binary",
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
        delay_seconds: float,
    ) -> dict[str, Any]:
        _ = timeout_seconds, delay_seconds
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

    with mock.patch.object(module, "download_binary", side_effect=fake_download):
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
        "download_binary",
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
