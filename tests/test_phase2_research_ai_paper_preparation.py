import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/prepare_research_ai_papers.py"
DOC_PATH = ROOT / "docs/34_phase2_research_ai_paper_discovery.md"


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
