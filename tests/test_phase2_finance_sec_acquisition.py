import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch

REGISTRY_PATH = Path("data/sources/finance_ticker_registry.json")
SCRIPT_PATH = Path("scripts/phase2/finance_sec_acquisition.py")
EXPECTED_TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"}
REQUIRED_COMPANY_FIELDS = {
    "company_name",
    "ticker",
    "cik",
    "cik_no_leading_zeros",
    "fiscal_year_end",
    "submissions_url",
    "companyfacts_url",
    "sec_company_browser_url",
    "local_raw_submissions_path",
    "local_raw_companyfacts_path",
    "local_processed_dir",
    "included_in_phase2a_finance_pilot",
}


class FakeUrlopenResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}

    def __enter__(self) -> "FakeUrlopenResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def _load_registry() -> dict[str, Any]:
    parsed_json = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed_json, dict)
    return parsed_json


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("finance_sec_acquisition", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _msft_company() -> dict[str, Any]:
    registry = _load_registry()
    companies = registry["companies"]
    assert isinstance(companies, list)
    return next(company for company in companies if company["ticker"] == "MSFT")


def _fake_submissions() -> dict[str, Any]:
    return {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0001193125-26-191507",
                    "0001193125-26-100000",
                    "0001193125-25-200000",
                    "0001193125-25-300000",
                    "0001193125-25-400000",
                ],
                "filingDate": [
                    "2026-04-25",
                    "2026-01-25",
                    "2025-07-30",
                    "2025-03-01",
                    "2025-04-01",
                ],
                "reportDate": [
                    "2026-03-31",
                    "2025-12-31",
                    "2025-06-30",
                    "2025-02-28",
                    "2025-03-31",
                ],
                "acceptanceDateTime": [
                    "20260425170000",
                    "20260125170000",
                    "20250730170000",
                    "20250301170000",
                    "20250401170000",
                ],
                "act": ["34", "34", "34", "34", "34"],
                "form": ["10-Q", "10-K", "8-K", "8-K", "4"],
                "fileNumber": ["001-00000", "001-00000", "001-00000", "001-00000", "001-00000"],
                "filmNumber": ["1", "2", "3", "4", "5"],
                "items": ["", "", "2.02,9.01", "8.01", ""],
                "size": [100, 200, 300, 400, 500],
                "isXBRL": [1, 1, 0, 0, 0],
                "isInlineXBRL": [1, 1, 0, 0, 0],
                "primaryDocument": [
                    "msft-20260331.htm",
                    "msft-20251231.htm",
                    "msft-8k.htm",
                    "msft-other-8k.htm",
                    "msft-ownership.htm",
                ],
                "primaryDocDescription": [
                    "10-Q",
                    "10-K",
                    "8-K",
                    "8-K",
                    "Statement of changes",
                ],
            }
        }
    }


def _fake_companyfacts() -> dict[str, Any]:
    return {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "description": "Revenue recognized during the period.",
                    "units": {
                        "USD": [
                            {
                                "form": "10-K",
                                "fy": 2025,
                                "fp": "FY",
                                "filed": "2026-01-25",
                                "end": "2025-12-31",
                                "val": 100,
                            },
                            {
                                "form": "10-Q",
                                "fy": 2026,
                                "fp": "Q1",
                                "filed": "2026-04-25",
                                "end": "2026-03-31",
                                "val": 30,
                            },
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "label": "Net Income Loss",
                    "description": "Net income or loss.",
                    "units": {
                        "USD": [
                            {
                                "form": "10-K",
                                "fy": 2025,
                                "fp": "FY",
                                "filed": "2026-01-25",
                                "end": "2025-12-31",
                                "val": 20,
                            }
                        ]
                    },
                },
            }
        }
    }


def _fake_selected_filing_rows() -> list[dict[str, Any]]:
    return [
        {
            "record_id": "finance_sec_MSFT_10Q_000119312526191507",
            "source_id": "finance_sec_edgar_xbrl",
            "vertical": "finance",
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "cik": "0000789019",
            "cik_no_leading_zeros": "789019",
            "fiscal_year_end": "0630",
            "form": "10-Q",
            "filing_date": "2026-04-25",
            "report_date": "2026-03-31",
            "accession_number": "0001193125-26-191507",
            "items": "",
            "primary_document": "msft-20260331.htm",
            "primary_doc_description": "10-Q",
            "derived_filing_url": (
                "https://www.sec.gov/Archives/edgar/data/"
                "789019/000119312526191507/msft-20260331.htm"
            ),
            "selection_reason": "Selected quarterly filing candidate from target year range.",
        },
        {
            "record_id": "finance_sec_MSFT_8K_000119312526200000",
            "source_id": "finance_sec_edgar_xbrl",
            "vertical": "finance",
            "ticker": "MSFT",
            "company_name": "Microsoft Corporation",
            "cik": "0000789019",
            "cik_no_leading_zeros": "789019",
            "fiscal_year_end": "0630",
            "form": "8-K",
            "filing_date": "2026-04-26",
            "report_date": "2026-04-26",
            "accession_number": "0001193125-26-200000",
            "items": "2.02,9.01",
            "primary_document": "msft-8k.htm",
            "primary_doc_description": "8-K",
            "derived_filing_url": (
                "https://www.sec.gov/Archives/edgar/data/789019/000119312526200000/msft-8k.htm"
            ),
            "selection_reason": "Selected earnings/results 8-K because item 2.02 is present.",
        },
        {
            "record_id": "finance_sec_AAPL_10K_000032019325000001",
            "source_id": "finance_sec_edgar_xbrl",
            "vertical": "finance",
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "cik": "0000320193",
            "cik_no_leading_zeros": "320193",
            "fiscal_year_end": "0926",
            "form": "10-K",
            "filing_date": "2025-11-01",
            "report_date": "2025-09-27",
            "accession_number": "0000320193-25-000001",
            "items": "",
            "primary_document": "aapl-20250927.htm",
            "primary_doc_description": "10-K",
            "derived_filing_url": (
                "https://www.sec.gov/Archives/edgar/data/"
                "320193/000032019325000001/aapl-20250927.htm"
            ),
            "selection_reason": "Selected annual filing candidate from target year range.",
        },
    ]


def _fake_document_row(form: str = "10-Q") -> dict[str, Any]:
    accession_number = "0001193125-26-191507"
    form_normalized = form.replace("-", "")
    return {
        "document_record_id": f"finance_doc_MSFT_{form_normalized}_000119312526191507",
        "source_manifest_record_id": "finance_sec_MSFT_10Q_000119312526191507",
        "source_id": "finance_sec_edgar_xbrl",
        "vertical": "finance",
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "cik": "0000789019",
        "cik_no_leading_zeros": "789019",
        "fiscal_year_end": "0630",
        "form": form,
        "filing_date": "2026-04-25",
        "report_date": "2026-03-31",
        "accession_number": accession_number,
        "items": "2.02,9.01" if form == "8-K" else "",
        "primary_document": "msft-20260331.htm",
        "primary_doc_description": form,
        "derived_filing_url": (
            "https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm"
        ),
        "local_html_path": "data/raw/finance/sec/filings/MSFT/10-Q/test.htm",
        "download_status": "downloaded",
        "file_size_bytes": 100,
        "sha256": "abc123",
        "selection_reason": "Selected quarterly filing candidate from target year range.",
        "downloaded_at_utc": "2026-05-17T00:00:00+00:00",
    }


def test_finance_sec_xbrl_pilot_doc_contains_required_terms() -> None:
    doc_path = Path("docs/32_phase2_finance_sec_xbrl_pilot.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Phase 2 Finance SEC/XBRL Pilot",
        "Finance Data Assets",
        "Approved Company Universe",
        "SEC URL Derivation",
        "SEC Access Rules",
        "2A-3A",
        "2A-3B",
        "2A-3C",
        "2A-3D",
    ]

    for term in required_terms:
        assert term in content


def test_finance_ticker_registry_exists_parses_and_contains_eight_companies() -> None:
    assert REGISTRY_PATH.exists()

    registry = _load_registry()
    companies = registry.get("companies")

    assert isinstance(companies, list)
    assert len(companies) == 8


def test_finance_ticker_registry_contains_expected_tickers() -> None:
    registry = _load_registry()
    companies = registry["companies"]
    assert isinstance(companies, list)

    tickers = {company["ticker"] for company in companies if isinstance(company, dict)}

    assert tickers == EXPECTED_TICKERS


def test_finance_ticker_registry_company_fields_and_urls() -> None:
    registry = _load_registry()
    companies = registry["companies"]
    assert isinstance(companies, list)

    for company in companies:
        assert isinstance(company, dict)
        assert REQUIRED_COMPANY_FIELDS.issubset(company)
        assert str(company["submissions_url"]).startswith("https://data.sec.gov/submissions/CIK")
        assert str(company["companyfacts_url"]).startswith(
            "https://data.sec.gov/api/xbrl/companyfacts/CIK"
        )


def test_normalize_forms() -> None:
    module = _load_script_module()
    normalize_forms = cast(Callable[[str], list[str]], module.__dict__["normalize_forms"])

    assert normalize_forms("10-K,10-Q,8-K") == ["10-K", "10-Q", "8-K"]
    assert normalize_forms(" 10-k , 8-k ") == ["10-K", "8-K"]


def test_parse_8k_items() -> None:
    module = _load_script_module()
    parse_8k_items = cast(Callable[[object], set[str]], module.__dict__["parse_8k_items"])

    assert parse_8k_items("2.02,9.01") == {"2.02", "9.01"}
    assert parse_8k_items(" 2.02 , 9.01 ") == {"2.02", "9.01"}
    assert parse_8k_items("5.02,9.01") == {"5.02", "9.01"}
    assert parse_8k_items("") == set()
    assert parse_8k_items(None) == set()


def test_select_companies() -> None:
    module = _load_script_module()
    select_companies = cast(
        Callable[[dict[str, Any], str], list[dict[str, Any]]],
        module.__dict__["select_companies"],
    )
    registry = _load_registry()

    assert len(select_companies(registry, "all")) == 8

    msft_companies = select_companies(registry, "MSFT")
    assert len(msft_companies) == 1
    assert msft_companies[0]["ticker"] == "MSFT"

    try:
        select_companies(registry, "UNKNOWN")
    except ValueError:
        pass
    else:
        msg = "Expected unknown ticker to raise ValueError"
        raise AssertionError(msg)


def test_build_dry_run_plan_for_msft() -> None:
    module = _load_script_module()
    build_dry_run_plan = cast(
        Callable[[argparse.Namespace], dict[str, Any]],
        module.__dict__["build_dry_run_plan"],
    )
    args = argparse.Namespace(
        registry_path=str(REGISTRY_PATH),
        company="MSFT",
        start_year=2024,
        end_year=2026,
        forms="10-K,10-Q,8-K",
    )

    plan = build_dry_run_plan(args)
    companies = plan["companies"]
    assert isinstance(companies, list)
    company = companies[0]
    assert isinstance(company, dict)

    assert plan["mode"] == "dry_run"
    assert plan["phase"] == "2A-3A"
    assert len(companies) == 1
    assert company["will_download"] is False
    assert company["planned_forms"] == ["10-K", "10-Q", "8-K"]
    assert company["planned_filing_years"] == [2024, 2025, 2026]


def test_download_json_handles_plain_json_response() -> None:
    module = _load_script_module()
    download_json = cast(
        Callable[[str, Path, str, float], dict[str, Any]],
        module.__dict__["download_json"],
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "sample.json"
        with patch(
            "urllib.request.urlopen",
            return_value=FakeUrlopenResponse(b'{"ok": true}'),
        ):
            parsed = download_json(
                "https://data.sec.gov/submissions/CIK0000789019.json",
                output_path,
                "test-agent",
                0,
            )

        assert parsed == {"ok": True}
        assert json.loads(output_path.read_text(encoding="utf-8")) == parsed


def test_download_json_handles_gzip_response_with_header() -> None:
    module = _load_script_module()
    download_json = cast(
        Callable[[str, Path, str, float], dict[str, Any]],
        module.__dict__["download_json"],
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "sample.json"
        with patch(
            "urllib.request.urlopen",
            return_value=FakeUrlopenResponse(
                gzip.compress(b'{"ok": true}'),
                {"Content-Encoding": "gzip"},
            ),
        ):
            parsed = download_json(
                "https://data.sec.gov/submissions/CIK0000789019.json",
                output_path,
                "test-agent",
                0,
            )

        assert parsed == {"ok": True}
        assert json.loads(output_path.read_text(encoding="utf-8")) == parsed


def test_download_json_handles_gzip_response_without_header_magic_bytes() -> None:
    module = _load_script_module()
    download_json = cast(
        Callable[[str, Path, str, float], dict[str, Any]],
        module.__dict__["download_json"],
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "sample.json"
        with patch(
            "urllib.request.urlopen",
            return_value=FakeUrlopenResponse(gzip.compress(b'{"ok": true}')),
        ):
            parsed = download_json(
                "https://data.sec.gov/submissions/CIK0000789019.json",
                output_path,
                "test-agent",
                0,
            )

        assert parsed == {"ok": True}
        assert json.loads(output_path.read_text(encoding="utf-8")) == parsed


def test_read_jsonl() -> None:
    module = _load_script_module()
    read_jsonl = cast(Callable[[Path], list[dict[str, Any]]], module.__dict__["read_jsonl"])

    with tempfile.TemporaryDirectory() as temporary_directory:
        jsonl_path = Path(temporary_directory) / "rows.jsonl"
        jsonl_path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

        assert read_jsonl(jsonl_path) == [{"a": 1}, {"b": 2}]

        invalid_path = Path(temporary_directory) / "invalid.jsonl"
        invalid_path.write_text('{"a": 1}\n{bad json}\n', encoding="utf-8")

        try:
            read_jsonl(invalid_path)
        except RuntimeError as exc:
            assert "line 2" in str(exc)
        else:
            msg = "Expected invalid JSONL to raise RuntimeError"
            raise AssertionError(msg)


def test_filter_manifest_rows() -> None:
    module = _load_script_module()
    filter_manifest_rows = cast(
        Callable[[list[dict[str, Any]], str, str, int], list[dict[str, Any]]],
        module.__dict__["filter_manifest_rows"],
    )
    rows = _fake_selected_filing_rows()

    assert [row["ticker"] for row in filter_manifest_rows(rows, "MSFT", "all", 0)] == [
        "MSFT",
        "MSFT",
    ]
    assert [row["form"] for row in filter_manifest_rows(rows, "all", "10-K", 0)] == ["10-K"]
    assert len(filter_manifest_rows(rows, "all", "all", 2)) == 2
    assert filter_manifest_rows(rows, "all", "all", 0) == rows


def test_build_local_filing_path() -> None:
    module = _load_script_module()
    build_local_filing_path = cast(
        Callable[[dict[str, Any], Path], Path],
        module.__dict__["build_local_filing_path"],
    )
    row = {
        "ticker": "MSFT",
        "form": "10-Q",
        "accession_number": "0001193125-26-191507",
        "primary_document": "msft-20260331.htm",
    }

    path = build_local_filing_path(row, Path("data/raw/finance/sec/filings"))

    assert path.parts[-4:] == (
        "MSFT",
        "10-Q",
        "000119312526191507",
        "msft-20260331.htm",
    )


def test_download_binary_plain_html_mocked() -> None:
    module = _load_script_module()
    download_binary = cast(
        Callable[[str, Path, str, float], dict[str, Any]],
        module.__dict__["download_binary"],
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "filing.htm"
        with patch(
            "urllib.request.urlopen",
            return_value=FakeUrlopenResponse(
                b"<html><body>ok</body></html>",
                {"Content-Type": "text/html"},
            ),
        ):
            metadata = download_binary(
                "https://www.sec.gov/Archives/edgar/data/789019/test/filing.htm",
                output_path,
                "test-agent",
                0,
            )

        assert output_path.read_bytes() == b"<html><body>ok</body></html>"
        assert metadata["status"] == "downloaded"
        assert metadata["bytes_written"] > 0
        assert metadata["sha256"]


def test_download_binary_gzip_html_mocked() -> None:
    module = _load_script_module()
    download_binary = cast(
        Callable[[str, Path, str, float], dict[str, Any]],
        module.__dict__["download_binary"],
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "filing.htm"
        with patch(
            "urllib.request.urlopen",
            return_value=FakeUrlopenResponse(
                gzip.compress(b"<html><body>ok</body></html>"),
                {"Content-Encoding": "gzip", "Content-Type": "text/html"},
            ),
        ):
            metadata = download_binary(
                "https://www.sec.gov/Archives/edgar/data/789019/test/filing.htm",
                output_path,
                "test-agent",
                0,
            )

        assert output_path.read_bytes() == b"<html><body>ok</body></html>"
        assert metadata["status"] == "downloaded"
        assert metadata["bytes_written"] > 0
        assert metadata["sha256"]


def test_strip_html_to_text() -> None:
    module = _load_script_module()
    strip_html_to_text = cast(Callable[[str], str], module.__dict__["strip_html_to_text"])
    html_content = """
    <html>
      <head>
        <title>Sample Filing</title>
        <style>.hidden { display: none; }</style>
        <script>var secret = "remove me";</script>
      </head>
      <body><p>Revenue &amp; margin improved.</p></body>
    </html>
    """

    text = strip_html_to_text(html_content)

    assert "remove me" not in text
    assert "display: none" not in text
    assert "Revenue & margin improved." in text
    assert "<p>" not in text


def test_build_extracted_text_path() -> None:
    module = _load_script_module()
    build_extracted_text_path = cast(
        Callable[[dict[str, Any], Path], Path],
        module.__dict__["build_extracted_text_path"],
    )
    row = {
        "ticker": "MSFT",
        "form": "10-Q",
        "accession_number": "0001193125-26-191507",
    }

    path = build_extracted_text_path(row, Path("data/processed/finance/sec/extracted_text"))

    assert path.parts[-3:] == (
        "MSFT",
        "10-Q",
        "000119312526191507.txt",
    )


def test_write_extracted_text() -> None:
    module = _load_script_module()
    write_extracted_text = cast(
        Callable[[str, Path], dict[str, Any]],
        module.__dict__["write_extracted_text"],
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "extracted.txt"
        metadata = write_extracted_text("Revenue increased during the period.", output_path)

        assert output_path.read_text(encoding="utf-8") == "Revenue increased during the period."
        assert metadata["char_count"] > 0
        assert metadata["word_count"] > 0
        assert metadata["sha256"]


def test_is_probable_section_heading() -> None:
    module = _load_script_module()
    is_probable_section_heading = cast(
        Callable[[str], bool],
        module.__dict__["is_probable_section_heading"],
    )

    true_cases = [
        "Item 1A.",
        "Item 7.",
        "Risk Factors",
        "Management\u2019s Discussion and Analysis",
        "Results of Operations",
        "Liquidity and Capital Resources",
        "Financial Statements",
        "Exhibit 99",
    ]
    for candidate in true_cases:
        assert is_probable_section_heading(candidate)

    long_paragraph = (
        "This report includes estimates, projections, statements relating to our business "
        "plans, objectives, and expected operating results that should not be treated as "
        "a clean section heading by the finance extractor."
    )
    assert not is_probable_section_heading(long_paragraph)
    assert not is_probable_section_heading(
        "This report includes estimates, projections, statements relating to our business "
        "plans, objectives, and expected operating results"
    )
    assert not is_probable_section_heading("")


def test_normalize_section_title() -> None:
    module = _load_script_module()
    normalize_section_title = cast(
        Callable[[str], str],
        module.__dict__["normalize_section_title"],
    )

    assert normalize_section_title("  Risk    Factors  ") == "Risk Factors"
    assert normalize_section_title("\nFinancial\tStatements: ") == "Financial Statements"


def test_is_suspicious_section_title() -> None:
    module = _load_script_module()
    is_suspicious_section_title = cast(
        Callable[[str], bool],
        module.__dict__["is_suspicious_section_title"],
    )
    paragraph = (
        "This report includes estimates, projections, statements relating to our business "
        "plans, objectives, and expected operating results"
    )

    assert is_suspicious_section_title(paragraph)
    assert not is_suspicious_section_title("Risk Factors")
    assert not is_suspicious_section_title("Item 1A.")


def test_detect_finance_sections_10k_or_10q() -> None:
    module = _load_script_module()
    detect_finance_sections = cast(
        Callable[[str, dict[str, Any], int], list[dict[str, Any]]],
        module.__dict__["detect_finance_sections"],
    )
    repeated_text = "The company describes material financial matters. " * 8
    text = "\n".join(
        [
            "Item 1A. Risk Factors",
            repeated_text,
            "Item 2. Management's Discussion and Analysis",
            repeated_text,
            "Item 8. Financial Statements",
            repeated_text,
        ]
    )
    document_row = {**_fake_document_row("10-Q"), "local_text_path": "extracted.txt"}

    sections = detect_finance_sections(text, document_row, 5)
    section_types = {section["section_type"] for section in sections}

    assert {"risk_factors", "management_discussion_and_analysis"} & section_types
    for section in sections:
        assert "section_type" in section
        assert "section_title" in section
        assert section["char_count"] > 0


def test_detect_finance_sections_8k() -> None:
    module = _load_script_module()
    detect_finance_sections = cast(
        Callable[[str, dict[str, Any], int], list[dict[str, Any]]],
        module.__dict__["detect_finance_sections"],
    )
    repeated_text = "Financial results and management commentary are summarized. " * 8
    text = "\n".join(
        [
            "Results of Operations and Financial Condition",
            repeated_text,
            "Exhibit 99",
            repeated_text,
        ]
    )
    document_row = {**_fake_document_row("8-K"), "local_text_path": "extracted.txt"}

    sections = detect_finance_sections(text, document_row, 5)
    section_types = {section["section_type"] for section in sections}

    assert {"results_of_operations", "exhibit_99"} & section_types


def test_detect_finance_sections_rejects_paragraph_fragments() -> None:
    module = _load_script_module()
    detect_finance_sections = cast(
        Callable[[str, dict[str, Any], int], list[dict[str, Any]]],
        module.__dict__["detect_finance_sections"],
    )
    paragraph = (
        "This paragraph discusses results of operations, expected operating results, "
        "and projected demand in a sentence-like format that should not become a heading."
    )
    body_text = "The company provides detailed analysis of revenue and operating expenses. " * 10
    text = "\n".join(
        [
            paragraph,
            "Item 2. Management's Discussion and Analysis",
            body_text,
        ]
    )
    document_row = {**_fake_document_row("10-Q"), "local_text_path": "extracted.txt"}

    sections = detect_finance_sections(text, document_row, 50)
    section_titles = {section["section_title"] for section in sections}

    assert paragraph not in section_titles
    assert any(
        section["section_title"] == "Item 2. Management's Discussion and Analysis"
        for section in sections
    )


def test_rows_from_recent_filings_transforms_column_arrays() -> None:
    module = _load_script_module()
    rows_from_recent_filings = cast(
        Callable[[dict[str, Any]], list[dict[str, Any]]],
        module.__dict__["rows_from_recent_filings"],
    )

    rows = rows_from_recent_filings(_fake_submissions())

    assert len(rows) == 5
    assert rows[0]["accession_number"] == "0001193125-26-191507"
    assert rows[0]["filing_date"] == "2026-04-25"
    assert rows[0]["primary_document"] == "msft-20260331.htm"
    assert rows[0]["primary_doc_description"] == "10-Q"
    assert rows[0]["is_xbrl"] == 1
    assert rows[0]["is_inline_xbrl"] == 1


def test_derive_filing_url() -> None:
    module = _load_script_module()
    derive_filing_url = cast(
        Callable[[str, str, str], str],
        module.__dict__["derive_filing_url"],
    )

    url = derive_filing_url(
        "789019",
        "0001193125-26-191507",
        "msft-20260331.htm",
    )

    assert (
        url == "https://www.sec.gov/Archives/edgar/data/789019/000119312526191507/msft-20260331.htm"
    )


def test_is_selected_finance_filing() -> None:
    module = _load_script_module()
    is_selected_finance_filing = cast(
        Callable[[dict[str, Any], int, int, list[str]], tuple[bool, str]],
        module.__dict__["is_selected_finance_filing"],
    )
    forms = ["10-K", "10-Q", "8-K"]

    assert is_selected_finance_filing(
        {"form": "10-K", "filing_date": "2025-01-01"}, 2024, 2026, forms
    )[0]
    assert is_selected_finance_filing(
        {"form": "10-Q", "filing_date": "2026-04-01"}, 2024, 2026, forms
    )[0]
    assert is_selected_finance_filing(
        {"form": "8-K", "filing_date": "2025-07-01", "items": "2.02,9.01"},
        2024,
        2026,
        forms,
    )[0]
    assert is_selected_finance_filing(
        {"form": "8-K", "filing_date": "2025-07-01", "items": "2.02"},
        2024,
        2026,
        forms,
    )[0]
    assert not is_selected_finance_filing(
        {"form": "8-K", "filing_date": "2025-07-01", "items": "5.02,9.01"},
        2024,
        2026,
        forms,
    )[0]
    assert not is_selected_finance_filing(
        {"form": "8-K", "filing_date": "2025-07-01", "items": "9.01"},
        2024,
        2026,
        forms,
    )[0]
    assert not is_selected_finance_filing(
        {"form": "8-K", "filing_date": "2025-07-01", "items": "8.01"},
        2024,
        2026,
        forms,
    )[0]
    assert not is_selected_finance_filing(
        {"form": "8-K", "filing_date": "2025-07-01", "items": ""},
        2024,
        2026,
        forms,
    )[0]
    assert not is_selected_finance_filing(
        {"form": "4", "filing_date": "2025-07-01"},
        2024,
        2026,
        forms,
    )[0]


def test_8k_901_only_is_not_selected_as_earnings_candidate() -> None:
    module = _load_script_module()
    is_selected_finance_filing = cast(
        Callable[[dict[str, Any], int, int, list[str]], tuple[bool, str]],
        module.__dict__["is_selected_finance_filing"],
    )
    selected, _reason = is_selected_finance_filing(
        {
            "form": "8-K",
            "filing_date": "2026-05-14",
            "report_date": "2026-05-13",
            "items": "5.02,9.01",
        },
        2024,
        2026,
        ["10-K", "10-Q", "8-K"],
    )

    assert selected is False


def test_build_selected_filings_manifest_from_fake_data() -> None:
    module = _load_script_module()
    build_selected_filings_manifest = cast(
        Callable[
            [list[dict[str, Any]], dict[str, dict[str, Any]], list[str], int, int, int],
            list[dict[str, Any]],
        ],
        module.__dict__["build_selected_filings_manifest"],
    )
    company = _msft_company()

    manifest_rows = build_selected_filings_manifest(
        [company],
        {"MSFT": _fake_submissions()},
        ["10-K", "10-Q", "8-K"],
        2024,
        2026,
        0,
    )

    assert len(manifest_rows) == 3
    first_row = manifest_rows[0]
    assert "record_id" in first_row
    assert first_row["ticker"] == "MSFT"
    assert "form" in first_row
    assert "derived_filing_url" in first_row
    assert "selection_reason" in first_row


def test_build_xbrl_concept_inventory_from_fake_companyfacts() -> None:
    module = _load_script_module()
    build_xbrl_concept_inventory = cast(
        Callable[[list[dict[str, Any]], dict[str, dict[str, Any]]], list[dict[str, Any]]],
        module.__dict__["build_xbrl_concept_inventory"],
    )
    inventory = build_xbrl_concept_inventory([_msft_company()], {"MSFT": _fake_companyfacts()})
    concepts = {row["concept"]: row for row in inventory}

    assert set(concepts) == {"Revenues", "NetIncomeLoss"}
    revenues = concepts["Revenues"]
    assert revenues["units"] == ["USD"]
    assert revenues["observation_count"] == 2
    assert revenues["forms_present"] == ["10-K", "10-Q"]
    assert revenues["fiscal_years_present"] == [2025, 2026]
    assert revenues["has_10k"] is True
    assert revenues["has_10q"] is True


def test_build_exploration_report_from_fake_manifest_and_inventory() -> None:
    module = _load_script_module()
    build_selected_filings_manifest = cast(
        Callable[
            [list[dict[str, Any]], dict[str, dict[str, Any]], list[str], int, int, int],
            list[dict[str, Any]],
        ],
        module.__dict__["build_selected_filings_manifest"],
    )
    build_xbrl_concept_inventory = cast(
        Callable[[list[dict[str, Any]], dict[str, dict[str, Any]]], list[dict[str, Any]]],
        module.__dict__["build_xbrl_concept_inventory"],
    )
    build_exploration_report = cast(
        Callable[
            [list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]],
            dict[str, Any],
        ],
        module.__dict__["build_exploration_report"],
    )
    companies = [_msft_company()]
    manifest_rows = build_selected_filings_manifest(
        companies,
        {"MSFT": _fake_submissions()},
        ["10-K", "10-Q", "8-K"],
        2024,
        2026,
        0,
    )
    inventory_rows = build_xbrl_concept_inventory(companies, {"MSFT": _fake_companyfacts()})
    report = build_exploration_report(
        companies,
        manifest_rows,
        inventory_rows,
        {
            "selected_filings_manifest_path": "manifest.jsonl",
            "xbrl_concept_inventory_path": "inventory.jsonl",
            "exploration_report_path": "report.json",
        },
    )

    assert report["phase"] == "2A-3B"
    assert "selected_filings_summary" in report
    assert "xbrl_summary" in report
    assert "important_concept_coverage" in report
    assert "next_step" in report
    selected_filings_summary = report["selected_filings_summary"]
    assert isinstance(selected_filings_summary, dict)
    assert selected_filings_summary["earnings_8k_selection_rule"] == (
        "8-K rows require item 2.02. Item 9.01 alone is not sufficient."
    )


def test_build_document_manifest_rows() -> None:
    module = _load_script_module()
    build_document_manifest_rows = cast(
        Callable[[list[dict[str, Any]], list[dict[str, Any]]], list[dict[str, Any]]],
        module.__dict__["build_document_manifest_rows"],
    )
    selected_rows = [_fake_selected_filing_rows()[0]]
    download_results = [
        {
            "destination": "data/raw/finance/sec/filings/MSFT/10-Q/test.htm",
            "status": "downloaded",
            "content_type": "text/html",
            "content_encoding": "",
            "bytes_written": 28,
            "sha256": "abc123",
            "downloaded_at_utc": "2026-05-17T00:00:00+00:00",
        }
    ]

    document_rows = build_document_manifest_rows(selected_rows, download_results)
    row = document_rows[0]

    assert row["document_record_id"] == "finance_doc_MSFT_10Q_000119312526191507"
    assert row["source_manifest_record_id"] == "finance_sec_MSFT_10Q_000119312526191507"
    assert row["ticker"] == "MSFT"
    assert row["form"] == "10-Q"
    assert row["local_html_path"] == "data/raw/finance/sec/filings/MSFT/10-Q/test.htm"
    assert row["download_status"] == "downloaded"
    assert row["sha256"] == "abc123"


def test_build_filing_download_report() -> None:
    module = _load_script_module()
    build_filing_download_report = cast(
        Callable[[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]], dict[str, Any]],
        module.__dict__["build_filing_download_report"],
    )
    document_rows = [
        {
            "ticker": "MSFT",
            "form": "10-Q",
            "download_status": "downloaded",
        },
        {
            "ticker": "AAPL",
            "form": "10-K",
            "download_status": "skipped_existing",
        },
    ]

    report = build_filing_download_report(
        document_rows,
        _fake_selected_filing_rows(),
        {
            "documents_manifest_path": "documents.jsonl",
            "download_report_path": "report.json",
        },
    )

    assert report["phase"] == "2A-3C"
    assert report["total_documents_attempted"] == 2
    assert report["total_documents_downloaded"] == 1
    assert report["counts_by_company"] == {"MSFT": 1, "AAPL": 1}
    assert report["counts_by_form"] == {"10-Q": 1, "10-K": 1}
    assert "Phase 2A-3D" in report["next_step"]


def test_build_text_manifest_row() -> None:
    module = _load_script_module()
    build_text_manifest_row = cast(
        Callable[..., dict[str, Any]],
        module.__dict__["build_text_manifest_row"],
    )
    metadata = {
        "path": "data/processed/finance/sec/extracted_text/MSFT/10-Q/test.txt",
        "char_count": 100,
        "word_count": 16,
        "sha256": "abc123",
        "section_count": 2,
    }

    row = build_text_manifest_row(_fake_document_row("10-Q"), metadata, "extracted")

    assert row["text_record_id"] == "finance_text_MSFT_10Q_000119312526191507"
    assert row["document_record_id"] == "finance_doc_MSFT_10Q_000119312526191507"
    assert row["local_text_path"] == metadata["path"]
    assert row["extraction_status"] == "extracted"
    assert row["text_char_count"] == 100
    assert row["text_word_count"] == 16
    assert row["section_count"] == 2


def test_build_text_extraction_report() -> None:
    module = _load_script_module()
    build_text_extraction_report = cast(
        Callable[[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, str]], dict[str, Any]],
        module.__dict__["build_text_extraction_report"],
    )
    text_rows = [
        {
            "document_record_id": "finance_doc_MSFT_10Q_000119312526191507",
            "ticker": "MSFT",
            "form": "10-Q",
            "extraction_status": "extracted",
            "section_count": 1,
        }
    ]
    section_rows = [
        {
            "ticker": "MSFT",
            "form": "10-Q",
            "section_type": "risk_factors",
        }
    ]

    report = build_text_extraction_report(
        text_rows,
        section_rows,
        3,
        {
            "text_manifest_path": "text.jsonl",
            "sections_manifest_path": "sections.jsonl",
            "text_extraction_report_path": "report.json",
        },
    )

    assert report["phase"] == "2A-3D"
    assert report["total_documents_attempted"] == 1
    assert report["total_documents_extracted"] == 1
    assert report["total_sections_extracted"] == 1
    assert report["sections_by_type"] == {"risk_factors": 1}
    assert "Phase 2A-3E" in report["next_step"]


def test_build_section_quality_report() -> None:
    module = _load_script_module()
    build_section_quality_report = cast(
        Callable[
            [list[dict[str, Any]], list[dict[str, Any]], str, str, int],
            dict[str, Any],
        ],
        module.__dict__["build_section_quality_report"],
    )
    text_rows = [
        {
            "document_record_id": "finance_doc_MSFT_10Q_000119312526191507",
            "ticker": "MSFT",
            "form": "10-Q",
            "accession_number": "0001193125-26-191507",
            "section_count": 2,
        }
    ]
    long_title = (
        "This report includes estimates, projections, statements relating to our business "
        "plans, objectives, and expected operating results"
    )
    section_rows = [
        {
            "section_record_id": "section_1",
            "document_record_id": "finance_doc_MSFT_10Q_000119312526191507",
            "ticker": "MSFT",
            "form": "10-Q",
            "accession_number": "0001193125-26-191507",
            "section_type": "results_of_operations",
            "section_title": long_title,
            "char_count": 300,
        },
        {
            "section_record_id": "section_2",
            "document_record_id": "finance_doc_MSFT_10Q_000119312526191507",
            "ticker": "MSFT",
            "form": "10-Q",
            "accession_number": "0001193125-26-191507",
            "section_type": "risk_factors",
            "section_title": "Risk Factors",
            "char_count": 600,
        },
    ]

    report = build_section_quality_report(text_rows, section_rows, "all", "all", 80)

    assert report["phase"] == "2A-3D-QA"
    assert report["total_text_rows"] == 1
    assert report["total_section_rows"] == 2
    assert report["suspicious_section_title_count"] == 1
    assert report["suspicious_section_title_examples"]
    assert "sections_per_document_summary" in report
    assert "recommended_next_step" in report


def test_finance_sec_acquisition_cli_dry_run_msft() -> None:
    completed_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--company",
            "MSFT",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0
    output = json.loads(completed_process.stdout)
    companies = output["companies"]

    assert output["mode"] == "dry_run"
    assert len(companies) == 1
    assert companies[0]["ticker"] == "MSFT"


def test_finance_sec_acquisition_cli_requires_a_mode() -> None:
    completed_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--company",
            "MSFT",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "Exactly one mode must be selected" in combined_output
    assert "Download mode is not implemented in Phase 2A-3A" in combined_output


def test_finance_sec_acquisition_cli_summarize_local_missing_files() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        company = dict(_msft_company())
        company["local_raw_submissions_path"] = str(temporary_path / "missing_submissions.json")
        company["local_raw_companyfacts_path"] = str(temporary_path / "missing_companyfacts.json")
        registry_path = temporary_path / "registry.json"
        registry_path.write_text(
            json.dumps(
                {
                    "registry_name": "test_registry",
                    "version": "1.0",
                    "companies": [company],
                }
            ),
            encoding="utf-8",
        )

        completed_process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--summarize-local",
                "--company",
                "MSFT",
                "--registry-path",
                str(registry_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "--download-json first" in combined_output


def test_finance_sec_acquisition_cli_download_filings_missing_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        missing_manifest_path = Path(temporary_directory) / "missing_manifest.jsonl"
        completed_process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--download-filings",
                "--company",
                "MSFT",
                "--manifest-path",
                str(missing_manifest_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "Missing selected filings manifest" in combined_output


def test_finance_sec_acquisition_cli_extract_text_missing_documents_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        missing_manifest_path = Path(temporary_directory) / "missing_documents.jsonl"
        completed_process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--extract-text",
                "--documents-manifest-path",
                str(missing_manifest_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "Missing selected filing documents manifest" in combined_output


def test_finance_sec_acquisition_cli_audit_sections_missing_manifest() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        missing_manifest_path = Path(temporary_directory) / "missing_text_manifest.jsonl"
        completed_process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--audit-sections",
                "--text-manifest-path",
                str(missing_manifest_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "Missing filing text manifest" in combined_output


def test_finance_sec_acquisition_cli_rejects_extract_audit_mode_conflict() -> None:
    completed_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--extract-text",
            "--audit-sections",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "Exactly one mode must be selected" in combined_output


def test_finance_sec_acquisition_cli_rejects_mode_conflict() -> None:
    completed_process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dry-run",
            "--download-json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = completed_process.stdout + completed_process.stderr

    assert completed_process.returncode != 0
    assert "Exactly one mode must be selected" in combined_output
