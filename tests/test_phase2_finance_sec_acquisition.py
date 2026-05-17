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
