import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

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


def test_finance_sec_acquisition_cli_requires_dry_run() -> None:
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
    assert "Download mode is not implemented in Phase 2A-3A" in combined_output
