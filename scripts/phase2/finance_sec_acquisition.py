"""SEC/XBRL acquisition planner and JSON-only explorer for Phase 2A finance."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = Path("data/sources/finance_ticker_registry.json")
DEFAULT_OUTPUT_DIR = Path("data/processed/finance/sec/")
ALLOWED_COMPANY_FILTERS = ("all", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD")
DEFAULT_FORMS = "10-K,10-Q,8-K"
DEFAULT_USER_AGENT = "LLM-Inference-Optimization-Suite research-contact@example.com"
IMPORTANT_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "CostOfRevenue",
    "GrossProfit",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "ResearchAndDevelopmentExpense",
    "EarningsPerShareDiluted",
    "CashAndCashEquivalentsAtCarryingValue",
    "NetCashProvidedByUsedInOperatingActivities",
)


def load_registry(path: Path) -> dict[str, Any]:
    """Load the finance ticker registry from a local JSON file."""

    if not path.exists():
        msg = f"Registry not found: {path}"
        raise FileNotFoundError(msg)

    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        msg = "Registry JSON must contain a top-level object"
        raise ValueError(msg)

    companies = registry.get("companies")
    if not isinstance(companies, list):
        msg = "Registry JSON must contain a companies list"
        raise ValueError(msg)

    return registry


def normalize_forms(forms: str) -> list[str]:
    """Normalize a comma-separated SEC form list."""

    normalized_forms = [form.strip().upper() for form in forms.split(",") if form.strip()]
    if not normalized_forms:
        msg = "At least one SEC form must be provided"
        raise ValueError(msg)
    return normalized_forms


def select_companies(registry: dict[str, Any], company_filter: str) -> list[dict[str, Any]]:
    """Select all companies or a single ticker from the registry."""

    raw_companies = registry.get("companies")
    if not isinstance(raw_companies, list):
        msg = "Registry JSON must contain a companies list"
        raise ValueError(msg)

    companies = [company for company in raw_companies if isinstance(company, dict)]
    normalized_filter = company_filter.strip().upper()

    if normalized_filter == "ALL":
        return companies

    selected_companies = [
        company
        for company in companies
        if str(company.get("ticker", "")).upper() == normalized_filter
    ]
    if not selected_companies:
        msg = f"Unknown company filter: {company_filter}"
        raise ValueError(msg)
    return selected_companies


def _selection_rules() -> dict[str, str]:
    return {
        "10-K": (
            "Form equals 10-K; filing date or report date from 2024 onward; "
            "target FY2024/FY2025 where available."
        ),
        "10-Q": (
            "Form equals 10-Q; filing date or report date from 2024 onward; "
            "prioritize 2026 quarterlies and retain 2024/2025 quarterlies for trends."
        ),
        "8-K": (
            "Form equals 8-K; filing date from 2024 onward; prioritize items 2.02 and/or 9.01."
        ),
    }


def build_planned_company(
    company: dict[str, Any],
    forms: list[str],
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    """Build a dry-run acquisition plan for one company."""

    filing_years = list(range(start_year, end_year + 1))
    return {
        "company_name": company["company_name"],
        "ticker": company["ticker"],
        "cik": company["cik"],
        "cik_no_leading_zeros": company["cik_no_leading_zeros"],
        "fiscal_year_end": company["fiscal_year_end"],
        "submissions_url": company["submissions_url"],
        "companyfacts_url": company["companyfacts_url"],
        "sec_company_browser_url": company["sec_company_browser_url"],
        "planned_raw_submissions_path": company["local_raw_submissions_path"],
        "planned_raw_companyfacts_path": company["local_raw_companyfacts_path"],
        "planned_processed_dir": company["local_processed_dir"],
        "planned_forms": forms,
        "planned_filing_years": filing_years,
        "planned_selection_rules": _selection_rules(),
        "will_download": False,
    }


def build_dry_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the dry-run acquisition plan without making network calls."""

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if start_year > end_year:
        msg = "start_year must be <= end_year"
        raise ValueError(msg)

    forms = normalize_forms(str(args.forms))
    registry_path = Path(str(args.registry_path))
    registry = load_registry(registry_path)
    company_filter = str(args.company)
    selected_companies = select_companies(registry, company_filter)

    return {
        "mode": "dry_run",
        "phase": "2A-3A",
        "registry_path": str(registry_path),
        "company_filter": company_filter,
        "start_year": start_year,
        "end_year": end_year,
        "target_forms": forms,
        "companies": [
            build_planned_company(
                company=company,
                forms=forms,
                start_year=start_year,
                end_year=end_year,
            )
            for company in selected_companies
        ],
        "warnings": [
            "Dry-run only: no SEC data was downloaded.",
            "Download mode is intentionally deferred until Phase 2A-3B.",
        ],
    }


def download_json(
    url: str,
    destination: Path,
    user_agent: str,
    delay_seconds: float,
) -> dict[str, Any]:
    """Download one SEC JSON endpoint and write a normalized local copy."""

    # SEC automated access should use a declared User-Agent and conservative
    # request pacing. The default delay is intentionally conservative for this
    # benchmark setup, even though SEC fair-access limits allow higher rates.
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_payload = response.read()
    except urllib.error.HTTPError as exc:
        msg = f"Failed to download SEC JSON from {url}: HTTP {exc.code} {exc.reason}"
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"Failed to download SEC JSON from {url}: {exc.reason}"
        raise RuntimeError(msg) from exc

    try:
        parsed_json = json.loads(raw_payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"SEC endpoint did not return valid JSON: {url}"
        raise RuntimeError(msg) from exc

    if not isinstance(parsed_json, dict):
        msg = f"SEC endpoint returned non-object JSON: {url}"
        raise RuntimeError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(parsed_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return parsed_json


def rows_from_recent_filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert SEC filings.recent column arrays into row dictionaries."""

    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return []

    field_map = {
        "accessionNumber": "accession_number",
        "filingDate": "filing_date",
        "reportDate": "report_date",
        "acceptanceDateTime": "acceptance_datetime",
        "act": "act",
        "form": "form",
        "fileNumber": "file_number",
        "filmNumber": "film_number",
        "items": "items",
        "size": "size",
        "isXBRL": "is_xbrl",
        "isInlineXBRL": "is_inline_xbrl",
        "primaryDocument": "primary_document",
        "primaryDocDescription": "primary_doc_description",
    }

    max_row_count = 0
    for values in recent.values():
        if isinstance(values, list):
            max_row_count = max(max_row_count, len(values))

    rows: list[dict[str, Any]] = []
    for index in range(max_row_count):
        row: dict[str, Any] = {}
        for source_field, output_field in field_map.items():
            values = recent.get(source_field)
            if isinstance(values, list) and index < len(values):
                row[output_field] = values[index]
            else:
                row[output_field] = None
        rows.append(row)
    return rows


def derive_filing_url(
    cik_no_leading_zeros: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """Derive an SEC Archives filing document URL."""

    accession_without_dashes = accession_number.replace("-", "")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_without_dashes}/{primary_document}"
    )


def _year_from_iso_date(value: object) -> int | None:
    if not isinstance(value, str) or len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def _row_has_selected_year(row: dict[str, Any], start_year: int, end_year: int) -> bool:
    years = [
        year
        for year in (
            _year_from_iso_date(row.get("filing_date")),
            _year_from_iso_date(row.get("report_date")),
        )
        if year is not None
    ]
    return any(start_year <= year <= end_year for year in years)


def is_selected_finance_filing(
    row: dict[str, Any],
    start_year: int,
    end_year: int,
    forms: list[str],
) -> tuple[bool, str]:
    """Return whether a submission row matches Phase 2A-3B finance rules."""

    form = str(row.get("form", "")).upper()
    target_forms = {target_form.upper() for target_form in forms}
    if form not in target_forms:
        return False, "form is not in target forms"

    if not _row_has_selected_year(row, start_year, end_year):
        return False, "filing_date/report_date is outside target years or missing"

    if form == "10-K":
        return True, "Selected annual filing candidate from target year range."

    if form == "10-Q":
        return True, "Selected quarterly filing candidate from target year range."

    if form == "8-K":
        items = str(row.get("items") or "")
        if "2.02" in items or "9.01" in items:
            return True, "Selected earnings/results candidate 8-K with item 2.02 and/or 9.01."
        return False, "8-K does not contain item 2.02 or 9.01"

    return False, "form-specific finance selection rule is not implemented"


def build_selected_filings_manifest(
    companies: list[dict[str, Any]],
    submissions_by_ticker: dict[str, dict[str, Any]],
    forms: list[str],
    start_year: int,
    end_year: int,
    max_filings_per_company: int = 0,
) -> list[dict[str, Any]]:
    """Build selected row-level filing manifest entries from local submissions JSON."""

    manifest_rows: list[dict[str, Any]] = []

    for company in companies:
        ticker = str(company["ticker"])
        selected_for_company: list[dict[str, Any]] = []
        for row in rows_from_recent_filings(submissions_by_ticker.get(ticker, {})):
            selected, selection_reason = is_selected_finance_filing(
                row=row,
                start_year=start_year,
                end_year=end_year,
                forms=forms,
            )
            if not selected:
                continue

            accession_number = str(row.get("accession_number") or "")
            primary_document = str(row.get("primary_document") or "")
            accession_without_dashes = accession_number.replace("-", "")
            form = str(row.get("form") or "")
            form_normalized = form.replace("-", "")
            record_id = f"finance_sec_{ticker}_{form_normalized}_{accession_without_dashes}"

            selected_for_company.append(
                {
                    "record_id": record_id,
                    "source_id": "finance_sec_edgar_xbrl",
                    "vertical": "finance",
                    "ticker": ticker,
                    "company_name": company["company_name"],
                    "cik": company["cik"],
                    "cik_no_leading_zeros": company["cik_no_leading_zeros"],
                    "fiscal_year_end": company["fiscal_year_end"],
                    "form": form,
                    "filing_date": row.get("filing_date"),
                    "report_date": row.get("report_date"),
                    "acceptance_datetime": row.get("acceptance_datetime"),
                    "accession_number": accession_number,
                    "items": row.get("items"),
                    "primary_document": primary_document,
                    "primary_doc_description": row.get("primary_doc_description"),
                    "is_xbrl": row.get("is_xbrl"),
                    "is_inline_xbrl": row.get("is_inline_xbrl"),
                    "size": row.get("size"),
                    "derived_filing_url": derive_filing_url(
                        cik_no_leading_zeros=str(company["cik_no_leading_zeros"]),
                        accession_number=accession_number,
                        primary_document=primary_document,
                    ),
                    "selection_reason": selection_reason,
                    "local_raw_submissions_path": company["local_raw_submissions_path"],
                    "local_raw_companyfacts_path": company["local_raw_companyfacts_path"],
                    "local_processed_dir": company["local_processed_dir"],
                }
            )

        selected_for_company.sort(
            key=lambda item: (
                str(item.get("ticker") or ""),
                str(item.get("filing_date") or ""),
                str(item.get("form") or ""),
                str(item.get("accession_number") or ""),
            ),
            reverse=True,
        )
        if max_filings_per_company > 0:
            selected_for_company = selected_for_company[:max_filings_per_company]
        manifest_rows.extend(selected_for_company)

    manifest_rows.sort(
        key=lambda item: (
            str(item.get("ticker") or ""),
            _reverse_date_sort_key(item.get("filing_date")),
            str(item.get("form") or ""),
            str(item.get("accession_number") or ""),
        )
    )
    return manifest_rows


def _reverse_date_sort_key(value: object) -> str:
    return "".join(str(9 - int(char)) if char.isdigit() else char for char in str(value or ""))


def _unique_sorted(values: list[Any]) -> list[Any]:
    return sorted({value for value in values if value not in (None, "")})


def _max_iso_date(values: list[Any]) -> str | None:
    dates = [value for value in values if isinstance(value, str) and value]
    return max(dates) if dates else None


def build_xbrl_concept_inventory(
    companies: list[dict[str, Any]],
    companyfacts_by_ticker: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one XBRL concept inventory row per company/taxonomy/concept."""

    inventory_rows: list[dict[str, Any]] = []

    for company in companies:
        ticker = str(company["ticker"])
        companyfacts = companyfacts_by_ticker.get(ticker, {})
        facts = companyfacts.get("facts", {})
        if not isinstance(facts, dict):
            continue

        us_gaap = facts.get("us-gaap", {})
        if not isinstance(us_gaap, dict):
            continue

        for concept, concept_payload in sorted(us_gaap.items()):
            if not isinstance(concept_payload, dict):
                continue
            units_payload = concept_payload.get("units", {})
            if not isinstance(units_payload, dict):
                units_payload = {}

            observations: list[dict[str, Any]] = []
            for unit_observations in units_payload.values():
                if isinstance(unit_observations, list):
                    observations.extend(
                        observation
                        for observation in unit_observations
                        if isinstance(observation, dict)
                    )

            forms_present = _unique_sorted(
                [observation.get("form") for observation in observations]
            )
            fiscal_years_present = _unique_sorted(
                [observation.get("fy") for observation in observations]
            )
            fiscal_periods_present = _unique_sorted(
                [observation.get("fp") for observation in observations]
            )

            inventory_rows.append(
                {
                    "record_id": f"finance_xbrl_{ticker}_{concept}",
                    "vertical": "finance",
                    "ticker": ticker,
                    "company_name": company["company_name"],
                    "cik": company["cik"],
                    "taxonomy": "us-gaap",
                    "concept": concept,
                    "label": concept_payload.get("label"),
                    "description": concept_payload.get("description"),
                    "units": sorted(str(unit) for unit in units_payload),
                    "observation_count": len(observations),
                    "forms_present": forms_present,
                    "fiscal_years_present": fiscal_years_present,
                    "fiscal_periods_present": fiscal_periods_present,
                    "latest_filed": _max_iso_date(
                        [observation.get("filed") for observation in observations]
                    ),
                    "latest_end": _max_iso_date(
                        [observation.get("end") for observation in observations]
                    ),
                    "has_10k": "10-K" in forms_present,
                    "has_10q": "10-Q" in forms_present,
                }
            )

    inventory_rows.sort(key=lambda row: (str(row["ticker"]), str(row["concept"])))
    return inventory_rows


def build_exploration_report(
    companies: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """Build a Phase 2A-3B exploration report from manifest and inventory rows."""

    company_tickers = [str(company["ticker"]) for company in companies]
    counts_by_company = Counter(str(row.get("ticker")) for row in manifest_rows)
    counts_by_form = Counter(str(row.get("form")) for row in manifest_rows)
    counts_by_company_and_form: dict[str, dict[str, int]] = {
        ticker: dict(
            Counter(str(row.get("form")) for row in manifest_rows if row.get("ticker") == ticker)
        )
        for ticker in company_tickers
    }
    filing_dates = [row.get("filing_date") for row in manifest_rows]
    report_dates = [row.get("report_date") for row in manifest_rows]

    concepts_by_company = Counter(str(row.get("ticker")) for row in inventory_rows)
    observations_by_company: dict[str, int] = defaultdict(int)
    concepts_by_ticker: dict[str, set[str]] = defaultdict(set)
    for row in inventory_rows:
        ticker = str(row.get("ticker"))
        observations_by_company[ticker] += int(row.get("observation_count") or 0)
        concepts_by_ticker[ticker].add(str(row.get("concept")))

    important_concept_coverage = {
        ticker: {
            concept: concept in concepts_by_ticker.get(ticker, set())
            for concept in IMPORTANT_CONCEPTS
        }
        for ticker in company_tickers
    }

    inventory_tickers = {str(row.get("ticker")) for row in inventory_rows}
    companies_missing_us_gaap = [
        ticker for ticker in company_tickers if ticker not in inventory_tickers
    ]

    return {
        "phase": "2A-3B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "company_count": len(companies),
        "companies": company_tickers,
        "selected_filings_summary": {
            "total_selected_filings": len(manifest_rows),
            "counts_by_company": dict(counts_by_company),
            "counts_by_form": dict(counts_by_form),
            "counts_by_company_and_form": counts_by_company_and_form,
            "filing_date_min": _max_or_min_iso_date(filing_dates, minimum=True),
            "filing_date_max": _max_or_min_iso_date(filing_dates, minimum=False),
            "report_date_min": _max_or_min_iso_date(report_dates, minimum=True),
            "report_date_max": _max_or_min_iso_date(report_dates, minimum=False),
            "earnings_8k_candidate_count": sum(
                1 for row in manifest_rows if row.get("form") == "8-K"
            ),
        },
        "xbrl_summary": {
            "total_inventory_rows": len(inventory_rows),
            "concepts_by_company": dict(concepts_by_company),
            "observations_by_company": dict(observations_by_company),
            "companies_missing_us_gaap": companies_missing_us_gaap,
        },
        "important_concept_coverage": important_concept_coverage,
        "output_files": output_paths,
        "warnings": [
            "This report summarizes SEC JSON acquisition artifacts only.",
            "Filing HTML/TXT downloads are deferred to Phase 2A-3C.",
            "Do not use these artifacts to make benchmark performance claims.",
        ],
        "next_step": (
            "Phase 2A-3C should download selected 10-K, 10-Q, and earnings-related "
            "8-K filing HTML documents using the selected filings manifest."
        ),
    }


def _max_or_min_iso_date(values: list[Any], *, minimum: bool) -> str | None:
    dates = [value for value in values if isinstance(value, str) and value]
    if not dates:
        return None
    return min(dates) if minimum else max(dates)


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Missing local SEC JSON file: {path}. Run --download-json first."
        raise FileNotFoundError(msg)
    parsed_json = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed_json, dict):
        msg = f"Local SEC JSON file must contain an object: {path}"
        raise ValueError(msg)
    return parsed_json


def _output_paths(output_dir: Path) -> dict[str, str]:
    return {
        "selected_filings_manifest_path": str(output_dir / "selected_filings_manifest.jsonl"),
        "xbrl_concept_inventory_path": str(output_dir / "xbrl_concept_inventory.jsonl"),
        "exploration_report_path": str(output_dir / "finance_sec_exploration_report.json"),
    }


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_common_args(args: argparse.Namespace) -> tuple[int, int, list[str], Path]:
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if start_year > end_year:
        msg = "start_year must be <= end_year"
        raise ValueError(msg)
    forms = normalize_forms(str(args.forms))
    output_dir = Path(str(args.output_dir))
    return start_year, end_year, forms, output_dir


def _build_outputs_from_local_data(
    *,
    companies: list[dict[str, Any]],
    submissions_by_ticker: dict[str, dict[str, Any]],
    companyfacts_by_ticker: dict[str, dict[str, Any]],
    forms: list[str],
    start_year: int,
    end_year: int,
    output_dir: Path,
    max_filings_per_company: int,
) -> dict[str, Any]:
    output_paths = _output_paths(output_dir)
    manifest_rows = build_selected_filings_manifest(
        companies=companies,
        submissions_by_ticker=submissions_by_ticker,
        forms=forms,
        start_year=start_year,
        end_year=end_year,
        max_filings_per_company=max_filings_per_company,
    )
    inventory_rows = build_xbrl_concept_inventory(
        companies=companies,
        companyfacts_by_ticker=companyfacts_by_ticker,
    )
    report = build_exploration_report(
        companies=companies,
        manifest_rows=manifest_rows,
        inventory_rows=inventory_rows,
        output_paths=output_paths,
    )

    _write_jsonl(manifest_rows, Path(output_paths["selected_filings_manifest_path"]))
    _write_jsonl(inventory_rows, Path(output_paths["xbrl_concept_inventory_path"]))
    _write_json(report, Path(output_paths["exploration_report_path"]))

    return {
        "output_paths": output_paths,
        "selected_filings_count": len(manifest_rows),
        "xbrl_inventory_count": len(inventory_rows),
        "warnings": report["warnings"],
    }


def build_download_json_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Download submissions/companyfacts JSON and build processed summaries."""

    start_year, end_year, forms, output_dir = _validate_common_args(args)
    registry = load_registry(Path(str(args.registry_path)))
    companies = select_companies(registry, str(args.company))

    submissions_by_ticker: dict[str, dict[str, Any]] = {}
    companyfacts_by_ticker: dict[str, dict[str, Any]] = {}
    for company in companies:
        ticker = str(company["ticker"])
        submissions_by_ticker[ticker] = download_json(
            url=str(company["submissions_url"]),
            destination=Path(str(company["local_raw_submissions_path"])),
            user_agent=str(args.user_agent),
            delay_seconds=float(args.request_delay_seconds),
        )
        companyfacts_by_ticker[ticker] = download_json(
            url=str(company["companyfacts_url"]),
            destination=Path(str(company["local_raw_companyfacts_path"])),
            user_agent=str(args.user_agent),
            delay_seconds=float(args.request_delay_seconds),
        )

    outputs = _build_outputs_from_local_data(
        companies=companies,
        submissions_by_ticker=submissions_by_ticker,
        companyfacts_by_ticker=companyfacts_by_ticker,
        forms=forms,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        max_filings_per_company=int(args.max_filings_per_company),
    )

    return {
        "mode": "download_json",
        "phase": "2A-3B",
        "company_filter": str(args.company),
        "companies_processed": [str(company["ticker"]) for company in companies],
        **outputs["output_paths"],
        "selected_filings_count": outputs["selected_filings_count"],
        "xbrl_inventory_count": outputs["xbrl_inventory_count"],
        "warnings": outputs["warnings"],
    }


def build_summarize_local_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Read local SEC JSON and rebuild processed summaries."""

    start_year, end_year, forms, output_dir = _validate_common_args(args)
    registry = load_registry(Path(str(args.registry_path)))
    companies = select_companies(registry, str(args.company))

    submissions_by_ticker: dict[str, dict[str, Any]] = {}
    companyfacts_by_ticker: dict[str, dict[str, Any]] = {}
    for company in companies:
        ticker = str(company["ticker"])
        submissions_by_ticker[ticker] = _read_json_file(
            Path(str(company["local_raw_submissions_path"]))
        )
        companyfacts_by_ticker[ticker] = _read_json_file(
            Path(str(company["local_raw_companyfacts_path"]))
        )

    outputs = _build_outputs_from_local_data(
        companies=companies,
        submissions_by_ticker=submissions_by_ticker,
        companyfacts_by_ticker=companyfacts_by_ticker,
        forms=forms,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        max_filings_per_company=int(args.max_filings_per_company),
    )

    return {
        "mode": "summarize_local",
        "phase": "2A-3B",
        "company_filter": str(args.company),
        "companies_processed": [str(company["ticker"]) for company in companies],
        **outputs["output_paths"],
        "selected_filings_count": outputs["selected_filings_count"],
        "xbrl_inventory_count": outputs["xbrl_inventory_count"],
        "warnings": outputs["warnings"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or run Phase 2A finance SEC/XBRL JSON acquisition."
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to the finance ticker registry JSON.",
    )
    parser.add_argument(
        "--company",
        default="all",
        choices=ALLOWED_COMPANY_FILTERS,
        help="Company ticker to plan, or all.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="First filing year to include in planning.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Last filing year to include in planning.",
    )
    parser.add_argument(
        "--forms",
        default=DEFAULT_FORMS,
        help="Comma-separated SEC forms to plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a JSON acquisition plan without downloading data.",
    )
    parser.add_argument(
        "--download-json",
        action="store_true",
        help="Download submissions and companyfacts JSON only.",
    )
    parser.add_argument(
        "--summarize-local",
        action="store_true",
        help="Read local SEC JSON and regenerate processed summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where processed finance exploration outputs should be written.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="Declared SEC User-Agent header.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.25,
        help="Conservative delay before each SEC JSON request.",
    )
    parser.add_argument(
        "--max-filings-per-company",
        type=int,
        default=0,
        help="Optional selected filing cap per company; 0 means uncapped.",
    )
    return parser


def _selected_mode_count(args: argparse.Namespace) -> int:
    return sum(bool(value) for value in (args.dry_run, args.download_json, args.summarize_local))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if _selected_mode_count(args) != 1:
        print(
            "Exactly one mode must be selected: --dry-run, --download-json, "
            "or --summarize-local. Download mode is not implemented in Phase "
            "2A-3A unless --download-json is explicitly selected.",
            file=sys.stderr,
        )
        return 2

    try:
        if int(args.max_filings_per_company) < 0:
            msg = "max_filings_per_company must be >= 0"
            raise ValueError(msg)

        if args.dry_run:
            payload = build_dry_run_plan(args)
        elif args.download_json:
            payload = build_download_json_summary(args)
        else:
            payload = build_summarize_local_summary(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
