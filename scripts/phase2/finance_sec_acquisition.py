"""SEC/XBRL acquisition planner and JSON-only explorer for Phase 2A finance."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import re
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
DEFAULT_SELECTED_FILINGS_MANIFEST_PATH = Path(
    "data/processed/finance/sec/selected_filings_manifest.jsonl"
)
DEFAULT_FILINGS_OUTPUT_DIR = Path("data/raw/finance/sec/filings/")
DEFAULT_DOCUMENTS_MANIFEST_PATH = Path(
    "data/processed/finance/sec/selected_filing_documents_manifest.jsonl"
)
DEFAULT_DOWNLOAD_REPORT_PATH = Path("data/processed/finance/sec/filing_download_report.json")
DEFAULT_EXTRACTED_TEXT_DIR = Path("data/processed/finance/sec/extracted_text/")
DEFAULT_TEXT_MANIFEST_PATH = Path("data/processed/finance/sec/filing_text_manifest.jsonl")
DEFAULT_SECTIONS_MANIFEST_PATH = Path("data/processed/finance/sec/filing_sections_manifest.jsonl")
DEFAULT_TEXT_EXTRACTION_REPORT_PATH = Path(
    "data/processed/finance/sec/finance_text_extraction_report.json"
)
ALLOWED_COMPANY_FILTERS = ("all", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD")
ALLOWED_FORM_FILTERS = ("all", "10-K", "10-Q", "8-K")
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
            "Form equals 8-K; filing date from 2024 onward; require item 2.02. "
            "Item 9.01 alone is not sufficient."
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


def _maybe_decompress_gzip(raw_payload: bytes, content_encoding: str, source_label: str) -> bytes:
    # SEC responses may be gzip-compressed when Accept-Encoding is requested, so
    # handle both explicit gzip headers and gzip magic bytes.
    if "gzip" not in content_encoding.lower() and not raw_payload.startswith(b"\x1f\x8b"):
        return raw_payload

    try:
        return gzip.decompress(raw_payload)
    except OSError as exc:
        msg = f"SEC endpoint returned invalid gzip data: {source_label}"
        raise RuntimeError(msg) from exc


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
            content_encoding = response.headers.get("Content-Encoding", "")
    except urllib.error.HTTPError as exc:
        msg = f"Failed to download SEC JSON from {url}: HTTP {exc.code} {exc.reason}"
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"Failed to download SEC JSON from {url}: {exc.reason}"
        raise RuntimeError(msg) from exc

    raw_payload = _maybe_decompress_gzip(raw_payload, content_encoding, url)

    try:
        response_text = raw_payload.decode("utf-8")
        parsed_json = json.loads(response_text)
    except UnicodeDecodeError as exc:
        msg = f"SEC endpoint returned non-UTF-8 JSON payload: {url}"
        raise RuntimeError(msg) from exc
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a newline-delimited JSON file into row dictionaries."""

    if not path.exists():
        msg = f"Missing selected filings manifest: {path}"
        raise RuntimeError(msg)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {path} on line {line_number}: {exc.msg}"
                raise RuntimeError(msg) from exc
            if not isinstance(row, dict):
                msg = f"Expected JSON object in {path} on line {line_number}"
                raise RuntimeError(msg)
            rows.append(row)
    return rows


def filter_manifest_rows(
    rows: list[dict[str, Any]],
    company_filter: str,
    form_filter: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Filter selected filings by ticker, form, and optional total limit."""

    normalized_company = company_filter.strip().upper()
    if normalized_company not in {company.upper() for company in ALLOWED_COMPANY_FILTERS}:
        msg = f"Unknown company filter: {company_filter}"
        raise ValueError(msg)

    normalized_form = form_filter.strip().upper()
    if normalized_form not in {form.upper() for form in ALLOWED_FORM_FILTERS}:
        msg = f"Unknown form filter: {form_filter}"
        raise ValueError(msg)

    if limit < 0:
        msg = "limit must be >= 0"
        raise ValueError(msg)

    filtered_rows = [
        row
        for row in rows
        if (normalized_company == "ALL" or str(row.get("ticker", "")).upper() == normalized_company)
        and (normalized_form == "ALL" or str(row.get("form", "")).upper() == normalized_form)
    ]

    if limit > 0:
        return filtered_rows[:limit]
    return filtered_rows


def safe_filename(value: str) -> str:
    """Return a readable filename segment without path separators."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.replace("\\", "_").replace("/", "_"))
    sanitized = sanitized.strip("._")
    return sanitized or "unknown"


def build_local_filing_path(row: dict[str, Any], filings_output_dir: Path) -> Path:
    """Build the local path for a downloaded filing document."""

    ticker = safe_filename(str(row.get("ticker") or "unknown").upper())
    form = safe_filename(str(row.get("form") or "unknown").upper())
    accession_without_dashes = safe_filename(
        str(row.get("accession_number") or "unknown").replace("-", "")
    )
    primary_document = safe_filename(str(row.get("primary_document") or "filing.html"))
    return filings_output_dir / ticker / form / accession_without_dashes / primary_document


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_binary(
    url: str,
    destination: Path,
    user_agent: str,
    delay_seconds: float,
) -> dict[str, Any]:
    """Download one selected SEC filing document as bytes without parsing HTML."""

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,*/*",
            "Accept-Encoding": "gzip, deflate",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw_payload = response.read()
            content_type = response.headers.get("Content-Type", "")
            content_encoding = response.headers.get("Content-Encoding", "")
    except urllib.error.HTTPError as exc:
        msg = f"Failed to download SEC filing document from {url}: HTTP {exc.code} {exc.reason}"
        raise RuntimeError(msg) from exc
    except urllib.error.URLError as exc:
        msg = f"Failed to download SEC filing document from {url}: {exc.reason}"
        raise RuntimeError(msg) from exc

    final_payload = _maybe_decompress_gzip(raw_payload, content_encoding, url)

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(final_payload)
    except OSError as exc:
        msg = f"Failed to write SEC filing document to {destination}: {exc}"
        raise RuntimeError(msg) from exc

    return {
        "url": url,
        "destination": str(destination),
        "status": "downloaded",
        "content_type": content_type,
        "content_encoding": content_encoding,
        "bytes_written": len(final_payload),
        "sha256": hashlib.sha256(final_payload).hexdigest(),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }


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


def parse_8k_items(items: object) -> set[str]:
    """Parse comma-separated 8-K item numbers into normalized item strings."""

    if not isinstance(items, str):
        return set()
    return {item.strip() for item in items.split(",") if item.strip()}


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
        items = parse_8k_items(row.get("items"))
        if "2.02" in items:
            return (
                True,
                "Selected earnings/results 8-K because item 2.02 is present; "
                "item 9.01 may provide supporting exhibits.",
            )
        return False, "8-K does not contain required earnings/results item 2.02"

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
                1
                for row in manifest_rows
                if row.get("form") == "8-K" and "2.02" in parse_8k_items(row.get("items"))
            ),
            "earnings_8k_selection_rule": (
                "8-K rows require item 2.02. Item 9.01 alone is not sufficient."
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


def build_document_manifest_rows(
    selected_rows: list[dict[str, Any]],
    download_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build document-level manifest rows from selected filing rows and downloads."""

    if len(selected_rows) != len(download_results):
        msg = "selected_rows and download_results must have the same length"
        raise ValueError(msg)

    document_rows: list[dict[str, Any]] = []
    for selected_row, download_result in zip(selected_rows, download_results, strict=True):
        ticker = str(selected_row.get("ticker") or "")
        form = str(selected_row.get("form") or "")
        accession_number = str(selected_row.get("accession_number") or "")
        accession_without_dashes = accession_number.replace("-", "")
        form_normalized = form.replace("-", "")
        document_record_id = f"finance_doc_{ticker}_{form_normalized}_{accession_without_dashes}"
        document_row = {
            "document_record_id": document_record_id,
            "source_manifest_record_id": selected_row.get("record_id"),
            "source_id": selected_row.get("source_id"),
            "vertical": selected_row.get("vertical"),
            "ticker": ticker,
            "company_name": selected_row.get("company_name"),
            "cik": selected_row.get("cik"),
            "cik_no_leading_zeros": selected_row.get("cik_no_leading_zeros"),
            "fiscal_year_end": selected_row.get("fiscal_year_end"),
            "form": form,
            "filing_date": selected_row.get("filing_date"),
            "report_date": selected_row.get("report_date"),
            "accession_number": accession_number,
            "items": selected_row.get("items"),
            "primary_document": selected_row.get("primary_document"),
            "primary_doc_description": selected_row.get("primary_doc_description"),
            "derived_filing_url": selected_row.get("derived_filing_url"),
            "local_html_path": download_result.get("destination"),
            "download_status": download_result.get("status"),
            "content_type": download_result.get("content_type"),
            "content_encoding": download_result.get("content_encoding"),
            "file_size_bytes": download_result.get("bytes_written", 0),
            "sha256": download_result.get("sha256"),
            "selection_reason": selected_row.get("selection_reason"),
            "downloaded_at_utc": download_result.get("downloaded_at_utc"),
        }
        if download_result.get("error_message"):
            document_row["error_message"] = download_result.get("error_message")
        document_rows.append(document_row)
    return document_rows


def build_filing_download_report(
    document_manifest_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """Build a Phase 2A-3C filing-document acquisition report."""

    status_counts = Counter(str(row.get("download_status")) for row in document_manifest_rows)
    return {
        "phase": "2A-3C",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_manifest_rows_available": len(selected_rows),
        "total_documents_attempted": len(document_manifest_rows),
        "total_documents_downloaded": status_counts.get("downloaded", 0),
        "total_documents_skipped_existing": status_counts.get("skipped_existing", 0),
        "total_documents_failed": status_counts.get("failed", 0),
        "counts_by_company": dict(
            Counter(str(row.get("ticker")) for row in document_manifest_rows)
        ),
        "counts_by_form": dict(Counter(str(row.get("form")) for row in document_manifest_rows)),
        "output_files": output_paths,
        "warnings": [
            "This report summarizes SEC filing document acquisition only.",
            "Text extraction and section parsing are deferred to Phase 2A-3D.",
            "Do not use these artifacts to make benchmark performance claims.",
        ],
        "next_step": (
            "Phase 2A-3D should extract readable text and finance-relevant sections "
            "from downloaded filing HTML documents."
        ),
    }


def strip_html_to_text(html_content: str) -> str:
    """Extract deterministic readable text from filing HTML using standard library tools."""

    text = re.sub(
        r"(?is)<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>",
        " ",
        html_content,
    )
    text = re.sub(
        r"(?i)<\s*(br|p|div|tr|table|section|article|h[1-6]|li)\b[^>]*>",
        "\n",
        text,
    )
    text = re.sub(
        r"(?i)<\s*/\s*(p|div|tr|table|section|article|h[1-6]|li)\s*>",
        "\n",
        text,
    )
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")

    normalized_lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped_line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        if stripped_line:
            normalized_lines.append(stripped_line)
    return "\n".join(normalized_lines)


def read_text_file(path: Path) -> str:
    """Read a local filing text/HTML file with clear missing-file errors."""

    if not path.exists():
        msg = f"Missing local SEC filing HTML file: {path}"
        raise RuntimeError(msg)

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def write_extracted_text(text: str, destination: Path) -> dict[str, Any]:
    """Write extracted filing text and return text-level metadata."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    encoded_text = text.encode("utf-8")
    return {
        "path": str(destination),
        "char_count": len(text),
        "word_count": len(re.findall(r"\S+", text)),
        "sha256": hashlib.sha256(encoded_text).hexdigest(),
    }


def build_extracted_text_path(document_row: dict[str, Any], extracted_text_dir: Path) -> Path:
    """Build the local path for extracted filing plain text."""

    ticker = safe_filename(str(document_row.get("ticker") or "unknown").upper())
    form = safe_filename(str(document_row.get("form") or "unknown").upper())
    accession_without_dashes = safe_filename(
        str(document_row.get("accession_number") or "unknown").replace("-", "")
    )
    return extracted_text_dir / ticker / form / f"{accession_without_dashes}.txt"


SECTION_PATTERNS_BY_FORM: dict[str, tuple[tuple[str, str], ...]] = {
    "10-K": (
        ("business", r"(?im)^\s*(?:item\s+1\.?\s+business\b|business\s*$)"),
        ("risk_factors", r"(?im)^\s*(?:item\s+1a\.?\s+risk factors\b|risk factors\s*$)"),
        (
            "legal_proceedings",
            r"(?im)^\s*(?:item\s+3\.?\s+legal proceedings\b|legal proceedings\s*$)",
        ),
        (
            "management_discussion_and_analysis",
            (
                r"(?im)^\s*(?:item\s+7\.?\s+management[’']?s discussion and analysis\b|"
                r"management[’']?s discussion and analysis\b)"
            ),
        ),
        (
            "quantitative_and_qualitative_disclosures",
            (
                r"(?im)^\s*(?:item\s+7a\.?\s+quantitative and qualitative disclosures\b|"
                r"quantitative and qualitative disclosures\b)"
            ),
        ),
        (
            "financial_statements",
            r"(?im)^\s*(?:item\s+8\.?\s+financial statements\b|financial statements\b)",
        ),
        (
            "controls_and_procedures",
            (
                r"(?im)^\s*(?:item\s+9a\.?\s+controls and procedures\b|"
                r"controls and procedures\b)"
            ),
        ),
        ("liquidity_and_capital_resources", r"(?im)^.*liquidity and capital resources.*$"),
        ("results_of_operations", r"(?im)^.*results of operations.*$"),
        ("segment_information", r"(?im)^.*segment information.*$"),
        ("notes_to_financial_statements", r"(?im)^.*notes to financial statements.*$"),
    ),
    "10-Q": (
        (
            "financial_statements",
            r"(?im)^\s*(?:item\s+1\.?\s+financial statements\b|financial statements\b)",
        ),
        (
            "management_discussion_and_analysis",
            (
                r"(?im)^\s*(?:item\s+2\.?\s+management[’']?s discussion and analysis\b|"
                r"management[’']?s discussion and analysis\b)"
            ),
        ),
        (
            "quantitative_and_qualitative_disclosures",
            (
                r"(?im)^\s*(?:item\s+3\.?\s+quantitative and qualitative disclosures\b|"
                r"quantitative and qualitative disclosures\b)"
            ),
        ),
        (
            "controls_and_procedures",
            (
                r"(?im)^\s*(?:item\s+4\.?\s+controls and procedures\b|"
                r"controls and procedures\b)"
            ),
        ),
        ("risk_factors", r"(?im)^\s*(?:item\s+1a\.?\s+risk factors\b|risk factors\s*$)"),
        ("legal_proceedings", r"(?im)^\s*(?:item\s+1\.?\s+legal proceedings\b)"),
        ("liquidity_and_capital_resources", r"(?im)^.*liquidity and capital resources.*$"),
        ("results_of_operations", r"(?im)^.*results of operations.*$"),
        ("segment_information", r"(?im)^.*segment information.*$"),
        ("notes_to_financial_statements", r"(?im)^.*notes to financial statements.*$"),
    ),
    "8-K": (
        (
            "results_of_operations",
            r"(?im)^.*results of operations and financial condition.*$|^.*results of operations.*$",
        ),
        ("financial_statements_and_exhibits", r"(?im)^.*financial statements and exhibits.*$"),
        ("exhibit_99", r"(?im)^.*(?:exhibit\s+99|ex-99).*?$"),
        ("earnings_release", r"(?im)^.*(?:earnings release|press release).*?$"),
        ("management_commentary", r"(?im)^.*management commentary.*$"),
        (
            "financial_tables",
            r"(?im)^.*(?:financial results|consolidated statements|condensed consolidated).*$",
        ),
    ),
}


def _section_patterns_for_form(form: str) -> tuple[tuple[str, str], ...]:
    normalized_form = form.upper()
    if normalized_form in {"10-K", "10-Q", "8-K"}:
        return SECTION_PATTERNS_BY_FORM[normalized_form]
    return ()


def _section_title_from_text(text: str, start: int, end: int) -> str:
    candidate = text[start : min(end, start + 240)].splitlines()[0].strip()
    return re.sub(r"\s+", " ", candidate)


def detect_finance_sections(
    text: str,
    document_row: dict[str, Any],
    min_section_chars: int = 500,
) -> list[dict[str, Any]]:
    """Detect deterministic finance-relevant section candidates in filing text."""

    form = str(document_row.get("form") or "").upper()
    matches: list[dict[str, Any]] = []
    seen_matches: set[tuple[str, int]] = set()
    for section_type, pattern in _section_patterns_for_form(form):
        for match in re.finditer(pattern, text):
            key = (section_type, match.start())
            if key in seen_matches:
                continue
            seen_matches.add(key)
            matches.append(
                {
                    "section_type": section_type,
                    "start": match.start(),
                    "title": _section_title_from_text(text, match.start(), match.end()),
                }
            )

    matches.sort(key=lambda item: int(item["start"]))
    section_rows: list[dict[str, Any]] = []
    ticker = str(document_row.get("ticker") or "")
    form_normalized = form.replace("-", "")
    accession_number = str(document_row.get("accession_number") or "")
    accession_without_dashes = accession_number.replace("-", "")
    document_record_id = str(document_row.get("document_record_id") or "")
    source_manifest_record_id = str(document_row.get("source_manifest_record_id") or "")
    local_text_path = str(document_row.get("local_text_path") or "")

    for match_index, section_match in enumerate(matches):
        start = int(section_match["start"])
        if match_index + 1 < len(matches):
            end = int(matches[match_index + 1]["start"])
        else:
            end = len(text)
        section_text = text[start:end].strip()
        char_count = len(section_text)
        allow_short_8k = form == "8-K" and len(text.strip()) <= min_section_chars
        if char_count < min_section_chars and not allow_short_8k:
            continue

        section_type = str(section_match["section_type"])
        section_index = len(section_rows) + 1
        section_record_id = (
            f"finance_section_{ticker}_{form_normalized}_{accession_without_dashes}_"
            f"{section_type}_{section_index}"
        )
        section_rows.append(
            {
                "section_record_id": section_record_id,
                "document_record_id": document_record_id,
                "source_manifest_record_id": source_manifest_record_id,
                "ticker": ticker,
                "company_name": document_row.get("company_name"),
                "form": form,
                "filing_date": document_row.get("filing_date"),
                "report_date": document_row.get("report_date"),
                "accession_number": accession_number,
                "primary_document": document_row.get("primary_document"),
                "section_type": section_type,
                "section_title": str(section_match["title"]),
                "section_start_char": start,
                "section_end_char": end,
                "char_count": char_count,
                "word_count": len(re.findall(r"\S+", section_text)),
                "extraction_method": "standard_library_heading_heuristic_v1",
                "local_text_path": local_text_path,
            }
        )

    return section_rows


def build_text_manifest_row(
    document_row: dict[str, Any],
    extracted_text_metadata: dict[str, Any],
    extraction_status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Build one extracted-text manifest row."""

    ticker = str(document_row.get("ticker") or "")
    form = str(document_row.get("form") or "")
    form_normalized = form.replace("-", "")
    accession_number = str(document_row.get("accession_number") or "")
    accession_without_dashes = accession_number.replace("-", "")
    text_record_id = f"finance_text_{ticker}_{form_normalized}_{accession_without_dashes}"

    return {
        "text_record_id": text_record_id,
        "document_record_id": document_row.get("document_record_id"),
        "source_manifest_record_id": document_row.get("source_manifest_record_id"),
        "source_id": document_row.get("source_id"),
        "vertical": document_row.get("vertical"),
        "ticker": ticker,
        "company_name": document_row.get("company_name"),
        "cik": document_row.get("cik"),
        "cik_no_leading_zeros": document_row.get("cik_no_leading_zeros"),
        "fiscal_year_end": document_row.get("fiscal_year_end"),
        "form": form,
        "filing_date": document_row.get("filing_date"),
        "report_date": document_row.get("report_date"),
        "accession_number": accession_number,
        "primary_document": document_row.get("primary_document"),
        "local_html_path": document_row.get("local_html_path"),
        "local_text_path": extracted_text_metadata.get("path"),
        "extraction_status": extraction_status,
        "file_size_bytes": document_row.get("file_size_bytes", 0),
        "text_char_count": extracted_text_metadata.get("char_count", 0),
        "text_word_count": extracted_text_metadata.get("word_count", 0),
        "text_sha256": extracted_text_metadata.get("sha256"),
        "section_count": extracted_text_metadata.get("section_count", 0),
        "error_message": error_message,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_text_extraction_report(
    text_manifest_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
    total_document_rows_available: int,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """Build a Phase 2A-3D text extraction report."""

    status_counts = Counter(str(row.get("extraction_status")) for row in text_manifest_rows)
    documents_with_zero_sections = [
        row.get("document_record_id")
        for row in text_manifest_rows
        if int(row.get("section_count") or 0) == 0
    ]
    return {
        "phase": "2A-3D",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_document_rows_available": total_document_rows_available,
        "total_documents_attempted": len(text_manifest_rows),
        "total_documents_extracted": status_counts.get("extracted", 0),
        "total_documents_failed": status_counts.get("failed", 0),
        "total_sections_extracted": len(section_rows),
        "counts_by_company": dict(Counter(str(row.get("ticker")) for row in text_manifest_rows)),
        "counts_by_form": dict(Counter(str(row.get("form")) for row in text_manifest_rows)),
        "sections_by_type": dict(Counter(str(row.get("section_type")) for row in section_rows)),
        "documents_with_zero_sections": documents_with_zero_sections,
        "output_files": output_paths,
        "warnings": [
            "This report summarizes local SEC filing text extraction only.",
            "Section extraction is heuristic and will be refined before RAG/context engineering.",
            (
                "Prompt generation, RAG, retrieval, and inference are deferred until all "
                "Phase 2A vertical data is prepared."
            ),
        ],
        "next_step": (
            "Phase 2A-3E should create curated finance source, KB, and gold/eval "
            "samples from extracted text and XBRL facts."
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


def build_download_filings_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Download selected SEC filing HTML documents from a local filing manifest."""

    manifest_path = Path(str(args.manifest_path))
    manifest_rows = read_jsonl(manifest_path)
    selected_rows = filter_manifest_rows(
        rows=manifest_rows,
        company_filter=str(args.company),
        form_filter=str(args.form),
        limit=int(args.limit),
    )
    if not selected_rows:
        msg = (
            "No selected filing rows matched the requested company/form/limit filters "
            f"from {manifest_path}"
        )
        raise ValueError(msg)

    filings_output_dir = Path(str(args.filings_output_dir))
    download_results: list[dict[str, Any]] = []
    for row in selected_rows:
        local_path = build_local_filing_path(row, filings_output_dir)
        url = str(row.get("derived_filing_url") or "")
        if not url:
            download_results.append(
                {
                    "url": url,
                    "destination": str(local_path),
                    "status": "failed",
                    "content_type": None,
                    "content_encoding": None,
                    "bytes_written": 0,
                    "sha256": None,
                    "error_message": "selected filing row is missing derived_filing_url",
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue

        if args.skip_existing and local_path.exists() and local_path.stat().st_size > 0:
            download_results.append(
                {
                    "url": url,
                    "destination": str(local_path),
                    "status": "skipped_existing",
                    "content_type": None,
                    "content_encoding": None,
                    "bytes_written": local_path.stat().st_size,
                    "sha256": _file_sha256(local_path),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue

        try:
            download_results.append(
                download_binary(
                    url=url,
                    destination=local_path,
                    user_agent=str(args.user_agent),
                    delay_seconds=float(args.request_delay_seconds),
                )
            )
        except RuntimeError as exc:
            download_results.append(
                {
                    "url": url,
                    "destination": str(local_path),
                    "status": "failed",
                    "content_type": None,
                    "content_encoding": None,
                    "bytes_written": 0,
                    "sha256": None,
                    "error_message": str(exc),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

    document_rows = build_document_manifest_rows(selected_rows, download_results)
    documents_manifest_path = Path(str(args.documents_manifest_path))
    download_report_path = Path(str(args.download_report_path))
    output_paths = {
        "documents_manifest_path": str(documents_manifest_path),
        "download_report_path": str(download_report_path),
    }
    report = build_filing_download_report(document_rows, manifest_rows, output_paths)

    _write_jsonl(document_rows, documents_manifest_path)
    _write_json(report, download_report_path)

    total_downloaded = int(report["total_documents_downloaded"])
    total_skipped = int(report["total_documents_skipped_existing"])
    total_failed = int(report["total_documents_failed"])
    all_attempted_downloads_failed = (
        len(document_rows) > 0 and total_failed == len(document_rows) and total_downloaded == 0
    )

    return {
        "mode": "download_filings",
        "phase": "2A-3C",
        "company_filter": str(args.company),
        "form_filter": str(args.form),
        "limit": int(args.limit),
        "documents_manifest_path": str(documents_manifest_path),
        "download_report_path": str(download_report_path),
        "total_documents_attempted": int(report["total_documents_attempted"]),
        "total_documents_downloaded": total_downloaded,
        "total_documents_skipped_existing": total_skipped,
        "total_documents_failed": total_failed,
        "warnings": report["warnings"],
        "all_attempted_downloads_failed": all_attempted_downloads_failed and total_skipped == 0,
    }


def build_extract_text_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Extract readable text and finance section candidates from local filing HTML."""

    documents_manifest_path = Path(str(args.documents_manifest_path))
    if not documents_manifest_path.exists():
        msg = f"Missing selected filing documents manifest: {documents_manifest_path}"
        raise RuntimeError(msg)

    document_rows = read_jsonl(documents_manifest_path)
    selected_rows = filter_manifest_rows(
        rows=document_rows,
        company_filter=str(args.company),
        form_filter=str(args.form),
        limit=int(args.limit),
    )
    if not selected_rows:
        msg = (
            "No selected filing document rows matched the requested company/form/limit "
            f"filters from {documents_manifest_path}"
        )
        raise ValueError(msg)

    extracted_text_dir = Path(str(args.extracted_text_dir))
    text_manifest_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        local_text_path = build_extracted_text_path(row, extracted_text_dir)
        row_with_text_path = {**row, "local_text_path": str(local_text_path)}
        try:
            html_content = read_text_file(Path(str(row.get("local_html_path") or "")))
            extracted_text = strip_html_to_text(html_content)
            extracted_metadata = write_extracted_text(extracted_text, local_text_path)
            document_section_rows = detect_finance_sections(
                extracted_text,
                row_with_text_path,
                min_section_chars=int(args.min_section_chars),
            )
            extracted_metadata["section_count"] = len(document_section_rows)
            text_manifest_rows.append(build_text_manifest_row(row, extracted_metadata, "extracted"))
            section_rows.extend(document_section_rows)
        except (OSError, RuntimeError, ValueError) as exc:
            failed_metadata = {
                "path": str(local_text_path),
                "char_count": 0,
                "word_count": 0,
                "sha256": None,
                "section_count": 0,
            }
            text_manifest_rows.append(
                build_text_manifest_row(
                    row,
                    failed_metadata,
                    "failed",
                    error_message=str(exc),
                )
            )

    text_manifest_path = Path(str(args.text_manifest_path))
    sections_manifest_path = Path(str(args.sections_manifest_path))
    text_extraction_report_path = Path(str(args.text_extraction_report_path))
    output_paths = {
        "text_manifest_path": str(text_manifest_path),
        "sections_manifest_path": str(sections_manifest_path),
        "text_extraction_report_path": str(text_extraction_report_path),
    }
    report = build_text_extraction_report(
        text_manifest_rows=text_manifest_rows,
        section_rows=section_rows,
        total_document_rows_available=len(document_rows),
        output_paths=output_paths,
    )

    _write_jsonl(text_manifest_rows, text_manifest_path)
    _write_jsonl(section_rows, sections_manifest_path)
    _write_json(report, text_extraction_report_path)

    total_extracted = int(report["total_documents_extracted"])
    total_failed = int(report["total_documents_failed"])
    all_attempted_extractions_failed = (
        len(text_manifest_rows) > 0
        and total_failed == len(text_manifest_rows)
        and total_extracted == 0
    )

    return {
        "mode": "extract_text",
        "phase": "2A-3D",
        "company_filter": str(args.company),
        "form_filter": str(args.form),
        "limit": int(args.limit),
        "text_manifest_path": str(text_manifest_path),
        "sections_manifest_path": str(sections_manifest_path),
        "text_extraction_report_path": str(text_extraction_report_path),
        "total_documents_attempted": int(report["total_documents_attempted"]),
        "total_documents_extracted": total_extracted,
        "total_documents_failed": total_failed,
        "total_sections_extracted": int(report["total_sections_extracted"]),
        "warnings": report["warnings"],
        "all_attempted_extractions_failed": all_attempted_extractions_failed,
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
        "--download-filings",
        action="store_true",
        help="Download selected SEC filing HTML documents from a local manifest.",
    )
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help="Extract readable text and finance section candidates from local filing HTML.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where processed finance exploration outputs should be written.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_SELECTED_FILINGS_MANIFEST_PATH),
        help="Path to the selected filings manifest JSONL from Phase 2A-3B.",
    )
    parser.add_argument(
        "--filings-output-dir",
        default=str(DEFAULT_FILINGS_OUTPUT_DIR),
        help="Directory where raw selected SEC filing documents should be written.",
    )
    parser.add_argument(
        "--documents-manifest-path",
        default=str(DEFAULT_DOCUMENTS_MANIFEST_PATH),
        help="Path for the selected filing documents manifest JSONL.",
    )
    parser.add_argument(
        "--download-report-path",
        default=str(DEFAULT_DOWNLOAD_REPORT_PATH),
        help="Path for the filing download report JSON.",
    )
    parser.add_argument(
        "--extracted-text-dir",
        default=str(DEFAULT_EXTRACTED_TEXT_DIR),
        help="Directory where extracted filing text files should be written.",
    )
    parser.add_argument(
        "--text-manifest-path",
        default=str(DEFAULT_TEXT_MANIFEST_PATH),
        help="Path for the filing text manifest JSONL.",
    )
    parser.add_argument(
        "--sections-manifest-path",
        default=str(DEFAULT_SECTIONS_MANIFEST_PATH),
        help="Path for the filing sections manifest JSONL.",
    )
    parser.add_argument(
        "--text-extraction-report-path",
        default=str(DEFAULT_TEXT_EXTRACTION_REPORT_PATH),
        help="Path for the text extraction report JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional total filing document download limit after filters; 0 means uncapped.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloads when the target local filing document already exists.",
    )
    parser.add_argument(
        "--form",
        default="all",
        choices=ALLOWED_FORM_FILTERS,
        help="Filing form to process, or all.",
    )
    parser.add_argument(
        "--min-section-chars",
        type=int,
        default=500,
        help="Minimum section candidate length for extracted finance sections.",
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
    return sum(
        bool(value)
        for value in (
            args.dry_run,
            args.download_json,
            args.summarize_local,
            args.download_filings,
            args.extract_text,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if _selected_mode_count(args) != 1:
        print(
            "Exactly one mode must be selected: --dry-run, --download-json, "
            "--summarize-local, --download-filings, or --extract-text. Download "
            "mode is not implemented in Phase 2A-3A unless an explicit download "
            "mode is selected.",
            file=sys.stderr,
        )
        return 2

    try:
        if int(args.max_filings_per_company) < 0:
            msg = "max_filings_per_company must be >= 0"
            raise ValueError(msg)
        if int(args.limit) < 0:
            msg = "limit must be >= 0"
            raise ValueError(msg)
        if int(args.min_section_chars) < 0:
            msg = "min_section_chars must be >= 0"
            raise ValueError(msg)

        if args.dry_run:
            payload = build_dry_run_plan(args)
        elif args.download_json:
            payload = build_download_json_summary(args)
        elif args.summarize_local:
            payload = build_summarize_local_summary(args)
        elif args.download_filings:
            payload = build_download_filings_summary(args)
        else:
            payload = build_extract_text_summary(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload.get("all_attempted_downloads_failed") or payload.get(
        "all_attempted_extractions_failed"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
