"""Dry-run planner for Phase 2A finance SEC/XBRL acquisition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_PATH = Path("data/sources/finance_ticker_registry.json")
ALLOWED_COMPANY_FILTERS = ("all", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD")
DEFAULT_FORMS = "10-K,10-Q,8-K"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan Phase 2A finance SEC/XBRL acquisition without downloading data."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dry_run:
        print(
            "Download mode is not implemented in Phase 2A-3A. Re-run with --dry-run.",
            file=sys.stderr,
        )
        return 2

    try:
        plan = build_dry_run_plan(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
