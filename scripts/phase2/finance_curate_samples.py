"""Build curated Finance Phase 2A seed prompt, KB, and gold samples."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SELECTED_FILINGS_MANIFEST = Path(
    "data/processed/finance/sec/selected_filings_manifest.jsonl"
)
DEFAULT_DOCUMENT_MANIFEST = Path(
    "data/processed/finance/sec/selected_filing_documents_manifest.jsonl"
)
DEFAULT_TEXT_MANIFEST = Path("data/processed/finance/sec/filing_text_manifest.jsonl")
DEFAULT_SECTIONS_MANIFEST = Path("data/processed/finance/sec/filing_sections_manifest.jsonl")
DEFAULT_XBRL_INVENTORY = Path("data/processed/finance/sec/xbrl_concept_inventory.jsonl")
DEFAULT_COMPANYFACTS_DIR = Path("data/raw/finance/sec/companyfacts/")
DEFAULT_OUTPUT_PROMPTS = Path("data/real_world_samples/finance_sample.jsonl")
DEFAULT_OUTPUT_KB = Path("data/kb/finance/kb_sample.jsonl")
DEFAULT_OUTPUT_GOLD = Path("data/eval/gold/finance_gold_sample.jsonl")
DEFAULT_CURATION_REPORT = Path("data/processed/finance/sec/finance_curation_report.json")

TICKER_ORDER = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD")
REVENUE_CONCEPTS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
NET_INCOME_CONCEPTS = ("NetIncomeLoss", "ProfitLoss")
R_AND_D_CONCEPTS = ("ResearchAndDevelopmentExpense",)
ASSET_CONCEPTS = ("Assets",)
OPERATING_INCOME_CONCEPTS = ("OperatingIncomeLoss",)
IMPORTANT_CONCEPTS = (
    *REVENUE_CONCEPTS,
    "CostOfRevenue",
    "GrossProfit",
    *OPERATING_INCOME_CONCEPTS,
    *NET_INCOME_CONCEPTS,
    *ASSET_CONCEPTS,
    "Liabilities",
    "StockholdersEquity",
    *R_AND_D_CONCEPTS,
    "CashAndCashEquivalentsAtCarryingValue",
    "NetCashProvidedByUsedInOperatingActivities",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from a local artifact."""

    if not path.exists():
        msg = f"Missing required finance artifact: {path}. Run Finance Phase 2A-3B/3C/3D first."
        raise RuntimeError(msg)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON in {path} on line {line_number}: {exc.msg}"
                raise RuntimeError(msg) from exc
            if not isinstance(row, dict):
                msg = f"Expected object row in {path} on line {line_number}"
                raise RuntimeError(msg)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows with stable key ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    """Load a local JSON object."""

    if not path.exists():
        msg = f"Missing required finance artifact: {path}"
        raise RuntimeError(msg)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"Expected JSON object in {path}"
        raise RuntimeError(msg)
    return parsed


def _ascii_clean(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text.encode("ascii", errors="ignore").decode("ascii")


def safe_text_excerpt(text: str, max_chars: int) -> str:
    """Return a compact ASCII-safe excerpt for committed curated samples."""

    compact = re.sub(r"\s+", " ", _ascii_clean(text)).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def group_by_ticker(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group manifest rows by ticker."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            grouped[ticker].append(row)
    return dict(grouped)


def load_companyfacts(companyfacts_dir: Path) -> dict[str, dict[str, Any]]:
    """Load local SEC companyfacts JSON files keyed by ticker from file names."""

    if not companyfacts_dir.exists():
        msg = (
            f"Missing companyfacts directory: {companyfacts_dir}. "
            "Run Finance Phase 2A-3B --download-json first."
        )
        raise RuntimeError(msg)

    companyfacts_by_ticker: dict[str, dict[str, Any]] = {}
    for path in sorted(companyfacts_dir.glob("*_CIK*.json")):
        ticker = path.name.split("_CIK", 1)[0].upper()
        companyfacts_by_ticker[ticker] = load_json(path)

    if not companyfacts_by_ticker:
        msg = f"No companyfacts JSON files found in {companyfacts_dir}"
        raise RuntimeError(msg)
    return companyfacts_by_ticker


def _fact_sort_key(observation: dict[str, Any]) -> tuple[int, int, str, str]:
    fp_rank = {"FY": 4, "Q4": 4, "Q3": 3, "Q2": 2, "Q1": 1}
    fy = observation.get("fy")
    year = int(fy) if isinstance(fy, int) else -1
    fp = str(observation.get("fp") or "")
    return (
        year,
        fp_rank.get(fp, 0),
        str(observation.get("filed") or ""),
        str(observation.get("end") or ""),
    )


def _normalize_observation(
    concept: str,
    unit: str,
    observation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "concept": concept,
        "value": observation.get("val"),
        "unit": unit,
        "fy": observation.get("fy"),
        "fp": observation.get("fp"),
        "form": observation.get("form"),
        "filed": observation.get("filed"),
        "start": observation.get("start"),
        "end": observation.get("end"),
        "accn": observation.get("accn"),
        "frame": observation.get("frame"),
    }


def get_company_fact_observations(
    companyfacts: dict[str, Any],
    concept: str,
    preferred_units: list[str],
) -> list[dict[str, Any]]:
    """Return normalized XBRL observations for a concept."""

    concept_payload = (
        companyfacts.get("facts", {}).get("us-gaap", {}).get(concept, {})
        if isinstance(companyfacts.get("facts"), dict)
        else {}
    )
    units_payload = concept_payload.get("units", {}) if isinstance(concept_payload, dict) else {}
    if not isinstance(units_payload, dict):
        return []

    ordered_units = [unit for unit in preferred_units if unit in units_payload]
    ordered_units.extend(unit for unit in sorted(units_payload) if unit not in set(ordered_units))
    observations: list[dict[str, Any]] = []
    for unit in ordered_units:
        unit_observations = units_payload.get(unit, [])
        if not isinstance(unit_observations, list):
            continue
        for observation in unit_observations:
            if isinstance(observation, dict):
                observations.append(_normalize_observation(concept, unit, observation))
    return sorted(observations, key=_fact_sort_key)


def _select_fact_for_concepts(
    companyfacts: dict[str, Any],
    concept_candidates: list[str],
    *,
    quarterly: bool,
    limit: int = 1,
) -> list[dict[str, Any]]:
    preferred_units = ["USD", "USD/shares", "shares", "pure"]
    candidate_records: list[tuple[int, dict[str, Any]]] = []
    for concept_index, concept in enumerate(concept_candidates):
        observations = get_company_fact_observations(companyfacts, concept, preferred_units)
        if quarterly:
            candidate_records.extend(
                (concept_index, observation)
                for observation in observations
                if observation.get("form") == "10-Q"
                and observation.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
                and observation.get("fy") in {2024, 2025, 2026}
                and isinstance(observation.get("value"), (int, float))
            )
            continue

        candidate_records.extend(
            (concept_index, observation)
            for observation in observations
            if observation.get("form") == "10-K"
            and observation.get("fp") == "FY"
            and isinstance(observation.get("value"), (int, float))
        )

    if not candidate_records:
        return []

    if quarterly:
        sorted_candidates = sorted(candidate_records, key=lambda item: _fact_sort_key(item[1]))
        return [observation for _concept_index, observation in sorted_candidates[-limit:]]

    preferred_records = [
        (concept_index, observation)
        for concept_index, observation in candidate_records
        if observation.get("fy") in {2024, 2025}
    ]
    selected_pool = preferred_records or candidate_records
    selected_pool.sort(key=lambda item: (_fact_sort_key(item[1]), -item[0]))
    return [selected_pool[-1][1]]


def select_latest_annual_fact(
    companyfacts: dict[str, Any],
    concept_candidates: list[str],
) -> dict[str, Any] | None:
    """Select a latest annual fact for preferred concepts."""

    selected = _select_fact_for_concepts(
        companyfacts,
        concept_candidates,
        quarterly=False,
        limit=1,
    )
    return selected[0] if selected else None


def select_recent_quarterly_facts(
    companyfacts: dict[str, Any],
    concept_candidates: list[str],
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Select recent quarterly facts for preferred concepts."""

    return _select_fact_for_concepts(
        companyfacts,
        concept_candidates,
        quarterly=True,
        limit=limit,
    )


def _section_priority(row: dict[str, Any]) -> tuple[int, int, str]:
    section_rank = {
        "management_discussion_and_analysis": 0,
        "risk_factors": 1,
        "results_of_operations": 2,
        "liquidity_and_capital_resources": 3,
        "financial_statements": 4,
        "exhibit_99": 5,
        "financial_statements_and_exhibits": 6,
        "controls_and_procedures": 7,
    }
    form_rank = {"10-K": 0, "10-Q": 1, "8-K": 2}
    return (
        section_rank.get(str(row.get("section_type")), 99),
        form_rank.get(str(row.get("form")), 99),
        str(row.get("filing_date") or ""),
    )


def _read_section_body(section_row: dict[str, Any], max_body_chars: int) -> str:
    text_path = Path(str(section_row.get("local_text_path") or ""))
    if not text_path.exists():
        return safe_text_excerpt(str(section_row.get("section_title") or ""), max_body_chars)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    start = int(section_row.get("section_start_char") or 0)
    end = int(section_row.get("section_end_char") or min(len(text), start + max_body_chars))
    return safe_text_excerpt(text[start:end], max_body_chars)


def _section_tags(row: dict[str, Any]) -> list[str]:
    tags = ["finance", "sec", str(row.get("form") or "").lower()]
    section_type = str(row.get("section_type") or "")
    if section_type:
        tags.append(section_type)
    if "risk" in section_type:
        tags.append("risk")
    if "management" in section_type:
        tags.append("mda")
    if str(row.get("form")) == "8-K":
        tags.extend(["earnings", "8-k"])
    return sorted({tag for tag in tags if tag})


def build_finance_kb_records(
    section_rows: list[dict[str, Any]],
    text_manifest_rows: list[dict[str, Any]],
    xbrl_inventory_rows: list[dict[str, Any]],
    max_body_chars: int,
) -> list[dict[str, Any]]:
    """Build curated finance KB/context seed records."""

    _ = text_manifest_rows
    selected_sections: list[dict[str, Any]] = []
    for ticker in TICKER_ORDER:
        ticker_sections = [row for row in section_rows if str(row.get("ticker")).upper() == ticker]
        seen_types: set[str] = set()
        for section in sorted(ticker_sections, key=_section_priority):
            section_type = str(section.get("section_type") or "")
            if section_type in seen_types and len(seen_types) >= 3:
                continue
            selected_sections.append(section)
            seen_types.add(section_type)
            if len([row for row in selected_sections if row.get("ticker") == ticker]) >= 3:
                break

    if len(selected_sections) < 24:
        existing_ids = {row.get("section_record_id") for row in selected_sections}
        for section in sorted(section_rows, key=_section_priority):
            if section.get("section_record_id") in existing_ids:
                continue
            selected_sections.append(section)
            existing_ids.add(section.get("section_record_id"))
            if len(selected_sections) >= 24:
                break

    kb_records: list[dict[str, Any]] = []
    section_doc_index = 1
    for section in selected_sections:
        ticker = str(section.get("ticker") or "")
        form = str(section.get("form") or "")
        accession = str(section.get("accession_number") or "").replace("-", "")
        section_type = str(section.get("section_type") or "section")
        doc_id = f"finance_kb_sec_{ticker}_{form.replace('-', '')}_{accession}_{section_type}"
        kb_records.append(
            {
                "doc_id": doc_id,
                "vertical": "finance",
                "title": (
                    f"{ticker} {form} {section_type.replace('_', ' ').title()} "
                    f"({section.get('filing_date')})"
                ),
                "document_type": "sec_filing_section",
                "source_type": "derived",
                "body": _read_section_body(section, max_body_chars),
                "version": "phase2a-3e-seed-v1",
                "tags": _section_tags(section),
                "source_id": "finance_sec_edgar_xbrl",
                "allowed_to_commit": True,
                "related_record_ids": [
                    str(section.get("document_record_id")),
                    str(section.get("section_record_id")),
                ],
                "metadata": {
                    "curation_index": section_doc_index,
                    "ticker": ticker,
                    "company_name": section.get("company_name"),
                    "form": form,
                    "filing_date": section.get("filing_date"),
                    "report_date": section.get("report_date"),
                    "accession_number": section.get("accession_number"),
                    "section_type": section_type,
                    "source_manifest_record_id": section.get("source_manifest_record_id"),
                    "document_record_id": section.get("document_record_id"),
                    "section_record_id": section.get("section_record_id"),
                },
            }
        )
        section_doc_index += 1

    inventory_by_ticker = group_by_ticker(xbrl_inventory_rows)
    for ticker in TICKER_ORDER:
        rows = inventory_by_ticker.get(ticker, [])
        important_rows = [row for row in rows if str(row.get("concept")) in set(IMPORTANT_CONCEPTS)]
        concept_fragments = []
        for row in sorted(important_rows, key=lambda item: str(item.get("concept")))[:16]:
            concept_fragments.append(
                f"{row.get('concept')} ({row.get('observation_count')} observations; "
                f"forms: {', '.join(str(form) for form in row.get('forms_present', []))})"
            )
        body = safe_text_excerpt(
            f"XBRL concept availability summary for {ticker}: " + "; ".join(concept_fragments),
            max_body_chars,
        )
        kb_records.append(
            {
                "doc_id": f"finance_kb_xbrl_{ticker}_concept_inventory",
                "vertical": "finance",
                "title": f"{ticker} XBRL important concept inventory",
                "document_type": "xbrl_fact_table",
                "source_type": "derived",
                "body": body,
                "version": "phase2a-3e-seed-v1",
                "tags": ["finance", "sec", "xbrl", "revenue", "net-income"],
                "source_id": "finance_sec_edgar_xbrl",
                "allowed_to_commit": True,
                "metadata": {
                    "ticker": ticker,
                    "concepts": [row.get("concept") for row in important_rows],
                },
            }
        )

    return kb_records


def _company_names_from_sections(section_rows: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in section_rows:
        ticker = str(row.get("ticker") or "").upper()
        company = str(row.get("company_name") or "").strip()
        if ticker and company and ticker not in names:
            names[ticker] = company
    return names


def _kb_by_ticker_and_type(
    kb_records: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in kb_records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        ticker = str(metadata.get("ticker") or "").upper()
        section_type = str(metadata.get("section_type") or record.get("document_type") or "")
        if ticker:
            grouped[ticker][section_type].append(record)
    return {ticker: dict(types) for ticker, types in grouped.items()}


def _xbrl_kb_doc_id(ticker: str) -> str:
    return f"finance_kb_xbrl_{ticker}_concept_inventory"


def _next_prompt_id(index: int) -> str:
    return f"finance_seed_{index:04d}"


def _prompt_record(
    *,
    index: int,
    prompt_category: str,
    task_type: str,
    question: str,
    expected_output_format: str,
    expected_status: str,
    company: str,
    ticker: str,
    source_doc_ids: list[str],
    metadata: dict[str, Any],
    filing_form: str | None = None,
    fiscal_year: int | None = None,
    fiscal_period: str | None = None,
    required_facts: list[str] | None = None,
    required_sections: list[str] | None = None,
) -> dict[str, Any]:
    prompt_metadata = {"prompt_category": prompt_category, **metadata}
    record: dict[str, Any] = {
        "prompt_id": _next_prompt_id(index),
        "vertical": "finance",
        "company": company,
        "ticker": ticker,
        "source_doc_ids": source_doc_ids,
        "task_type": task_type,
        "question": question,
        "expected_output_format": expected_output_format,
        "expected_status": expected_status,
        "metadata": prompt_metadata,
    }
    if filing_form:
        record["filing_form"] = filing_form
    if fiscal_year is not None:
        record["fiscal_year"] = fiscal_year
    if fiscal_period:
        record["fiscal_period"] = fiscal_period
    if required_facts:
        record["required_facts"] = required_facts
    if required_sections:
        record["required_sections"] = required_sections
    return record


def build_finance_prompt_records(
    kb_records: list[dict[str, Any]],
    companyfacts_by_ticker: dict[str, dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build exactly 40 deterministic curated finance prompt/source records."""

    company_names = _company_names_from_sections(section_rows)
    kb_by_ticker_type = _kb_by_ticker_and_type(kb_records)
    prompts: list[dict[str, Any]] = []

    def add_prompt(**kwargs: Any) -> None:
        prompts.append(_prompt_record(index=len(prompts) + 1, **kwargs))

    for ticker in TICKER_ORDER:
        fact = select_latest_annual_fact(companyfacts_by_ticker[ticker], list(REVENUE_CONCEPTS))
        if fact is None:
            continue
        add_prompt(
            prompt_category="direct_numeric_fact",
            task_type="answer_short",
            question=(
                f"What {fact['concept']} value did {company_names[ticker]} report "
                f"for fiscal year {fact.get('fy')}?"
            ),
            expected_output_format="text",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[_xbrl_kb_doc_id(ticker)],
            filing_form=str(fact.get("form") or ""),
            fiscal_year=int(fact.get("fy") or 0),
            fiscal_period=str(fact.get("fp") or ""),
            required_facts=[str(fact["concept"])],
            metadata={"fact": fact},
        )

    grounded_tickers = TICKER_ORDER[:6]
    for ticker in grounded_tickers:
        section_record = (
            kb_by_ticker_type.get(ticker, {}).get("management_discussion_and_analysis")
            or next(iter(kb_by_ticker_type.get(ticker, {}).values()))
        )[0]
        metadata = section_record["metadata"]
        add_prompt(
            prompt_category="single_document_grounded_qa",
            task_type="answer_grounded",
            question=(
                f"Using only the cited {ticker} {metadata.get('form')} section, "
                "summarize the main finance-relevant point in two to four sentences."
            ),
            expected_output_format="text",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[section_record["doc_id"]],
            filing_form=str(metadata.get("form") or ""),
            required_sections=[str(metadata.get("section_type") or "")],
            metadata={"section_record_id": metadata.get("section_record_id")},
        )

    structured_tickers = TICKER_ORDER[:6]
    for ticker in structured_tickers:
        section_record = (
            kb_by_ticker_type.get(ticker, {}).get("financial_statements")
            or next(iter(kb_by_ticker_type.get(ticker, {}).values()))
        )[0]
        metadata = section_record["metadata"]
        add_prompt(
            prompt_category="structured_json_extraction",
            task_type="extract_structured",
            question=(
                "Return JSON with keys ticker, filing_form, filing_date, section_type, "
                f"and cited_doc_id for the cited {ticker} filing section."
            ),
            expected_output_format="json",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[section_record["doc_id"]],
            filing_form=str(metadata.get("form") or ""),
            required_sections=[str(metadata.get("section_type") or "")],
            metadata={"section_record_id": metadata.get("section_record_id")},
        )

    for ticker in TICKER_ORDER[:5]:
        facts = select_recent_quarterly_facts(
            companyfacts_by_ticker[ticker],
            list(REVENUE_CONCEPTS),
            limit=4,
        )
        add_prompt(
            prompt_category="trend_analysis",
            task_type="trend_summary",
            question=(
                f"Using the cited XBRL fact summary, describe the recent quarterly "
                f"revenue trend for {company_names[ticker]} without making projections."
            ),
            expected_output_format="text",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[_xbrl_kb_doc_id(ticker)],
            required_facts=[str(facts[0]["concept"]) if facts else "revenue"],
            metadata={"facts": facts},
        )

    comparison_sets = [
        ("AAPL", "MSFT", REVENUE_CONCEPTS, "revenue"),
        ("NVDA", "AMD", R_AND_D_CONCEPTS, "research and development expense"),
        ("GOOGL", "META", NET_INCOME_CONCEPTS, "net income"),
        ("AMZN", "TSLA", ASSET_CONCEPTS, "assets"),
    ]
    for left, right, concepts, metric_label in comparison_sets:
        add_prompt(
            prompt_category="cross_company_comparison",
            task_type="compare_companies",
            question=(
                f"Create a markdown table comparing latest annual {metric_label} for "
                f"{left} and {right} using the cited XBRL summaries."
            ),
            expected_output_format="markdown_table",
            expected_status="answer",
            company=f"{company_names[left]} and {company_names[right]}",
            ticker="MULTI",
            source_doc_ids=[_xbrl_kb_doc_id(left), _xbrl_kb_doc_id(right)],
            required_facts=list(concepts),
            metadata={"comparison_tickers": [left, right], "metric_label": metric_label},
        )

    summary_specs = [
        ("AAPL", "risk_factors", "risk_summary"),
        ("MSFT", "management_discussion_and_analysis", "answer_grounded"),
        ("NVDA", "risk_factors", "risk_summary"),
        ("AMZN", "results_of_operations", "answer_grounded"),
    ]
    for ticker, section_type, task_type in summary_specs:
        section_record = (
            kb_by_ticker_type.get(ticker, {}).get(section_type)
            or next(iter(kb_by_ticker_type.get(ticker, {}).values()))
        )[0]
        metadata = section_record["metadata"]
        add_prompt(
            prompt_category="summarization_risk_mda",
            task_type=task_type,
            question=(
                f"Summarize the cited {ticker} {section_type.replace('_', ' ')} section "
                "for a finance analyst, staying within the evidence."
            ),
            expected_output_format="text",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[section_record["doc_id"]],
            filing_form=str(metadata.get("form") or ""),
            required_sections=[section_type],
            metadata={"section_record_id": metadata.get("section_record_id")},
        )

    for ticker in ("AAPL", "MSFT", "NVDA"):
        revenue = select_latest_annual_fact(companyfacts_by_ticker[ticker], list(REVENUE_CONCEPTS))
        net_income = select_latest_annual_fact(
            companyfacts_by_ticker[ticker],
            list(NET_INCOME_CONCEPTS),
        )
        add_prompt(
            prompt_category="calculation",
            task_type="calculation_answer",
            question=(
                f"Calculate latest annual net margin for {company_names[ticker]} using "
                "net income divided by revenue. Show the formula."
            ),
            expected_output_format="text",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[_xbrl_kb_doc_id(ticker)],
            required_facts=["NetIncomeLoss", "Revenue"],
            metadata={"revenue_fact": revenue, "net_income_fact": net_income},
        )

    for ticker, section_type in (("META", "risk_factors"), ("TSLA", "results_of_operations")):
        section_record = (
            kb_by_ticker_type.get(ticker, {}).get(section_type)
            or next(iter(kb_by_ticker_type.get(ticker, {}).values()))
        )[0]
        metadata = section_record["metadata"]
        add_prompt(
            prompt_category="evidence_citation_lookup",
            task_type="answer_grounded",
            question=(
                f"Identify the cited document and section ID that should support an answer "
                f"about {ticker} {section_type.replace('_', ' ')}."
            ),
            expected_output_format="text",
            expected_status="answer",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[section_record["doc_id"]],
            filing_form=str(metadata.get("form") or ""),
            required_sections=[section_type],
            metadata={"section_record_id": metadata.get("section_record_id")},
        )

    for ticker, topic in (
        ("GOOGL", "confidential internal cloud margin target for next quarter"),
        ("AMD", "unannounced board-approved acquisition budget"),
    ):
        add_prompt(
            prompt_category="escalation_insufficient_evidence",
            task_type="escalation_response",
            question=(
                f"Using public SEC filings only, provide the {topic} for {company_names[ticker]}."
            ),
            expected_output_format="text",
            expected_status="insufficient_evidence",
            company=company_names[ticker],
            ticker=ticker,
            source_doc_ids=[_xbrl_kb_doc_id(ticker)],
            metadata={"insufficient_evidence_topic": topic},
        )

    if len(prompts) != 40:
        msg = f"Expected 40 curated finance prompts, built {len(prompts)}"
        raise RuntimeError(msg)
    return prompts


def _format_fact_value(fact: dict[str, Any] | None) -> str:
    if not fact:
        return "not available"
    return f"{fact.get('value')} {fact.get('unit')} (FY {fact.get('fy')}, {fact.get('fp')})"


def _doc_ids_for_prompt(prompt: dict[str, Any], kb_ids: set[str]) -> list[str]:
    return [doc_id for doc_id in prompt.get("source_doc_ids", []) if doc_id in kb_ids]


def build_finance_gold_records(
    prompt_records: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    companyfacts_by_ticker: dict[str, dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build matching deterministic gold/eval records for finance prompts."""

    _ = companyfacts_by_ticker
    section_ids = {str(row.get("section_record_id")) for row in section_rows}
    kb_ids = {str(record.get("doc_id")) for record in kb_records}
    gold_records: list[dict[str, Any]] = []
    for prompt in prompt_records:
        metadata = prompt.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        category = str(metadata.get("prompt_category") or "")
        gold: dict[str, Any] = {
            "prompt_id": prompt["prompt_id"],
            "vertical": "finance",
            "task_type": prompt["task_type"],
            "expected_status": prompt["expected_status"],
            "must_include": ["public SEC evidence"],
            "must_not_include": [
                "unsupported claims",
                "private/internal budgets",
                "unverifiable projections",
            ],
            "required_doc_ids": _doc_ids_for_prompt(prompt, kb_ids),
            "metadata": {"prompt_category": category, "ticker": prompt.get("ticker")},
        }

        if category == "direct_numeric_fact":
            fact = metadata.get("fact") if isinstance(metadata.get("fact"), dict) else None
            if fact is None:
                msg = f"Missing numeric fact metadata for prompt {prompt['prompt_id']}"
                raise RuntimeError(msg)
            numeric_value = float(fact["value"]) if fact.get("value") is not None else 0.0
            gold.update(
                {
                    "reference_answer": (
                        f"{prompt['company']} reported {fact.get('concept')} of "
                        f"{fact.get('value')} {fact.get('unit')} for FY {fact.get('fy')}."
                    ),
                    "must_include": [
                        str(fact.get("concept")),
                        str(fact.get("fy")),
                        str(fact.get("value")),
                    ],
                    "numeric_answer": numeric_value,
                    "tolerance": 0,
                    "required_citations": [str(fact.get("accn") or prompt["source_doc_ids"][0])],
                    "metadata": {**gold["metadata"], "fact": fact},
                }
            )
        elif category == "calculation":
            revenue = (
                metadata.get("revenue_fact")
                if isinstance(metadata.get("revenue_fact"), dict)
                else None
            )
            net_income = (
                metadata.get("net_income_fact")
                if isinstance(metadata.get("net_income_fact"), dict)
                else None
            )
            revenue_value = float(revenue["value"]) if revenue and revenue.get("value") else 0.0
            net_income_value = (
                float(net_income["value"]) if net_income and net_income.get("value") else 0.0
            )
            net_margin = net_income_value / revenue_value if revenue_value else 0.0
            gold.update(
                {
                    "reference_answer": (
                        f"Net margin = net income / revenue = {net_income_value} / "
                        f"{revenue_value} = {net_margin:.4f}."
                    ),
                    "must_include": ["Net margin", "net income", "revenue"],
                    "formula": "Net margin = Net income / revenue",
                    "numeric_answer": net_margin,
                    "tolerance": 0.01,
                    "metadata": {
                        **gold["metadata"],
                        "revenue_fact": revenue,
                        "net_income_fact": net_income,
                    },
                }
            )
        elif category == "structured_json_extraction":
            gold.update(
                {
                    "reference_answer": (
                        "Return JSON with ticker, filing_form, filing_date, section_type, "
                        "and cited_doc_id."
                    ),
                    "must_include": [
                        "ticker",
                        "filing_form",
                        "filing_date",
                        "section_type",
                    ],
                }
            )
        elif category == "trend_analysis":
            facts = metadata.get("facts") if isinstance(metadata.get("facts"), list) else []
            gold.update(
                {
                    "reference_answer": (
                        "Describe the direction and variability of recent quarterly revenue "
                        "using the cited XBRL observations; do not forecast."
                    ),
                    "must_include": ["quarterly", "revenue", "no projections"],
                    "metadata": {**gold["metadata"], "facts": facts},
                }
            )
        elif category == "cross_company_comparison":
            gold.update(
                {
                    "reference_answer": (
                        "Provide a markdown table comparing the cited metric for the "
                        "requested companies."
                    ),
                    "must_include": ["markdown table", "company", "metric"],
                }
            )
        elif category == "escalation_insufficient_evidence":
            gold.update(
                {
                    "expected_status": "insufficient_evidence",
                    "expected_escalation": True,
                    "escalation_reason": (
                        "Public SEC filings do not contain enough evidence for the requested "
                        "internal or confidential information."
                    ),
                    "reference_answer": (
                        "Insufficient evidence in public SEC filings; do not guess internal "
                        "confidential information."
                    ),
                    "must_include": ["insufficient evidence", "public SEC filings"],
                    "must_not_include": [
                        "guessed internal target",
                        "confidential budget",
                        "unsupported projection",
                    ],
                }
            )
        else:
            section_id = str(metadata.get("section_record_id") or "")
            citations = (
                [section_id] if section_id in section_ids else list(prompt["source_doc_ids"])
            )
            gold.update(
                {
                    "reference_answer": (
                        "Answer only from the cited finance filing section and cite the "
                        "document or section ID."
                    ),
                    "must_include": [
                        str(prompt.get("ticker")),
                        "filing section",
                        "evidence",
                    ],
                    "required_chunk_ids": [section_id] if section_id else [],
                    "required_citations": citations,
                }
            )
        gold_records.append(gold)

    if len(gold_records) != 40:
        msg = f"Expected 40 curated finance gold records, built {len(gold_records)}"
        raise RuntimeError(msg)
    return gold_records


def build_finance_curation_report(
    prompt_records: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a local Finance Phase 2A-3E curation report."""

    return {
        "phase": "2A-3E",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_record_count": len(prompt_records),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "prompt_counts_by_task_type": dict(
            Counter(str(record.get("task_type")) for record in prompt_records)
        ),
        "prompt_counts_by_expected_output_format": dict(
            Counter(str(record.get("expected_output_format")) for record in prompt_records)
        ),
        "prompt_counts_by_expected_status": dict(
            Counter(str(record.get("expected_status")) for record in prompt_records)
        ),
        "prompt_counts_by_ticker": dict(
            Counter(str(record.get("ticker")) for record in prompt_records)
        ),
        "kb_counts_by_document_type": dict(
            Counter(str(record.get("document_type")) for record in kb_records)
        ),
        "gold_counts_by_expected_status": dict(
            Counter(str(record.get("expected_status")) for record in gold_records)
        ),
        "escalation_count": sum(
            1
            for record in gold_records
            if record.get("expected_escalation") is True
            or record.get("expected_status") in {"escalate", "insufficient_evidence"}
        ),
        "answerable_count": sum(
            1 for record in prompt_records if record.get("expected_status") == "answer"
        ),
        "warnings": [
            "This is a curated Finance seed dataset, not the full 10,000-prompt dataset.",
            (
                "RAG, retrieval, prompt assembly, and inference are deferred until all "
                "five Phase 2A vertical datasets are prepared."
            ),
            (
                "Gold/eval records are intended for data-contract validation and early "
                "evaluation design, not final benchmark claims."
            ),
        ],
        "next_step": (
            "Proceed to Phase 2A-4 Airline and Healthcare synthetic generator pilots "
            "after reviewing the Finance curated samples."
        ),
    }


def build_curated_samples(args: argparse.Namespace) -> dict[str, Any]:
    """Build and write Finance curated seed samples."""

    selected_filings = read_jsonl(Path(str(args.selected_filings_manifest)))
    document_rows = read_jsonl(Path(str(args.document_manifest)))
    text_rows = read_jsonl(Path(str(args.text_manifest)))
    section_rows = read_jsonl(Path(str(args.sections_manifest)))
    xbrl_rows = read_jsonl(Path(str(args.xbrl_inventory)))
    companyfacts = load_companyfacts(Path(str(args.companyfacts_dir)))
    if not (selected_filings and document_rows and text_rows and section_rows and xbrl_rows):
        msg = "Finance curation requires non-empty Phase 2A-3B/3C/3D artifacts."
        raise RuntimeError(msg)

    kb_records = build_finance_kb_records(
        section_rows=section_rows,
        text_manifest_rows=text_rows,
        xbrl_inventory_rows=xbrl_rows,
        max_body_chars=int(args.max_kb_body_chars),
    )
    prompt_records = build_finance_prompt_records(kb_records, companyfacts, section_rows)
    gold_records = build_finance_gold_records(
        prompt_records,
        kb_records,
        companyfacts,
        section_rows,
    )
    report = build_finance_curation_report(prompt_records, kb_records, gold_records)

    write_jsonl(Path(str(args.output_prompts)), prompt_records)
    write_jsonl(Path(str(args.output_kb)), kb_records)
    write_jsonl(Path(str(args.output_gold)), gold_records)
    report_path = Path(str(args.curation_report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "mode": "build_curated_samples",
        "phase": "2A-3E",
        "prompt_record_count": len(prompt_records),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "output_prompts": str(args.output_prompts),
        "output_kb": str(args.output_kb),
        "output_gold": str(args.output_gold),
        "curation_report": str(args.curation_report),
        "warnings": report["warnings"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build curated Finance Phase 2A seed samples.")
    parser.add_argument("--build-curated-samples", action="store_true")
    parser.add_argument(
        "--selected-filings-manifest", default=str(DEFAULT_SELECTED_FILINGS_MANIFEST)
    )
    parser.add_argument("--document-manifest", default=str(DEFAULT_DOCUMENT_MANIFEST))
    parser.add_argument("--text-manifest", default=str(DEFAULT_TEXT_MANIFEST))
    parser.add_argument("--sections-manifest", default=str(DEFAULT_SECTIONS_MANIFEST))
    parser.add_argument("--xbrl-inventory", default=str(DEFAULT_XBRL_INVENTORY))
    parser.add_argument("--companyfacts-dir", default=str(DEFAULT_COMPANYFACTS_DIR))
    parser.add_argument("--output-prompts", default=str(DEFAULT_OUTPUT_PROMPTS))
    parser.add_argument("--output-kb", default=str(DEFAULT_OUTPUT_KB))
    parser.add_argument("--output-gold", default=str(DEFAULT_OUTPUT_GOLD))
    parser.add_argument("--curation-report", default=str(DEFAULT_CURATION_REPORT))
    parser.add_argument("--max-kb-body-chars", type=int, default=3000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.build_curated_samples:
        print("Select exactly one mode: --build-curated-samples.", file=sys.stderr)
        return 2
    if int(args.max_kb_body_chars) <= 0:
        print("max-kb-body-chars must be > 0.", file=sys.stderr)
        return 2

    try:
        summary = build_curated_samples(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
