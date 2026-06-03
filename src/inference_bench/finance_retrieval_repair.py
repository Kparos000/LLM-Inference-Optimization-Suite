"""Finance prompt/gold metadata repair audit for retrieval planning.

This module audits Finance prompt and gold ambiguity, derives human-readable
retrieval metadata from linked benchmark context, and measures whether rewritten
queries improve Finance retrieval. It does not run model inference, GPU work, or
external API calls.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, cast

from inference_bench.context_schema import ContextRecord
from inference_bench.gold_evidence_audit import (
    finance_prompt_has_entity,
    finance_prompt_has_metric,
    finance_prompt_has_period,
    gold_ids_from_gold_record,
)
from inference_bench.memory_workloads import (
    build_retrievers,
    close_retrievers,
    load_context_corpora,
    load_prompts_and_gold,
    prompt_query_text,
    recall_at_candidate_k,
    retrieve_for_mode,
)
from inference_bench.retrieval import (
    DEFAULT_FINAL_TOP_K,
    FINANCE_METRIC_SYNONYMS,
    CompanyTickerResolver,
    RetrievalResult,
    context_match_ids,
    enrich_query_text,
    evaluate_retrieval_results,
    normalize_identifier,
    scrub_direct_evidence_identifiers,
    split_identifier_text,
    tokenize,
)

FINANCE_REPAIR_SCOPE = "offline_finance_retrieval_repair_no_inference_no_gpu_no_api"
FINANCE_PROMPT_QUALITY_FIELDS = [
    "total_prompts",
    "company_or_ticker_present_count",
    "company_or_ticker_missing_count",
    "metric_present_count",
    "metric_missing_count",
    "period_present_count",
    "period_missing_count",
    "filing_type_present_count",
    "filing_type_missing_count",
    "section_present_count",
    "section_missing_count",
]
FINANCE_GOLD_QUALITY_FIELDS = [
    "total_gold_records",
    "metric_explicitly_recoverable_count",
    "metric_explicitly_missing_count",
    "period_explicitly_recoverable_count",
    "period_explicitly_missing_count",
    "filing_explicitly_recoverable_count",
    "filing_explicitly_missing_count",
    "section_explicitly_recoverable_count",
    "section_explicitly_missing_count",
    "metric_recoverable_from_linked_context_count",
    "period_recoverable_from_linked_context_count",
    "filing_recoverable_from_linked_context_count",
    "section_recoverable_from_linked_context_count",
]
FINANCE_RETRIEVAL_IMPACT_FIELDS = [
    "measurement",
    "ablation_mode",
    "record_count",
    "candidate_recall_at_20",
    "candidate_recall_at_50",
    "final_recall_at_5",
    "mrr",
    "avg_blocked_direct_hint_count",
    "rewritten_query_count",
    "dense_backend",
    "vector_store",
]

FORM_RE = re.compile(r"\b(?:10\s*[- ]?\s*k|10\s*[- ]?\s*q|8\s*[- ]?\s*k)\b", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
QUARTER_RE = re.compile(r"\b(?:q[1-4]|quarter\s+[1-4]|quarterly|annual|fiscal year|fy)\b", re.I)
DIRECT_ID_RE = re.compile(
    r"(?:finance_kb[A-Za-z0-9_:/.\-]*|finance_sec[A-Za-z0-9_:/.\-]*|sec://\S+|"
    r"xbrl://\S+|\b(?:required_doc_ids|required_evidence_ids|required_chunk_ids|"
    r"source_id|parent_id|document_id|filing_id)\b)",
    re.I,
)
SECTION_TERMS = {
    "management discussion": "management_discussion_and_analysis",
    "md&a": "management_discussion_and_analysis",
    "risk factor": "risk_factors",
    "risk factors": "risk_factors",
    "balance sheet": "balance_sheet",
    "income statement": "income_statement",
    "cash flow": "cash_flow_statement",
    "filing event": "filing_event",
    "financial statements": "financial_statements",
    "notes": "notes_to_financial_statements",
    "business": "business",
}
FORM_LABELS = {
    "10-K": "Form 10-K annual filing fiscal year",
    "10-Q": "Form 10-Q quarterly filing fiscal quarter",
    "8-K": "Form 8-K current report filing event",
}


@dataclass(frozen=True)
class FinanceEnrichment:
    """Human-readable retrieval metadata derived for one finance prompt."""

    prompt_id: str
    ticker: str | None = None
    company: str | None = None
    filing_type: str | None = None
    filing_section: str | None = None
    period: str | None = None
    fiscal_quarter: str | None = None
    fiscal_year: str | None = None
    xbrl_concept: str | None = None
    metric_family: str | None = None
    derived_from: list[str] = field(default_factory=list)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a sorted JSON object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write CSV rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def normalize_form(value: str | None) -> str | None:
    """Normalize a filing form string."""

    if not value:
        return None
    normalized = value.upper().replace(" ", "-")
    normalized = normalized.replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
    match = FORM_RE.search(normalized)
    if not match:
        return None
    raw = match.group(0).upper().replace(" ", "").replace("-", "")
    if raw == "10K":
        return "10-K"
    if raw == "10Q":
        return "10-Q"
    if raw == "8K":
        return "8-K"
    return None


def visible_prompt_text(prompt: dict[str, Any], *, include_metadata: bool = True) -> str:
    """Return prompt text and public prompt metadata used for quality audit."""

    parts = [
        str(prompt.get("question") or ""),
        str(prompt.get("issue") or ""),
        str(prompt.get("company") or ""),
        str(prompt.get("ticker") or ""),
        str(prompt.get("filing_form") or ""),
        str(prompt.get("task_type") or ""),
    ]
    if include_metadata:
        metadata = prompt.get("metadata")
        if isinstance(metadata, dict):
            for field_name in ("prompt_category", "evidence_type", "required_section_types"):
                value = metadata.get(field_name)
                if isinstance(value, list):
                    parts.extend(str(item) for item in value if item)
                elif value:
                    parts.append(str(value))
    return " ".join(part for part in parts if part)


def gold_text(gold_record: dict[str, Any]) -> str:
    """Return non-ID gold/eval text for explicit recoverability checks."""

    parts = [
        str(gold_record.get("reference_answer") or ""),
        str(gold_record.get("task_type") or ""),
        str(gold_record.get("expected_status") or ""),
    ]
    for field_name in ("must_include", "must_not_include"):
        value = gold_record.get(field_name)
        if isinstance(value, list):
            for item in value:
                item_text = str(item)
                scrubbed, _ = scrub_direct_evidence_identifiers(item_text)
                if scrubbed:
                    parts.append(scrubbed)
    metadata = gold_record.get("metadata")
    if isinstance(metadata, dict):
        for field_name in ("source_subject", "prompt_category", "expected_output_format"):
            value = metadata.get(field_name)
            if value:
                parts.append(str(value))
    scrubbed_text, _ = scrub_direct_evidence_identifiers(" ".join(parts))
    return scrubbed_text


def detect_period(text: str) -> str | None:
    """Detect a year/quarter/period phrase from text."""

    years = sorted(set(YEAR_RE.findall(text)))
    quarters = sorted({match.group(0).lower() for match in QUARTER_RE.finditer(text)})
    if years and quarters:
        return f"{' '.join(quarters)} {' '.join(years)}"
    if years:
        return " ".join(years)
    if quarters:
        return " ".join(quarters)
    return None


def detect_section(text: str) -> str | None:
    """Detect a human-readable filing section from text."""

    normalized = text.lower()
    for phrase, section in SECTION_TERMS.items():
        if phrase in normalized:
            return section
    return None


def detect_metric_family(text: str) -> str | None:
    """Detect a finance metric family from text or XBRL-style concept names."""

    normalized = " ".join(tokenize(split_identifier_text(text))).lower()
    for metric_family, synonyms in FINANCE_METRIC_SYNONYMS.items():
        triggers = {metric_family, *synonyms}
        if any(trigger in normalized for trigger in triggers):
            return metric_family
    return None


def context_records_by_match_id(records: list[ContextRecord]) -> dict[str, list[ContextRecord]]:
    """Return context records keyed by every gold-compatible match ID."""

    by_match_id: dict[str, list[ContextRecord]] = defaultdict(list)
    for record in records:
        for match_id in context_match_ids(record):
            by_match_id[match_id].append(record)
    return by_match_id


def linked_context_records(
    gold_record: dict[str, Any] | None,
    by_match_id: dict[str, list[ContextRecord]],
) -> list[ContextRecord]:
    """Return deduplicated context records linked to a gold/eval row."""

    if gold_record is None:
        return []
    linked: list[ContextRecord] = []
    seen: set[str] = set()
    for gold_id in gold_ids_from_gold_record(gold_record):
        for variant in {gold_id, normalize_identifier(gold_id)}:
            for record in by_match_id.get(variant, []):
                if record.context_id not in seen:
                    linked.append(record)
                    seen.add(record.context_id)
    return linked


def first_non_empty(values: list[Any]) -> str | None:
    """Return the first non-empty string-like value."""

    for value in values:
        if value:
            text = str(value).strip()
            if text and text.lower() != "none":
                return text
    return None


def context_metadata_values(records: list[ContextRecord], field_name: str) -> list[str]:
    """Return metadata values from linked context records."""

    values: list[str] = []
    for record in records:
        value = record.metadata.get(field_name)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return values


def fiscal_quarter_from_date(value: str | None) -> str | None:
    """Infer a rough fiscal quarter label from a YYYY-MM-DD date."""

    if not value:
        return None
    match = re.match(r"(\d{4})-(\d{2})-\d{2}", value)
    if not match:
        return None
    month = int(match.group(2))
    quarter = ((month - 1) // 3) + 1
    return f"Q{quarter}"


def fiscal_year_from_date(value: str | None) -> str | None:
    """Extract a fiscal year from a date-like value."""

    if not value:
        return None
    match = YEAR_RE.search(value)
    return match.group(0) if match else None


def audit_finance_prompts(prompts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit whether Finance prompts expose entity, metric, period, filing, and section cues."""

    prompt_rows: list[dict[str, Any]] = []
    by_task_type: dict[str, Counter[str]] = defaultdict(Counter)
    for prompt in prompts:
        text = visible_prompt_text(prompt)
        form = normalize_form(str(prompt.get("filing_form") or "")) or normalize_form(text)
        section = detect_section(text)
        row = {
            "prompt_id": str(prompt.get("prompt_id") or ""),
            "task_type": str(prompt.get("task_type") or ""),
            "company_or_ticker_present": finance_prompt_has_entity(prompt),
            "metric_present": finance_prompt_has_metric(prompt),
            "period_present": finance_prompt_has_period(prompt),
            "filing_type_present": form is not None,
            "section_present": section is not None,
            "detected_filing_type": form,
            "detected_section": section,
        }
        prompt_rows.append(row)
        task_counter = by_task_type[str(row["task_type"])]
        for field_name in (
            "company_or_ticker_present",
            "metric_present",
            "period_present",
            "filing_type_present",
            "section_present",
        ):
            task_counter[f"{field_name}_count"] += int(bool(row[field_name]))
            task_counter[f"{field_name}_missing_count"] += int(not bool(row[field_name]))

    total = len(prompt_rows)
    summary = {
        "total_prompts": total,
        "company_or_ticker_present_count": sum(
            int(row["company_or_ticker_present"]) for row in prompt_rows
        ),
        "company_or_ticker_missing_count": sum(
            int(not row["company_or_ticker_present"]) for row in prompt_rows
        ),
        "metric_present_count": sum(int(row["metric_present"]) for row in prompt_rows),
        "metric_missing_count": sum(int(not row["metric_present"]) for row in prompt_rows),
        "period_present_count": sum(int(row["period_present"]) for row in prompt_rows),
        "period_missing_count": sum(int(not row["period_present"]) for row in prompt_rows),
        "filing_type_present_count": sum(int(row["filing_type_present"]) for row in prompt_rows),
        "filing_type_missing_count": sum(
            int(not row["filing_type_present"]) for row in prompt_rows
        ),
        "section_present_count": sum(int(row["section_present"]) for row in prompt_rows),
        "section_missing_count": sum(int(not row["section_present"]) for row in prompt_rows),
    }
    report = {
        "generated_at_utc": utc_now(),
        "scope": FINANCE_REPAIR_SCOPE,
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "total_prompts": total,
        "summary": summary,
        "by_task_type": {
            task_type: dict(counter) for task_type, counter in sorted(by_task_type.items())
        },
        "prompt_rows": prompt_rows,
    }
    return report, summary


def build_linked_recoverability(
    linked_records: list[ContextRecord],
) -> dict[str, str | None]:
    """Return metric/period/filing/section values recoverable from linked context."""

    form = first_non_empty(context_metadata_values(linked_records, "form"))
    section = first_non_empty(context_metadata_values(linked_records, "section_type"))
    concept = first_non_empty(context_metadata_values(linked_records, "concept"))
    report_date = first_non_empty(context_metadata_values(linked_records, "report_date"))
    filing_date = first_non_empty(context_metadata_values(linked_records, "filing_date"))
    fiscal_year = first_non_empty(context_metadata_values(linked_records, "fiscal_year"))
    period = first_non_empty([fiscal_year, report_date, filing_date])
    metric_family = detect_metric_family(
        " ".join([concept or "", *[r.text for r in linked_records]])
    )
    return {
        "metric_family": metric_family,
        "period": period,
        "filing_type": normalize_form(form),
        "filing_section": section,
        "xbrl_concept": concept,
    }


def audit_finance_gold(
    gold_by_prompt_id: dict[str, dict[str, Any]],
    by_match_id: dict[str, list[ContextRecord]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit whether Finance gold/eval records expose recoverable retrieval metadata."""

    gold_rows: list[dict[str, Any]] = []
    for prompt_id, gold_record in sorted(gold_by_prompt_id.items()):
        text = gold_text(gold_record)
        linked = linked_context_records(gold_record, by_match_id)
        linked_recoverability = build_linked_recoverability(linked)
        row = {
            "prompt_id": prompt_id,
            "metric_explicitly_recoverable": detect_metric_family(text) is not None,
            "period_explicitly_recoverable": detect_period(text) is not None,
            "filing_explicitly_recoverable": normalize_form(text) is not None,
            "section_explicitly_recoverable": detect_section(text) is not None,
            "metric_recoverable_from_linked_context": linked_recoverability["metric_family"]
            is not None,
            "period_recoverable_from_linked_context": linked_recoverability["period"] is not None,
            "filing_recoverable_from_linked_context": linked_recoverability["filing_type"]
            is not None,
            "section_recoverable_from_linked_context": linked_recoverability["filing_section"]
            is not None,
            "linked_context_count": len(linked),
        }
        gold_rows.append(row)

    total = len(gold_rows)
    summary = {
        "total_gold_records": total,
        "metric_explicitly_recoverable_count": sum(
            int(row["metric_explicitly_recoverable"]) for row in gold_rows
        ),
        "metric_explicitly_missing_count": sum(
            int(not row["metric_explicitly_recoverable"]) for row in gold_rows
        ),
        "period_explicitly_recoverable_count": sum(
            int(row["period_explicitly_recoverable"]) for row in gold_rows
        ),
        "period_explicitly_missing_count": sum(
            int(not row["period_explicitly_recoverable"]) for row in gold_rows
        ),
        "filing_explicitly_recoverable_count": sum(
            int(row["filing_explicitly_recoverable"]) for row in gold_rows
        ),
        "filing_explicitly_missing_count": sum(
            int(not row["filing_explicitly_recoverable"]) for row in gold_rows
        ),
        "section_explicitly_recoverable_count": sum(
            int(row["section_explicitly_recoverable"]) for row in gold_rows
        ),
        "section_explicitly_missing_count": sum(
            int(not row["section_explicitly_recoverable"]) for row in gold_rows
        ),
        "metric_recoverable_from_linked_context_count": sum(
            int(row["metric_recoverable_from_linked_context"]) for row in gold_rows
        ),
        "period_recoverable_from_linked_context_count": sum(
            int(row["period_recoverable_from_linked_context"]) for row in gold_rows
        ),
        "filing_recoverable_from_linked_context_count": sum(
            int(row["filing_recoverable_from_linked_context"]) for row in gold_rows
        ),
        "section_recoverable_from_linked_context_count": sum(
            int(row["section_recoverable_from_linked_context"]) for row in gold_rows
        ),
    }
    report = {
        "generated_at_utc": utc_now(),
        "scope": FINANCE_REPAIR_SCOPE,
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "total_gold_records": total,
        "summary": summary,
        "gold_rows": gold_rows,
    }
    return report, summary


def derive_finance_enrichment(
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    linked_records: list[ContextRecord],
) -> FinanceEnrichment:
    """Derive human-readable finance metadata for one prompt."""

    prompt_text_value = visible_prompt_text(prompt)
    gold_text_value = gold_text(gold_record) if gold_record else ""
    derived_from: list[str] = []

    ticker = first_non_empty(
        [prompt.get("ticker"), *context_metadata_values(linked_records, "ticker")]
    )
    if ticker:
        derived_from.append("prompt_or_context_ticker")
    company = first_non_empty(
        [
            prompt.get("company"),
            *context_metadata_values(linked_records, "company_name"),
            *context_metadata_values(linked_records, "company"),
        ]
    )
    if company:
        derived_from.append("prompt_or_context_company")

    form = (
        normalize_form(str(prompt.get("filing_form") or ""))
        or normalize_form(prompt_text_value)
        or normalize_form(gold_text_value)
        or normalize_form(first_non_empty(context_metadata_values(linked_records, "form")))
    )
    if form:
        derived_from.append("prompt_gold_or_context_form")

    section = (
        detect_section(prompt_text_value)
        or detect_section(gold_text_value)
        or first_non_empty(context_metadata_values(linked_records, "section_type"))
    )
    if section:
        derived_from.append("prompt_gold_or_context_section")

    report_date = first_non_empty(context_metadata_values(linked_records, "report_date"))
    filing_date = first_non_empty(context_metadata_values(linked_records, "filing_date"))
    fiscal_year = (
        first_non_empty(context_metadata_values(linked_records, "fiscal_year"))
        or fiscal_year_from_date(report_date)
        or fiscal_year_from_date(filing_date)
    )
    fiscal_quarter = first_non_empty(context_metadata_values(linked_records, "fiscal_quarter"))
    if not fiscal_quarter:
        fiscal_quarter = fiscal_quarter_from_date(report_date) or fiscal_quarter_from_date(
            filing_date
        )
    period = detect_period(prompt_text_value) or detect_period(gold_text_value)
    if not period and fiscal_year:
        period = f"fiscal year {fiscal_year}"
    if not period and report_date:
        period = f"report date {report_date}"
    if period:
        derived_from.append("prompt_gold_or_context_period")

    concept = first_non_empty(
        [
            *context_metadata_values(linked_records, "concept"),
            *context_metadata_values(linked_records, "concepts"),
        ]
    )
    metric_family = (
        detect_metric_family(prompt_text_value)
        or detect_metric_family(gold_text_value)
        or detect_metric_family(
            " ".join([concept or "", *[record.text for record in linked_records]])
        )
    )
    if concept:
        derived_from.append("context_xbrl_concept")
    if metric_family:
        derived_from.append("prompt_gold_or_context_metric")

    return FinanceEnrichment(
        prompt_id=str(prompt.get("prompt_id") or ""),
        ticker=ticker,
        company=company,
        filing_type=form,
        filing_section=section,
        period=period,
        fiscal_quarter=fiscal_quarter,
        fiscal_year=fiscal_year,
        xbrl_concept=concept,
        metric_family=metric_family,
        derived_from=list(dict.fromkeys(derived_from)),
    )


def humanize_identifier(value: str | None) -> str:
    """Return a readable form of an identifier-like string."""

    if not value:
        return ""
    return " ".join(tokenize(split_identifier_text(value)))


def metric_terms_for_family(metric_family: str | None) -> list[str]:
    """Return deterministic metric terms for a metric family."""

    if not metric_family:
        return []
    terms = {metric_family, *FINANCE_METRIC_SYNONYMS.get(metric_family, set())}
    return sorted(term for term in terms if term)


def rewritten_retrieval_query(
    prompt: dict[str, Any],
    enrichment: FinanceEnrichment,
    *,
    ablation_mode: str,
    resolver: CompanyTickerResolver | None = None,
    concept_map: dict[str, set[str]] | None = None,
) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
    """Build a human-readable repaired retrieval query without direct evidence IDs."""

    raw_query = prompt_query_text(
        prompt,
        ablation_mode,
        company_ticker_resolver=resolver,
        xbrl_concept_map=concept_map,
    )
    parts = [raw_query.query_text]
    if enrichment.ticker:
        parts.append(enrichment.ticker)
    if enrichment.company:
        parts.append(enrichment.company)
    if enrichment.filing_type:
        parts.append(FORM_LABELS.get(enrichment.filing_type, enrichment.filing_type))
    if enrichment.filing_section:
        parts.append(f"filing section {humanize_identifier(enrichment.filing_section)}")
    if enrichment.period:
        parts.append(enrichment.period)
    if enrichment.fiscal_year:
        parts.append(f"fiscal year {enrichment.fiscal_year}")
    if enrichment.fiscal_quarter:
        parts.append(f"fiscal quarter {enrichment.fiscal_quarter}")
    if enrichment.metric_family:
        parts.extend(metric_terms_for_family(enrichment.metric_family))
    if enrichment.xbrl_concept:
        parts.append(humanize_identifier(enrichment.xbrl_concept))
    candidate_query = " ".join(part for part in parts if part)
    scrubbed_query, repair_blocked_count = scrub_direct_evidence_identifiers(candidate_query)
    enriched = enrich_query_text(
        scrubbed_query,
        vertical="finance",
        allow_direct_identifiers=False,
        resolver=resolver,
        concept_map=concept_map,
        metadata_terms=set(tokenize(scrubbed_query)),
    )
    return (
        enriched.query_text,
        enriched.expanded_queries,
        enriched.expansion_types,
        raw_query.blocked_direct_hint_count
        + repair_blocked_count
        + enriched.blocked_direct_hint_count,
    )


def build_finance_metadata_enrichment_report(
    prompts: list[dict[str, Any]],
    gold_by_prompt_id: dict[str, dict[str, Any]],
    by_match_id: dict[str, list[ContextRecord]],
) -> tuple[dict[str, Any], dict[str, FinanceEnrichment]]:
    """Build Finance metadata enrichment records and coverage report."""

    enrichments: dict[str, FinanceEnrichment] = {}
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt.get("prompt_id") or "")
        gold_record = gold_by_prompt_id.get(prompt_id)
        linked = linked_context_records(gold_record, by_match_id)
        enrichment = derive_finance_enrichment(prompt, gold_record, linked)
        enrichments[prompt_id] = enrichment
        rows.append(
            {
                "prompt_id": prompt_id,
                "ticker": enrichment.ticker,
                "company": enrichment.company,
                "filing_type": enrichment.filing_type,
                "filing_section": enrichment.filing_section,
                "period": enrichment.period,
                "fiscal_quarter": enrichment.fiscal_quarter,
                "fiscal_year": enrichment.fiscal_year,
                "xbrl_concept": enrichment.xbrl_concept,
                "metric_family": enrichment.metric_family,
                "derived_from": enrichment.derived_from,
                "linked_context_count": len(linked),
            }
        )
    summary = {
        "total_prompts": len(rows),
        "ticker_enriched_count": sum(int(bool(row["ticker"])) for row in rows),
        "company_enriched_count": sum(int(bool(row["company"])) for row in rows),
        "filing_type_enriched_count": sum(int(bool(row["filing_type"])) for row in rows),
        "filing_section_enriched_count": sum(int(bool(row["filing_section"])) for row in rows),
        "period_enriched_count": sum(int(bool(row["period"])) for row in rows),
        "fiscal_year_enriched_count": sum(int(bool(row["fiscal_year"])) for row in rows),
        "xbrl_concept_enriched_count": sum(int(bool(row["xbrl_concept"])) for row in rows),
        "metric_family_enriched_count": sum(int(bool(row["metric_family"])) for row in rows),
        "rows_with_linked_context_count": sum(
            int(cast(int, row["linked_context_count"]) > 0) for row in rows
        ),
    }
    report = {
        "generated_at_utc": utc_now(),
        "scope": FINANCE_REPAIR_SCOPE,
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "summary": summary,
        "enrichment_rows": rows,
        "leakage_policy": {
            "direct_evidence_ids_allowed": False,
            "raw_gold_evidence_ids_in_rewritten_queries": False,
            "human_readable_linked_metadata_used_for_repair_measurement": True,
        },
    }
    return report, enrichments


def candidate_results_from_ids(
    context_ids: list[str],
    records_by_context_id: dict[str, ContextRecord],
    retrieval_mode: str,
) -> list[RetrievalResult]:
    """Rebuild candidate result objects from diagnostic context IDs."""

    return [
        RetrievalResult(
            context_record=records_by_context_id[context_id],
            score=0.0,
            rank=index,
            retrieval_mode=retrieval_mode,
            component_scores={},
        )
        for index, context_id in enumerate(context_ids, start=1)
        if context_id in records_by_context_id
    ]


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Finance retrieval repair rows."""

    if not rows:
        return {
            "record_count": 0,
            "candidate_recall_at_20": 0.0,
            "candidate_recall_at_50": 0.0,
            "final_recall_at_5": 0.0,
            "mrr": 0.0,
            "avg_blocked_direct_hint_count": 0.0,
            "rewritten_query_count": 0,
            "dense_backend": "unavailable",
            "vector_store": "none",
        }
    return {
        "record_count": len(rows),
        "candidate_recall_at_20": round(
            mean(float(row["candidate_recall_at_20"]) for row in rows), 6
        ),
        "candidate_recall_at_50": round(
            mean(float(row["candidate_recall_at_50"]) for row in rows), 6
        ),
        "final_recall_at_5": round(mean(float(row["final_recall_at_5"]) for row in rows), 6),
        "mrr": round(mean(float(row["mrr"]) for row in rows), 6),
        "avg_blocked_direct_hint_count": round(
            mean(float(row["blocked_direct_hint_count"]) for row in rows),
            6,
        ),
        "rewritten_query_count": sum(int(bool(row["query_rewritten"])) for row in rows),
        "dense_backend": ",".join(sorted({str(row["dense_backend"]) for row in rows})),
        "vector_store": ",".join(sorted({str(row["vector_store"]) for row in rows})),
    }


def measure_finance_retrieval_repair(
    *,
    prompts: list[dict[str, Any]],
    gold_by_prompt_id: dict[str, dict[str, Any]],
    finance_records: list[ContextRecord],
    enrichments: dict[str, FinanceEnrichment],
    dense_backend: str,
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = True,
    ablation_modes: tuple[str, ...] = ("prompt_text_only", "prompt_plus_metadata"),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure Finance-only before/after retrieval impact for repaired queries."""

    retrievers = build_retrievers(
        {"finance": finance_records},
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    try:
        finance_retrievers = retrievers["finance"]
        resolver = cast(
            CompanyTickerResolver | None, finance_retrievers.get("company_ticker_resolver")
        )
        concept_map = cast(dict[str, set[str]], finance_retrievers.get("xbrl_concept_map") or {})
        records_by_context_id = cast(
            dict[str, ContextRecord], finance_retrievers["records_by_context_id"]
        )
        all_rows: list[dict[str, Any]] = []
        retrieval_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        for ablation_mode in ablation_modes:
            for prompt in prompts:
                prompt_id = str(prompt.get("prompt_id") or "")
                gold_record = gold_by_prompt_id.get(prompt_id)
                gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
                baseline_query = prompt_query_text(
                    prompt,
                    ablation_mode,
                    company_ticker_resolver=resolver,
                    xbrl_concept_map=concept_map,
                )
                enrichment = enrichments[prompt_id]
                repaired_query, expanded_queries, expansion_types, blocked_count = (
                    rewritten_retrieval_query(
                        prompt,
                        enrichment,
                        ablation_mode=ablation_mode,
                        resolver=resolver,
                        concept_map=concept_map,
                    )
                )
                for measurement, query_text, active_expanded, active_types, blocked in (
                    (
                        "before",
                        baseline_query.query_text,
                        baseline_query.expanded_queries,
                        baseline_query.expansion_types,
                        baseline_query.blocked_direct_hint_count,
                    ),
                    (
                        "after_metadata_repair",
                        repaired_query,
                        expanded_queries,
                        expansion_types,
                        blocked_count,
                    ),
                ):
                    retrieval = retrieve_for_mode(
                        memory_mode="mm2_hybrid_top5",
                        query=query_text,
                        expanded_queries=active_expanded,
                        expansion_types=active_types,
                        source_hints_used=False,
                        vertical="finance",
                        retrievers=retrievers,
                        top_k=DEFAULT_FINAL_TOP_K,
                        final_top_k=DEFAULT_FINAL_TOP_K,
                        retrieval_cache=retrieval_cache,
                    )
                    evaluation = evaluate_retrieval_results(
                        gold_evidence_ids=gold_ids,
                        results=retrieval.results,
                    )
                    candidate_ids = [
                        str(context_id)
                        for context_id in retrieval.diagnostics.get("candidate_context_ids", [])
                    ]
                    candidate_results = candidate_results_from_ids(
                        candidate_ids,
                        records_by_context_id,
                        retrieval.retrieval_type,
                    )
                    query_rewritten = (
                        measurement == "after_metadata_repair"
                        and normalize_identifier(repaired_query)
                        != normalize_identifier(baseline_query.query_text)
                    )
                    all_rows.append(
                        {
                            "measurement": measurement,
                            "ablation_mode": ablation_mode,
                            "prompt_id": prompt_id,
                            "gold_evidence_id_count": len(gold_ids),
                            "candidate_recall_at_20": recall_at_candidate_k(
                                gold_ids=gold_ids,
                                candidate_results=candidate_results,
                                top_k=20,
                            ),
                            "candidate_recall_at_50": recall_at_candidate_k(
                                gold_ids=gold_ids,
                                candidate_results=candidate_results,
                                top_k=50,
                            ),
                            "final_recall_at_5": evaluation["recall_at_5"],
                            "mrr": evaluation["mrr"],
                            "blocked_direct_hint_count": blocked,
                            "query_rewritten": query_rewritten,
                            "direct_id_leakage_detected": DIRECT_ID_RE.search(query_text)
                            is not None,
                            "dense_backend": retrieval.backend_label,
                            "vector_store": retrieval.vector_store,
                            "retrieval_latency_ms": round(retrieval.latency_ms, 6),
                            "candidate_count": len(candidate_results),
                            "matched_gold_evidence_ids": evaluation["matched_gold_evidence_ids"],
                            "selected_context_ids": [
                                result.context_record.context_id for result in retrieval.results
                            ],
                        }
                    )

        summary_rows: list[dict[str, Any]] = []
        for (measurement, ablation_mode), grouped_rows in sorted(
            group_rows(all_rows, "measurement", "ablation_mode").items()
        ):
            summary = metrics_for_rows(grouped_rows)
            summary_rows.append(
                {
                    "measurement": measurement,
                    "ablation_mode": ablation_mode,
                    **summary,
                }
            )
        impact_by_ablation: dict[str, dict[str, Any]] = {}
        for ablation_mode in ablation_modes:
            before = metrics_for_rows(
                [
                    row
                    for row in all_rows
                    if row["ablation_mode"] == ablation_mode and row["measurement"] == "before"
                ]
            )
            after = metrics_for_rows(
                [
                    row
                    for row in all_rows
                    if row["ablation_mode"] == ablation_mode
                    and row["measurement"] == "after_metadata_repair"
                ]
            )
            impact_by_ablation[ablation_mode] = {
                "before": before,
                "after_metadata_repair": after,
                "delta": {
                    "candidate_recall_at_20": round(
                        float(after["candidate_recall_at_20"])
                        - float(before["candidate_recall_at_20"]),
                        6,
                    ),
                    "candidate_recall_at_50": round(
                        float(after["candidate_recall_at_50"])
                        - float(before["candidate_recall_at_50"]),
                        6,
                    ),
                    "final_recall_at_5": round(
                        float(after["final_recall_at_5"]) - float(before["final_recall_at_5"]),
                        6,
                    ),
                    "mrr": round(float(after["mrr"]) - float(before["mrr"]), 6),
                },
            }
        report = {
            "generated_at_utc": utc_now(),
            "scope": FINANCE_REPAIR_SCOPE,
            "measurement_scope": "finance_only_mm2_hybrid_top5_final_10000",
            "no_model_inference_triggered": True,
            "no_gpu_work_triggered": True,
            "no_external_api_calls_triggered": True,
            "dense_backend_requested": dense_backend,
            "impact_by_ablation": impact_by_ablation,
            "leakage_guard": {
                "direct_id_leakage_detected_count": sum(
                    int(bool(row["direct_id_leakage_detected"])) for row in all_rows
                ),
                "source_hints_used": False,
                "gold_ids_added_to_query": False,
            },
            "sample_repaired_rows": [
                row
                for row in all_rows
                if row["measurement"] == "after_metadata_repair" and row["query_rewritten"]
            ][:25],
        }
        return report, summary_rows
    finally:
        close_retrievers(retrievers)


def group_rows(
    rows: list[dict[str, Any]],
    first_key: str,
    second_key: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group rows by two string keys."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[first_key]), str(row[second_key]))].append(row)
    return grouped


def build_finance_retrieval_repair(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    output_root: str | Path,
    dense_backend: str = "qdrant_vector",
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = True,
) -> dict[str, Any]:
    """Build Finance prompt/gold repair reports and write them to disk."""

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    finance_prompts = prompts_by_vertical["finance"]
    finance_gold = gold_by_vertical["finance"]
    finance_records = corpora_by_vertical["finance"]
    by_match_id = context_records_by_match_id(finance_records)

    prompt_report, prompt_summary = audit_finance_prompts(finance_prompts)
    gold_report, gold_summary = audit_finance_gold(finance_gold, by_match_id)
    enrichment_report, enrichments = build_finance_metadata_enrichment_report(
        finance_prompts,
        finance_gold,
        by_match_id,
    )
    impact_report, impact_summary_rows = measure_finance_retrieval_repair(
        prompts=finance_prompts,
        gold_by_prompt_id=finance_gold,
        finance_records=finance_records,
        enrichments=enrichments,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )

    output_path = Path(output_root)
    write_json(output_path / "finance_prompt_quality_report.json", prompt_report)
    write_csv(
        output_path / "finance_prompt_quality_summary.csv",
        [prompt_summary],
        FINANCE_PROMPT_QUALITY_FIELDS,
    )
    write_json(output_path / "finance_gold_quality_report.json", gold_report)
    write_csv(
        output_path / "finance_gold_quality_summary.csv",
        [gold_summary],
        FINANCE_GOLD_QUALITY_FIELDS,
    )
    write_json(
        output_path / "finance_metadata_enrichment_report.json",
        enrichment_report,
    )
    write_json(output_path / "finance_retrieval_repair_impact_report.json", impact_report)
    write_csv(
        output_path / "finance_retrieval_repair_impact_summary.csv",
        impact_summary_rows,
        FINANCE_RETRIEVAL_IMPACT_FIELDS,
    )

    aggregate_report = {
        "generated_at_utc": utc_now(),
        "scope": FINANCE_REPAIR_SCOPE,
        "prompt_quality_summary": prompt_summary,
        "gold_quality_summary": gold_summary,
        "metadata_enrichment_summary": enrichment_report["summary"],
        "retrieval_impact": impact_report["impact_by_ablation"],
        "output_files": {
            "finance_prompt_quality_report": str(
                output_path / "finance_prompt_quality_report.json"
            ),
            "finance_prompt_quality_summary": str(
                output_path / "finance_prompt_quality_summary.csv"
            ),
            "finance_gold_quality_report": str(output_path / "finance_gold_quality_report.json"),
            "finance_gold_quality_summary": str(output_path / "finance_gold_quality_summary.csv"),
            "finance_metadata_enrichment_report": str(
                output_path / "finance_metadata_enrichment_report.json"
            ),
            "finance_retrieval_repair_impact_report": str(
                output_path / "finance_retrieval_repair_impact_report.json"
            ),
            "finance_retrieval_repair_impact_summary": str(
                output_path / "finance_retrieval_repair_impact_summary.csv"
            ),
        },
    }
    write_json(output_path / "finance_retrieval_repair_report.json", aggregate_report)
    return aggregate_report
