"""Canonical non-leaking retrieval key extraction.

The keys in this module are derived from prompt-visible fields and realistic
metadata only. They intentionally exclude gold evidence identifiers, source IDs,
and answer-side hints so retrieval validation can stay honest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from inference_bench.finance_retrieval_repair import (
    detect_metric_family,
    detect_period,
    detect_section,
    normalize_form,
)
from inference_bench.retrieval import (
    normalize_identifier,
    scrub_direct_evidence_identifiers,
    tokenize,
)

FORBIDDEN_KEY_NAMES = {
    "gold_evidence_ids",
    "required_evidence_ids",
    "required_doc_ids",
    "required_chunk_ids",
    "required_policy_ids",
    "source_id",
    "source_ids",
    "source_parent_asins",
    "source_product_ids",
    "parent_id",
    "document_id",
    "filing_id",
}
DIRECT_ID_RE = re.compile(
    r"\b(?:CA-POL|MCH-POL)-[A-Za-z0-9-]+\b|"
    r"\b(?:finance|retail|research_ai|airline|healthcare_admin)_(?:kb|doc|section|"
    r"policy|review|summary|chunk|text|corpus)[A-Za-z0-9_:/.\-]*|"
    r"sec://\S+|xbrl://\S+",
    re.I,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
QUARTER_RE = re.compile(r"\b(?:q([1-4])|quarter\s+([1-4]))\b", re.I)
TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)|\b[A-Z]{2,5}\b")


@dataclass(frozen=True)
class RetrievalKeys:
    """Canonical retrieval keys for a single prompt."""

    vertical: str
    prompt_id: str
    values: dict[str, str | list[str] | bool | None] = field(default_factory=dict)
    blocked_direct_hint_count: int = 0

    def term_values(self) -> list[str]:
        """Return key values as query-safe text terms."""

        terms: list[str] = []
        for key, value in self.values.items():
            if key in FORBIDDEN_KEY_NAMES or value is None or value is False:
                continue
            if isinstance(value, list):
                terms.extend(str(item) for item in value if item)
            elif value is True:
                terms.append(key)
            else:
                terms.append(str(value))
        scrubbed, _blocked = scrub_direct_evidence_identifiers(" ".join(terms))
        return [term for term in split_safe_terms(scrubbed) if term]


def split_safe_terms(text: str) -> list[str]:
    """Split cleaned terms while keeping compact multi-token phrases useful."""

    terms: list[str] = []
    for value in re.split(r"[|;,]+", text):
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned or DIRECT_ID_RE.search(cleaned):
            continue
        terms.append(cleaned)
    return terms


def safe_value(value: Any) -> str | None:
    """Return a scrubbed string value or None."""

    if value is None:
        return None
    scrubbed, _blocked = scrub_direct_evidence_identifiers(str(value))
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    if not scrubbed or DIRECT_ID_RE.search(scrubbed):
        return None
    return scrubbed


def prompt_text(prompt: dict[str, Any]) -> str:
    """Return visible prompt text."""

    return " ".join(
        str(prompt.get(field) or "")
        for field in ("question", "issue", "company", "product_title", "topic")
    ).strip()


def first_year(text: str) -> str | None:
    """Return first visible year."""

    match = YEAR_RE.search(text)
    return match.group(0) if match else None


def first_quarter(text: str) -> str | None:
    """Return first visible quarter."""

    match = QUARTER_RE.search(text)
    if not match:
        return None
    return f"Q{match.group(1) or match.group(2)}"


def compact_title_terms(title: str, *, limit: int = 10) -> list[str]:
    """Return stable title terms without boilerplate."""

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "using",
        "only",
        "cited",
        "evidence",
        "answer",
        "based",
        "about",
    }
    return [token for token in tokenize(title) if token not in stopwords][:limit]


def metadata_values(prompt: dict[str, Any], key: str) -> list[str]:
    """Return prompt metadata values for a field."""

    metadata = prompt.get("metadata")
    if not isinstance(metadata, dict):
        return []
    value = metadata.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def derive_finance_keys(
    prompt: dict[str, Any],
    text: str,
) -> dict[str, str | list[str] | bool | None]:
    """Derive Finance canonical keys from prompt-visible data."""

    company = safe_value(prompt.get("company"))
    ticker = safe_value(prompt.get("ticker"))
    if not ticker:
        ticker_match = re.search(r"\(([A-Z]{1,5})\)", text)
        ticker = ticker_match.group(1) if ticker_match else None
    form = normalize_form(safe_value(prompt.get("filing_form")))
    period = detect_period(text)
    section = detect_section(text)
    metric = detect_metric_family(text)
    return {
        "company": company,
        "ticker": ticker,
        "filing_type": form,
        "filing_section": section,
        "period": period,
        "fiscal_quarter": first_quarter(text),
        "fiscal_year": first_year(text),
        "metric_family": metric,
        "xbrl_concept_family": metric,
    }


def derive_retail_keys(
    prompt: dict[str, Any],
    text: str,
) -> dict[str, str | list[str] | bool | None]:
    """Derive Retail canonical keys from prompt-visible data."""

    issue_type = safe_value(prompt.get("issue_type"))
    product_title = safe_value(prompt.get("product_title"))
    category = safe_value(prompt.get("category"))
    issue_tokens = compact_title_terms(" ".join([text, issue_type or ""]), limit=14)
    return {
        "category": category,
        "product_title": product_title,
        "product_title_terms": compact_title_terms(product_title or "", limit=12),
        "support_intent": issue_type,
        "review_issue_terms": issue_tokens,
        "policy_context": "return refund support policy" if "policy" in text.lower() else None,
    }


def derive_research_ai_keys(
    prompt: dict[str, Any],
    text: str,
) -> dict[str, str | list[str] | bool | None]:
    """Derive Research AI canonical keys from prompt-visible paper metadata."""

    source_titles = [safe_value(value) for value in metadata_values(prompt, "source_titles")]
    evidence_type = [safe_value(value) for value in metadata_values(prompt, "evidence_type")]
    topics = [safe_value(value) for value in metadata_values(prompt, "topics")]
    section_terms = [
        item
        for item in evidence_type
        if item
        and normalize_identifier(item)
        in {
            "abstract",
            "introduction",
            "method",
            "methods",
            "experiments",
            "results",
            "limitations",
            "appendix",
        }
    ]
    return {
        "topic": safe_value(prompt.get("topic")),
        "paper_title": next((value for value in source_titles if value), None),
        "paper_title_terms": compact_title_terms(" ".join(value or "" for value in source_titles)),
        "section_type": section_terms[0] if section_terms else None,
        "section_types": section_terms,
        "topic_terms": [value for value in topics if value],
        "method_signal": "method" in normalize_identifier(text),
        "results_signal": "result" in normalize_identifier(text)
        or "evaluation" in normalize_identifier(text),
    }


def derive_airline_keys(
    prompt: dict[str, Any],
    text: str,
) -> dict[str, str | list[str] | bool | None]:
    """Derive Airline canonical keys from prompt-visible support metadata."""

    support_type = safe_value(prompt.get("support_type"))
    route = safe_value(prompt.get("route"))
    normalized = normalize_identifier(text)
    issue_terms = compact_title_terms(" ".join([support_type or "", text]), limit=14)
    return {
        "support_type": support_type,
        "route": route,
        "policy_issue_terms": issue_terms,
        "booking_refund_signal": any(term in normalized for term in ("refund", "cancel")),
        "delay_disruption_signal": any(term in normalized for term in ("delay", "disruption")),
        "baggage_signal": "baggage" in normalized or "bag" in normalized,
        "escalation_signal": any(term in normalized for term in ("fraud", "chargeback", "escalat")),
    }


def derive_healthcare_keys(
    prompt: dict[str, Any],
    text: str,
) -> dict[str, str | list[str] | bool | None]:
    """Derive Healthcare Admin canonical keys from prompt-visible metadata."""

    support_type = safe_value(prompt.get("support_type"))
    department = safe_value(prompt.get("department"))
    safety_boundary = safe_value(prompt.get("safety_boundary"))
    normalized = normalize_identifier(text)
    return {
        "support_type": support_type,
        "department": department,
        "safety_boundary": safety_boundary,
        "admin_procedure_terms": compact_title_terms(" ".join([support_type or "", text])),
        "privacy_signal": "privacy" in normalized or bool(prompt.get("privacy_sensitive")),
        "identity_signal": "identity" in normalized or "verification" in normalized,
    }


def derive_retrieval_keys(
    prompt: dict[str, Any],
    *,
    ablation_mode: str = "prompt_plus_metadata",
) -> RetrievalKeys:
    """Derive canonical non-leaking retrieval keys for a prompt."""

    vertical = str(prompt.get("vertical") or "")
    prompt_id = str(prompt.get("prompt_id") or "")
    text, blocked = scrub_direct_evidence_identifiers(prompt_text(prompt))
    if ablation_mode == "prompt_text_only":
        visible_prompt = {"vertical": vertical, "prompt_id": prompt_id, "question": text}
    else:
        visible_prompt = {
            key: value for key, value in prompt.items() if key not in FORBIDDEN_KEY_NAMES
        }
    visible_text = prompt_text(visible_prompt)
    if vertical == "finance":
        values = derive_finance_keys(visible_prompt, visible_text)
    elif vertical == "retail":
        values = derive_retail_keys(visible_prompt, visible_text)
    elif vertical == "research_ai":
        values = derive_research_ai_keys(visible_prompt, visible_text)
    elif vertical == "airline":
        values = derive_airline_keys(visible_prompt, visible_text)
    elif vertical == "healthcare_admin":
        values = derive_healthcare_keys(visible_prompt, visible_text)
    else:
        values = {"prompt_terms": compact_title_terms(visible_text)}
    scrubbed_values: dict[str, str | list[str] | bool | None] = {}
    extra_blocked = 0
    for key, value in values.items():
        if key in FORBIDDEN_KEY_NAMES:
            continue
        if isinstance(value, list):
            clean_values: list[str] = []
            for item in value:
                safe, item_blocked = scrub_direct_evidence_identifiers(str(item))
                extra_blocked += item_blocked
                if safe.strip() and not DIRECT_ID_RE.search(safe):
                    clean_values.append(safe.strip())
            scrubbed_values[key] = clean_values
        elif isinstance(value, bool) or value is None:
            scrubbed_values[key] = value
        else:
            safe, item_blocked = scrub_direct_evidence_identifiers(str(value))
            extra_blocked += item_blocked
            scrubbed_values[key] = (
                safe.strip() if safe.strip() and not DIRECT_ID_RE.search(safe) else None
            )
    return RetrievalKeys(
        vertical=vertical,
        prompt_id=prompt_id,
        values=scrubbed_values,
        blocked_direct_hint_count=blocked + extra_blocked,
    )


def retrieval_key_terms(keys: RetrievalKeys) -> list[str]:
    """Return deterministic key terms for query rendering."""

    terms = keys.term_values()
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        normalized = normalize_identifier(term)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(term)
    return deduped
