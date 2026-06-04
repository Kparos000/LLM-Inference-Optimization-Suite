"""All-vertical retrieval SLO repair audit and staged validation.

This module adds vertical-aware prompt metadata enrichment and staged retrieval
validation without running inference, GPU work, or external API calls. Gold IDs
are used only for offline recall measurement, never as retrieval query terms.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, TypeAlias, cast

from inference_bench.context_schema import ContextRecord
from inference_bench.finance_retrieval_repair import (
    detect_metric_family,
    detect_period,
    detect_section,
    normalize_form,
)
from inference_bench.gold_evidence_audit import gold_ids_from_gold_record
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
    CompanyTickerResolver,
    RetrievalResult,
    context_match_ids,
    enrich_query_text,
    evaluate_retrieval_results,
    normalize_identifier,
    scrub_direct_evidence_identifiers,
    tokenize,
)
from inference_bench.slo import SLO_VERTICALS, load_slo_config

VALIDATION_FIELDS = [
    "vertical",
    "stage_size",
    "ablation_mode",
    "measurement",
    "dense_backend",
    "vector_store",
    "candidate_recall_at_20",
    "candidate_recall_at_50",
    "final_recall_at_5",
    "mrr",
    "slo_status",
    "primary_blocker",
    "recommended_next_action",
    "record_count",
    "query_rewrite_count",
]
DEFAULT_ABLATION_MODE = "prompt_plus_metadata"
PreparedQuery: TypeAlias = tuple[str, tuple[str, ...], tuple[str, ...], int]
DIRECT_HINT_RE = re.compile(
    r"(?:\b(?:required_doc_ids|required_evidence_ids|required_chunk_ids|source_id|parent_id|"
    r"document_id|filing_id|gold_evidence_ids)\b|"
    r"\b(?:finance|retail|research_ai|airline|healthcare_admin)_(?:kb|doc|section|policy|"
    r"review|summary|chunk|text|corpus)[A-Za-z0-9_:/.\-]*|"
    r"\b(?:CA-POL|MCH-POL)-[A-Za-z0-9-]+\b|sec://\S+|xbrl://\S+)",
    re.I,
)
ROUTE_RE = re.compile(r"\b[A-Z]{3}\s*-\s*[A-Z]{3}\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PAPER_ID_RE = re.compile(r"\bresearch_ai_[a-z0-9_]+", re.I)


@dataclass(frozen=True)
class VerticalRepairProfile:
    """Vertical-specific audit and enrichment profile."""

    vertical: str
    audit_fields: tuple[str, ...]
    enrichment_fields: tuple[str, ...]
    synonym_groups: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class EnrichmentResult:
    """Non-ID retrieval metadata derived from prompt-visible fields."""

    vertical: str
    prompt_id: str
    fields: dict[str, str | bool | list[str] | None] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    blocked_direct_hint_count: int = 0


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def repair_profiles() -> dict[str, VerticalRepairProfile]:
    """Return all vertical retrieval repair profiles."""

    return {
        "airline": VerticalRepairProfile(
            vertical="airline",
            audit_fields=(
                "policy_type_present",
                "travel_issue_present",
                "customer_scenario_present",
                "gold_policy_section_recoverable",
                "chunk_width_label",
            ),
            enrichment_fields=(
                "policy_type",
                "travel_issue",
                "route_region",
                "escalation_type",
            ),
            synonym_groups={
                "refund": (
                    "refund",
                    "reimbursement",
                    "fare credit",
                    "travel credit",
                    "refund processing",
                    "payment method",
                ),
                "cancellation": (
                    "cancellation",
                    "cancel",
                    "24 hour",
                    "non refundable",
                    "travel credit",
                ),
                "baggage": (
                    "baggage",
                    "bag",
                    "checked bag",
                    "carry on",
                    "claim",
                    "damaged bag",
                    "delayed bag",
                ),
                "delay": (
                    "delay",
                    "disruption",
                    "rebooking",
                    "missed connection",
                    "schedule change",
                    "weather disruption",
                    "compensation eligibility",
                ),
                "accessibility": (
                    "accessibility",
                    "assistance",
                    "mobility",
                    "accommodation",
                    "wheelchair",
                ),
                "identity": (
                    "identity",
                    "verification",
                    "fraud",
                    "document",
                    "payment dispute",
                    "manual account review",
                ),
                "codeshare": (
                    "codeshare",
                    "partner airline",
                    "marketing carrier",
                    "operating carrier",
                    "ticketing carrier",
                    "irregular operations",
                ),
                "visa": (
                    "visa",
                    "passport",
                    "documentation",
                    "international travel",
                    "entry documents",
                ),
            },
        ),
        "healthcare_admin": VerticalRepairProfile(
            vertical="healthcare_admin",
            audit_fields=(
                "admin_task_type_present",
                "admin_issue_present",
                "safety_boundary_present",
                "gold_procedure_recoverable",
                "chunk_width_label",
            ),
            enrichment_fields=(
                "admin_task_type",
                "safety_boundary",
                "privacy_sensitive",
                "policy_type",
            ),
            synonym_groups={
                "appointment": (
                    "appointment",
                    "booking",
                    "scheduling",
                    "visit",
                    "reschedule",
                    "identity verification",
                ),
                "billing": (
                    "billing",
                    "invoice",
                    "payment",
                    "claim",
                    "insurance",
                    "account specific",
                ),
                "referral": ("referral", "authorization", "specialist", "routing", "status"),
                "identity": (
                    "identity",
                    "verification",
                    "patient matching",
                    "secure workflow",
                    "account activation",
                ),
                "privacy": (
                    "privacy",
                    "hipaa",
                    "secure message",
                    "consent",
                    "authorization",
                    "proxy access",
                    "disclosure",
                ),
                "clinical_boundary": (
                    "clinical",
                    "triage",
                    "diagnosis",
                    "treatment",
                    "urgent clinical redirect",
                ),
            },
        ),
        "retail": VerticalRepairProfile(
            vertical="retail",
            audit_fields=(
                "product_title_or_category_present",
                "review_issue_present",
                "support_policy_present",
                "parent_child_link_recoverable",
                "chunk_width_label",
            ),
            enrichment_fields=(
                "product_category",
                "product_title_terms",
                "review_issue_type",
                "sentiment_signal",
                "policy_type",
            ),
            synonym_groups={
                "return": ("return", "refund", "exchange", "eligibility"),
                "defect": ("defect", "broken", "damaged", "quality issue"),
                "sentiment": ("sentiment", "rating", "review", "positive", "negative"),
                "shipping": ("shipping", "delivery", "late", "package"),
                "authenticity": ("authentic", "counterfeit", "fake"),
            },
        ),
        "finance": VerticalRepairProfile(
            vertical="finance",
            audit_fields=(
                "company_or_ticker_present",
                "metric_present",
                "period_present",
                "filing_type_present",
                "section_or_xbrl_present",
            ),
            enrichment_fields=(
                "company",
                "ticker",
                "metric_family",
                "period",
                "fiscal_year",
                "fiscal_quarter",
                "filing_type",
                "section_type",
                "xbrl_concept",
            ),
            synonym_groups={
                "revenue": ("revenue", "sales", "net sales", "total revenue"),
                "income": ("income", "earnings", "profit", "operating income"),
                "cash_flow": ("cash flow", "operating cash flow", "free cash flow"),
                "capex": ("capex", "capital expenditures", "property equipment"),
                "risk": ("risk", "risk factors", "uncertainty"),
            },
        ),
        "research_ai": VerticalRepairProfile(
            vertical="research_ai",
            audit_fields=(
                "paper_title_or_topic_present",
                "section_type_present",
                "method_result_limitation_cue_present",
                "gold_paper_section_recoverable",
                "chunk_width_label",
            ),
            enrichment_fields=(
                "paper_id_public",
                "topic",
                "section_type",
                "method_signal",
                "result_signal",
                "limitation_signal",
            ),
            synonym_groups={
                "method": ("method", "architecture", "algorithm", "approach"),
                "results": ("results", "experiments", "evaluation", "benchmark"),
                "limitations": ("limitations", "failure cases", "future work"),
                "inference": ("inference", "serving", "decoding", "generation"),
                "rag": ("rag", "retrieval augmented generation", "retrieval"),
            },
        ),
    }


def prompt_text(prompt: dict[str, Any]) -> str:
    """Return prompt-visible text and public metadata excluding direct source IDs."""

    parts = [
        str(prompt.get("question") or ""),
        str(prompt.get("issue") or ""),
        str(prompt.get("support_type") or ""),
        str(prompt.get("department") or ""),
        str(prompt.get("category") or ""),
        str(prompt.get("product_title") or ""),
        str(prompt.get("issue_type") or ""),
        str(prompt.get("topic") or ""),
        str(prompt.get("company") or ""),
        str(prompt.get("ticker") or ""),
        str(prompt.get("filing_form") or ""),
        str(prompt.get("task_type") or ""),
        str(prompt.get("expected_action") or ""),
    ]
    metadata = prompt.get("metadata")
    if isinstance(metadata, dict):
        for field_name in (
            "prompt_category",
            "evidence_type",
            "topics",
            "category",
            "difficulty",
        ):
            value = metadata.get(field_name)
            if isinstance(value, list):
                parts.extend(str(item) for item in value if item)
            elif value:
                parts.append(str(value))
    scrubbed, _ = scrub_direct_evidence_identifiers(" ".join(parts))
    return scrubbed


def contains_any(text: str, terms: tuple[str, ...] | list[str] | set[str]) -> bool:
    """Return whether any term appears in text."""

    normalized = text.lower()
    return any(term.lower() in normalized for term in terms)


def sanitize_query_terms(terms: list[str]) -> tuple[list[str], int]:
    """Remove direct evidence/source hints from query terms."""

    clean_terms: list[str] = []
    blocked_count = 0
    for term in terms:
        scrubbed, blocked = scrub_direct_evidence_identifiers(str(term))
        blocked_count += blocked
        scrubbed = DIRECT_HINT_RE.sub(" ", scrubbed)
        scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
        if scrubbed and scrubbed.lower() not in {value.lower() for value in clean_terms}:
            clean_terms.append(scrubbed)
    return clean_terms, blocked_count


def linked_context_records(
    gold_record: dict[str, Any] | None,
    records_by_match_id: dict[str, list[ContextRecord]],
) -> list[ContextRecord]:
    """Return context records linked to a gold row for audit/measurement only."""

    if gold_record is None:
        return []
    seen: set[str] = set()
    linked: list[ContextRecord] = []
    for gold_id in gold_ids_from_gold_record(gold_record):
        for match_id in (gold_id, normalize_identifier(gold_id)):
            for record in records_by_match_id.get(match_id, []):
                if record.context_id not in seen:
                    linked.append(record)
                    seen.add(record.context_id)
    return linked


def context_records_by_match_id(records: list[ContextRecord]) -> dict[str, list[ContextRecord]]:
    """Index context records by gold-compatible match IDs."""

    indexed: dict[str, list[ContextRecord]] = defaultdict(list)
    for record in records:
        for match_id in context_match_ids(record):
            indexed[match_id].append(record)
    return indexed


def chunk_width_label(records: list[ContextRecord]) -> str:
    """Classify linked context chunk width for diagnosis."""

    if not records:
        return "not_recoverable"
    avg_tokens = mean(record.token_estimate for record in records)
    if avg_tokens > 900:
        return "chunk_too_broad"
    if avg_tokens < 12:
        return "chunk_too_narrow"
    return "reasonable"


def audit_prompt(
    *,
    vertical: str,
    prompt: dict[str, Any],
    gold_record: dict[str, Any] | None,
    linked_records: list[ContextRecord],
) -> dict[str, Any]:
    """Build one vertical-specific audit row."""

    text = prompt_text(prompt)
    metadata = prompt.get("metadata") if isinstance(prompt.get("metadata"), dict) else {}
    metadata = cast(dict[str, Any], metadata)
    linked_metadata_text = " ".join(
        " ".join(str(value) for value in record.metadata.values()) for record in linked_records
    )
    row: dict[str, Any] = {
        "vertical": vertical,
        "prompt_id": str(prompt.get("prompt_id") or ""),
        "chunk_width_label": chunk_width_label(linked_records),
    }
    if vertical == "airline":
        issue_terms: tuple[str, ...] = (
            "refund",
            "baggage",
            "delay",
            "cancellation",
            "accessibility",
            "booking",
        )
        row.update(
            {
                "policy_type_present": bool(prompt.get("support_type"))
                or contains_any(text, issue_terms),
                "travel_issue_present": contains_any(text, issue_terms),
                "customer_scenario_present": "scenario" in text.lower()
                or bool(prompt.get("ticket_id")),
                "gold_policy_section_recoverable": bool(linked_records)
                and contains_any(linked_metadata_text, ("policy", "CA-POL")),
            }
        )
    elif vertical == "healthcare_admin":
        admin_terms = ("appointment", "privacy", "identity", "billing", "referral", "scheduling")
        safety_terms = ("clinical", "diagnosis", "treatment", "privacy", "identity", "safety")
        row.update(
            {
                "admin_task_type_present": bool(prompt.get("support_type"))
                or bool(prompt.get("department")),
                "admin_issue_present": contains_any(text, admin_terms),
                "safety_boundary_present": bool(prompt.get("safety_boundary"))
                or contains_any(text, safety_terms),
                "gold_procedure_recoverable": bool(linked_records)
                and contains_any(linked_metadata_text, ("MCH-POL", "policy", "procedure")),
            }
        )
    elif vertical == "retail":
        issue_terms = ("return", "refund", "review", "rating", "defect", "broken", "sentiment")
        row.update(
            {
                "product_title_or_category_present": bool(prompt.get("product_title"))
                or bool(prompt.get("category")),
                "review_issue_present": bool(prompt.get("issue_type"))
                or contains_any(text, issue_terms),
                "support_policy_present": contains_any(text, ("return", "refund", "policy")),
                "parent_child_link_recoverable": bool(linked_records)
                and contains_any(
                    linked_metadata_text,
                    ("parent_asin", "product_title", "review", "summary"),
                ),
            }
        )
    elif vertical == "finance":
        row.update(
            {
                "company_or_ticker_present": bool(prompt.get("ticker"))
                or bool(prompt.get("company")),
                "metric_present": detect_metric_family(text) is not None,
                "period_present": detect_period(text) is not None,
                "filing_type_present": normalize_form(str(prompt.get("filing_form") or ""))
                is not None,
                "section_or_xbrl_present": detect_section(text) is not None
                or bool(metadata.get("required_section_types")),
            }
        )
    elif vertical == "research_ai":
        source_titles = metadata.get("source_titles")
        evidence_type = metadata.get("evidence_type")
        section_terms = (
            "abstract",
            "introduction",
            "method",
            "experiments",
            "results",
            "limitations",
        )
        row.update(
            {
                "paper_title_or_topic_present": bool(prompt.get("topic")) or bool(source_titles),
                "section_type_present": contains_any(text, section_terms) or bool(evidence_type),
                "method_result_limitation_cue_present": contains_any(
                    text,
                    ("method", "experiment", "result", "limitation", "evaluation"),
                ),
                "gold_paper_section_recoverable": bool(linked_records)
                and contains_any(linked_metadata_text, ("paper", "section", "abstract")),
            }
        )
    return row


def enrich_prompt_metadata(
    *,
    vertical: str,
    prompt: dict[str, Any],
) -> EnrichmentResult:
    """Derive non-ID vertical retrieval metadata from prompt-visible fields."""

    text = prompt_text(prompt)
    metadata = prompt.get("metadata") if isinstance(prompt.get("metadata"), dict) else {}
    metadata = cast(dict[str, Any], metadata)
    fields: dict[str, str | bool | list[str] | None] = {}
    terms: list[str] = []

    if vertical == "airline":
        issue = first_matching_key(
            text,
            {
                "refund": ("refund", "reimbursement"),
                "cancellation": ("cancel", "cancellation"),
                "baggage": ("baggage", "bag"),
                "delay": ("delay", "disruption"),
                "accessibility": ("accessibility", "assistance"),
                "identity": ("identity", "verification", "fraud"),
            },
        )
        route = ROUTE_RE.search(str(prompt.get("route") or text))
        fields = {
            "policy_type": str(prompt.get("support_type") or metadata.get("prompt_category") or ""),
            "travel_issue": issue,
            "route_region": route.group(0) if route else None,
            "escalation_type": "fraud_or_identity"
            if contains_any(text, ("fraud", "identity", "verification", "escalation"))
            else None,
        }
    elif vertical == "healthcare_admin":
        boundary = str(prompt.get("safety_boundary") or "")
        fields = {
            "admin_task_type": str(
                prompt.get("support_type") or metadata.get("prompt_category") or ""
            ),
            "safety_boundary": boundary
            or (
                "clinical_boundary"
                if contains_any(text, ("clinical", "diagnosis", "treatment"))
                else ""
            ),
            "privacy_sensitive": bool(prompt.get("privacy_sensitive"))
            or contains_any(text, ("privacy", "identity", "consent")),
            "policy_type": str(prompt.get("department") or prompt.get("expected_queue") or ""),
        }
    elif vertical == "retail":
        fields = {
            "product_category": str(prompt.get("category") or metadata.get("category") or ""),
            "product_title_terms": title_terms(str(prompt.get("product_title") or "")),
            "review_issue_type": str(
                prompt.get("issue_type") or metadata.get("prompt_category") or ""
            ),
            "sentiment_signal": first_matching_key(
                text,
                {
                    "negative": ("negative", "bad", "broken", "defect", "refund"),
                    "positive": ("positive", "good", "great", "five star"),
                    "mixed": ("mixed", "summary", "compare"),
                },
            ),
            "policy_type": "return_refund"
            if contains_any(text, ("return", "refund", "exchange"))
            else None,
        }
    elif vertical == "finance":
        concept = detect_metric_family(text)
        filing_type = normalize_form(str(prompt.get("filing_form") or ""))
        fields = {
            "company": str(prompt.get("company") or ""),
            "ticker": str(prompt.get("ticker") or ""),
            "metric_family": concept,
            "period": detect_period(text),
            "fiscal_year": first_year(text),
            "fiscal_quarter": first_quarter(text),
            "filing_type": filing_type,
            "section_type": detect_section(text),
            "xbrl_concept": None,
        }
    elif vertical == "research_ai":
        source_titles = metadata.get("source_titles")
        title = source_titles[0] if isinstance(source_titles, list) and source_titles else ""
        evidence_type = metadata.get("evidence_type")
        section_type = (
            evidence_type[0]
            if isinstance(evidence_type, list) and evidence_type
            else str(evidence_type or "")
        )
        fields = {
            "paper_id_public": public_paper_label(str(title)),
            "topic": str(prompt.get("topic") or " ".join(metadata.get("topics", [])) or ""),
            "section_type": section_type,
            "method_signal": contains_any(text, ("method", "approach", "architecture")),
            "result_signal": contains_any(text, ("result", "experiment", "evaluation")),
            "limitation_signal": contains_any(text, ("limitation", "failure", "future work")),
        }

    for value in fields.values():
        if isinstance(value, list):
            terms.extend(value)
        elif isinstance(value, bool):
            continue
        elif value:
            terms.append(str(value))
    profile = repair_profiles()[vertical]
    for field_value in fields.values():
        if isinstance(field_value, str):
            for key, synonyms in profile.synonym_groups.items():
                if key in field_value.lower() or contains_any(field_value, synonyms):
                    terms.extend(synonyms)
    if vertical == "finance" and fields.get("metric_family"):
        terms.extend(profile.synonym_groups.get(str(fields["metric_family"]), ()))
    if vertical == "finance":
        terms.extend(finance_materialized_terms(fields, str(metadata.get("evidence_type") or "")))
    if vertical == "airline":
        terms.extend(airline_materialized_terms(prompt, fields))
    if vertical == "healthcare_admin":
        terms.extend(healthcare_materialized_terms(prompt, fields))

    clean_terms, blocked_count = sanitize_query_terms(terms)
    missing = [field_name for field_name in profile.enrichment_fields if not fields.get(field_name)]
    return EnrichmentResult(
        vertical=vertical,
        prompt_id=str(prompt.get("prompt_id") or ""),
        fields=fields,
        missing_fields=missing,
        query_terms=clean_terms,
        blocked_direct_hint_count=blocked_count,
    )


def finance_materialized_terms(
    fields: dict[str, str | bool | list[str] | None],
    evidence_type: str,
) -> list[str]:
    """Return non-ID Finance expansion terms from prompt-visible metadata."""

    terms: list[str] = []
    company = str(fields.get("company") or "")
    ticker = str(fields.get("ticker") or "")
    filing_type = str(fields.get("filing_type") or "")
    if company:
        terms.append(company)
        terms.extend(tokenize(company))
    if ticker:
        terms.append(ticker)
    if filing_type == "10-K":
        terms.extend(["10-K", "annual report", "annual filing", "fiscal year"])
    elif filing_type == "10-Q":
        terms.extend(["10-Q", "quarterly report", "quarterly filing", "fiscal quarter"])
    elif filing_type == "8-K":
        terms.extend(["8-K", "current report", "filing event", "material event"])
    if "xbrl" in evidence_type.lower():
        terms.extend(
            [
                "xbrl",
                "financial statement facts",
                "revenue",
                "net sales",
                "income",
                "cash flow",
                "assets",
                "liabilities",
                "fiscal period",
            ]
        )
    if filing_type in {"10-K", "10-Q"}:
        terms.extend(
            [
                "management discussion and analysis",
                "financial statements",
                "risk factors",
                "results of operations",
            ]
        )
    return terms


def airline_materialized_terms(
    prompt: dict[str, Any],
    fields: dict[str, str | bool | list[str] | None],
) -> list[str]:
    """Return non-ID Airline expansion terms from prompt-visible metadata."""

    support_type = str(fields.get("policy_type") or prompt.get("support_type") or "").lower()
    travel_issue = str(fields.get("travel_issue") or "").lower()
    route = str(fields.get("route_region") or prompt.get("route") or "")
    travel_type = str(prompt.get("travel_type") or "")
    partner_involved = bool(prompt.get("partner_airline_involved"))
    normalized = f"{support_type} {travel_issue}"
    terms: list[str] = []
    airline_map: dict[str, tuple[str, ...]] = {
        "ticket_purchase": (
            "ticket purchase",
            "booking",
            "reservation",
            "fare rules",
            "fare ownership",
            "payment verification",
            "24 hour cancellation",
            "refundable fare",
        ),
        "cancellation_refund": (
            "cancellation",
            "refund",
            "24 hour cancellation",
            "non refundable",
            "travel credit",
            "refund processing",
            "payment method",
            "timeline",
        ),
        "ticket_change": (
            "ticket change",
            "same day change",
            "fare difference",
            "route restrictions",
            "payment verification",
            "account review",
        ),
        "disruption": (
            "disruption",
            "delay",
            "rebooking",
            "schedule change",
            "weather disruption",
            "compensation eligibility",
            "arrival delay",
            "notice period",
            "manual review",
        ),
        "missed_flight": (
            "missed flight",
            "missed connection",
            "protected itinerary",
            "rebooking",
            "separate ticket",
            "manual review",
        ),
        "baggage_delay": (
            "baggage delay",
            "delayed bag",
            "checked bag",
            "bag delivery",
            "interim expense",
        ),
        "baggage_damage": (
            "baggage damage",
            "damaged bag",
            "checked bag",
            "damage report",
            "claim",
        ),
        "codeshare": (
            "codeshare",
            "marketing carrier",
            "operating carrier",
            "ticketing carrier",
            "partner airline",
            "irregular operations",
            "escalation",
            "compensation edge cases",
        ),
        "partner_airline": (
            "partner airline",
            "operating carrier",
            "validating carrier",
            "ticketing carrier",
            "manual review",
            "irregular operations",
        ),
        "visa_passport": (
            "visa",
            "passport",
            "documentation",
            "entry documents",
            "transit documents",
            "international travel",
            "admissibility",
            "official government guidance",
        ),
        "accessibility": (
            "accessibility",
            "assistance",
            "mobility",
            "accommodation",
            "wheelchair",
            "accessible travel",
        ),
        "fraud_or_chargeback": (
            "fraud",
            "chargeback",
            "payment dispute",
            "identity verification",
            "manual account review",
            "account access",
        ),
        "loyalty": (
            "loyalty",
            "points",
            "miles",
            "membership",
            "account access",
            "identity verification",
        ),
    }
    for key, values in airline_map.items():
        if key in normalized or key.replace("_", " ") in normalized:
            terms.extend(values)
    if route:
        terms.extend(["route", "itinerary"])
    if travel_type:
        terms.extend([travel_type, "travel type", "itinerary"])
    if partner_involved:
        terms.extend(["partner airline", "operating carrier", "manual review"])
    return terms


def healthcare_materialized_terms(
    prompt: dict[str, Any],
    fields: dict[str, str | bool | list[str] | None],
) -> list[str]:
    """Return non-ID Healthcare Admin expansion terms from prompt-visible metadata."""

    support_type = str(fields.get("admin_task_type") or prompt.get("support_type") or "").lower()
    department = str(fields.get("policy_type") or prompt.get("department") or "").lower()
    boundary = str(fields.get("safety_boundary") or prompt.get("safety_boundary") or "").lower()
    normalized = f"{support_type} {department} {boundary}"
    terms: list[str] = []
    healthcare_map: dict[str, tuple[str, ...]] = {
        "appointment_booking": (
            "appointment booking",
            "scheduling",
            "visit reason",
            "preferred clinic",
            "date range",
            "identity verification",
            "administrative channel",
        ),
        "appointment_reschedule": (
            "appointment reschedule",
            "rescheduling",
            "scheduling",
            "appointment context",
            "identity verification",
            "approved administrative channel",
        ),
        "appointment_cancellation": (
            "appointment cancellation",
            "cancel visit",
            "no show",
            "scheduling",
            "late cancellation",
        ),
        "billing_question": (
            "billing",
            "invoice",
            "payment",
            "claim",
            "insurance",
            "identity verification",
            "account specific",
        ),
        "payment_plan_request": (
            "payment plan",
            "billing",
            "payment",
            "financial assistance",
            "identity verification",
        ),
        "insurance_verification": (
            "insurance verification",
            "coverage",
            "eligibility",
            "payer",
            "identity verification",
        ),
        "prior_authorization_status": (
            "prior authorization",
            "authorization",
            "insurance",
            "status",
            "payer",
            "specialist",
        ),
        "referral_status": (
            "referral",
            "specialist",
            "authorization",
            "routing",
            "status",
        ),
        "medical_records_request": (
            "medical records",
            "records release",
            "authorization",
            "identity verification",
            "privacy review",
            "timeline",
            "receiving party",
        ),
        "portal_access": (
            "portal access",
            "login recovery",
            "account activation",
            "secure message",
            "identity verification",
        ),
        "privacy_request": (
            "privacy",
            "proxy access",
            "disclosure",
            "authorization",
            "privacy office",
            "identity verification",
        ),
        "new_patient_registration": (
            "new patient registration",
            "intake",
            "identity verification",
            "insurance workflow",
            "consent forms",
        ),
        "lab_result_availability": (
            "lab result",
            "result availability",
            "portal",
            "clinical boundary",
            "secure message",
        ),
        "prescription_refill_routing": (
            "prescription refill",
            "refill routing",
            "clinical staff review",
            "medication",
            "safety boundary",
        ),
        "clinic_location_hours": ("clinic location", "hours", "directions", "administrative"),
        "telehealth_setup": ("telehealth", "video visit", "portal", "setup", "technical support"),
        "complaint_or_grievance": (
            "complaint",
            "grievance",
            "patient relations",
            "escalation",
            "manual review",
        ),
        "transportation_or_accessibility_request": (
            "transportation",
            "accessibility",
            "accommodation",
            "administrative support",
        ),
    }
    for key, values in healthcare_map.items():
        if key in normalized or key.replace("_", " ") in normalized:
            terms.extend(values)
    if fields.get("privacy_sensitive"):
        terms.extend(["privacy", "consent", "authorization", "secure workflow"])
    if "administrative" in boundary:
        terms.extend(["administrative", "non clinical", "policy", "procedure"])
    if "privacy" in boundary:
        terms.extend(["privacy", "authorization", "disclosure", "privacy office"])
    if "urgent" in boundary or "clinical" in boundary:
        terms.extend(["urgent clinical redirect", "clinical boundary", "triage", "safety"])
    return terms


def first_matching_key(text: str, mapping: dict[str, tuple[str, ...]]) -> str | None:
    """Return the first key whose terms appear in text."""

    for key, terms in mapping.items():
        if contains_any(text, terms):
            return key
    return None


def title_terms(title: str, max_terms: int = 8) -> list[str]:
    """Return useful title terms without direct identifiers."""

    stopwords = {"the", "and", "for", "with", "case", "pack", "set"}
    return [term for term in tokenize(title) if term not in stopwords][:max_terms]


def first_year(text: str) -> str | None:
    """Return the first year in text."""

    match = YEAR_RE.search(text)
    return match.group(0) if match else None


def first_quarter(text: str) -> str | None:
    """Return the first quarter in text."""

    match = re.search(r"\b(?:q([1-4])|quarter\s+([1-4]))\b", text, re.I)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return f"Q{value}"


def public_paper_label(title: str) -> str | None:
    """Return a non-ID paper label from a title."""

    scrubbed = PAPER_ID_RE.sub(" ", title)
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return scrubbed or None


def repaired_query(
    *,
    prompt: dict[str, Any],
    enrichment: EnrichmentResult,
    resolver: CompanyTickerResolver | None,
    concept_map: dict[str, set[str]],
    ablation_mode: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
    """Build a repaired query under strict no-source-ID rules."""

    base = prompt_query_text(
        prompt,
        ablation_mode,
        company_ticker_resolver=resolver,
        xbrl_concept_map=concept_map,
    )
    query_text = " ".join([base.query_text, *enrichment.query_terms])
    scrubbed, blocked = scrub_direct_evidence_identifiers(query_text)
    enriched = enrich_query_text(
        scrubbed,
        vertical=str(prompt.get("vertical") or ""),
        allow_direct_identifiers=False,
        resolver=resolver,
        concept_map=concept_map,
        metadata_terms=set(tokenize(" ".join(enrichment.query_terms))),
    )
    return (
        enriched.query_text,
        enriched.expanded_queries,
        enriched.expansion_types,
        base.blocked_direct_hint_count + blocked + enriched.blocked_direct_hint_count,
    )


def candidate_results_from_ids(
    context_ids: list[str],
    records_by_context_id: dict[str, ContextRecord],
    retrieval_mode: str,
) -> list[RetrievalResult]:
    """Rebuild candidate results from diagnostic context IDs."""

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


def select_stage_prompts(prompts: list[dict[str, Any]], stage_size: int) -> list[dict[str, Any]]:
    """Return deterministic stage prompts."""

    return sorted(prompts, key=lambda row: str(row.get("prompt_id") or ""))[:stage_size]


def warm_qdrant_repair_queries(
    *,
    retrievers: dict[str, dict[str, Any]],
    queries_by_vertical: dict[str, set[str]],
    top_k: int,
) -> dict[str, int]:
    """Warm local Qdrant query embeddings/searches when Qdrant is active."""

    qdrant_searchers = retrievers.get("_qdrant_searchers", {})
    warmed: dict[str, int] = {}
    for vertical, queries in queries_by_vertical.items():
        searcher = qdrant_searchers.get(vertical)
        if searcher is None:
            continue
        cast(Any, searcher).warm_snapshot_search_results(sorted(queries), top_k=top_k)
        warmed[vertical] = len(queries)
    return warmed


def aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    slo_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate per-prompt staged rows into report rows."""

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["vertical"]),
                int(row["stage_size"]),
                str(row["ablation_mode"]),
                str(row["measurement"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (vertical, stage_size, ablation_mode, measurement), group in sorted(grouped.items()):
        metrics = {
            "candidate_recall_at_20": round(
                mean(float(row["candidate_recall_at_20"]) for row in group),
                6,
            ),
            "candidate_recall_at_50": round(
                mean(float(row["candidate_recall_at_50"]) for row in group),
                6,
            ),
            "final_recall_at_5": round(
                mean(float(row["final_recall_at_5"]) for row in group),
                6,
            ),
            "mrr": round(mean(float(row["mrr"]) for row in group), 6),
        }
        status, blocker, action = slo_status_for_metrics(
            vertical=vertical,
            metrics=metrics,
            slo_config=slo_config,
        )
        summary_rows.append(
            {
                "vertical": vertical,
                "stage_size": stage_size,
                "ablation_mode": ablation_mode,
                "measurement": measurement,
                "dense_backend": ",".join(sorted({str(row["dense_backend"]) for row in group})),
                "vector_store": ",".join(sorted({str(row["vector_store"]) for row in group})),
                **metrics,
                "slo_status": status,
                "primary_blocker": blocker,
                "recommended_next_action": action,
                "record_count": len(group),
                "query_rewrite_count": sum(int(bool(row["query_rewritten"])) for row in group),
            }
        )
    return summary_rows


def slo_status_for_metrics(
    *,
    vertical: str,
    metrics: dict[str, float],
    slo_config: dict[str, Any],
) -> tuple[str, str, str]:
    """Return aggregate retrieval SLO status, blocker, and action."""

    retrieval_slo = cast(
        dict[str, Any],
        cast(dict[str, Any], cast(dict[str, Any], slo_config["verticals"])[vertical])[
            "retrieval_slo"
        ],
    )
    checks = [
        ("candidate_recall_at_20", "candidate_recall_at_20_min", "candidate_retrieval_top20"),
        ("candidate_recall_at_50", "candidate_recall_at_50_min", "candidate_retrieval_top50"),
        ("final_recall_at_5", "final_recall_at_5_min", "final_top5_selection"),
        ("mrr", "mrr_min", "rank_ordering"),
    ]
    for observed_name, target_name, blocker in checks:
        if float(metrics[observed_name]) < float(retrieval_slo[target_name]):
            return (
                "FAILED",
                blocker,
                recommended_action_for_blocker(vertical, blocker),
            )
    return "PASSED", "none", "Proceed to the next staged validation or inference smoke gate."


def recommended_action_for_blocker(vertical: str, blocker: str) -> str:
    """Return a vertical-aware repair recommendation."""

    if vertical == "finance":
        return (
            "Materialize non-ID period/metric/filing metadata and rerun Qdrant strict validation."
        )
    if blocker.startswith("candidate"):
        return f"Improve {vertical} query enrichment, chunk metadata, and candidate recall."
    if blocker == "final_top5_selection":
        return f"Tune {vertical} final top-5 evidence selection and reranking."
    if blocker == "rank_ordering":
        return f"Improve {vertical} rank ordering and tie-breaking."
    return f"Inspect {vertical} retrieval failures before inference scaling."


def build_audit_report(
    *,
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
) -> tuple[dict[str, Any], dict[str, dict[str, EnrichmentResult]]]:
    """Build all-vertical audit and metadata enrichment report."""

    profiles = repair_profiles()
    audit_rows: list[dict[str, Any]] = []
    enrichment_rows: list[dict[str, Any]] = []
    enrichments: dict[str, dict[str, EnrichmentResult]] = {
        vertical: {} for vertical in SLO_VERTICALS
    }
    by_match_id = {
        vertical: context_records_by_match_id(corpora_by_vertical[vertical])
        for vertical in SLO_VERTICALS
    }
    for vertical in SLO_VERTICALS:
        for prompt in prompts_by_vertical[vertical]:
            prompt_id = str(prompt.get("prompt_id") or "")
            gold_record = gold_by_vertical[vertical].get(prompt_id)
            linked = linked_context_records(gold_record, by_match_id[vertical])
            audit_rows.append(
                audit_prompt(
                    vertical=vertical,
                    prompt=prompt,
                    gold_record=gold_record,
                    linked_records=linked,
                )
            )
            enrichment = enrich_prompt_metadata(vertical=vertical, prompt=prompt)
            enrichments[vertical][prompt_id] = enrichment
            enrichment_rows.append(
                {
                    "vertical": vertical,
                    "prompt_id": prompt_id,
                    "fields": enrichment.fields,
                    "missing_fields": enrichment.missing_fields,
                    "query_terms": enrichment.query_terms,
                    "blocked_direct_hint_count": enrichment.blocked_direct_hint_count,
                }
            )
    summary = {
        vertical: summarize_audit_rows(
            [row for row in audit_rows if row["vertical"] == vertical],
            profiles[vertical],
        )
        for vertical in SLO_VERTICALS
    }
    report = {
        "generated_at_utc": utc_now(),
        "scope": "all_vertical_retrieval_repair_no_inference_no_gpu_no_api",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "profiles": {vertical: asdict(profile) for vertical, profile in profiles.items()},
        "audit_summary": summary,
        "audit_rows_sample": audit_rows[:100],
        "enrichment_rows_sample": enrichment_rows[:100],
        "leakage_policy": {
            "gold_ids_used_as_query_terms": False,
            "source_ids_used_as_query_terms": False,
            "direct_hint_regex": DIRECT_HINT_RE.pattern,
        },
    }
    return report, enrichments


def summarize_audit_rows(
    rows: list[dict[str, Any]],
    profile: VerticalRepairProfile,
) -> dict[str, Any]:
    """Summarize audit rows for one vertical."""

    summary: dict[str, Any] = {"record_count": len(rows)}
    for field_name in profile.audit_fields:
        if field_name == "chunk_width_label":
            summary["chunk_width_counts"] = dict(
                Counter(str(row.get(field_name) or "unknown") for row in rows)
            )
            continue
        summary[f"{field_name}_count"] = sum(int(bool(row.get(field_name))) for row in rows)
        summary[f"{field_name}_missing_count"] = sum(
            int(not bool(row.get(field_name))) for row in rows
        )
    return summary


def validate_stages(
    *,
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
    enrichments: dict[str, dict[str, EnrichmentResult]],
    slo_config: dict[str, Any],
    stage_sizes: list[int],
    dense_backend: str,
    vector_store_config_path: str | Path,
    vector_store_key: str,
    allow_dense_fallback: bool,
    ablation_mode: str = DEFAULT_ABLATION_MODE,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run staged retrieval validation with repaired queries."""

    retrievers = build_retrievers(
        corpora_by_vertical,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    try:
        query_cache: dict[tuple[str, str, str, tuple[str, ...], int], Any] = {}
        queries_to_warm: dict[str, set[str]] = {vertical: set() for vertical in SLO_VERTICALS}
        prepared: list[tuple[str, int, dict[str, Any], PreparedQuery]] = []
        for vertical in SLO_VERTICALS:
            vertical_retrievers = retrievers[vertical]
            resolver = cast(
                CompanyTickerResolver | None,
                vertical_retrievers.get("company_ticker_resolver"),
            )
            concept_map = cast(
                dict[str, set[str]],
                vertical_retrievers.get("xbrl_concept_map") or {},
            )
            max_stage_size = max(stage_sizes)
            for prompt in select_stage_prompts(prompts_by_vertical[vertical], max_stage_size):
                enrichment = enrichments[vertical][str(prompt.get("prompt_id") or "")]
                query_tuple = repaired_query(
                    prompt=prompt,
                    enrichment=enrichment,
                    resolver=resolver,
                    concept_map=concept_map,
                    ablation_mode=ablation_mode,
                )
                queries_to_warm[vertical].add(query_tuple[0])
                prepared.append((vertical, max_stage_size, prompt, query_tuple))

        warmed = warm_qdrant_repair_queries(
            retrievers=retrievers,
            queries_by_vertical=queries_to_warm,
            top_k=50,
        )
        prepared_by_vertical = defaultdict(list)
        for vertical, _max_stage_size, prompt, query_tuple in prepared:
            prepared_by_vertical[vertical].append((prompt, query_tuple))

        for stage_size in stage_sizes:
            for vertical in SLO_VERTICALS:
                vertical_retrievers = retrievers[vertical]
                records_by_context_id = cast(
                    dict[str, ContextRecord],
                    vertical_retrievers["records_by_context_id"],
                )
                for prompt, query_tuple in prepared_by_vertical[vertical][:stage_size]:
                    prompt_id = str(prompt.get("prompt_id") or "")
                    gold_record = gold_by_vertical[vertical].get(prompt_id)
                    gold_ids = gold_ids_from_gold_record(gold_record) if gold_record else []
                    query_text, expanded_queries, expansion_types, blocked_count = query_tuple
                    retrieval = retrieve_for_mode(
                        memory_mode="mm2_hybrid_top5",
                        query=query_text,
                        expanded_queries=expanded_queries,
                        expansion_types=expansion_types,
                        source_hints_used=False,
                        vertical=vertical,
                        retrievers=retrievers,
                        top_k=DEFAULT_FINAL_TOP_K,
                        final_top_k=DEFAULT_FINAL_TOP_K,
                        retrieval_cache=query_cache,
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
                    evaluation = evaluate_retrieval_results(
                        gold_evidence_ids=gold_ids,
                        results=retrieval.results,
                    )
                    row = {
                        "vertical": vertical,
                        "stage_size": stage_size,
                        "ablation_mode": ablation_mode,
                        "measurement": "after_repair_staged",
                        "prompt_id": prompt_id,
                        "dense_backend": retrieval.backend_label,
                        "vector_store": retrieval.vector_store,
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
                        "query_rewritten": True,
                        "blocked_direct_hint_count": blocked_count,
                        "direct_hint_leakage_detected": DIRECT_HINT_RE.search(query_text)
                        is not None,
                    }
                    rows.append(row)
                    if (
                        float(row["final_recall_at_5"]) < 1.0
                        and len([item for item in examples if item["vertical"] == vertical]) < 20
                    ):
                        examples.append(
                            {
                                "vertical": vertical,
                                "stage_size": stage_size,
                                "prompt_id": prompt_id,
                                "ablation_mode": ablation_mode,
                                "dense_backend": retrieval.backend_label,
                                "candidate_recall_at_50": row["candidate_recall_at_50"],
                                "final_recall_at_5": row["final_recall_at_5"],
                                "mrr": row["mrr"],
                                "missing_enrichment_fields": enrichments[vertical][
                                    prompt_id
                                ].missing_fields,
                                "recommended_next_action": recommended_action_for_blocker(
                                    vertical,
                                    "final_top5_selection",
                                ),
                            }
                        )
        summary_rows = aggregate_rows(rows, slo_config=slo_config)
        report = {
            "generated_at_utc": utc_now(),
            "scope": "all_vertical_retrieval_repair_no_inference_no_gpu_no_api",
            "no_model_inference_triggered": True,
            "no_gpu_work_triggered": True,
            "no_external_api_calls_triggered": True,
            "dense_backend_requested": dense_backend,
            "qdrant_warmed_query_counts": warmed,
            "stage_sizes": stage_sizes,
            "ablation_mode": ablation_mode,
            "summary_rows": summary_rows,
            "stage_timed_out": False,
            "direct_hint_leakage_detected_count": sum(
                int(bool(row["direct_hint_leakage_detected"])) for row in rows
            ),
        }
        return report, summary_rows, examples
    finally:
        close_retrievers(retrievers)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write JSON to disk."""

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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write JSONL rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
    output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return output_path


def build_all_vertical_retrieval_repair(
    *,
    dataset_root: str | Path,
    context_root: str | Path,
    slo_config_path: str | Path,
    output_root: str | Path,
    stage_sizes: list[int],
    dense_backend: str = "qdrant_vector",
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
    allow_dense_fallback: bool = True,
) -> dict[str, Any]:
    """Build all-vertical retrieval repair reports and write them to disk."""

    prompts_by_vertical, gold_by_vertical = load_prompts_and_gold(dataset_root)
    corpora_by_vertical = load_context_corpora(context_root)
    slo_config = load_slo_config(slo_config_path)
    audit_report, enrichments = build_audit_report(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
    )
    validation_report, summary_rows, examples = validate_stages(
        prompts_by_vertical=prompts_by_vertical,
        gold_by_vertical=gold_by_vertical,
        corpora_by_vertical=corpora_by_vertical,
        enrichments=enrichments,
        slo_config=slo_config,
        stage_sizes=stage_sizes,
        dense_backend=dense_backend,
        vector_store_config_path=vector_store_config_path,
        vector_store_key=vector_store_key,
        allow_dense_fallback=allow_dense_fallback,
    )
    output_path = Path(output_root)
    combined_report = {
        **validation_report,
        "audit_summary": audit_report["audit_summary"],
        "profiles": audit_report["profiles"],
    }
    write_json(output_path / "all_vertical_retrieval_repair_report.json", combined_report)
    write_csv(
        output_path / "all_vertical_retrieval_repair_summary.csv",
        summary_rows,
        VALIDATION_FIELDS,
    )
    write_jsonl(output_path / "all_vertical_retrieval_repair_examples.jsonl", examples)
    return combined_report
