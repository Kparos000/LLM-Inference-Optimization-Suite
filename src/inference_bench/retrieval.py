"""Dependency-light retrieval and compression utilities for Phase 3.

This module does not call model APIs or run inference. The dense retriever can
use a local Qdrant vector store when an index exists, and it keeps the
deterministic local fallback explicit for offline tests.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from inference_bench.context_schema import ContextRecord

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_CANDIDATE_TOP_K_DENSE = 50
DEFAULT_CANDIDATE_TOP_K_LEXICAL = 50
DEFAULT_FINAL_TOP_K = 5
FINANCE_METRIC_TERMS = {
    "revenue",
    "sales",
    "operating",
    "income",
    "net",
    "margin",
    "capex",
    "cash",
    "flow",
    "guidance",
    "risk",
    "segment",
    "r&d",
    "research",
    "development",
    "assets",
    "liabilities",
    "equity",
    "expense",
    "profit",
    "loss",
}
FINANCE_METRIC_SYNONYMS = {
    "revenue": {"revenue", "sales", "net sales", "total revenue", "contract customer revenue"},
    "sales": {"revenue", "sales", "net sales", "total revenue"},
    "operating income": {"operating income", "operating profit", "income from operations"},
    "operating profit": {"operating income", "operating profit", "income from operations"},
    "net income": {"net income", "earnings", "profit"},
    "earnings": {"net income", "earnings", "profit"},
    "margin": {"margin", "gross margin", "operating margin"},
    "gross margin": {"margin", "gross margin"},
    "operating margin": {"margin", "operating margin"},
    "capex": {
        "capex",
        "capital expenditure",
        "capital expenditures",
        "property and equipment",
    },
    "capital expenditure": {
        "capex",
        "capital expenditure",
        "capital expenditures",
        "property and equipment",
    },
    "cash flow": {
        "cash flow",
        "operating cash flow",
        "free cash flow",
        "cash provided by operations",
        "cash from operations",
    },
    "operating cash flow": {
        "cash flow",
        "operating cash flow",
        "cash provided by operations",
        "cash from operations",
    },
    "free cash flow": {"cash flow", "free cash flow"},
    "r&d": {
        "r&d",
        "research and development",
        "research development",
        "research expense",
    },
    "research and development": {
        "r&d",
        "research and development",
        "research development",
        "research expense",
    },
    "cloud": {"cloud", "azure", "aws", "google cloud"},
    "azure": {"cloud", "azure", "microsoft cloud"},
    "aws": {"cloud", "aws", "amazon web services"},
    "google cloud": {"cloud", "google cloud", "alphabet cloud"},
    "risk": {"risk", "risks", "risk factor", "risk factors"},
    "segment": {"segment", "segments", "business segment"},
    "guidance": {"guidance", "outlook", "forecast", "expectation"},
}
FINANCE_COMPANY_ALIASES = {
    "aapl": {"aapl", "apple", "apple inc"},
    "apple": {"aapl", "apple", "apple inc"},
    "msft": {"msft", "microsoft", "microsoft corporation", "azure"},
    "microsoft": {"msft", "microsoft", "microsoft corporation", "azure"},
    "nvda": {"nvda", "nvidia", "nvidia corporation"},
    "nvidia": {"nvda", "nvidia", "nvidia corporation"},
    "tsla": {"tsla", "tesla", "tesla inc"},
    "tesla": {"tsla", "tesla", "tesla inc"},
    "amzn": {"amzn", "amazon", "amazon com", "aws"},
    "amazon": {"amzn", "amazon", "amazon com", "aws"},
    "googl": {"googl", "alphabet", "google", "google cloud"},
    "alphabet": {"googl", "alphabet", "google", "google cloud"},
    "google": {"googl", "alphabet", "google", "google cloud"},
    "meta": {"meta", "facebook", "meta platforms"},
    "facebook": {"meta", "facebook", "meta platforms"},
    "amd": {"amd", "advanced micro devices"},
    "advanced": {"amd", "advanced micro devices"},
}
RETAIL_REVIEW_INTENT_TERMS = {
    "review_summary",
    "quality_complaint",
    "recommendation",
    "suspicious_review",
    "general_review_signal",
    "scent_texture",
    "fit_compatibility",
    "positive_signal",
    "catalog_metadata",
    "review",
    "reviews",
    "rating",
    "ratings",
    "complaint",
    "quality",
    "defect",
    "defective",
    "broken",
    "damaged",
    "scent",
    "smell",
    "texture",
    "fit",
    "compatibility",
    "recommend",
    "suspicious",
}
RETAIL_POLICY_INTENT_TERMS = {
    "policy_reasoning",
    "return_refund",
    "delivery_packaging",
    "product_question",
    "return",
    "refund",
    "exchange",
    "policy",
    "eligibility",
    "shipping",
    "delivery",
    "package",
    "packaging",
    "wrong",
    "missing",
}
RETAIL_MULTICATEGORY_INTENT_TERMS = {
    "issue_identification",
    "product_comparison",
    "evidence_lookup",
    "compare",
    "comparison",
    "identify",
    "lookup",
    "selected",
    "signal",
}
RETAIL_ISSUE_SYNONYMS = {
    "return_refund": {"return", "refund", "exchange", "eligibility"},
    "quality_complaint": {"quality", "complaint", "defect", "defective", "broken", "damaged"},
    "delivery_packaging": {"delivery", "shipping", "package", "packaging", "missing"},
    "suspicious_review": {"suspicious", "spam", "fake", "authenticity"},
    "review_summary": {"review", "summary", "rating", "signal"},
    "recommendation": {"recommend", "recommendation", "compare", "rating", "review"},
    "product_comparison": {"compare", "comparison", "versus", "vs", "review"},
    "issue_identification": {"issue", "identify", "complaint", "signal"},
    "evidence_lookup": {"evidence", "lookup", "selected", "record"},
    "policy_reasoning": {"policy", "reasoning", "support", "eligibility"},
}
RESEARCH_AI_SYNONYMS = {
    "inference": {"inference", "serving", "decoding", "generation"},
    "long context": {"long context", "long-context", "extended context"},
    "attention": {"attention", "kv cache", "key value cache"},
    "rag": {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
    "agentic": {"agentic", "agent", "agents", "workflow"},
    "workflow": {"workflow", "workflows", "agentic"},
    "benchmark": {"benchmark", "evaluation", "eval"},
}
DIRECT_EVIDENCE_ID_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:airline|healthcare_admin|retail|finance|research_ai)_(?:kb|doc|section|policy|review|summary|chunk|text|corpus)_[A-Za-z0-9:_\-/\.]+",
        r"\b(?:required|source)_(?:doc|evidence|chunk|paper|policy|parent|product)_ids?\b",
        r"\bCA-POL-[A-Za-z0-9-]+\b",
        r"\bMCH-[A-Za-z0-9-]+\b",
        r"\b(?:sec|xbrl)://\S+",
        r"\b\d{10}-\d{2}-\d{6}\b",
    )
]
STRICT_DIRECT_HINT_TOKENS = {
    "source_id",
    "parent_id",
    "document_id",
    "filing_id",
    "gold_evidence_ids",
    "required_doc_ids",
    "required_evidence_ids",
    "required_chunk_ids",
}


@dataclass(frozen=True)
class QueryEnrichmentResult:
    """Normalized query text and audit details for strict retrieval ablations."""

    query_text: str
    enrichment_terms: tuple[str, ...]
    blocked_direct_hint_count: int
    expanded_queries: tuple[str, ...] = ()
    expansion_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved context record and its score."""

    context_record: ContextRecord
    score: float
    rank: int
    retrieval_mode: str
    component_scores: dict[str, float]


@dataclass(frozen=True)
class TimedRetrieval:
    """Retrieval results with latency and backend metadata."""

    results: list[RetrievalResult]
    latency_ms: float
    backend_label: str
    retrieval_type: str
    vector_store: str = "none"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DenseRetrieverProtocol(Protocol):
    """Common interface for dense retrievers."""

    backend_label: str
    vector_store: str

    def retrieve(self, query: str, top_k: int) -> TimedRetrieval:
        """Retrieve dense context records."""


@dataclass(frozen=True)
class CompressionResult:
    """Compressed context selection metadata."""

    results: list[RetrievalResult]
    original_token_count: int
    compressed_token_count: int
    token_reduction: int
    compression_ratio: float
    dropped_context_ids: list[str]


@dataclass(frozen=True)
class BoostFeatures:
    """Precomputed metadata features used for hybrid score boosts."""

    match_ids: set[str]
    metadata_tokens: set[str]
    title_tokens: set[str]
    text_tokens: set[str]
    ticker: str
    form_normalized: str
    company_tokens: set[str]
    concept_tokens: set[str]
    section_tokens: set[str]
    date_tokens: set[str]
    record_metric_terms: set[str]
    retail_product_title_tokens: set[str] = field(default_factory=set)
    retail_category_tokens: set[str] = field(default_factory=set)
    retail_issue_tokens: set[str] = field(default_factory=set)
    retail_policy_tokens: set[str] = field(default_factory=set)
    retail_evidence_kind: str = ""
    retail_parent_key: str = ""


@dataclass(frozen=True)
class CompanyTickerResolver:
    """Corpus-derived company/ticker lookup for finance query expansion."""

    ticker_to_company: dict[str, str]
    alias_to_ticker: dict[str, str]

    @classmethod
    def from_records(cls, records: list[ContextRecord]) -> CompanyTickerResolver:
        """Build a resolver from finance context metadata only."""

        ticker_to_company: dict[str, str] = {}
        alias_to_ticker: dict[str, str] = {}
        for record in records:
            metadata = record.metadata
            ticker = str(metadata.get("ticker") or "").strip().upper()
            company = str(metadata.get("company_name") or metadata.get("company") or "").strip()
            if not ticker or ticker.lower() == "none":
                continue
            if company:
                ticker_to_company.setdefault(ticker, company)
                for alias in company_aliases_from_name(company):
                    alias_to_ticker.setdefault(alias, ticker)
            alias_to_ticker.setdefault(ticker.lower(), ticker)
        for alias, values in FINANCE_COMPANY_ALIASES.items():
            matching_tickers = {
                value.upper() for value in values if value.upper() in ticker_to_company
            }
            if matching_tickers:
                alias_to_ticker.setdefault(alias.lower(), sorted(matching_tickers)[0])
        return cls(ticker_to_company=ticker_to_company, alias_to_ticker=alias_to_ticker)

    def resolve_terms(self, text: str) -> set[str]:
        """Return company/ticker terms visible in text and supported by the corpus."""

        normalized = normalize_query_for_retrieval(text).lower()
        tokens = set(tokenize(normalized))
        terms: set[str] = set()
        for ticker, company in self.ticker_to_company.items():
            ticker_lower = ticker.lower()
            if ticker_lower in tokens:
                terms.add(ticker_lower)
                terms.update(tokenize(company))
        for alias, ticker in self.alias_to_ticker.items():
            if alias and alias in normalized:
                terms.add(alias)
                terms.add(ticker.lower())
                terms.update(tokenize(self.ticker_to_company.get(ticker, "")))
        return {term for term in terms if term}


def company_aliases_from_name(company_name: str) -> set[str]:
    """Return conservative aliases derived from a corpus company name."""

    normalized = re.sub(
        r"\b(inc|incorporated|corporation|corp|company|co|ltd)\b\.?", " ", company_name, flags=re.I
    )
    normalized = re.sub(r"[^A-Za-z0-9 ]+", " ", normalized)
    aliases = {company_name.lower(), normalized.lower().strip()}
    first_word = normalized.split()[0].lower() if normalized.split() else ""
    if first_word:
        aliases.add(first_word)
    if "alphabet" in normalized.lower():
        aliases.add("google")
    if "amazon" in normalized.lower():
        aliases.add("aws")
    if "microsoft" in normalized.lower():
        aliases.add("azure")
    if "meta" in normalized.lower():
        aliases.add("facebook")
    return {alias for alias in aliases if alias}


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalize_query_for_retrieval(text: str) -> str:
    """Normalize punctuation and common forms without adding hidden evidence hints."""

    normalized = text.replace("\\text", " ").replace("$", " ")
    normalized = re.sub(r"\bscenario\s+\d+\b", "scenario", normalized, flags=re.I)
    normalized = re.sub(
        r"\brecords?\s+(?:for|about|from|using)?\b",
        "records",
        normalized,
        flags=re.I,
    )
    normalized = re.sub(r"\b[A-Z]{3}\s*-\s*[A-Z]{3}\b", "route", normalized)
    normalized = normalized.replace("R&D", "research and development")
    normalized = normalized.replace("r&d", "research and development")
    normalized = re.sub(r"\b10\s*[- ]\s*k\b", "10-K annual filing", normalized, flags=re.I)
    normalized = re.sub(r"\b10\s*[- ]\s*q\b", "10-Q quarterly filing", normalized, flags=re.I)
    normalized = re.sub(r"\b8\s*[- ]\s*k\b", "8-K current report", normalized, flags=re.I)
    normalized = re.sub(r"\bQ([1-4])\b", r"quarter \1 q\1", normalized, flags=re.I)
    normalized = re.sub(r"\bfy\s*(20\d{2})\b", r"fiscal year \1", normalized, flags=re.I)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_period_terms(text: str) -> set[str]:
    """Extract visible year, quarter, and fiscal-period terms from query text."""

    normalized = normalize_query_for_retrieval(text).lower()
    terms = set(re.findall(r"\b20\d{2}\b", normalized))
    for quarter in re.findall(r"\bq([1-4])\b", normalized, flags=re.I):
        terms.add(f"q{quarter}")
        terms.add(f"quarter {quarter}")
    for quarter in re.findall(r"\bquarter\s+([1-4])\b", normalized, flags=re.I):
        terms.add(f"q{quarter}")
        terms.add(f"quarter {quarter}")
    if "fiscal year" in normalized:
        terms.add("fiscal year")
    if "annual" in normalized or "10-k" in normalized:
        terms.update({"annual", "10-k", "fiscal year"})
    if (
        "latest quarter" in normalized
        or "most recent quarter" in normalized
        or "10-q" in normalized
    ):
        terms.update({"latest quarter", "quarterly", "10-q"})
    if "latest year" in normalized or "most recent year" in normalized:
        terms.update({"latest year", "annual"})
    return {term for term in terms if term}


def scrub_direct_evidence_identifiers(text: str) -> tuple[str, int]:
    """Remove generated source/evidence IDs from strict retrieval query text."""

    scrubbed = text
    blocked_count = 0
    for pattern in DIRECT_EVIDENCE_ID_PATTERNS:
        scrubbed, count = pattern.subn(" ", scrubbed)
        blocked_count += count
    for token in STRICT_DIRECT_HINT_TOKENS:
        if token in scrubbed:
            blocked_count += scrubbed.count(token)
            scrubbed = scrubbed.replace(token, " ")
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return scrubbed, blocked_count


def finance_metric_expansion_terms(text: str) -> set[str]:
    """Return metric synonyms that are visible or implied by visible query text."""

    normalized = normalize_query_for_retrieval(text).lower()
    terms: set[str] = set()
    for trigger, expansions in FINANCE_METRIC_SYNONYMS.items():
        if trigger in normalized:
            terms.update(expansions)
    return terms


def build_xbrl_concept_map(records: list[ContextRecord]) -> dict[str, set[str]]:
    """Map natural metric terms to XBRL concepts found in the supplied corpus."""

    concept_tokens_by_name: dict[str, set[str]] = {}
    for record in records:
        concept_values = [record.metadata.get("concept")]
        raw_concepts = record.metadata.get("concepts")
        if isinstance(raw_concepts, list):
            concept_values.extend(raw_concepts)
        for concept_value in concept_values:
            if not concept_value:
                continue
            concept = str(concept_value)
            concept_tokens_by_name[concept] = set(tokenize(split_identifier_text(concept)))

    concept_map: dict[str, set[str]] = defaultdict(set)
    for metric, synonyms in FINANCE_METRIC_SYNONYMS.items():
        metric_tokens = set(tokenize(" ".join({metric, *synonyms})))
        for concept, concept_tokens in concept_tokens_by_name.items():
            if metric_tokens & concept_tokens:
                concept_map[metric].add(concept)
    return {metric: concepts for metric, concepts in concept_map.items() if concepts}


def concept_terms_for_query(
    text: str,
    concept_map: dict[str, set[str]] | None = None,
) -> set[str]:
    """Return concept-like terms found in the corpus and relevant to the query."""

    if not concept_map:
        return set()
    normalized = normalize_query_for_retrieval(text).lower()
    terms: set[str] = set()
    for metric, concepts in concept_map.items():
        triggers = {metric, *FINANCE_METRIC_SYNONYMS.get(metric, set())}
        if any(trigger in normalized for trigger in triggers):
            for concept in concepts:
                terms.add(concept)
                terms.update(tokenize(split_identifier_text(concept)))
    return terms


def synonym_terms_for_query(text: str, vertical: str | None = None) -> set[str]:
    """Return deterministic query expansion terms allowed by strict ablation policy."""

    normalized = normalize_query_for_retrieval(text).lower()
    terms: set[str] = set()
    synonym_maps = [FINANCE_METRIC_SYNONYMS]
    if vertical == "research_ai":
        synonym_maps.append(RESEARCH_AI_SYNONYMS)
    for synonym_map in synonym_maps:
        for trigger, expansions in synonym_map.items():
            if trigger in normalized:
                terms.update(expansions)
    for token in tokenize(normalized):
        terms.update(FINANCE_COMPANY_ALIASES.get(token, set()))
    if "latest quarter" in normalized or "most recent quarter" in normalized:
        terms.update({"quarter", "quarterly", "10-q"})
    if "annual" in normalized or "fiscal year" in normalized:
        terms.update({"annual", "fiscal year", "10-k"})
    terms.update(extract_period_terms(normalized))
    return {term for term in terms if term}


def build_expanded_queries(
    *,
    original_query: str,
    normalized_query: str,
    vertical: str | None,
    enrichment_terms: set[str],
    resolver_terms: set[str] | None = None,
    concept_terms: set[str] | None = None,
    metadata_terms: set[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Build deterministic expanded query variants for candidate generation."""

    queries: list[str] = []
    expansion_types: list[str] = []

    def add_query(query: str, expansion_type: str) -> None:
        clean_query = re.sub(r"\s+", " ", query).strip()
        if clean_query and clean_query not in queries:
            queries.append(clean_query)
            expansion_types.append(expansion_type)

    add_query(normalized_query, "normalized_original")
    if enrichment_terms:
        add_query(
            " ".join([normalized_query, " ".join(sorted(enrichment_terms))]),
            "synonym_expanded",
        )
    if vertical == "finance":
        metric_terms = finance_metric_expansion_terms(normalized_query)
        if metric_terms:
            add_query(
                " ".join([normalized_query, " ".join(sorted(metric_terms))]),
                "finance_metric_expanded",
            )
        period_terms = extract_period_terms(normalized_query)
        if period_terms:
            add_query(
                " ".join([normalized_query, " ".join(sorted(period_terms))]),
                "period_normalized",
            )
        active_resolver_terms = resolver_terms or set()
        if active_resolver_terms:
            add_query(
                " ".join([normalized_query, " ".join(sorted(active_resolver_terms))]),
                "entity_normalized",
            )
        active_concept_terms = concept_terms or set()
        if active_concept_terms:
            add_query(
                " ".join([normalized_query, " ".join(sorted(active_concept_terms))]),
                "xbrl_concept_expanded",
            )
    active_metadata_terms = metadata_terms or set()
    if active_metadata_terms:
        add_query(
            " ".join([normalized_query, " ".join(sorted(active_metadata_terms))]),
            "metadata_expanded",
        )
    return tuple(queries), tuple(expansion_types)


def enrich_query_text(
    text: str,
    *,
    vertical: str | None = None,
    allow_direct_identifiers: bool = False,
    resolver: CompanyTickerResolver | None = None,
    concept_map: dict[str, set[str]] | None = None,
    metadata_terms: set[str] | None = None,
) -> QueryEnrichmentResult:
    """Build retrieval query text with allowed deterministic enrichment."""

    blocked_count = 0
    query_text = text
    if not allow_direct_identifiers:
        query_text, blocked_count = scrub_direct_evidence_identifiers(query_text)
    normalized = normalize_query_for_retrieval(query_text)
    enrichment_terms_set = synonym_terms_for_query(normalized, vertical)
    if resolver is not None:
        enrichment_terms_set.update(resolver.resolve_terms(normalized))
    enrichment_terms_set.update(concept_terms_for_query(normalized, concept_map))
    enrichment_terms = tuple(sorted(enrichment_terms_set))
    enriched = " ".join(part for part in [normalized, " ".join(enrichment_terms)] if part)
    enriched = re.sub(r"\s+", " ", enriched).strip()
    expanded_queries, expansion_types = build_expanded_queries(
        original_query=query_text,
        normalized_query=normalized.lower(),
        vertical=vertical,
        enrichment_terms=enrichment_terms_set,
        resolver_terms=resolver.resolve_terms(normalized) if resolver else set(),
        concept_terms=concept_terms_for_query(normalized, concept_map),
        metadata_terms=metadata_terms,
    )
    return QueryEnrichmentResult(
        query_text=enriched,
        enrichment_terms=enrichment_terms,
        blocked_direct_hint_count=blocked_count,
        expanded_queries=expanded_queries,
        expansion_types=expansion_types,
    )


def split_identifier_text(value: str) -> str:
    """Split camel-case and separators in compact metadata identifiers."""

    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    return re.sub(r"[_\-/]+", " ", spaced)


def normalize_identifier(value: str) -> str:
    """Normalize identifiers for cross-field matching."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def identifier_variants(value: str) -> set[str]:
    """Return exact and normalized identifier variants."""

    if not value:
        return set()
    normalized = normalize_identifier(value)
    variants = {value}
    if normalized:
        variants.add(normalized)
    return variants


def context_match_ids(record: ContextRecord) -> set[str]:
    """Return identifiers that can match a gold evidence ID."""

    values = {
        record.context_id,
        record.source_id,
        record.parent_id,
        record.chunk_id,
        str(record.metadata.get("original_doc_id") or ""),
    }
    values.update(metadata_search_values(record.metadata))
    match_ids: set[str] = set()
    for value in values:
        if value:
            match_ids.update(identifier_variants(value))
    return match_ids


def metadata_search_values(value: Any) -> list[str]:
    """Flatten metadata values into searchable strings."""

    if isinstance(value, dict):
        values: list[str] = []
        for nested_value in value.values():
            values.extend(metadata_search_values(nested_value))
        return values
    if isinstance(value, list):
        values = []
        for nested_value in value:
            values.extend(metadata_search_values(nested_value))
        return values
    if value is None:
        return []
    return [str(value)]


def record_search_text(record: ContextRecord) -> str:
    """Build retrieval text from body, stable IDs, and metadata."""

    parts = [
        record.context_id,
        record.source_id,
        record.parent_id,
        record.chunk_id,
        record.chunk_strategy,
        record.source_type,
        record.title,
        record.text,
        record.provenance,
        *metadata_search_values(record.metadata),
    ]
    return " ".join(part for part in parts if part)


def metadata_text(record: ContextRecord) -> str:
    """Return searchable metadata-only text for boosting."""

    return " ".join(
        [
            record.context_id,
            record.source_id,
            record.parent_id,
            record.chunk_id,
            record.chunk_strategy,
            record.source_type,
            record.title,
            record.provenance,
            *metadata_search_values(record.metadata),
        ]
    )


def retail_evidence_kind(record: ContextRecord) -> str:
    """Classify Retail context into a non-gold evidence kind."""

    metadata = record.metadata
    explicit_text = " ".join(
        str(value or "")
        for value in (
            metadata.get("evidence_type"),
            metadata.get("document_type"),
            record.source_type,
        )
    ).lower()
    if "policy" in explicit_text:
        return "policy"
    if "multicategory" in explicit_text:
        return "multicategory"
    if "summary" in explicit_text:
        return "summary"
    if "review_evidence" in explicit_text:
        return "review"
    if "product_metadata" in explicit_text:
        return "metadata"

    tags_value = metadata.get("tags")
    tags_text = (
        " ".join(str(item) for item in tags_value if item) if isinstance(tags_value, list) else ""
    )
    haystack = " ".join(
        str(value or "")
        for value in (
            metadata.get("evidence_type"),
            metadata.get("document_type"),
            metadata.get("source_type"),
            record.source_type,
            record.title,
            tags_text,
        )
    ).lower()
    if "policy" in haystack:
        return "policy"
    if "summary" in haystack:
        return "summary"
    if "multicategory" in haystack or "support evidence" in haystack:
        return "multicategory"
    if "review_evidence" in haystack or re.search(r"\breview\b", haystack):
        return "review"
    if "metadata" in haystack:
        return "metadata"
    return "other"


def build_boost_features(record: ContextRecord) -> BoostFeatures:
    """Precompute metadata features for one context record."""

    metadata = record.metadata
    issue_values = metadata.get("issue_terms")
    retail_issue_tokens: set[str] = set()
    if isinstance(issue_values, list):
        retail_issue_tokens.update(
            token for value in issue_values for token in tokenize(split_identifier_text(str(value)))
        )
    retail_issue_tokens.update(
        tokenize(split_identifier_text(str(metadata.get("issue_type") or "")))
    )
    retail_policy_tokens = set(
        tokenize(split_identifier_text(str(metadata.get("policy_key") or "")))
    )
    tags_value = metadata.get("tags")
    tags_text = (
        " ".join(str(item) for item in tags_value if item) if isinstance(tags_value, list) else ""
    )
    retail_policy_tokens.update(
        token
        for token in tokenize(split_identifier_text(tags_text))
        if token in RETAIL_POLICY_INTENT_TERMS
    )
    concept_values = metadata.get("concepts")
    concepts = set(tokenize(split_identifier_text(str(metadata.get("concept") or ""))))
    if isinstance(concept_values, list):
        concepts.update(
            token
            for value in concept_values
            for token in tokenize(split_identifier_text(str(value)))
        )
    date_tokens: set[str] = set()
    for date_field in (
        "filing_date",
        "report_date",
        "period",
        "fiscal_year",
        "fiscal_periods_present",
        "fiscal_years_present",
        "latest_end",
        "latest_filed",
    ):
        date_tokens.update(tokenize(str(metadata.get(date_field) or "")))
    return BoostFeatures(
        match_ids=context_match_ids(record),
        metadata_tokens=set(tokenize(metadata_text(record))),
        title_tokens=set(tokenize(record.title)),
        text_tokens=set(tokenize(record.text)),
        ticker=str(metadata.get("ticker") or "").lower(),
        form_normalized=normalize_identifier(str(metadata.get("form") or "")),
        company_tokens=set(tokenize(str(metadata.get("company_name") or ""))),
        concept_tokens=concepts,
        section_tokens=set(tokenize(str(metadata.get("section_type") or ""))),
        date_tokens=date_tokens,
        record_metric_terms=set(tokenize(record_search_text(record))) & FINANCE_METRIC_TERMS,
        retail_product_title_tokens=set(
            tokenize(str(metadata.get("product_title") or record.title))
        ),
        retail_category_tokens=set(
            tokenize(split_identifier_text(str(metadata.get("category") or "")))
        ),
        retail_issue_tokens=retail_issue_tokens,
        retail_policy_tokens=retail_policy_tokens,
        retail_evidence_kind=retail_evidence_kind(record),
        retail_parent_key=str(
            metadata.get("parent_asin") or metadata.get("asin") or record.parent_id
        ),
    )


def metadata_boost_score(
    query: str,
    record: ContextRecord,
    features: BoostFeatures | None = None,
) -> float:
    """Return a deterministic metadata boost for hybrid retrieval."""

    query_tokens = set(tokenize(query))
    query_normalized = normalize_identifier(query)
    active_features = features or build_boost_features(record)
    return metadata_boost_score_from_features(
        query_tokens=query_tokens,
        query_normalized=query_normalized,
        record=record,
        features=active_features,
    )


def metadata_boost_score_from_features(
    *,
    query_tokens: set[str],
    query_normalized: str,
    record: ContextRecord,
    features: BoostFeatures,
) -> float:
    """Return a deterministic metadata boost using precomputed features."""

    boost = 0.0
    for match_id in features.match_ids:
        if len(match_id) >= 8 and match_id in query_normalized:
            boost += 2.5

    boost += min(0.8, 0.08 * len(query_tokens & features.metadata_tokens))

    if record.vertical == "finance":
        boost += finance_metadata_boost_score(
            query_tokens=query_tokens,
            query_normalized=query_normalized,
            features=features,
        )
    elif record.vertical == "retail":
        boost += retail_metadata_boost_score(
            query_tokens=query_tokens,
            query_normalized=query_normalized,
            features=features,
        )
    return boost


def finance_metadata_boost_score(
    *,
    query_tokens: set[str],
    query_normalized: str,
    features: BoostFeatures,
) -> float:
    """Return finance-specific boosts from prompt/corpus metadata."""

    boost = 0.0
    if features.ticker and features.ticker in query_tokens:
        boost += 1.2

    if features.form_normalized and features.form_normalized in query_normalized:
        boost += 0.8

    if features.company_tokens:
        overlap = len(features.company_tokens & query_tokens) / len(features.company_tokens)
        boost += 0.8 * overlap

    concept_overlap = len(features.concept_tokens & query_tokens)
    metric_overlap = len(features.record_metric_terms & query_tokens)
    boost += min(1.2, concept_overlap * 0.45 + metric_overlap * 0.2)

    if features.section_tokens and features.section_tokens & query_tokens:
        boost += 0.6

    if any(len(token) >= 4 and token in query_tokens for token in features.date_tokens):
        boost += 0.35

    return boost


def expanded_retail_query_terms(query_tokens: set[str]) -> set[str]:
    """Return Retail issue synonyms allowed from visible query terms."""

    terms = set(query_tokens)
    for token in query_tokens:
        terms.update(RETAIL_ISSUE_SYNONYMS.get(token, set()))
    return {term for term in tokenize(" ".join(terms))}


def retail_intent_flags(query_tokens: set[str], query_normalized: str) -> dict[str, bool]:
    """Infer Retail retrieval intent from visible query terms."""

    expanded_terms = expanded_retail_query_terms(query_tokens)
    joined = " ".join(sorted(expanded_terms | query_tokens | set(tokenize(query_normalized))))
    direct_review_terms = RETAIL_REVIEW_INTENT_TERMS - {
        "review_summary",
        "review",
        "reviews",
        "rating",
        "ratings",
    }
    direct_review_intent = bool(query_tokens & direct_review_terms)
    review_intent = (
        direct_review_intent
        or bool(expanded_terms & RETAIL_REVIEW_INTENT_TERMS)
        or any(term in query_normalized for term in RETAIL_REVIEW_INTENT_TERMS)
    )
    summary_intent = (
        "review_summary" in joined
        or "summary" in expanded_terms
        or "review summary" in query_normalized
    )
    policy_intent = bool(expanded_terms & RETAIL_POLICY_INTENT_TERMS) or any(
        term in query_normalized for term in RETAIL_POLICY_INTENT_TERMS
    )
    multicategory_intent = bool(expanded_terms & RETAIL_MULTICATEGORY_INTENT_TERMS) or any(
        term in query_normalized for term in RETAIL_MULTICATEGORY_INTENT_TERMS
    )
    if review_intent or summary_intent:
        multicategory_intent = (
            "product_comparison" in joined
            or "evidence_lookup" in joined
            or "issue_identification" in joined
        )
    if multicategory_intent and not (direct_review_intent or summary_intent):
        review_intent = False
    return {
        "review": review_intent,
        "direct_review": direct_review_intent,
        "policy": policy_intent,
        "multicategory": multicategory_intent,
        "summary": summary_intent,
        "comparison": "product_comparison" in joined or "comparison" in expanded_terms,
    }


def retail_kind_intent_score(
    *,
    evidence_kind: str,
    intent_flags: dict[str, bool],
) -> float:
    """Score how well a Retail evidence kind matches visible query intent."""

    if evidence_kind == "summary":
        if intent_flags["summary"]:
            return 1.0
        if intent_flags["review"]:
            return 0.7
    if evidence_kind == "review":
        if intent_flags["review"]:
            return 1.0
        if intent_flags["summary"]:
            return 0.8
    if evidence_kind == "policy":
        return 1.0 if intent_flags["policy"] else 0.15
    if evidence_kind == "multicategory":
        if intent_flags["direct_review"]:
            return 0.2
        if intent_flags["summary"]:
            return 1.0
        if intent_flags["multicategory"] or intent_flags["comparison"]:
            return 1.0
    if evidence_kind == "metadata":
        return 0.6 if intent_flags["comparison"] else 0.25
    return 0.2


def retail_metadata_boost_score(
    *,
    query_tokens: set[str],
    query_normalized: str,
    features: BoostFeatures,
) -> float:
    """Return Retail-specific metadata boosts from prompt-visible signals."""

    expanded_terms = expanded_retail_query_terms(query_tokens)
    intent_flags = retail_intent_flags(query_tokens, query_normalized)
    product_overlap = lexical_overlap_ratio(query_tokens, features.retail_product_title_tokens)
    category_overlap = lexical_overlap_ratio(query_tokens, features.retail_category_tokens)
    issue_overlap = lexical_overlap_ratio(expanded_terms, features.retail_issue_tokens)
    policy_overlap = lexical_overlap_ratio(expanded_terms, features.retail_policy_tokens)
    kind_score = retail_kind_intent_score(
        evidence_kind=features.retail_evidence_kind,
        intent_flags=intent_flags,
    )

    boost = 0.0
    boost += min(1.1, 1.1 * product_overlap)
    boost += min(0.45, 0.45 * category_overlap)
    boost += min(0.9, 0.9 * issue_overlap)
    boost += min(0.75, 0.75 * policy_overlap)
    boost += 0.65 * kind_score
    if (
        features.retail_evidence_kind == "multicategory"
        and intent_flags["direct_review"]
        and not intent_flags["multicategory"]
    ):
        boost -= 0.45
    return boost


def lexical_overlap_ratio(query_tokens: set[str], record_tokens: set[str]) -> float:
    """Return overlap share from query to record tokens."""

    if not query_tokens:
        return 0.0
    return len(query_tokens & record_tokens) / len(query_tokens)


def rerank_boost_score(
    query: str,
    record: ContextRecord,
    features: BoostFeatures | None = None,
) -> float:
    """Return deterministic post-fusion rerank boost for strict retrieval."""

    query_tokens = set(tokenize(query))
    query_normalized = normalize_identifier(query)
    active_features = features or build_boost_features(record)
    return rerank_boost_score_from_features(
        query_tokens=query_tokens,
        query_normalized=query_normalized,
        record=record,
        features=active_features,
    )


def rerank_boost_score_from_features(
    *,
    query_tokens: set[str],
    query_normalized: str,
    record: ContextRecord,
    features: BoostFeatures,
) -> float:
    """Return deterministic rerank boost using precomputed query and record features."""

    title_overlap = lexical_overlap_ratio(query_tokens, features.title_tokens)
    metadata_overlap = lexical_overlap_ratio(query_tokens, features.metadata_tokens)
    text_overlap = lexical_overlap_ratio(query_tokens, features.text_tokens)
    boost = 0.35 * title_overlap + 0.2 * metadata_overlap + 0.1 * text_overlap
    if record.vertical == "finance":
        boost += finance_rerank_boost_score(
            query_tokens=query_tokens,
            query_normalized=query_normalized,
            features=features,
        )
    elif record.vertical == "retail":
        boost += retail_rerank_boost_score(
            query_tokens=query_tokens,
            query_normalized=query_normalized,
            features=features,
        )
    elif record.vertical == "research_ai":
        section_overlap = features.section_tokens & query_tokens
        boost += min(0.45, 0.18 * len(section_overlap))
        if features.title_tokens & query_tokens:
            boost += 0.25
    return boost


def finance_rerank_boost_score(
    *,
    query_tokens: set[str],
    query_normalized: str,
    features: BoostFeatures,
) -> float:
    """Return finance-specific deterministic reranking score."""

    score = 0.0
    if features.ticker and features.ticker in query_tokens:
        score += 1.4
    if features.form_normalized and features.form_normalized in query_normalized:
        score += 0.9
    if features.company_tokens:
        score += 0.8 * lexical_overlap_ratio(query_tokens, features.company_tokens)
    metric_terms = set()
    for token in query_tokens:
        metric_terms.update(tokenize(" ".join(FINANCE_METRIC_SYNONYMS.get(token, set()))))
    metric_overlap = (query_tokens | metric_terms) & (
        features.concept_tokens | features.record_metric_terms
    )
    score += min(1.6, 0.35 * len(metric_overlap))
    if features.section_tokens & query_tokens:
        score += 0.5
    if any(len(token) >= 4 and token in query_tokens for token in features.date_tokens):
        score += 0.45
    return score


def retail_rerank_boost_score(
    *,
    query_tokens: set[str],
    query_normalized: str,
    features: BoostFeatures,
) -> float:
    """Return Retail-specific deterministic reranking score."""

    expanded_terms = expanded_retail_query_terms(query_tokens)
    intent_flags = retail_intent_flags(query_tokens, query_normalized)
    score = 0.0
    score += min(
        1.2, 1.25 * lexical_overlap_ratio(query_tokens, features.retail_product_title_tokens)
    )
    score += min(0.5, 0.5 * lexical_overlap_ratio(query_tokens, features.retail_category_tokens))
    score += min(1.0, 1.1 * lexical_overlap_ratio(expanded_terms, features.retail_issue_tokens))
    score += min(0.8, 0.9 * lexical_overlap_ratio(expanded_terms, features.retail_policy_tokens))
    score += 0.9 * retail_kind_intent_score(
        evidence_kind=features.retail_evidence_kind,
        intent_flags=intent_flags,
    )
    if (
        features.retail_evidence_kind == "multicategory"
        and intent_flags["direct_review"]
        and not intent_flags["multicategory"]
    ):
        score -= 0.7
    return score


def section_intent_terms(query_tokens: set[str]) -> set[str]:
    """Infer section terms from visible query intent."""

    terms: set[str] = set()
    if query_tokens & {"risk", "risks", "uncertainty", "headwinds"}:
        terms.update({"risk", "factors", "risk_factors"})
    if query_tokens & {"revenue", "sales", "income", "margin", "cash", "flow", "capex"}:
        terms.update(
            {
                "financial",
                "statements",
                "results",
                "operations",
                "management",
                "discussion",
                "analysis",
                "xbrl",
            }
        )
    if query_tokens & {"guidance", "outlook", "forecast", "expectation"}:
        terms.update({"outlook", "guidance", "management", "discussion"})
    if query_tokens & {"business", "segment", "segments"}:
        terms.update({"business", "segment", "segments", "operations"})
    return terms


def rerank_feature_breakdown(
    *,
    query_tokens: set[str],
    query_normalized: str,
    record: ContextRecord,
    features: BoostFeatures,
    source_hints_used: bool,
) -> dict[str, float]:
    """Return deterministic reranker feature values for one candidate."""

    section_terms = section_intent_terms(query_tokens)
    source_hint_match = 0.0
    if source_hints_used and any(
        len(match_id) >= 8 and match_id in query_normalized for match_id in features.match_ids
    ):
        source_hint_match = 1.0
    metric_terms = set(query_tokens)
    for token in query_tokens:
        metric_terms.update(tokenize(" ".join(FINANCE_METRIC_SYNONYMS.get(token, set()))))
    breakdown = {
        "title_match": lexical_overlap_ratio(query_tokens, features.title_tokens),
        "metadata_match": lexical_overlap_ratio(query_tokens, features.metadata_tokens),
        "text_match": lexical_overlap_ratio(query_tokens, features.text_tokens),
        "company_ticker_match": 1.0
        if (
            (features.ticker and features.ticker in query_tokens)
            or bool(features.company_tokens & query_tokens)
        )
        else 0.0,
        "metric_match": min(
            1.0,
            len(metric_terms & (features.concept_tokens | features.record_metric_terms)) / 4,
        ),
        "period_match": 1.0
        if any(len(token) >= 4 and token in query_tokens for token in features.date_tokens)
        else 0.0,
        "section_match": 1.0
        if bool((features.section_tokens | features.metadata_tokens) & section_terms)
        else 0.0,
        "source_hint_match": source_hint_match,
    }
    if record.vertical == "retail":
        expanded_terms = expanded_retail_query_terms(query_tokens)
        intent_flags = retail_intent_flags(query_tokens, query_normalized)
        breakdown.update(
            {
                "retail_product_match": lexical_overlap_ratio(
                    query_tokens,
                    features.retail_product_title_tokens,
                ),
                "retail_category_match": lexical_overlap_ratio(
                    query_tokens,
                    features.retail_category_tokens,
                ),
                "retail_issue_match": lexical_overlap_ratio(
                    expanded_terms,
                    features.retail_issue_tokens,
                ),
                "retail_policy_match": lexical_overlap_ratio(
                    expanded_terms,
                    features.retail_policy_tokens,
                ),
                "retail_kind_match": retail_kind_intent_score(
                    evidence_kind=features.retail_evidence_kind,
                    intent_flags=intent_flags,
                ),
            }
        )
        breakdown["metric_match"] = min(
            1.0,
            (
                breakdown["retail_product_match"]
                + breakdown["retail_issue_match"]
                + breakdown["retail_kind_match"]
            )
            / 2.0,
        )
        breakdown["section_match"] = breakdown["retail_policy_match"]
    elif record.vertical != "finance":
        breakdown["metric_match"] = min(
            1.0,
            len(query_tokens & (features.metadata_tokens | features.title_tokens)) / 8,
        )
    return breakdown


def score_candidate_from_features(
    *,
    lexical_score: float,
    dense_score: float,
    max_lexical: float,
    max_dense: float,
    query_tokens: set[str],
    query_normalized: str,
    record: ContextRecord,
    features: BoostFeatures,
    lexical_weight: float,
    dense_weight: float,
    query_hit_count: int,
    source_hints_used: bool,
) -> tuple[float, dict[str, float]]:
    """Score one candidate using fused retrieval and reranker features."""

    lexical_norm = lexical_score / max_lexical if max_lexical > 0 else 0.0
    dense_norm = dense_score / max_dense if max_dense > 0 else 0.0
    metadata_boost = metadata_boost_score_from_features(
        query_tokens=query_tokens,
        query_normalized=query_normalized,
        record=record,
        features=features,
    )
    base_rerank_boost = rerank_boost_score_from_features(
        query_tokens=query_tokens,
        query_normalized=query_normalized,
        record=record,
        features=features,
    )
    breakdown = rerank_feature_breakdown(
        query_tokens=query_tokens,
        query_normalized=query_normalized,
        record=record,
        features=features,
        source_hints_used=source_hints_used,
    )
    if record.vertical == "retail":
        feature_score = (
            0.85 * breakdown.get("retail_product_match", 0.0)
            + 0.45 * breakdown.get("retail_category_match", 0.0)
            + 0.75 * breakdown.get("retail_issue_match", 0.0)
            + 0.6 * breakdown.get("retail_policy_match", 0.0)
            + 0.95 * breakdown.get("retail_kind_match", 0.0)
            + 0.25 * breakdown["title_match"]
            + 0.15 * breakdown["metadata_match"]
            + (1.25 * breakdown["source_hint_match"] if source_hints_used else 0.0)
        )
    else:
        feature_score = (
            0.45 * breakdown["company_ticker_match"]
            + 0.55 * breakdown["metric_match"]
            + 0.35 * breakdown["period_match"]
            + 0.35 * breakdown["section_match"]
            + 0.3 * breakdown["title_match"]
            + 0.2 * breakdown["metadata_match"]
            + (1.25 * breakdown["source_hint_match"] if source_hints_used else 0.0)
        )
    multi_query_boost = min(0.35, 0.04 * max(0, query_hit_count - 1))
    rerank_score = (
        lexical_weight * lexical_norm
        + dense_weight * dense_norm
        + metadata_boost
        + base_rerank_boost
        + feature_score
        + multi_query_boost
    )
    breakdown.update(
        {
            "bm25_score": lexical_score,
            "dense_score": dense_score,
            "bm25_norm": lexical_norm,
            "dense_norm": dense_norm,
            "metadata_boost": metadata_boost,
            "base_rerank_boost": base_rerank_boost,
            "multi_query_boost": multi_query_boost,
            "rerank_score": rerank_score,
        }
    )
    return rerank_score, breakdown


def evidence_selector_strategy_for_record(record: ContextRecord) -> str:
    """Return the final top-5 selector strategy for a candidate record."""

    if record.vertical == "finance":
        return "finance_calibrated_top5"
    if record.vertical == "retail":
        return "retail_balanced_top5"
    return "calibrated_top5"


def selection_reason_from_features(
    *,
    record: ContextRecord,
    feature_breakdown: dict[str, float],
    source_hints_used: bool,
) -> str:
    """Return a compact human-readable selection reason."""

    reasons: list[str] = []
    if feature_breakdown.get("company_ticker_match", 0.0) > 0:
        reasons.append("company_or_ticker_match")
    if feature_breakdown.get("metric_match", 0.0) > 0:
        reasons.append("metric_or_concept_match")
    if feature_breakdown.get("period_match", 0.0) > 0:
        reasons.append("period_match")
    if feature_breakdown.get("section_match", 0.0) > 0:
        reasons.append("section_match")
    if feature_breakdown.get("title_match", 0.0) > 0:
        reasons.append("title_overlap")
    if feature_breakdown.get("metadata_match", 0.0) > 0:
        reasons.append("metadata_overlap")
    if record.vertical == "retail":
        if feature_breakdown.get("retail_product_match", 0.0) > 0:
            reasons.append("retail_product_title_match")
        if feature_breakdown.get("retail_category_match", 0.0) > 0:
            reasons.append("retail_category_match")
        if feature_breakdown.get("retail_issue_match", 0.0) > 0:
            reasons.append("retail_review_issue_match")
        if feature_breakdown.get("retail_policy_match", 0.0) > 0:
            reasons.append("retail_policy_match")
        if feature_breakdown.get("retail_kind_match", 0.0) > 0:
            reasons.append("retail_evidence_kind_match")
    if source_hints_used and feature_breakdown.get("source_hint_match", 0.0) > 0:
        reasons.append("source_hint_match_assisted")
    if not reasons:
        reasons.append("hybrid_score_rank")
    if record.vertical == "finance":
        reasons.append("finance_selector")
    if record.vertical == "retail":
        reasons.append("retail_selector")
    return ",".join(dict.fromkeys(reasons))


def select_retail_balanced_candidates(
    *,
    ranked: list[tuple[str, float, dict[str, float]]],
    records_by_id: dict[str, ContextRecord],
    query_tokens: set[str],
    query_normalized: str,
    final_top_k: int,
) -> list[tuple[str, float, dict[str, float]]]:
    """Select Retail top-k candidates with evidence-kind and parent-child balance."""

    retail_ranked = [
        item for item in ranked if item[1] > 0 and records_by_id[item[0]].vertical == "retail"
    ]
    if not retail_ranked:
        return ranked

    intent_flags = retail_intent_flags(query_tokens, query_normalized)
    if intent_flags["direct_review"]:
        desired_kinds = ["review", "summary", "policy", "multicategory", "metadata"]
    elif intent_flags["summary"]:
        desired_kinds = ["multicategory", "summary", "review", "policy", "metadata"]
    elif intent_flags["multicategory"] or intent_flags["comparison"]:
        desired_kinds = ["multicategory", "policy", "metadata", "review", "summary"]
    elif intent_flags["policy"]:
        desired_kinds = ["policy", "multicategory", "review", "summary", "metadata"]
    elif intent_flags["review"]:
        desired_kinds = ["summary", "review", "policy", "multicategory", "metadata"]
    else:
        desired_kinds = ["summary", "review", "multicategory", "policy", "metadata"]

    selected: list[tuple[str, float, dict[str, float]]] = []
    selected_ids: set[str] = set()

    def add_candidate(item: tuple[str, float, dict[str, float]]) -> None:
        if len(selected) >= final_top_k:
            return
        context_id = item[0]
        if context_id in selected_ids:
            return
        selected.append(item)
        selected_ids.add(context_id)

    def add_first_for_kind(kind: str, parent_key: str | None = None) -> None:
        for item in retail_ranked:
            record = records_by_id[item[0]]
            features = build_boost_features(record)
            if features.retail_evidence_kind != kind:
                continue
            if parent_key is not None and features.retail_parent_key != parent_key:
                continue
            add_candidate(item)
            return

    parent_order: list[str] = []
    for context_id, _score, _breakdown in retail_ranked:
        parent_key = build_boost_features(records_by_id[context_id]).retail_parent_key
        if parent_key and parent_key not in parent_order:
            parent_order.append(parent_key)

    if intent_flags["comparison"] and len(parent_order) > 1:
        for parent_key in parent_order[:2]:
            for kind in ("review", "summary", "multicategory", "metadata"):
                add_first_for_kind(kind, parent_key=parent_key)
                if len(selected) >= final_top_k:
                    break
            if len(selected) >= final_top_k:
                break

    if intent_flags["direct_review"]:
        review_parent_order: list[str | None] = list(parent_order[:2])
        if not review_parent_order:
            review_parent_order = [None]
        for active_parent_key in review_parent_order:
            for kind in ("review", "summary"):
                add_first_for_kind(kind, parent_key=active_parent_key)

    for kind in desired_kinds:
        add_first_for_kind(kind)

    for item in retail_ranked:
        add_candidate(item)
        if len(selected) >= final_top_k:
            break

    if len(selected) < final_top_k:
        for item in ranked:
            if item[1] <= 0:
                continue
            add_candidate(item)
            if len(selected) >= final_top_k:
                break

    ordered_ids = {context_id for context_id, _score, _breakdown in selected}
    remainder = [item for item in ranked if item[0] not in ordered_ids]
    return [*selected, *remainder]


def rerank_candidate_results(
    *,
    query: str,
    candidate_results: list[RetrievalResult],
    final_top_k: int,
    retrieval_mode: str,
    lexical_weight: float = 0.55,
    dense_weight: float = 0.45,
    source_hints_used: bool = False,
    candidate_top_k_dense: int = DEFAULT_CANDIDATE_TOP_K_DENSE,
    candidate_top_k_lexical: int = DEFAULT_CANDIDATE_TOP_K_LEXICAL,
    expanded_query_count: int = 1,
    expansion_types: tuple[str, ...] = (),
    started: float | None = None,
    backend_label: str = "local_fallback",
    vector_store: str = "none",
    boost_features_by_context_id: dict[str, BoostFeatures] | None = None,
) -> TimedRetrieval:
    """Merge, deduplicate, rerank candidates, and return final top-k results."""

    start_time = started if started is not None else time.perf_counter()
    records_by_id: dict[str, ContextRecord] = {}
    lexical_scores: dict[str, float] = defaultdict(float)
    dense_scores: dict[str, float] = defaultdict(float)
    query_hit_counts: Counter[str] = Counter()
    pre_rerank_order: list[str] = []
    for result in candidate_results:
        context_id = result.context_record.context_id
        records_by_id[context_id] = result.context_record
        query_hit_counts[context_id] += 1
        if context_id not in pre_rerank_order:
            pre_rerank_order.append(context_id)
        lexical_scores[context_id] = max(
            lexical_scores[context_id],
            float(result.component_scores.get("bm25", 0.0)),
        )
        dense_scores[context_id] = max(
            dense_scores[context_id],
            float(result.component_scores.get("dense", 0.0)),
        )

    max_lexical = max(lexical_scores.values(), default=0.0)
    max_dense = max(dense_scores.values(), default=0.0)
    query_tokens = set(tokenize(query))
    query_normalized = normalize_identifier(query)
    boost_features = boost_features_by_context_id or {}

    ranked_candidates: list[tuple[str, float, dict[str, float]]] = []
    for context_id, record in records_by_id.items():
        score, feature_breakdown = score_candidate_from_features(
            lexical_score=lexical_scores.get(context_id, 0.0),
            dense_score=dense_scores.get(context_id, 0.0),
            max_lexical=max_lexical,
            max_dense=max_dense,
            query_tokens=query_tokens,
            query_normalized=query_normalized,
            record=record,
            features=boost_features.get(context_id) or build_boost_features(record),
            lexical_weight=lexical_weight,
            dense_weight=dense_weight,
            query_hit_count=query_hit_counts[context_id],
            source_hints_used=source_hints_used,
        )
        ranked_candidates.append((context_id, score, feature_breakdown))

    ranked = sorted(ranked_candidates, key=lambda item: (-item[1], item[0]))
    if any(records_by_id[context_id].vertical == "retail" for context_id, _score, _ in ranked):
        ranked = select_retail_balanced_candidates(
            ranked=ranked,
            records_by_id=records_by_id,
            query_tokens=query_tokens,
            query_normalized=query_normalized,
            final_top_k=final_top_k,
        )
    results: list[RetrievalResult] = []
    seen_texts: set[str] = set()
    selection_reasons_by_context_id: dict[str, str] = {}
    selector_strategy = "calibrated_top5"
    if any(records_by_id[context_id].vertical == "finance" for context_id, _score, _ in ranked):
        selector_strategy = "finance_calibrated_top5"
    if any(records_by_id[context_id].vertical == "retail" for context_id, _score, _ in ranked):
        selector_strategy = "retail_balanced_top5"
    for context_id, score, feature_breakdown in ranked:
        if score <= 0:
            continue
        record = records_by_id[context_id]
        normalized_text = normalize_identifier(record.text)
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        selection_reason = selection_reason_from_features(
            record=record,
            feature_breakdown=feature_breakdown,
            source_hints_used=source_hints_used,
        )
        selection_reasons_by_context_id[context_id] = selection_reason
        results.append(
            RetrievalResult(
                context_record=record,
                score=float(score),
                rank=len(results) + 1,
                retrieval_mode=retrieval_mode,
                component_scores={
                    "bm25": float(lexical_scores.get(context_id, 0.0)),
                    "dense": float(dense_scores.get(context_id, 0.0)),
                    "lexical_weight": lexical_weight,
                    "dense_weight": dense_weight,
                    "metadata_boost": feature_breakdown["metadata_boost"],
                    "rerank_boost": feature_breakdown["base_rerank_boost"],
                    "reranking_used": 1.0,
                    "rerank_score": feature_breakdown["rerank_score"],
                    "company_ticker_match": feature_breakdown["company_ticker_match"],
                    "metric_match": feature_breakdown["metric_match"],
                    "period_match": feature_breakdown["period_match"],
                    "section_match": feature_breakdown["section_match"],
                    "source_hint_match": feature_breakdown["source_hint_match"],
                    "retail_product_match": feature_breakdown.get("retail_product_match", 0.0),
                    "retail_category_match": feature_breakdown.get("retail_category_match", 0.0),
                    "retail_issue_match": feature_breakdown.get("retail_issue_match", 0.0),
                    "retail_policy_match": feature_breakdown.get("retail_policy_match", 0.0),
                    "retail_kind_match": feature_breakdown.get("retail_kind_match", 0.0),
                },
            )
        )
        if len(results) >= final_top_k:
            break
    if len(results) < final_top_k:
        selected_ids = {result.context_record.context_id for result in results}
        for context_id, score, feature_breakdown in ranked:
            if len(results) >= final_top_k:
                break
            if context_id in selected_ids or score <= 0:
                continue
            record = records_by_id[context_id]
            normalized_text = normalize_identifier(record.text)
            if normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)
            selected_ids.add(context_id)
            selection_reason = selection_reason_from_features(
                record=record,
                feature_breakdown=feature_breakdown,
                source_hints_used=source_hints_used,
            )
            selection_reasons_by_context_id[context_id] = selection_reason
            results.append(
                RetrievalResult(
                    context_record=record,
                    score=float(score),
                    rank=len(results) + 1,
                    retrieval_mode=retrieval_mode,
                    component_scores={
                        "bm25": float(lexical_scores.get(context_id, 0.0)),
                        "dense": float(dense_scores.get(context_id, 0.0)),
                        "lexical_weight": lexical_weight,
                        "dense_weight": dense_weight,
                        "metadata_boost": feature_breakdown["metadata_boost"],
                        "rerank_boost": feature_breakdown["base_rerank_boost"],
                        "reranking_used": 1.0,
                        "rerank_score": feature_breakdown["rerank_score"],
                        "company_ticker_match": feature_breakdown["company_ticker_match"],
                        "metric_match": feature_breakdown["metric_match"],
                        "period_match": feature_breakdown["period_match"],
                        "section_match": feature_breakdown["section_match"],
                        "source_hint_match": feature_breakdown["source_hint_match"],
                        "retail_product_match": feature_breakdown.get(
                            "retail_product_match",
                            0.0,
                        ),
                        "retail_category_match": feature_breakdown.get(
                            "retail_category_match",
                            0.0,
                        ),
                        "retail_issue_match": feature_breakdown.get(
                            "retail_issue_match",
                            0.0,
                        ),
                        "retail_policy_match": feature_breakdown.get(
                            "retail_policy_match",
                            0.0,
                        ),
                        "retail_kind_match": feature_breakdown.get("retail_kind_match", 0.0),
                    },
                )
            )

    candidate_context_ids = [
        context_id
        for context_id, _score, _breakdown in ranked[
            : max(candidate_top_k_dense, candidate_top_k_lexical, final_top_k)
        ]
    ]
    diagnostics = {
        "candidate_top_k_dense": candidate_top_k_dense,
        "candidate_top_k_lexical": candidate_top_k_lexical,
        "final_top_k": final_top_k,
        "reranker_enabled": True,
        "candidates_before_dedupe": len(candidate_results),
        "candidates_after_dedupe": len(records_by_id),
        "candidate_context_ids": candidate_context_ids,
        "pre_rerank_top_context_ids": pre_rerank_order[:final_top_k],
        "expanded_query_count": expanded_query_count,
        "expansion_types": list(expansion_types),
        "source_hints_used": source_hints_used,
        "reranked": True,
        "reranker_backend": "calibrated_linear",
        "calibrated_reranker_enabled": True,
        "evidence_selector_strategy": selector_strategy,
        "selection_reasons_by_context_id": selection_reasons_by_context_id,
        "oracle_strategy_available_for_diagnostics_only": True,
    }
    return TimedRetrieval(
        results=results,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        backend_label=backend_label,
        retrieval_type=retrieval_mode,
        vector_store=vector_store,
        diagnostics=diagnostics,
    )


def retrieval_record_payload(result: RetrievalResult) -> dict[str, Any]:
    """Return a compact JSON-safe representation of one retrieval result."""

    record = result.context_record
    return {
        "context_id": record.context_id,
        "chunk_id": record.chunk_id,
        "source_id": record.source_id,
        "parent_id": record.parent_id,
        "title": record.title,
        "score": round(result.score, 6),
        "rank": result.rank,
        "retrieval_mode": result.retrieval_mode,
        "component_scores": {
            key: round(value, 6) for key, value in result.component_scores.items()
        },
    }


class BM25Retriever:
    """Lightweight BM25-style lexical retriever."""

    backend_label = "local_bm25"

    def __init__(self, records: list[ContextRecord], k1: float = 1.5, b: float = 0.75) -> None:
        self.records = records
        self.k1 = k1
        self.b = b
        self._cache: dict[tuple[str, int], TimedRetrieval] = {}
        self.term_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: list[int] = []
        for index, record in enumerate(records):
            terms = tokenize(record_search_text(record))
            counts = Counter(terms)
            self.doc_lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.term_postings[term].append((index, count))
        self.average_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

    def retrieve(self, query: str, top_k: int) -> TimedRetrieval:
        """Retrieve top-k contexts with BM25 scoring."""

        cache_key = (normalize_query_for_retrieval(query).lower(), top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]
        started = time.perf_counter()
        query_terms = Counter(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        total_docs = len(self.records)
        for term, query_count in query_terms.items():
            postings = self.term_postings.get(term, [])
            if not postings:
                continue
            document_frequency = len(postings)
            idf = math.log(1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
            for doc_index, term_frequency in postings:
                doc_length = self.doc_lengths[doc_index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.average_doc_length, 1.0)
                )
                scores[doc_index] += (
                    query_count * idf * (term_frequency * (self.k1 + 1)) / denominator
                )
        ranked = sorted(
            scores.items(), key=lambda item: (-item[1], self.records[item[0]].context_id)
        )
        results = [
            RetrievalResult(
                context_record=self.records[doc_index],
                score=float(score),
                rank=rank,
                retrieval_mode="bm25",
                component_scores={"bm25": float(score)},
            )
            for rank, (doc_index, score) in enumerate(ranked[:top_k], start=1)
            if score > 0
        ]
        retrieval = TimedRetrieval(
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_label=self.backend_label,
            retrieval_type="bm25",
        )
        self._cache[cache_key] = retrieval
        return retrieval


class LocalFallbackDenseRetriever:
    """Deterministic local fallback for the dense-retrieval interface."""

    backend_label = "local_fallback"
    vector_store = "none"

    def __init__(self, records: list[ContextRecord]) -> None:
        self.records = records
        self.feature_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.doc_norms: list[float] = []
        for index, record in enumerate(records):
            features = self._features(record_search_text(record))
            norm = math.sqrt(sum(value * value for value in features.values()))
            self.doc_norms.append(norm)
            for feature, value in features.items():
                self.feature_postings[feature].append((index, value))

    def _features(self, text: str) -> Counter[str]:
        terms = tokenize(text)
        features: Counter[str] = Counter()
        features.update(f"tok:{term}" for term in terms)
        features.update(f"bi:{terms[index]}_{terms[index + 1]}" for index in range(len(terms) - 1))
        return features

    def retrieve(self, query: str, top_k: int) -> TimedRetrieval:
        """Retrieve top-k contexts with deterministic sparse cosine scoring."""

        started = time.perf_counter()
        query_features = self._features(query)
        query_norm = math.sqrt(sum(value * value for value in query_features.values()))
        scores: dict[int, float] = defaultdict(float)
        if query_norm > 0:
            for feature, query_value in query_features.items():
                for doc_index, doc_value in self.feature_postings.get(feature, []):
                    scores[doc_index] += query_value * doc_value
            for doc_index in list(scores):
                denominator = query_norm * max(self.doc_norms[doc_index], 1e-9)
                scores[doc_index] = scores[doc_index] / denominator
        ranked = sorted(
            scores.items(), key=lambda item: (-item[1], self.records[item[0]].context_id)
        )
        results = [
            RetrievalResult(
                context_record=self.records[doc_index],
                score=float(score),
                rank=rank,
                retrieval_mode="dense",
                component_scores={"dense": float(score)},
            )
            for rank, (doc_index, score) in enumerate(ranked[:top_k], start=1)
            if score > 0
        ]
        return TimedRetrieval(
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_label=self.backend_label,
            retrieval_type="dense",
            vector_store=self.vector_store,
        )


class QdrantDenseRetriever:
    """Dense retriever backed by a local Qdrant vector collection."""

    backend_label = "qdrant_vector"
    vector_store = "qdrant_local"

    def __init__(self, searcher: Any) -> None:
        self.searcher = searcher

    def retrieve(self, query: str, top_k: int) -> TimedRetrieval:
        """Retrieve top-k contexts using local Qdrant vector search."""

        started = time.perf_counter()
        search_results = self.searcher.retrieve(query, top_k)
        results = [
            RetrievalResult(
                context_record=result.context_record,
                score=result.score,
                rank=rank,
                retrieval_mode="dense",
                component_scores={"dense": result.score, "qdrant": result.score},
            )
            for rank, result in enumerate(search_results, start=1)
            if result.score > 0
        ]
        return TimedRetrieval(
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_label=self.backend_label,
            retrieval_type="dense",
            vector_store=self.vector_store,
        )


class HybridRetriever:
    """Hybrid retriever with score fusion over BM25, dense retrieval, and boosts."""

    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        dense_retriever: DenseRetrieverProtocol,
        lexical_weight: float = 0.55,
        dense_weight: float = 0.45,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.boost_features = {
            record.context_id: build_boost_features(record) for record in lexical_retriever.records
        }

    @property
    def backend_label(self) -> str:
        """Return the dense backend status for report compatibility."""

        return self.dense_retriever.backend_label

    @property
    def vector_store(self) -> str:
        """Return the vector store label used by the dense component."""

        return self.dense_retriever.vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        *,
        expanded_queries: tuple[str, ...] | None = None,
        candidate_top_k_dense: int = DEFAULT_CANDIDATE_TOP_K_DENSE,
        candidate_top_k_lexical: int = DEFAULT_CANDIDATE_TOP_K_LEXICAL,
        source_hints_used: bool = False,
        expansion_types: tuple[str, ...] = (),
    ) -> TimedRetrieval:
        """Retrieve top-k contexts with weighted score fusion."""

        started = time.perf_counter()
        active_queries = expanded_queries or (query,)
        candidate_results: list[RetrievalResult] = []
        for expanded_query in active_queries:
            lexical = self.lexical_retriever.retrieve(expanded_query, candidate_top_k_lexical)
            candidate_results.extend(lexical.results)
        dense = self.dense_retriever.retrieve(query, candidate_top_k_dense)
        candidate_results.extend(dense.results)
        return rerank_candidate_results(
            query=query,
            candidate_results=candidate_results,
            final_top_k=top_k,
            retrieval_mode="hybrid",
            lexical_weight=self.lexical_weight,
            dense_weight=self.dense_weight,
            source_hints_used=source_hints_used,
            candidate_top_k_dense=candidate_top_k_dense,
            candidate_top_k_lexical=candidate_top_k_lexical,
            expanded_query_count=len(active_queries),
            expansion_types=expansion_types,
            started=started,
            backend_label=self.backend_label,
            vector_store=self.vector_store,
            boost_features_by_context_id=self.boost_features,
        )


def compress_retrieval_results(
    results: list[RetrievalResult],
    max_context_tokens: int,
    minimum_score_ratio: float = 0.02,
    target_token_ratio: float = 0.72,
) -> CompressionResult:
    """Deterministically deduplicate and compress retrieved context."""

    original_token_count = sum(result.context_record.token_estimate for result in results)
    max_score = max((result.score for result in results), default=0.0)
    seen: set[str] = set()
    selected: list[RetrievalResult] = []
    dropped: list[str] = []
    running_tokens = 0

    for result in sorted(results, key=lambda item: (-item.score, item.context_record.context_id)):
        record = result.context_record
        dedupe_key = normalize_identifier(record.text)
        is_low_score = max_score > 0 and result.score < max_score * minimum_score_ratio
        if dedupe_key in seen or is_low_score:
            dropped.append(record.context_id)
            continue
        compressed_record = compress_context_record_text(record, target_token_ratio)
        if running_tokens + compressed_record.token_estimate > max_context_tokens:
            dropped.append(record.context_id)
            continue
        selected.append(
            RetrievalResult(
                context_record=compressed_record,
                score=result.score,
                rank=len(selected) + 1,
                retrieval_mode=result.retrieval_mode,
                component_scores=result.component_scores,
            )
        )
        seen.add(dedupe_key)
        running_tokens += compressed_record.token_estimate

    if not selected and results and max_context_tokens > 0:
        best = max(results, key=lambda item: item.score)
        compressed_record = compress_context_record_text(best.context_record, target_token_ratio)
        selected = [
            RetrievalResult(
                context_record=compressed_record,
                score=best.score,
                rank=1,
                retrieval_mode=best.retrieval_mode,
                component_scores=best.component_scores,
            )
        ]
        running_tokens = min(compressed_record.token_estimate, max_context_tokens)
        dropped = [result.context_record.context_id for result in results if result is not best]

    compression_ratio = running_tokens / original_token_count if original_token_count > 0 else 0.0
    return CompressionResult(
        results=selected,
        original_token_count=original_token_count,
        compressed_token_count=running_tokens,
        token_reduction=max(0, original_token_count - running_tokens),
        compression_ratio=round(compression_ratio, 6),
        dropped_context_ids=dropped,
    )


def compress_context_record_text(record: ContextRecord, target_token_ratio: float) -> ContextRecord:
    """Return a context record with deterministic extractive text compression."""

    words = record.text.split()
    if not words:
        return record
    target_tokens = max(24, int(len(words) * target_token_ratio))
    target_tokens = min(target_tokens, len(words))
    if target_tokens >= len(words):
        return record

    head_count = max(12, int(target_tokens * 0.72))
    tail_count = max(0, target_tokens - head_count)
    if tail_count:
        compressed_words = words[:head_count] + words[-tail_count:]
    else:
        compressed_words = words[:head_count]
    compressed_text = " ".join(compressed_words)
    metadata = dict(record.metadata)
    metadata["compression"] = {
        "type": "deterministic_extractive_truncation",
        "original_token_estimate": record.token_estimate,
        "compressed_token_estimate": len(compressed_words),
    }
    return replace(
        record,
        text=compressed_text,
        metadata=metadata,
        token_estimate=len(compressed_words),
    )


def evaluate_retrieval_results(
    *,
    gold_evidence_ids: list[str],
    results: list[RetrievalResult],
) -> dict[str, Any]:
    """Evaluate retrieval against gold evidence identifiers."""

    required = [evidence_id for evidence_id in dict.fromkeys(gold_evidence_ids) if evidence_id]
    if not required:
        return {
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "gold_evidence_included": False,
            "missing_gold_evidence_count": 0,
            "matched_gold_evidence_ids": [],
        }

    matched: set[str] = set()
    reciprocal_rank = 0.0
    for result in results:
        match_ids = context_match_ids(result.context_record)
        current_matches = {
            evidence_id
            for evidence_id in required
            if evidence_id in match_ids or normalize_identifier(evidence_id) in match_ids
        }
        if current_matches and reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / result.rank
        matched.update(current_matches)

    return {
        "recall_at_5": len(matched) / len(required),
        "mrr": reciprocal_rank,
        "gold_evidence_included": bool(matched),
        "missing_gold_evidence_count": len(required) - len(matched),
        "matched_gold_evidence_ids": sorted(matched),
    }
