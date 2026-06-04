"""Canonical retrieval query materialization.

Canonical queries make the retrieval path explicit: prompt-visible text is
normalized, enriched with non-leaking retrieval keys, and rendered into the text
used by BM25/Qdrant/hybrid retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from inference_bench.retrieval import (
    CompanyTickerResolver,
    enrich_query_text,
    normalize_identifier,
    scrub_direct_evidence_identifiers,
    tokenize,
)
from inference_bench.retrieval_keys import (
    RetrievalKeys,
    derive_retrieval_keys,
    retrieval_key_terms,
)

QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "using",
    "only",
    "cited",
    "evidence",
    "answer",
    "question",
    "request",
    "scenario",
    "prompt",
    "record",
    "records",
}


@dataclass(frozen=True)
class CanonicalQuery:
    """Rendered retrieval queries and key diagnostics."""

    raw_prompt: str
    normalized_prompt: str
    vertical_enriched_query: str
    compact_keyword_query: str
    qdrant_query: str
    expanded_queries: tuple[str, ...]
    expansion_types: tuple[str, ...]
    retrieval_keys: RetrievalKeys
    blocked_direct_hint_count: int = 0

    @property
    def query_text(self) -> str:
        """Return the canonical query used by hybrid retrieval."""

        return self.qdrant_query


def visible_prompt_text(prompt: dict[str, Any]) -> str:
    """Return prompt-visible natural language text."""

    text = " ".join(
        str(prompt.get(field) or "")
        for field in ("question", "issue", "company", "product_title", "topic")
    )
    scrubbed, _blocked = scrub_direct_evidence_identifiers(text)
    return re.sub(r"\s+", " ", scrubbed).strip()


def compact_keywords(*parts: str, max_terms: int = 48) -> str:
    """Return a compact deterministic keyword query."""

    seen: set[str] = set()
    terms: list[str] = []
    for token in tokenize(" ".join(parts)):
        if token in QUERY_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return " ".join(terms)


def qdrant_enriched_text(
    *,
    raw_prompt: str,
    vertical: str,
    key_terms: list[str],
    compact_keyword_query: str,
) -> str:
    """Return text optimized for Qdrant/BM25 retrieval."""

    vertical_prefix = {
        "airline": "airline policy route customer support",
        "healthcare_admin": "healthcare administrative procedure privacy safety",
        "retail": "retail product review category support issue policy",
        "finance": "SEC filing company ticker financial metric period section XBRL",
        "research_ai": "research paper section evidence citation",
    }.get(vertical, vertical)
    return " ".join(
        part
        for part in (
            raw_prompt,
            vertical_prefix,
            " ".join(key_terms),
            compact_keyword_query,
        )
        if part
    )


def build_canonical_query(
    prompt: dict[str, Any],
    *,
    ablation_mode: str = "prompt_plus_metadata",
    resolver: CompanyTickerResolver | None = None,
    concept_map: dict[str, set[str]] | None = None,
) -> CanonicalQuery:
    """Build canonical non-leaking query forms for a prompt."""

    vertical = str(prompt.get("vertical") or "")
    raw_prompt = visible_prompt_text(prompt)
    normalized_prompt = normalize_identifier(raw_prompt)
    retrieval_keys = derive_retrieval_keys(prompt, ablation_mode=ablation_mode)
    key_terms = retrieval_key_terms(retrieval_keys)
    vertical_enriched_query = " ".join(
        part for part in (normalized_prompt, " ".join(key_terms)) if part
    )
    compact_keyword_query = compact_keywords(raw_prompt, vertical_enriched_query)
    qdrant_query = qdrant_enriched_text(
        raw_prompt=raw_prompt,
        vertical=vertical,
        key_terms=key_terms,
        compact_keyword_query=compact_keyword_query,
    )
    scrubbed_query, blocked = scrub_direct_evidence_identifiers(qdrant_query)
    enriched = enrich_query_text(
        scrubbed_query,
        vertical=vertical,
        allow_direct_identifiers=False,
        resolver=resolver,
        concept_map=concept_map or {},
        metadata_terms=set(tokenize(" ".join(key_terms))),
    )
    return CanonicalQuery(
        raw_prompt=raw_prompt,
        normalized_prompt=normalized_prompt,
        vertical_enriched_query=vertical_enriched_query,
        compact_keyword_query=compact_keyword_query,
        qdrant_query=enriched.query_text,
        expanded_queries=enriched.expanded_queries,
        expansion_types=tuple(dict.fromkeys(("canonical_query", *enriched.expansion_types))),
        retrieval_keys=retrieval_keys,
        blocked_direct_hint_count=(
            retrieval_keys.blocked_direct_hint_count + blocked + enriched.blocked_direct_hint_count
        ),
    )
