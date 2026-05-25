"""Dependency-light retrieval and compression utilities for Phase 3.

This module does not call model APIs, build dense embedding indexes, or run
inference. The dense retriever is an interface with a deterministic local
fallback so workload generation and tests can run without external services.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from inference_bench.context_schema import ContextRecord

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


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


@dataclass(frozen=True)
class CompressionResult:
    """Compressed context selection metadata."""

    results: list[RetrievalResult]
    original_token_count: int
    compressed_token_count: int
    token_reduction: int
    compression_ratio: float
    dropped_context_ids: list[str]


def tokenize(text: str) -> list[str]:
    """Tokenize text for local retrieval."""

    return [token.lower() for token in TOKEN_RE.findall(text)]


def context_match_ids(record: ContextRecord) -> set[str]:
    """Return identifiers that can match a gold evidence ID."""

    values = {
        record.context_id,
        record.source_id,
        record.parent_id,
        record.chunk_id,
        str(record.metadata.get("original_doc_id") or ""),
    }
    return {value for value in values if value}


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
        return TimedRetrieval(
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_label=self.backend_label,
            retrieval_type="bm25",
        )


class LocalFallbackDenseRetriever:
    """Deterministic local fallback for the dense-retrieval interface."""

    backend_label = "local_fallback"

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
        )


class HybridRetriever:
    """Hybrid retriever with simple score fusion over BM25 and dense fallback."""

    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        dense_retriever: LocalFallbackDenseRetriever,
        lexical_weight: float = 0.55,
        dense_weight: float = 0.45,
    ) -> None:
        self.lexical_retriever = lexical_retriever
        self.dense_retriever = dense_retriever
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight

    @property
    def backend_label(self) -> str:
        """Return the dense backend status for report compatibility."""

        return self.dense_retriever.backend_label

    def retrieve(self, query: str, top_k: int) -> TimedRetrieval:
        """Retrieve top-k contexts with weighted score fusion."""

        started = time.perf_counter()
        candidate_k = max(top_k * 8, 25)
        lexical = self.lexical_retriever.retrieve(query, candidate_k)
        dense = self.dense_retriever.retrieve(query, candidate_k)
        lexical_scores = {
            result.context_record.context_id: result.score for result in lexical.results
        }
        dense_scores = {result.context_record.context_id: result.score for result in dense.results}
        records_by_id = {
            result.context_record.context_id: result.context_record
            for result in lexical.results + dense.results
        }
        max_lexical = max(lexical_scores.values(), default=0.0)
        max_dense = max(dense_scores.values(), default=0.0)

        fused: list[tuple[str, float, float, float]] = []
        for context_id in records_by_id:
            lexical_score = lexical_scores.get(context_id, 0.0)
            dense_score = dense_scores.get(context_id, 0.0)
            lexical_norm = lexical_score / max_lexical if max_lexical > 0 else 0.0
            dense_norm = dense_score / max_dense if max_dense > 0 else 0.0
            fused_score = self.lexical_weight * lexical_norm + self.dense_weight * dense_norm
            fused.append((context_id, fused_score, lexical_score, dense_score))

        ranked = sorted(fused, key=lambda item: (-item[1], item[0]))
        results = [
            RetrievalResult(
                context_record=records_by_id[context_id],
                score=float(fused_score),
                rank=rank,
                retrieval_mode="hybrid",
                component_scores={
                    "bm25": float(lexical_score),
                    "dense": float(dense_score),
                    "lexical_weight": self.lexical_weight,
                    "dense_weight": self.dense_weight,
                },
            )
            for rank, (context_id, fused_score, lexical_score, dense_score) in enumerate(
                ranked[:top_k], start=1
            )
            if fused_score > 0
        ]
        return TimedRetrieval(
            results=results,
            latency_ms=(time.perf_counter() - started) * 1000,
            backend_label=self.backend_label,
            retrieval_type="hybrid",
        )


def compress_retrieval_results(
    results: list[RetrievalResult],
    max_context_tokens: int,
    minimum_score_ratio: float = 0.08,
) -> CompressionResult:
    """Deterministically deduplicate and trim retrieved context."""

    original_token_count = sum(result.context_record.token_estimate for result in results)
    max_score = max((result.score for result in results), default=0.0)
    seen: set[str] = set()
    selected: list[RetrievalResult] = []
    dropped: list[str] = []
    running_tokens = 0

    for result in sorted(results, key=lambda item: (-item.score, item.context_record.context_id)):
        record = result.context_record
        dedupe_key = f"{record.context_id}:{record.text.strip()}"
        is_low_score = max_score > 0 and result.score < max_score * minimum_score_ratio
        if dedupe_key in seen or is_low_score:
            dropped.append(record.context_id)
            continue
        if running_tokens + record.token_estimate > max_context_tokens:
            dropped.append(record.context_id)
            continue
        selected.append(result)
        seen.add(dedupe_key)
        running_tokens += record.token_estimate

    if not selected and results and max_context_tokens > 0:
        best = max(results, key=lambda item: item.score)
        selected = [best]
        running_tokens = min(best.context_record.token_estimate, max_context_tokens)
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
        current_matches = {evidence_id for evidence_id in required if evidence_id in match_ids}
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
