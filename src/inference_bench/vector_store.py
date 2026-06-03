"""Local Qdrant vector store utilities for Phase 3 retrieval.

This module builds and queries a local embedded Qdrant store. It does not call
LLM APIs, run inference, or require an external Qdrant server.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from inference_bench.config import VectorStoreConfig, resolve_vector_store
from inference_bench.context_corpora import VERTICALS, read_jsonl
from inference_bench.context_schema import ContextRecord
from inference_bench.retrieval import normalize_query_for_retrieval, split_identifier_text

QDRANT_BUILD_COMMAND = (
    "python scripts/phase3/build_qdrant_index.py --context-root "
    "data/generated/context_engineering --output-root data/generated/context_engineering "
    "--vector-store-config configs/vector_stores.yaml"
)


class EmbeddingProvider(Protocol):
    """Small embedding interface used by local Qdrant indexing and search."""

    backend_label: str
    model_name: str
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for each input text."""


@dataclass(frozen=True)
class VectorSearchResult:
    """One Qdrant vector search result."""

    context_record: ContextRecord
    score: float


@dataclass(frozen=True)
class QdrantIndexBuildResult:
    """Qdrant index reports."""

    report: dict[str, Any]
    summary_rows: list[dict[str, Any]]


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object to disk."""

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


def load_context_corpora(context_root: str | Path) -> dict[str, list[ContextRecord]]:
    """Load generated context corpora from a context-engineering root."""

    root = Path(context_root)
    corpora: dict[str, list[ContextRecord]] = {}
    missing_paths: list[Path] = []
    for vertical in VERTICALS:
        path = root / "corpora" / f"{vertical}_context_corpus.jsonl"
        if not path.exists():
            missing_paths.append(path)
            continue
        corpora[vertical] = [ContextRecord(**row) for row in read_jsonl(path)]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        msg = (
            "Missing context corpora required for Qdrant indexing:\n"
            f"{missing}\nRegenerate them with:\n"
            "python scripts/phase3/build_context_corpora.py --dataset-root "
            "data/scaleup_2000_full --output-root data/generated/context_engineering"
        )
        raise RuntimeError(msg)
    return corpora


class SentenceTransformerEmbeddingProvider:
    """Local sentence-transformers embedding provider."""

    def __init__(self, model_name: str, batch_size: int) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = max(batch_size, 256)
        self.model = SentenceTransformer(model_name, local_files_only=True)
        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        self.dimension = int(dimension or 0)
        self.backend_label = "sentence_transformers"

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with a locally cached sentence-transformers model."""

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], vectors.astype(float).tolist())


class HashingEmbeddingProvider:
    """Deterministic offline embedding fallback for tests and uncached models."""

    backend_label = "local_hashing_fallback"

    def __init__(self, model_name: str = "local_hashing_fallback", dimension: int = 384) -> None:
        self.model_name = model_name
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts with a deterministic signed hashing vector."""

        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def build_embedding_provider(
    config: VectorStoreConfig,
    *,
    allow_hash_fallback: bool = True,
) -> EmbeddingProvider:
    """Build the configured embedding provider."""

    if config.embedding_backend == "sentence_transformers":
        try:
            return SentenceTransformerEmbeddingProvider(
                model_name=config.embedding_model,
                batch_size=config.batch_size,
            )
        except Exception as exc:
            if not allow_hash_fallback:
                msg = (
                    f"Could not load local sentence-transformers model "
                    f"'{config.embedding_model}'. Install/cache it or rerun with a supported "
                    f"offline embedding backend. Original error: {exc}"
                )
                raise RuntimeError(msg) from exc
            return HashingEmbeddingProvider()

    if config.embedding_backend == "deterministic_hash":
        return HashingEmbeddingProvider(model_name=config.embedding_model)

    msg = f"Unsupported embedding backend '{config.embedding_backend}'"
    raise ValueError(msg)


def qdrant_collection_name(config: VectorStoreConfig, vertical: str) -> str:
    """Return the collection name for one vertical."""

    return f"{config.collection_prefix}_{vertical}"


def point_id_for_context(context_id: str) -> str:
    """Return a deterministic Qdrant point UUID for a context ID."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, context_id))


def qdrant_distance(distance: str) -> Any:
    """Return a Qdrant distance enum."""

    from qdrant_client import models

    if distance.lower() == "cosine":
        return models.Distance.COSINE
    msg = f"Unsupported Qdrant distance '{distance}'"
    raise ValueError(msg)


def build_qdrant_client(config: VectorStoreConfig) -> Any:
    """Create a local embedded Qdrant client."""

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        msg = "qdrant-client is required for Qdrant vector retrieval"
        raise RuntimeError(msg) from exc

    storage_path = Path(config.storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def context_payload(record: ContextRecord) -> dict[str, Any]:
    """Return payload stored in Qdrant for one context record."""

    return {
        "context_record": asdict(record),
        "context_id": record.context_id,
        "vertical": record.vertical,
        "source_id": record.source_id,
        "parent_id": record.parent_id,
        "chunk_id": record.chunk_id,
        "chunk_strategy": record.chunk_strategy,
        "source_type": record.source_type,
        "title": record.title,
        "indexed_text": vector_text(record),
        "indexed_text_strategy": "title_text_selected_metadata_finance_concepts_v3",
        "metadata": record.metadata,
    }


def vector_text(record: ContextRecord) -> str:
    """Return the text embedded into Qdrant."""

    metadata = record.metadata
    selected_metadata_keys = (
        "ticker",
        "company_name",
        "company",
        "form",
        "filing_date",
        "report_date",
        "period",
        "fiscal_year",
        "fiscal_periods_present",
        "fiscal_years_present",
        "forms_present",
        "latest_end",
        "latest_filed",
        "concept",
        "concepts",
        "label",
        "record_id",
        "section_type",
        "section_title",
        "category",
        "product_title",
        "rating",
        "paper_id",
        "title",
        "topic",
        "topics",
        "evidence_type",
    )
    selected_metadata_values = [
        str(metadata[key]) for key in selected_metadata_keys if metadata.get(key) is not None
    ]
    concept_values = [
        str(metadata[key])
        for key in ("concept", "concepts", "label")
        if metadata.get(key) is not None
    ]
    period_values = [
        str(metadata[key])
        for key in (
            "filing_date",
            "report_date",
            "fiscal_periods_present",
            "fiscal_years_present",
            "forms_present",
            "latest_end",
            "latest_filed",
        )
        if metadata.get(key) is not None
    ]
    metadata_values = " ".join(str(value) for value in flatten_metadata(metadata))
    return " ".join(
        part
        for part in (
            f"title: {record.title}",
            f"vertical: {record.vertical}",
            f"source type: {record.source_type}",
            f"chunk strategy: {record.chunk_strategy}",
            f"selected metadata: {' '.join(selected_metadata_values)}",
            split_identifier_text(" ".join(selected_metadata_values)),
            f"finance concepts: {' '.join(concept_values)}",
            split_identifier_text(" ".join(concept_values)),
            f"finance periods: {' '.join(period_values)}",
            record.title,
            record.source_type,
            record.chunk_strategy,
            metadata_values,
            f"text: {record.text}",
        )
        if part
    )


def flatten_metadata(value: Any) -> Iterable[str]:
    """Yield scalar metadata values as strings."""

    if isinstance(value, dict):
        for nested_value in value.values():
            yield from flatten_metadata(nested_value)
        return
    if isinstance(value, list):
        for nested_value in value:
            yield from flatten_metadata(nested_value)
        return
    if value is not None:
        yield str(value)


def batched(records: list[ContextRecord], batch_size: int) -> Iterable[list[ContextRecord]]:
    """Yield records in fixed-size batches."""

    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def batched_strings(values: list[str], batch_size: int) -> Iterable[list[str]]:
    """Yield strings in fixed-size batches."""

    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def compact_query_for_vector(query: str, max_terms: int = 160) -> str:
    """Return a compact query string for vector embedding."""

    normalized = normalize_query_for_retrieval(query).lower()
    compact_terms: list[str] = []
    term_counts: dict[str, int] = {}
    for term in normalized.split():
        count = term_counts.get(term, 0)
        if count >= 2:
            continue
        compact_terms.append(term)
        term_counts[term] = count + 1
        if len(compact_terms) >= max_terms:
            break
    return " ".join(compact_terms)


def build_qdrant_index(
    *,
    context_root: str | Path,
    output_root: str | Path,
    vector_store_config_path: str | Path = "configs/vector_stores.yaml",
    vector_store_key: str = "qdrant_local",
) -> QdrantIndexBuildResult:
    """Build local Qdrant collections from normalized context corpora."""

    config = resolve_vector_store(vector_store_key, vector_store_config_path)
    corpora_by_vertical = load_context_corpora(context_root)
    embedding_provider = build_embedding_provider(config)
    client = build_qdrant_client(config)
    from qdrant_client import models

    summary_rows: list[dict[str, Any]] = []
    collections: dict[str, Any] = {}
    payload_fields = [
        "context_record",
        "context_id",
        "vertical",
        "source_id",
        "parent_id",
        "chunk_id",
        "chunk_strategy",
        "source_type",
        "title",
        "indexed_text",
        "indexed_text_strategy",
        "metadata",
    ]

    try:
        for vertical in VERTICALS:
            records = corpora_by_vertical.get(vertical, [])
            collection_name = qdrant_collection_name(config, vertical)
            started = time.perf_counter()
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_provider.dimension,
                    distance=qdrant_distance(config.distance),
                ),
            )

            indexed_count = 0
            failed_records: list[str] = []
            for batch in batched(records, config.batch_size):
                try:
                    vectors = embedding_provider.encode([vector_text(record) for record in batch])
                    points = [
                        models.PointStruct(
                            id=point_id_for_context(record.context_id),
                            vector=vector,
                            payload=context_payload(record),
                        )
                        for record, vector in zip(batch, vectors, strict=True)
                    ]
                    client.upsert(collection_name=collection_name, points=points, wait=True)
                    indexed_count += len(points)
                except Exception:
                    failed_records.extend(record.context_id for record in batch)

            elapsed = time.perf_counter() - started
            payload = {
                "collection_name": collection_name,
                "vertical": vertical,
                "indexed_chunks": indexed_count,
                "embedding_backend_configured": config.embedding_backend,
                "embedding_backend_effective": embedding_provider.backend_label,
                "embedding_model": embedding_provider.model_name,
                "vector_dimension": embedding_provider.dimension,
                "distance": config.distance,
                "payload_fields_stored": payload_fields,
                "indexed_text_strategy": "title_text_selected_metadata_finance_concepts_v3",
                "indexing_time_seconds": round(elapsed, 6),
                "skipped_records": len(failed_records),
                "failed_record_ids": failed_records[:25],
            }
            collections[vertical] = payload
            summary_rows.append(payload)
    finally:
        client.close()

    report = {
        "generated_at_utc": utc_now(),
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
        "vector_store": vector_store_key,
        "provider": config.provider,
        "mode": config.mode,
        "storage_path": config.storage_path,
        "collection_prefix": config.collection_prefix,
        "embedding_backend_configured": config.embedding_backend,
        "embedding_backend_effective": embedding_provider.backend_label,
        "embedding_model": embedding_provider.model_name,
        "vector_dimension": embedding_provider.dimension,
        "distance": config.distance,
        "indexed_text_strategy": "title_text_selected_metadata_finance_concepts_v3",
        "collections": collections,
    }
    output_path = Path(output_root)
    write_json(output_path / "qdrant_index_report.json", report)
    write_csv(
        output_path / "qdrant_index_summary.csv",
        summary_rows,
        [
            "collection_name",
            "vertical",
            "indexed_chunks",
            "embedding_backend_configured",
            "embedding_backend_effective",
            "embedding_model",
            "vector_dimension",
            "distance",
            "payload_fields_stored",
            "indexed_text_strategy",
            "indexing_time_seconds",
            "skipped_records",
            "failed_record_ids",
        ],
    )
    return QdrantIndexBuildResult(report=report, summary_rows=summary_rows)


class QdrantVectorSearcher:
    """Query local Qdrant collections and return context records."""

    vector_store_label = "qdrant_local"

    def __init__(
        self,
        *,
        config: VectorStoreConfig,
        vertical: str,
        embedding_provider: EmbeddingProvider | None = None,
        client: Any | None = None,
        records_by_id: dict[str, ContextRecord] | None = None,
    ) -> None:
        self.config = config
        self.vertical = vertical
        self.collection_name = qdrant_collection_name(config, vertical)
        self.embedding_provider = embedding_provider or build_embedding_provider(config)
        self.client = client or build_qdrant_client(config)
        self.records_by_id = records_by_id or {}
        self._owns_client = client is None
        self._embedding_cache: dict[str, list[float]] = {}
        self._search_cache: dict[str, list[VectorSearchResult]] = {}
        self._snapshot_matrix: Any | None = None
        self._snapshot_records: list[ContextRecord] = []
        if not self.client.collection_exists(self.collection_name):
            if self._owns_client:
                self.client.close()
            msg = (
                f"Qdrant collection '{self.collection_name}' is missing. Build the local "
                f"index first with:\n{QDRANT_BUILD_COMMAND}"
            )
            raise RuntimeError(msg)

    def close(self) -> None:
        """Close the local Qdrant client."""

        if self._owns_client:
            self.client.close()

    def retrieve(self, query: str, top_k: int) -> list[VectorSearchResult]:
        """Return top-k vector results for one query."""

        normalized_query = compact_query_for_vector(query)
        cached = self._search_cache.get(normalized_query)
        if cached is not None and len(cached) >= top_k:
            return cached[:top_k]

        if self._snapshot_matrix is not None:
            results = self._retrieve_from_snapshot(normalized_query, top_k)
            self._search_cache[normalized_query] = results
            return results[:top_k]

        query_vector = self._embedding_cache.get(normalized_query)
        if query_vector is None:
            query_vector = self.embedding_provider.encode([normalized_query])[0]
            self._embedding_cache[normalized_query] = query_vector
        limit = max(top_k, 120)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = self._points_to_results(response.points)
        self._search_cache[normalized_query] = results
        return results[:top_k]

    def _points_to_results(self, points: list[Any]) -> list[VectorSearchResult]:
        """Convert Qdrant points to vector search results."""

        results: list[VectorSearchResult] = []
        for point in points:
            payload = point.payload or {}
            context_id = str(payload.get("context_id") or "")
            if context_id in self.records_by_id:
                results.append(
                    VectorSearchResult(
                        context_record=self.records_by_id[context_id],
                        score=float(point.score),
                    )
                )
                continue
            context_payload_value = payload.get("context_record")
            if not isinstance(context_payload_value, dict):
                continue
            results.append(
                VectorSearchResult(
                    context_record=ContextRecord(**context_payload_value),
                    score=float(point.score),
                )
            )
        return results

    def warm_query_embeddings(self, queries: list[str]) -> None:
        """Batch-encode query embeddings before a large workload build."""

        normalized_queries = sorted(
            {
                compact_query_for_vector(query)
                for query in queries
                if compact_query_for_vector(query)
            }
        )
        missing = [query for query in normalized_queries if query not in self._embedding_cache]
        for batch in batched_strings(missing, self.config.batch_size):
            vectors = self.embedding_provider.encode(batch)
            for query, vector in zip(batch, vectors, strict=True):
                self._embedding_cache[query] = vector

    def load_vector_snapshot(self) -> None:
        """Load Qdrant collection vectors into memory for high-volume local scoring."""

        if self._snapshot_matrix is not None:
            return
        import numpy as np

        vectors: list[list[float]] = []
        records: list[ContextRecord] = []
        offset: Any | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                vector = point.vector
                if not isinstance(vector, list):
                    continue
                payload = point.payload or {}
                context_id = str(payload.get("context_id") or "")
                if context_id in self.records_by_id:
                    record = self.records_by_id[context_id]
                else:
                    context_payload_value = payload.get("context_record")
                    if not isinstance(context_payload_value, dict):
                        continue
                    record = ContextRecord(**context_payload_value)
                vectors.append([float(value) for value in vector])
                records.append(record)
            if offset is None:
                break
        self._snapshot_matrix = np.asarray(vectors, dtype="float32")
        self._snapshot_records = records

    def _retrieve_from_snapshot(
        self,
        normalized_query: str,
        top_k: int,
    ) -> list[VectorSearchResult]:
        """Score a normalized query against the in-memory Qdrant vector snapshot."""

        import numpy as np

        if self._snapshot_matrix is None or len(self._snapshot_records) == 0:
            return []
        query_vector = self._embedding_cache.get(normalized_query)
        if query_vector is None:
            query_vector = self.embedding_provider.encode([normalized_query])[0]
            self._embedding_cache[normalized_query] = query_vector
        query_array = np.asarray(query_vector, dtype="float32")
        scores = self._snapshot_matrix @ query_array
        limit = min(max(top_k, 120), len(scores))
        if limit <= 0:
            return []
        candidate_indices = np.argpartition(-scores, limit - 1)[:limit]
        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: (
                -float(scores[index]),
                self._snapshot_records[int(index)].context_id,
            ),
        )
        return [
            VectorSearchResult(
                context_record=self._snapshot_records[int(index)],
                score=float(scores[index]),
            )
            for index in ranked_indices
            if float(scores[index]) > 0
        ]

    def warm_search_results(self, queries: list[str], top_k: int = 120) -> None:
        """Batch-populate vector search results for large workload builds."""

        self.warm_query_embeddings(queries)
        normalized_queries = sorted(
            {
                compact_query_for_vector(query)
                for query in queries
                if compact_query_for_vector(query)
            }
        )
        missing = [
            query
            for query in normalized_queries
            if query not in self._search_cache or len(self._search_cache[query]) < top_k
        ]
        if not missing:
            return

        from qdrant_client import models

        limit = max(top_k, 120)
        query_batch_size = max(1, min(max(self.config.batch_size, 256), 512))
        for batch in batched_strings(missing, query_batch_size):
            requests = [
                models.QueryRequest(
                    query=self._embedding_cache[query],
                    limit=limit,
                    with_payload=True,
                    with_vector=False,
                )
                for query in batch
            ]
            responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=requests,
            )
            for query, response in zip(batch, responses, strict=True):
                self._search_cache[query] = self._points_to_results(response.points)

    def warm_snapshot_search_results(self, queries: list[str], top_k: int = 120) -> None:
        """Batch-populate search results from the in-memory Qdrant vector snapshot."""

        self.load_vector_snapshot()
        self.warm_query_embeddings(queries)
        if self._snapshot_matrix is None or len(self._snapshot_records) == 0:
            return
        import numpy as np

        normalized_queries = sorted(
            {
                compact_query_for_vector(query)
                for query in queries
                if compact_query_for_vector(query)
            }
        )
        missing = [
            query
            for query in normalized_queries
            if query not in self._search_cache or len(self._search_cache[query]) < top_k
        ]
        limit = min(max(top_k, 120), len(self._snapshot_records))
        if not missing or limit <= 0:
            return
        query_batch_size = max(1, min(max(self.config.batch_size, 256), 512))
        for batch in batched_strings(missing, query_batch_size):
            query_matrix = np.asarray(
                [self._embedding_cache[query] for query in batch],
                dtype="float32",
            )
            scores_matrix = query_matrix @ self._snapshot_matrix.T
            candidate_indices = np.argpartition(-scores_matrix, limit - 1, axis=1)[:, :limit]
            for query_index, query in enumerate(batch):
                scores = scores_matrix[query_index]
                ranked_indices = sorted(
                    candidate_indices[query_index],
                    key=lambda index: (
                        -float(scores[index]),
                        self._snapshot_records[int(index)].context_id,
                    ),
                )
                self._search_cache[query] = [
                    VectorSearchResult(
                        context_record=self._snapshot_records[int(index)],
                        score=float(scores[index]),
                    )
                    for index in ranked_indices
                    if float(scores[index]) > 0
                ]
