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
        self.batch_size = batch_size
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
        "metadata": record.metadata,
    }


def vector_text(record: ContextRecord) -> str:
    """Return the text embedded into Qdrant."""

    metadata_values = " ".join(str(value) for value in flatten_metadata(record.metadata))
    return " ".join(
        part
        for part in (
            record.title,
            record.source_type,
            record.chunk_strategy,
            metadata_values,
            record.text,
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
    ) -> None:
        self.config = config
        self.vertical = vertical
        self.collection_name = qdrant_collection_name(config, vertical)
        self.embedding_provider = embedding_provider or build_embedding_provider(config)
        self.client = client or build_qdrant_client(config)
        self._owns_client = client is None
        self._embedding_cache: dict[str, list[float]] = {}
        self._search_cache: dict[str, list[VectorSearchResult]] = {}
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

        cached = self._search_cache.get(query)
        if cached is not None and len(cached) >= top_k:
            return cached[:top_k]

        query_vector = self._embedding_cache.get(query)
        if query_vector is None:
            query_vector = self.embedding_provider.encode([query])[0]
            self._embedding_cache[query] = query_vector
        limit = max(top_k, 50)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results: list[VectorSearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            context_payload_value = payload.get("context_record")
            if not isinstance(context_payload_value, dict):
                continue
            results.append(
                VectorSearchResult(
                    context_record=ContextRecord(**context_payload_value),
                    score=float(point.score),
                )
            )
        self._search_cache[query] = results
        return results[:top_k]
