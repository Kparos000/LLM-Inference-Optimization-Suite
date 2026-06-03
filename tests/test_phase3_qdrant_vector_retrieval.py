import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from inference_bench.config import resolve_vector_store
from inference_bench.context_corpora import VERTICALS
from inference_bench.context_schema import ContextRecord
from inference_bench.memory_workloads import build_retrievers
from inference_bench.retrieval import HybridRetriever, QdrantDenseRetriever
from inference_bench.vector_store import (
    QDRANT_BUILD_COMMAND,
    QdrantVectorSearcher,
    build_qdrant_index,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def context_record(vertical: str, *, suffix: str = "target") -> ContextRecord:
    text_by_vertical = {
        "airline": "Refund policy context for Canada Air cancellation evidence.",
        "healthcare_admin": "Scheduling staff verify identity and avoid diagnosis.",
        "retail": "Home kitchen towel review says soft absorbent cotton.",
        "finance": "AAPL 10-K revenue metric for fiscal year 2024.",
        "research_ai": "Method section discusses retrieval evaluation and recall.",
    }
    metadata = {
        "ticker": "AAPL" if vertical == "finance" else "",
        "company_name": "Apple Inc." if vertical == "finance" else "",
        "concept": "Revenue" if vertical == "finance" else "",
        "form": "10-K" if vertical == "finance" else "",
        "fiscal_year": "2024" if vertical == "finance" else "",
    }
    return ContextRecord(
        context_id=f"{vertical}_{suffix}",
        vertical=vertical,
        source_id=f"{vertical}_source_{suffix}",
        parent_id=f"{vertical}_parent_{suffix}",
        chunk_id=f"{vertical}_chunk_{suffix}",
        chunk_strategy="test_chunk",
        source_type="fixture",
        title=f"{vertical} fixture",
        text=text_by_vertical[vertical],
        metadata=metadata,
        token_estimate=len(text_by_vertical[vertical].split()),
        provenance="test fixture",
        is_gold_linked=True,
    )


def write_context_corpora(context_root: Path) -> dict[str, list[ContextRecord]]:
    corpora: dict[str, list[ContextRecord]] = {}
    for vertical in VERTICALS:
        records = [context_record(vertical)]
        corpora[vertical] = records
        write_jsonl(
            context_root / "corpora" / f"{vertical}_context_corpus.jsonl",
            [asdict(record) for record in records],
        )
    return corpora


def write_vector_config(path: Path, storage_path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "qdrant_local:",
                "  provider: qdrant",
                "  mode: local",
                f"  storage_path: {storage_path.as_posix()}",
                "  collection_prefix: test_suite",
                "  distance: cosine",
                "  embedding_backend: deterministic_hash",
                "  embedding_model: local_hashing",
                "  batch_size: 4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def qdrant_test_storage_path() -> Path:
    return Path("data/generated/vector_store/test_qdrant") / uuid.uuid4().hex


def test_vector_store_config_loads() -> None:
    config = resolve_vector_store("qdrant_local", "configs/vector_stores.yaml")

    assert config.provider == "qdrant"
    assert config.mode == "local"
    assert config.embedding_backend == "deterministic_hash"


def test_qdrant_index_builder_works_on_small_fixture(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    output_root = tmp_path / "reports"
    config_path = write_vector_config(tmp_path / "vector_stores.yaml", qdrant_test_storage_path())
    write_context_corpora(context_root)

    result = build_qdrant_index(
        context_root=context_root,
        output_root=output_root,
        vector_store_config_path=config_path,
    )

    assert (output_root / "qdrant_index_report.json").exists()
    assert (output_root / "qdrant_index_summary.csv").exists()
    assert result.report["provider"] == "qdrant"
    assert result.report["embedding_backend_effective"] == "local_hashing_fallback"
    assert all(row["indexed_chunks"] == 1 for row in result.summary_rows)


def test_qdrant_retriever_returns_valid_context_records(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = write_vector_config(tmp_path / "vector_stores.yaml", qdrant_test_storage_path())
    write_context_corpora(context_root)
    build_qdrant_index(
        context_root=context_root,
        output_root=tmp_path / "reports",
        vector_store_config_path=config_path,
    )
    config = resolve_vector_store("qdrant_local", config_path)
    searcher = QdrantVectorSearcher(config=config, vertical="finance")
    try:
        result = QdrantDenseRetriever(searcher).retrieve("AAPL revenue fiscal 2024", top_k=1)
    finally:
        searcher.close()

    assert result.backend_label == "qdrant_vector"
    assert result.vector_store == "qdrant_local"
    assert result.results
    assert result.results[0].context_record.vertical == "finance"


def test_dense_backend_label_becomes_qdrant_vector_when_index_exists(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = write_vector_config(tmp_path / "vector_stores.yaml", qdrant_test_storage_path())
    corpora = write_context_corpora(context_root)
    build_qdrant_index(
        context_root=context_root,
        output_root=tmp_path / "reports",
        vector_store_config_path=config_path,
    )

    retrievers = build_retrievers(
        corpora,
        dense_backend="qdrant_vector",
        vector_store_config_path=config_path,
    )
    try:
        result = retrievers["finance"]["dense"].retrieve("AAPL revenue", top_k=1)
    finally:
        retrievers["_qdrant_client"]["client"].close()

    assert result.backend_label == "qdrant_vector"
    assert result.vector_store == "qdrant_local"


def test_retrieval_fails_clearly_when_qdrant_index_required_but_missing(
    tmp_path: Path,
) -> None:
    config_path = write_vector_config(tmp_path / "vector_stores.yaml", qdrant_test_storage_path())
    corpora = {vertical: [context_record(vertical)] for vertical in VERTICALS}

    with pytest.raises(RuntimeError, match="Build the local index first") as exc_info:
        build_retrievers(
            corpora,
            dense_backend="qdrant_vector",
            vector_store_config_path=config_path,
        )

    assert QDRANT_BUILD_COMMAND in str(exc_info.value)


def test_hybrid_retrieval_combines_bm25_and_qdrant_scores(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config_path = write_vector_config(tmp_path / "vector_stores.yaml", qdrant_test_storage_path())
    corpora = write_context_corpora(context_root)
    build_qdrant_index(
        context_root=context_root,
        output_root=tmp_path / "reports",
        vector_store_config_path=config_path,
    )
    retrievers = build_retrievers(
        corpora,
        dense_backend="qdrant_vector",
        vector_store_config_path=config_path,
    )
    try:
        hybrid = retrievers["finance"]["hybrid"]
        assert isinstance(hybrid, HybridRetriever)
        result = hybrid.retrieve("AAPL 10-K revenue metric", top_k=1)
    finally:
        retrievers["_qdrant_client"]["client"].close()

    assert result.backend_label == "qdrant_vector"
    assert result.results
    component_scores = result.results[0].component_scores
    assert "bm25" in component_scores
    assert "dense" in component_scores
    assert "metadata_boost" in component_scores


def test_no_model_inference_or_gpu_api_calls_are_triggered(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    output_root = tmp_path / "reports"
    config_path = write_vector_config(tmp_path / "vector_stores.yaml", qdrant_test_storage_path())
    write_context_corpora(context_root)

    result = build_qdrant_index(
        context_root=context_root,
        output_root=output_root,
        vector_store_config_path=config_path,
    )

    assert result.report["no_model_inference_triggered"] is True
    assert result.report["no_gpu_work_triggered"] is True
    assert result.report["no_external_api_calls_triggered"] is True
