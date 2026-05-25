"""Phase 3 context corpus registry and vertical-specific chunk builders.

This module normalizes committed benchmark KB/evidence records into context
records. It does not implement retrieval, embeddings, inference, or GPU work.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from inference_bench.context_schema import VALID_VERTICALS, ContextRecord

VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
CONTEXT_CORPORA_DIR = "corpora"

REQUIRED_METADATA_FIELDS = {
    "airline": ("base_doc_id", "fictional_airline"),
    "healthcare_admin": ("base_doc_id", "fictional_provider"),
    "retail": ("category", "parent_asin", "product_title", "rating"),
    "finance": ("ticker", "company_name", "form", "filing_date", "concept", "section_type"),
    "research_ai": ("paper_id", "title", "section_type", "section_title"),
}

EXPECTED_RESEARCH_AI_SECTION_TYPES = {
    "abstract",
    "introduction",
    "method",
    "methods",
    "experiments",
    "results",
    "limitations",
    "appendix",
}

REFERENCE_FIELDS = ("required_doc_ids", "required_evidence_ids", "required_chunk_ids")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL objects from disk."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                msg = f"Expected JSON object in {jsonl_path} line {line_number}"
                raise ValueError(msg)
            rows.append(cast(dict[str, Any], loaded))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON object to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_context_jsonl(path: str | Path, records: list[ContextRecord]) -> Path:
    """Write normalized context records as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(asdict(record), ensure_ascii=True, sort_keys=True) for record in records
    )
    output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return output_path


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write the context corpus summary CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "vertical",
        "corpus_id",
        "chunk_strategy",
        "source_rows",
        "context_rows",
        "avg_text_length",
        "token_min",
        "token_mean",
        "token_median",
        "token_p95",
        "token_max",
        "missing_metadata_warning_count",
        "enough_context_rows_for_retrieval_testing",
        "output_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def token_estimate(text: str) -> int:
    """Estimate tokens with a deterministic whitespace approximation."""

    return len(text.split())


def text_length(text: str) -> int:
    """Return character length after trimming outer whitespace."""

    return len(text.strip())


def percentile(values: list[int], percentile_value: float) -> float:
    """Return a simple percentile for sorted integer values."""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def distribution(values: list[int]) -> dict[str, float]:
    """Return a compact distribution summary."""

    if not values:
        return {"min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "mean": round(sum(values) / len(values), 3),
        "median": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": float(max(values)),
    }


def benchmark_paths(dataset_root: str | Path, vertical: str) -> dict[str, Path]:
    """Return the promoted prompt/gold/KB paths for a vertical."""

    root = Path(dataset_root)
    return {
        "prompts": root / vertical / f"{vertical}_prompts_2000.jsonl",
        "gold": root / vertical / f"{vertical}_gold_2000.jsonl",
        "kb": root / vertical / f"{vertical}_kb_2000.jsonl",
    }


def referenced_evidence_ids(gold_rows: list[dict[str, Any]]) -> set[str]:
    """Collect evidence identifiers used by gold/eval records."""

    referenced: set[str] = set()
    for row in gold_rows:
        for field_name in REFERENCE_FIELDS:
            values = row.get(field_name)
            if isinstance(values, list):
                referenced.update(str(value) for value in values if value)
    return referenced


def metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of row metadata when it exists."""

    raw_metadata = row.get("metadata")
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    return {}


def row_doc_id(row: dict[str, Any]) -> str:
    """Return a stable document ID for a KB row."""

    return str(row.get("doc_id") or row.get("id") or row.get("source_id") or "")


def compact_text(value: object) -> str:
    """Return one-line text for generated context chunks."""

    return " ".join(str(value or "").split())


def split_text_windows(text: str, max_tokens: int, overlap_tokens: int = 24) -> list[str]:
    """Split long text into sentence-aware windows with a word-window fallback."""

    normalized = compact_text(text)
    if not normalized:
        return []
    if token_estimate(normalized) <= max_tokens:
        return [normalized]

    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(normalized) if sentence.strip()]
    if len(sentences) <= 1:
        return split_word_windows(normalized, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = token_estimate(sentence)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.extend(
                split_word_windows(sentence, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            )
            continue
        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            overlap_words = " ".join(current).split()[-overlap_tokens:]
            current = [" ".join(overlap_words)] if overlap_words else []
            current_tokens = len(overlap_words)
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def split_word_windows(text: str, max_tokens: int, overlap_tokens: int = 24) -> list[str]:
    """Split text by word windows."""

    words = compact_text(text).split()
    if not words:
        return []
    step = max(1, max_tokens - overlap_tokens)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_tokens]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + max_tokens >= len(words):
            break
    return chunks


def source_provenance(row: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Return the best available provenance string for a context record."""

    return str(
        row.get("provenance_url")
        or metadata.get("provenance_url")
        or row.get("source_id")
        or row.get("source_type")
        or "promoted_benchmark_kb"
    )


def build_context_record(
    *,
    vertical: str,
    row: dict[str, Any],
    chunk_text: str,
    chunk_strategy: str,
    chunk_index: int,
    chunk_count: int,
    referenced_ids: set[str],
    parent_id: str,
    metadata_extra: dict[str, Any] | None = None,
) -> ContextRecord:
    """Build and validate a normalized context record."""

    if vertical not in VALID_VERTICALS:
        msg = f"Unsupported vertical: {vertical}"
        raise ValueError(msg)

    doc_id = row_doc_id(row)
    metadata = metadata_from_row(row)
    if metadata_extra:
        metadata.update(metadata_extra)
    metadata.update(
        {
            "allowed_to_commit": bool(row.get("allowed_to_commit", False)),
            "chunk_count": chunk_count,
            "chunk_index": chunk_index,
            "document_type": row.get("document_type"),
            "original_doc_id": doc_id,
            "source_type": row.get("source_type"),
            "tags": row.get("tags", []),
        }
    )
    chunk_id = doc_id if chunk_count == 1 else f"{doc_id}::chunk_{chunk_index:03d}"
    context_id = f"{vertical}:{chunk_id}"
    return ContextRecord(
        context_id=context_id,
        vertical=vertical,
        source_id=str(row.get("source_id") or row.get("source_type") or "benchmark_kb"),
        parent_id=parent_id or doc_id,
        chunk_id=chunk_id,
        chunk_strategy=chunk_strategy,
        source_type=str(row.get("document_type") or row.get("source_type") or "benchmark_kb"),
        title=str(row.get("title") or doc_id),
        text=compact_text(chunk_text),
        metadata=metadata,
        token_estimate=token_estimate(chunk_text),
        provenance=source_provenance(row, metadata),
        is_gold_linked=doc_id in referenced_ids or chunk_id in referenced_ids,
    )


def build_airline_context_records(
    kb_rows: list[dict[str, Any]],
    referenced_ids: set[str],
) -> list[ContextRecord]:
    """Build airline policy-section context records with recursive fallback."""

    records: list[ContextRecord] = []
    for row in kb_rows:
        metadata = metadata_from_row(row)
        doc_id = row_doc_id(row)
        parent_id = str(metadata.get("base_doc_id") or doc_id)
        chunks = split_text_windows(str(row.get("body") or ""), max_tokens=180)
        strategy = "airline_policy_section_recursive_fallback"
        for index, chunk in enumerate(chunks, start=1):
            records.append(
                build_context_record(
                    vertical="airline",
                    row=row,
                    chunk_text=chunk,
                    chunk_strategy=strategy,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    referenced_ids=referenced_ids,
                    parent_id=parent_id,
                    metadata_extra={
                        "policy_family_id": parent_id,
                        "policy_tags": row.get("tags", []),
                    },
                )
            )
    return records


def build_healthcare_admin_context_records(
    kb_rows: list[dict[str, Any]],
    referenced_ids: set[str],
) -> list[ContextRecord]:
    """Build healthcare admin procedure and safety-boundary context records."""

    records: list[ContextRecord] = []
    for row in kb_rows:
        metadata = metadata_from_row(row)
        doc_id = row_doc_id(row)
        parent_id = str(metadata.get("base_doc_id") or doc_id)
        chunks = split_text_windows(str(row.get("body") or ""), max_tokens=180)
        strategy = "healthcare_admin_procedure_safety_boundary"
        tags = [str(tag).lower() for tag in row.get("tags", []) if tag]
        for index, chunk in enumerate(chunks, start=1):
            records.append(
                build_context_record(
                    vertical="healthcare_admin",
                    row=row,
                    chunk_text=chunk,
                    chunk_strategy=strategy,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    referenced_ids=referenced_ids,
                    parent_id=parent_id,
                    metadata_extra={
                        "admin_procedure_family_id": parent_id,
                        "contains_privacy_boundary": any("privacy" in tag for tag in tags),
                        "contains_identity_boundary": any("identity" in tag for tag in tags),
                        "contains_clinical_boundary": any(
                            tag in {"clinical", "triage", "diagnosis", "treatment"} for tag in tags
                        ),
                        "contains_escalation_boundary": any("escalation" in tag for tag in tags),
                    },
                )
            )
    return records


def build_retail_context_records(
    kb_rows: list[dict[str, Any]],
    referenced_ids: set[str],
) -> list[ContextRecord]:
    """Build retail parent-child product/category/review context records."""

    records: list[ContextRecord] = []
    for row in kb_rows:
        metadata = metadata_from_row(row)
        doc_id = row_doc_id(row)
        parent_id = str(metadata.get("parent_asin") or metadata.get("asin") or doc_id)
        chunks = split_text_windows(str(row.get("body") or ""), max_tokens=180)
        strategy = "retail_parent_child_product_review"
        for index, chunk in enumerate(chunks, start=1):
            records.append(
                build_context_record(
                    vertical="retail",
                    row=row,
                    chunk_text=chunk,
                    chunk_strategy=strategy,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    referenced_ids=referenced_ids,
                    parent_id=parent_id,
                    metadata_extra={
                        "category": metadata.get("category"),
                        "parent_asin": metadata.get("parent_asin"),
                        "product_title": metadata.get("product_title"),
                        "rating": metadata.get("rating"),
                        "average_rating": metadata.get("average_rating"),
                        "issue_terms": metadata.get("issue_terms", []),
                    },
                )
            )
    return records


def build_finance_context_records(
    kb_rows: list[dict[str, Any]],
    referenced_ids: set[str],
) -> list[ContextRecord]:
    """Build SEC/XBRL-aware finance context records."""

    records: list[ContextRecord] = []
    for row in kb_rows:
        metadata = metadata_from_row(row)
        doc_id = row_doc_id(row)
        document_type = str(row.get("document_type") or "")
        parent_id = str(
            metadata.get("document_record_id")
            or metadata.get("section_record_id")
            or metadata.get("record_id")
            or doc_id
        )
        if document_type in {"xbrl_fact_evidence", "xbrl_fact_table", "xbrl_concept_inventory"}:
            chunks = [compact_text(row.get("body"))]
            strategy = "finance_sec_xbrl_atomic_fact"
        elif document_type == "sec_filing_event":
            chunks = [compact_text(row.get("body"))]
            strategy = "finance_sec_filing_event"
        else:
            chunks = split_text_windows(str(row.get("body") or ""), max_tokens=220)
            strategy = "finance_filing_section_sentence_window"

        for index, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                continue
            records.append(
                build_context_record(
                    vertical="finance",
                    row=row,
                    chunk_text=chunk,
                    chunk_strategy=strategy,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    referenced_ids=referenced_ids,
                    parent_id=parent_id,
                    metadata_extra={
                        "ticker": metadata.get("ticker"),
                        "company_name": metadata.get("company_name"),
                        "form": metadata.get("form"),
                        "filing_date": metadata.get("filing_date"),
                        "report_date": metadata.get("report_date"),
                        "concept": metadata.get("concept"),
                        "concepts": metadata.get("concepts", []),
                        "section_type": metadata.get("section_type"),
                        "section_title": metadata.get("section_title"),
                        "accession_number": metadata.get("accession_number"),
                    },
                )
            )
    return records


def build_research_ai_context_records(
    kb_rows: list[dict[str, Any]],
    referenced_ids: set[str],
) -> list[ContextRecord]:
    """Build Research AI paper-section context records."""

    records: list[ContextRecord] = []
    for row in kb_rows:
        metadata = metadata_from_row(row)
        doc_id = row_doc_id(row)
        parent_id = str(metadata.get("paper_id") or metadata.get("text_record_id") or doc_id)
        document_type = str(row.get("document_type") or "")
        if document_type in {"paper_section", "paper_section_evidence", "paper_abstract"}:
            chunks = split_text_windows(str(row.get("body") or ""), max_tokens=220)
            strategy = "research_ai_paper_section_sentence_window"
        else:
            chunks = [compact_text(row.get("body"))]
            strategy = "research_ai_paper_metadata"

        for index, chunk in enumerate(chunks, start=1):
            if not chunk.strip():
                continue
            records.append(
                build_context_record(
                    vertical="research_ai",
                    row=row,
                    chunk_text=chunk,
                    chunk_strategy=strategy,
                    chunk_index=index,
                    chunk_count=len(chunks),
                    referenced_ids=referenced_ids,
                    parent_id=parent_id,
                    metadata_extra={
                        "paper_id": metadata.get("paper_id"),
                        "paper_title": metadata.get("title"),
                        "section_record_id": metadata.get("section_record_id"),
                        "section_title": metadata.get("section_title"),
                        "section_type": metadata.get("section_type"),
                        "topic": metadata.get("topic"),
                        "topics": metadata.get("topics", []),
                        "venue": metadata.get("venue"),
                        "year": metadata.get("year"),
                    },
                )
            )
    return records


CHUNK_BUILDERS: dict[str, Callable[[list[dict[str, Any]], set[str]], list[ContextRecord]]] = {
    "airline": build_airline_context_records,
    "healthcare_admin": build_healthcare_admin_context_records,
    "retail": build_retail_context_records,
    "finance": build_finance_context_records,
    "research_ai": build_research_ai_context_records,
}

CHUNK_STRATEGY_NOTES = {
    "airline": "policy-section chunking plus recursive fallback",
    "healthcare_admin": "admin-procedure chunking plus safety-boundary metadata",
    "retail": "parent-child product/category/review chunking",
    "finance": "SEC/XBRL-aware structured chunking with sentence-window fallback",
    "research_ai": "paper-section chunking with sentence-window fallback",
}


def validate_unique_context_ids(records: list[ContextRecord]) -> bool:
    """Return whether all context IDs are unique."""

    context_ids = [record.context_id for record in records]
    return len(context_ids) == len(set(context_ids))


def metadata_coverage(
    vertical: str,
    records: list[ContextRecord],
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Return metadata coverage counts and explicit warning strings."""

    coverage: dict[str, dict[str, int]] = {}
    warnings: list[str] = []
    fields = REQUIRED_METADATA_FIELDS[vertical]
    total = len(records)
    for field_name in fields:
        present = sum(1 for record in records if record.metadata.get(field_name) not in (None, ""))
        missing = total - present
        coverage[field_name] = {"present": present, "missing": missing, "total": total}
        if missing:
            warnings.append(
                f"{vertical}: metadata field '{field_name}' missing on {missing}/{total} "
                "context records."
            )
    return coverage, warnings


def research_ai_section_metadata_summary(records: list[ContextRecord]) -> dict[str, Any]:
    """Summarize preserved Research AI section metadata."""

    section_type_counts: dict[str, int] = {}
    paper_ids: set[str] = set()
    for record in records:
        section_type = str(record.metadata.get("section_type") or "missing")
        section_type_counts[section_type] = section_type_counts.get(section_type, 0) + 1
        paper_id = record.metadata.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            paper_ids.add(paper_id)
    preserved_expected_types = sorted(
        section_type
        for section_type in section_type_counts
        if section_type in EXPECTED_RESEARCH_AI_SECTION_TYPES
    )
    return {
        "paper_id_count": len(paper_ids),
        "section_type_counts": dict(sorted(section_type_counts.items())),
        "preserved_expected_section_types": preserved_expected_types,
    }


def finance_metadata_summary(records: list[ContextRecord]) -> dict[str, Any]:
    """Summarize Finance metadata coverage for reporting."""

    fields = (
        "ticker",
        "company_name",
        "form",
        "filing_date",
        "concept",
        "concepts",
        "section_type",
    )
    summary: dict[str, Any] = {}
    for field_name in fields:
        present = 0
        values: set[str] = set()
        for record in records:
            value = record.metadata.get(field_name)
            if isinstance(value, list):
                if value:
                    present += 1
                    values.update(str(item) for item in value if item)
            elif value not in (None, ""):
                present += 1
                values.add(str(value))
        summary[field_name] = {
            "present": present,
            "missing": len(records) - present,
            "unique_values_sample": sorted(values)[:20],
        }
    return summary


def context_report_for_vertical(
    *,
    vertical: str,
    paths: dict[str, Path],
    output_path: Path,
    kb_rows: list[dict[str, Any]],
    records: list[ContextRecord],
) -> dict[str, Any]:
    """Build a report payload for one vertical."""

    token_values = [record.token_estimate for record in records]
    text_lengths = [text_length(record.text) for record in records]
    coverage, warnings = metadata_coverage(vertical, records)
    report: dict[str, Any] = {
        "vertical": vertical,
        "corpus_id": f"{vertical}_benchmark_context_corpus",
        "corpus_role": "benchmark_kb",
        "source_files": {key: str(value) for key, value in paths.items()},
        "output_path": str(output_path),
        "chunk_strategy": CHUNK_STRATEGY_NOTES[vertical],
        "chunk_builder": CHUNK_BUILDERS[vertical].__name__,
        "source_row_count": len(kb_rows),
        "context_row_count": len(records),
        "average_text_length": round(sum(text_lengths) / len(text_lengths), 3)
        if text_lengths
        else 0.0,
        "token_estimate_distribution": distribution(token_values),
        "missing_metadata_warnings": warnings,
        "metadata_coverage": coverage,
        "context_ids_unique": validate_unique_context_ids(records),
        "validated_context_records": True,
        "enough_context_rows_for_retrieval_testing": len(records) >= 50,
    }
    if vertical == "finance":
        report["finance_metadata_summary"] = finance_metadata_summary(records)
    if vertical == "research_ai":
        report["research_ai_section_metadata_summary"] = research_ai_section_metadata_summary(
            records
        )
    return report


def registry_entries(dataset_root: str | Path, output_root: str | Path) -> list[dict[str, Any]]:
    """Return corpus registry entries."""

    root = Path(dataset_root)
    output = Path(output_root)
    entries: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        context_output_path = output / CONTEXT_CORPORA_DIR / f"{vertical}_context_corpus.jsonl"
        entries.append(
            {
                "vertical": vertical,
                "corpus_id": f"{vertical}_benchmark_context_corpus",
                "corpus_role": "benchmark_kb",
                "input_path": str(root / vertical / f"{vertical}_kb_2000.jsonl"),
                "output_path": str(context_output_path),
                "chunk_builder": CHUNK_BUILDERS[vertical].__name__,
                "notes": CHUNK_STRATEGY_NOTES[vertical],
            }
        )
        entries.append(
            {
                "vertical": vertical,
                "corpus_id": f"{vertical}_gold_linked_context_corpus",
                "corpus_role": "gold_linked_evidence",
                "input_path": str(root / vertical / f"{vertical}_gold_2000.jsonl"),
                "output_path": str(context_output_path),
                "chunk_builder": CHUNK_BUILDERS[vertical].__name__,
                "notes": (
                    "Gold/eval records provide required evidence IDs used to flag context rows."
                ),
            }
        )

    research_corpus = Path(
        "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_full_sections_corpus.jsonl"
    )
    entries.append(
        {
            "vertical": "research_ai",
            "corpus_id": "research_ai_full_sections_future_retrieval_corpus",
            "corpus_role": "full_retrieval_corpus",
            "input_path": str(research_corpus),
            "output_path": "",
            "chunk_builder": "not_built_in_phase3_block2",
            "notes": (
                "Future retrieval corpus exported during data preparation. Block 2 normalizes the "
                "promoted benchmark KB only."
            ),
        }
    )
    return entries


def build_corpus_registry(dataset_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Build the corpus registry payload."""

    entries = registry_entries(dataset_root, output_root)
    return {
        "generated_at_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "entries": entries,
    }


def build_context_corpora(
    dataset_root: str | Path = "data/scaleup_2000_full",
    output_root: str | Path = "data/generated/context_engineering",
) -> dict[str, Any]:
    """Build registry, normalized context corpora, and reports."""

    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    corpora_dir = output_path / CONTEXT_CORPORA_DIR
    corpora_dir.mkdir(parents=True, exist_ok=True)

    registry = build_corpus_registry(dataset_path, output_path)
    report_by_vertical: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    invalid_context_count = 0

    for vertical in VERTICALS:
        paths = benchmark_paths(dataset_path, vertical)
        kb_rows = read_jsonl(paths["kb"])
        gold_rows = read_jsonl(paths["gold"])
        referenced_ids = referenced_evidence_ids(gold_rows)
        records = CHUNK_BUILDERS[vertical](kb_rows, referenced_ids)
        invalid_context_count += sum(
            1 for record in records if not isinstance(record, ContextRecord)
        )
        if not validate_unique_context_ids(records):
            msg = f"Duplicate context IDs generated for {vertical}"
            raise ValueError(msg)

        vertical_output_path = corpora_dir / f"{vertical}_context_corpus.jsonl"
        write_context_jsonl(vertical_output_path, records)
        vertical_report = context_report_for_vertical(
            vertical=vertical,
            paths=paths,
            output_path=vertical_output_path,
            kb_rows=kb_rows,
            records=records,
        )
        report_by_vertical[vertical] = vertical_report

        token_dist = vertical_report["token_estimate_distribution"]
        summary_rows.append(
            {
                "vertical": vertical,
                "corpus_id": vertical_report["corpus_id"],
                "chunk_strategy": vertical_report["chunk_strategy"],
                "source_rows": vertical_report["source_row_count"],
                "context_rows": vertical_report["context_row_count"],
                "avg_text_length": vertical_report["average_text_length"],
                "token_min": token_dist["min"],
                "token_mean": token_dist["mean"],
                "token_median": token_dist["median"],
                "token_p95": token_dist["p95"],
                "token_max": token_dist["max"],
                "missing_metadata_warning_count": len(vertical_report["missing_metadata_warnings"]),
                "enough_context_rows_for_retrieval_testing": vertical_report[
                    "enough_context_rows_for_retrieval_testing"
                ],
                "output_path": str(vertical_output_path),
            }
        )

    report = {
        "generated_at_utc": utc_now(),
        "dataset_root": str(dataset_path),
        "output_root": str(output_path),
        "no_retrieval_no_embeddings_no_inference": True,
        "registry_path": str(output_path / "corpus_registry.json"),
        "corpora_dir": str(corpora_dir),
        "invalid_context_count": invalid_context_count,
        "all_context_records_validated": invalid_context_count == 0,
        "by_vertical": report_by_vertical,
    }

    write_json(output_path / "corpus_registry.json", registry)
    write_json(output_path / "corpus_build_report.json", report)
    write_summary_csv(output_path / "corpus_build_summary.csv", summary_rows)
    return {
        "registry": registry,
        "report": report,
        "summary_rows": summary_rows,
    }
