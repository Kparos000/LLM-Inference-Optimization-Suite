"""Export the full Research AI section pool as a future retrieval corpus.

This script exports local processed paper sections only. It does not build RAG,
retrieval indexes, embeddings, prompt assembly, model calls, GPU runs, or
inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-16B"
DEFAULT_SECTIONS_MANIFEST = Path("data/processed/research_ai/paper_sections_manifest.jsonl")
DEFAULT_APPROVED_PAPERS = Path("data/sources/research_ai_approved_papers.jsonl")
DEFAULT_BENCHMARK_KB = Path("data/scaleup_2000_full/research_ai/research_ai_kb_2000.jsonl")
DEFAULT_BENCHMARK_GOLD = Path("data/scaleup_2000_full/research_ai/research_ai_gold_2000.jsonl")
DEFAULT_OUTPUT_CORPUS = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_full_sections_corpus.jsonl"
)
DEFAULT_OUTPUT_MANIFEST = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_full_sections_manifest.json"
)
DEFAULT_OUTPUT_MAPPING = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_benchmark_kb_to_source_mapping.jsonl"
)
DEFAULT_OUTPUT_QUALITY_REPORT = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_retrieval_corpus_quality_report.json"
)

ALLOWED_SHORT_SECTION_TYPES = {"abstract", "title", "summary"}
UNUSABLE_SECTION_TYPES = {"references", "bibliography", "reference", "acknowledgements"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(
            f"Missing required Research AI file: {path}. The full retrieval corpus "
            "requires local processed Research AI section artifacts. Run the paper "
            "text/section preparation workflow before exporting the full corpus."
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object in {path} line {line_number}.")
        rows.append(parsed)
    return rows


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def approved_registry_by_paper_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row.get("paper_id") or row.get("id") or "")
        if paper_id:
            registry[paper_id] = row
    return registry


def section_text(section: dict[str, Any]) -> str:
    inline_text = section.get("text") or section.get("section_text") or section.get("body")
    if isinstance(inline_text, str) and inline_text.strip():
        return normalize_whitespace(inline_text)

    text_path_value = section.get("local_text_path")
    if not text_path_value:
        return ""
    text_path = Path(str(text_path_value))
    if not text_path.exists():
        return ""
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    start = int(section.get("section_start_char") or 0)
    end = int(section.get("section_end_char") or len(text))
    if end <= start:
        end = len(text)
    return normalize_whitespace(text[start:end])


def exclusion_reason(section: dict[str, Any], text: str) -> str | None:
    section_type = str(section.get("section_type") or "").lower()
    section_title = str(section.get("section_title") or "").lower()
    word_count = int(section.get("word_count") or len(text.split()))
    if not text.strip():
        return "empty_text"
    if section_type in UNUSABLE_SECTION_TYPES or section_title in UNUSABLE_SECTION_TYPES:
        return "references_or_bibliography"
    if word_count < 20 and section_type not in ALLOWED_SHORT_SECTION_TYPES:
        return "too_short"
    return None


def corpus_row(
    *,
    section: dict[str, Any],
    text: str,
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paper_id = str(section.get("paper_id") or "")
    registry_row = registry.get(paper_id, {})
    section_id = str(section.get("section_record_id") or section.get("section_id") or "")
    section_type = str(section.get("section_type") or "section")
    title = str(section.get("title") or registry_row.get("title") or "")
    venue = str(registry_row.get("venue_or_source") or registry_row.get("venue") or "Research AI")
    return {
        "corpus_id": section_id,
        "paper_id": paper_id,
        "paper_title": title,
        "venue_or_source": venue,
        "publication_year": registry_row.get("publication_year") or registry_row.get("year"),
        "source_url": registry_row.get("source_url")
        or registry_row.get("openreview_url")
        or registry_row.get("arxiv_url")
        or registry_row.get("pdf_url"),
        "section_id": section_id,
        "section_title": str(section.get("section_title") or section_type),
        "section_type": section_type,
        "word_count": int(section.get("word_count") or len(text.split())),
        "char_count": len(text),
        "text": text,
        "metadata": {
            "source_quality": "processed_section",
            "not_benchmark_gold_kb": True,
            "future_phase2b_retrieval_corpus": True,
        },
    }


def referenced_doc_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in ["required_doc_ids", "required_evidence_ids", "source_doc_ids"]:
        value = row.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return list(dict.fromkeys(ids))


def benchmark_mapping_rows(
    *,
    benchmark_kb: list[dict[str, Any]],
    benchmark_gold: list[dict[str, Any]],
    corpus_ids: set[str],
) -> list[dict[str, Any]]:
    referenced_kb_ids: Counter[str] = Counter()
    for gold in benchmark_gold:
        referenced_kb_ids.update(referenced_doc_ids(gold))

    rows: list[dict[str, Any]] = []
    for kb in benchmark_kb:
        metadata = kb.get("metadata") if isinstance(kb.get("metadata"), dict) else {}
        source_section_id = str(
            metadata.get("section_record_id")
            or metadata.get("source_section_id")
            or kb.get("section_record_id")
            or ""
        )
        paper_id = str(metadata.get("paper_id") or kb.get("paper_id") or "")
        benchmark_doc_id = str(kb.get("doc_id") or "")
        mapped = bool(source_section_id and source_section_id in corpus_ids)
        rows.append(
            {
                "benchmark_doc_id": benchmark_doc_id,
                "paper_id": paper_id,
                "source_section_id": source_section_id,
                "mapping_status": "mapped" if mapped else "unmapped",
                "gold_reference_count": referenced_kb_ids.get(benchmark_doc_id, 0),
            }
        )
    return rows


def export_full_corpus(args: argparse.Namespace) -> dict[str, Any]:
    approved_rows = read_jsonl(Path(args.approved_papers))
    sections = read_jsonl(Path(args.sections_manifest))
    benchmark_kb = read_jsonl_if_exists(Path(args.benchmark_kb))
    benchmark_gold = read_jsonl_if_exists(Path(args.benchmark_gold))
    registry = approved_registry_by_paper_id(approved_rows)

    corpus_rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for section in sections:
        text = section_text(section)
        reason = exclusion_reason(section, text)
        if reason:
            excluded[reason] += 1
            continue
        row = corpus_row(section=section, text=text, registry=registry)
        if row["corpus_id"]:
            corpus_rows.append(row)

    corpus_ids = {str(row["corpus_id"]) for row in corpus_rows}
    mapping_rows = benchmark_mapping_rows(
        benchmark_kb=benchmark_kb,
        benchmark_gold=benchmark_gold,
        corpus_ids=corpus_ids,
    )
    mapped_count = sum(1 for row in mapping_rows if row["mapping_status"] == "mapped")
    papers_with_sections = {str(row["paper_id"]) for row in corpus_rows if row.get("paper_id")}
    approved_paper_ids = set(registry)
    section_type_counts = dict(Counter(str(row["section_type"]) for row in corpus_rows))
    warnings: list[str] = []
    if not corpus_rows:
        warnings.append("No usable Research AI sections were exported.")
    if benchmark_kb and mapped_count == 0:
        warnings.append("No promoted benchmark KB rows mapped to exported source sections.")

    quality_report = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "approved_paper_count": len(approved_paper_ids),
        "sections_loaded_count": len(sections),
        "exported_corpus_count": len(corpus_rows),
        "excluded_section_count": sum(excluded.values()),
        "exclusion_counts": dict(excluded),
        "section_type_counts": section_type_counts,
        "paper_coverage_count": len(papers_with_sections),
        "papers_with_no_exported_sections": sorted(approved_paper_ids - papers_with_sections),
        "benchmark_kb_count": len(benchmark_kb),
        "mapped_benchmark_kb_count": mapped_count,
        "unmapped_benchmark_kb_count": len(mapping_rows) - mapped_count,
        "retrieval_corpus_ready_for_phase2b": bool(corpus_rows) and len(warnings) == 0,
        "warnings": warnings,
        "next_step": "Use this corpus beside benchmark KB during Phase 2B context engineering.",
    }
    manifest = {
        "phase": PHASE,
        "generated_at_utc": quality_report["generated_at_utc"],
        "corpus_name": "research_ai_full_sections_corpus",
        "corpus_path": str(args.output_corpus),
        "quality_report_path": str(args.output_quality_report),
        "mapping_path": str(args.output_mapping),
        "approved_paper_count": quality_report["approved_paper_count"],
        "sections_loaded_count": quality_report["sections_loaded_count"],
        "exported_corpus_count": quality_report["exported_corpus_count"],
        "benchmark_kb_path": str(args.benchmark_kb),
        "full_corpus_is_not_gold_linked_benchmark_kb": True,
        "no_embeddings_no_index_no_inference": True,
    }

    write_jsonl(Path(args.output_corpus), corpus_rows)
    write_json(Path(args.output_manifest), manifest)
    write_jsonl(Path(args.output_mapping), mapping_rows)
    write_json(Path(args.output_quality_report), quality_report)

    return {
        "phase": PHASE,
        "mode": "export_full_corpus",
        "approved_paper_count": quality_report["approved_paper_count"],
        "sections_loaded_count": quality_report["sections_loaded_count"],
        "exported_corpus_count": quality_report["exported_corpus_count"],
        "excluded_section_count": quality_report["excluded_section_count"],
        "benchmark_kb_count": quality_report["benchmark_kb_count"],
        "mapped_benchmark_kb_count": quality_report["mapped_benchmark_kb_count"],
        "unmapped_benchmark_kb_count": quality_report["unmapped_benchmark_kb_count"],
        "retrieval_corpus_ready_for_phase2b": quality_report["retrieval_corpus_ready_for_phase2b"],
        "warnings": warnings,
        "corpus_path": str(args.output_corpus),
        "manifest_path": str(args.output_manifest),
        "mapping_path": str(args.output_mapping),
        "quality_report_path": str(args.output_quality_report),
        "next_step": quality_report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-full-corpus", action="store_true")
    parser.add_argument("--sections-manifest", default=str(DEFAULT_SECTIONS_MANIFEST))
    parser.add_argument("--approved-papers", default=str(DEFAULT_APPROVED_PAPERS))
    parser.add_argument("--benchmark-kb", default=str(DEFAULT_BENCHMARK_KB))
    parser.add_argument("--benchmark-gold", default=str(DEFAULT_BENCHMARK_GOLD))
    parser.add_argument("--output-corpus", default=str(DEFAULT_OUTPUT_CORPUS))
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--output-mapping", default=str(DEFAULT_OUTPUT_MAPPING))
    parser.add_argument("--output-quality-report", default=str(DEFAULT_OUTPUT_QUALITY_REPORT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.export_full_corpus:
        parser.error("Pass --export-full-corpus to export the Research AI retrieval corpus.")
    try:
        summary = export_full_corpus(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
