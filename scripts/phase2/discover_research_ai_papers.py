"""Discover AI Research Assistant candidate papers from arXiv metadata."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QUERY_PLAN = Path("data/sources/research_ai_query_plan.json")
DEFAULT_OUTPUT_CANDIDATES = Path("data/generated/research_ai/candidate_papers.jsonl")
DEFAULT_OUTPUT_REVIEW_CSV = Path("data/generated/research_ai/candidate_papers_review.csv")
DEFAULT_OUTPUT_SAMPLE = Path("data/sources/research_ai_candidate_papers_sample.jsonl")
DEFAULT_OUTPUT_REPORT = Path("data/generated/research_ai/research_ai_discovery_report.json")

ARXIV_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_EXTENSION_NAMESPACE = "http://arxiv.org/schemas/atom"
ATOM = {"atom": ARXIV_NAMESPACE, "arxiv": ARXIV_EXTENSION_NAMESPACE}
DEFAULT_CATEGORIES_ALLOWED = {"cs.CL", "cs.LG", "cs.AI", "cs.DC", "cs.SE", "cs.IR"}
USER_AGENT = "LLM-Inference-Optimization-Suite research-discovery"
TITLE_SIGNAL_TERMS = (
    "llm",
    "inference",
    "rag",
    "routing",
    "agent",
    "vllm",
    "decoding",
    "kv cache",
    "serving",
    "small language model",
)
RELATED_TERMS = (
    "llm",
    "language model",
    "inference",
    "rag",
    "retrieval",
    "agent",
    "routing",
    "serving",
    "decoding",
    "machine learning",
    "transformer",
    "neural",
)
STATUS_TAXONOMY = {
    "answer": "The question is in scope and answerable from the provided data.",
    "escalate": (
        "The question is in scope, but unclear, ambiguous, conflicting, or requires "
        "human/expert review."
    ),
    "insufficient_evidence": (
        "The question is in scope, but the available corpus does not provide enough evidence."
    ),
    "out_of_scope": (
        "The question is outside the selected vertical/corpus. The model may know the "
        "answer from general world knowledge, but a grounded/RAG/agentic system should "
        "not answer from this corpus."
    ),
    "spam_or_fraud": (
        "The request is spam, abusive, fraudulent, or intentionally irrelevant for "
        "support-style verticals."
    ),
    "boundary_response": (
        "The request hits a safety/privacy/clinical/admin boundary and requires a safe "
        "boundary response."
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Missing required JSON file: {path}"
        raise RuntimeError(msg)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"Expected JSON object in {path}"
        raise RuntimeError(msg)
    return parsed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def normalize_arxiv_id(entry_id_or_url: str) -> tuple[str, str]:
    """Return arXiv ID without version and with version when present."""

    candidate = entry_id_or_url.rstrip("/").rsplit("/", 1)[-1]
    candidate = candidate.split("?", 1)[0]
    versionless = re.sub(r"v\d+$", "", candidate)
    return versionless, candidate


def paper_id_from_arxiv_id(arxiv_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", arxiv_id).strip("_")
    return f"research_ai_arxiv_{safe_id}"


def build_arxiv_url(
    api_base_url: str,
    search_query: str,
    max_results: int,
    sort_by: str,
    sort_order: str,
) -> str:
    query_params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    return f"{api_base_url}?{urllib.parse.urlencode(query_params)}"


def fetch_arxiv_atom(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/atom+xml, application/xml, text/xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - convert library errors to a clear CLI error.
        msg = f"Failed to fetch arXiv metadata from {url}: {exc}"
        raise RuntimeError(msg) from exc


def _entry_text(entry: ET.Element, tag: str) -> str:
    element = entry.find(f"atom:{tag}", ATOM)
    return normalize_text(element.text if element is not None else "")


def _entry_links(entry: ET.Element) -> tuple[str, str]:
    abstract_url = ""
    pdf_url = ""
    for link in entry.findall("atom:link", ATOM):
        href = link.attrib.get("href", "")
        rel = link.attrib.get("rel", "")
        title = link.attrib.get("title", "")
        link_type = link.attrib.get("type", "")
        if rel == "alternate" and href:
            abstract_url = href
        if href and (title.lower() == "pdf" or link_type == "application/pdf" or "/pdf/" in href):
            pdf_url = href
    if not pdf_url and abstract_url:
        pdf_url = abstract_url.replace("/abs/", "/pdf/")
    return abstract_url, pdf_url


def parse_arxiv_atom(xml_text: str, query_id: str, topic: str) -> list[dict[str, Any]]:
    """Parse arXiv Atom XML into candidate paper records."""

    root = ET.fromstring(xml_text)
    records: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM):
        entry_id = _entry_text(entry, "id")
        arxiv_id, arxiv_id_version = normalize_arxiv_id(entry_id)
        title = _entry_text(entry, "title")
        abstract = _entry_text(entry, "summary")
        authors = [
            normalize_text(name.text if name is not None else "")
            for name in entry.findall("atom:author/atom:name", ATOM)
        ]
        authors = [author for author in authors if author]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM)
            if category.attrib.get("term")
        ]
        primary_element = entry.find("arxiv:primary_category", ATOM)
        primary_category = (
            primary_element.attrib.get("term", "")
            if primary_element is not None
            else categories[0]
            if categories
            else ""
        )
        abstract_url, pdf_url = _entry_links(entry)
        license_element = entry.find("arxiv:license", ATOM)
        record = {
            "paper_id": paper_id_from_arxiv_id(arxiv_id),
            "arxiv_id": arxiv_id,
            "arxiv_id_version": arxiv_id_version,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "published": _entry_text(entry, "published"),
            "updated": _entry_text(entry, "updated"),
            "primary_category": primary_category,
            "categories": categories,
            "abstract_url": abstract_url,
            "pdf_url": pdf_url,
            "source": "arXiv",
            "query_ids": [query_id],
            "topics": [topic],
            "matched_keywords": [],
            "score": 0,
            "selection_status": "candidate",
            "review_notes": "",
            "license": normalize_text(license_element.text)
            if license_element is not None
            else None,
            "provenance_url": abstract_url,
        }
        records.append(record)
    return records


def matched_keywords_for_candidate(
    candidate: dict[str, Any], query_group: dict[str, Any]
) -> list[str]:
    haystack = f"{candidate.get('title', '')} {candidate.get('abstract', '')}".lower()
    keywords = []
    for keyword in query_group.get("required_keywords", []):
        if str(keyword).lower() in haystack:
            keywords.append(str(keyword))
    for keyword in query_group.get("preferred_keywords", []):
        if str(keyword).lower() in haystack:
            keywords.append(str(keyword))
    return sorted(set(keywords), key=str.lower)


def score_candidate(candidate: dict[str, Any], query_group: dict[str, Any]) -> int:
    """Score candidate metadata for Phase 2A review priority."""

    title = str(candidate.get("title") or "")
    abstract = str(candidate.get("abstract") or "")
    haystack = f"{title} {abstract}".lower()
    title_lower = title.lower()
    score = 0

    if any(
        str(keyword).lower() in haystack for keyword in query_group.get("required_keywords", [])
    ):
        score += 5
    for keyword in query_group.get("preferred_keywords", []):
        if str(keyword).lower() in haystack:
            score += 3

    categories_allowed = set(query_group.get("categories_allowed", DEFAULT_CATEGORIES_ALLOWED))
    categories = set(candidate.get("categories", []))
    if candidate.get("primary_category") in categories_allowed or categories & categories_allowed:
        score += 2

    if any(term in title_lower for term in TITLE_SIGNAL_TERMS):
        score += 2

    published = str(candidate.get("published") or "")
    year_match = re.match(r"(\d{4})", published)
    if year_match and int(year_match.group(1)) >= 2023:
        score += 1

    if not any(term in haystack for term in RELATED_TERMS):
        score -= 3
    return score


def _version_sort_key(candidate: dict[str, Any]) -> tuple[int, str]:
    version = str(candidate.get("arxiv_id_version") or "")
    match = re.search(r"v(\d+)$", version)
    version_number = int(match.group(1)) if match else 0
    return version_number, str(candidate.get("updated") or "")


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by versionless arXiv ID and merge query metadata."""

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        arxiv_id = str(candidate.get("arxiv_id") or "")
        if not arxiv_id:
            continue
        existing = deduped.get(arxiv_id)
        if existing is None:
            deduped[arxiv_id] = {**candidate}
            continue

        existing["query_ids"] = sorted(
            set(existing.get("query_ids", [])) | set(candidate.get("query_ids", []))
        )
        existing["topics"] = sorted(
            set(existing.get("topics", [])) | set(candidate.get("topics", []))
        )
        existing["matched_keywords"] = sorted(
            set(existing.get("matched_keywords", [])) | set(candidate.get("matched_keywords", [])),
            key=str.lower,
        )
        existing["score"] = max(int(existing.get("score") or 0), int(candidate.get("score") or 0))
        if _version_sort_key(candidate) > _version_sort_key(existing):
            keep_fields = {"query_ids", "topics", "matched_keywords", "score"}
            preserved = {field: existing[field] for field in keep_fields}
            deduped[arxiv_id] = {**candidate, **preserved}
    return list(deduped.values())


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda candidate: str(candidate.get("title") or "").lower(),
    )
    ranked = sorted(
        ranked, key=lambda candidate: str(candidate.get("published") or ""), reverse=True
    )
    ranked = sorted(ranked, key=lambda candidate: str(candidate.get("updated") or ""), reverse=True)
    return sorted(ranked, key=lambda candidate: int(candidate.get("score") or 0), reverse=True)


def write_review_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "score",
        "paper_id",
        "arxiv_id",
        "title",
        "authors",
        "published",
        "updated",
        "primary_category",
        "categories",
        "topics",
        "abstract_url",
        "pdf_url",
        "selection_status",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, candidate in enumerate(candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "score": candidate.get("score"),
                    "paper_id": candidate.get("paper_id"),
                    "arxiv_id": candidate.get("arxiv_id"),
                    "title": candidate.get("title"),
                    "authors": "; ".join(candidate.get("authors", [])),
                    "published": candidate.get("published"),
                    "updated": candidate.get("updated"),
                    "primary_category": candidate.get("primary_category"),
                    "categories": "; ".join(candidate.get("categories", [])),
                    "topics": "; ".join(candidate.get("topics", [])),
                    "abstract_url": candidate.get("abstract_url"),
                    "pdf_url": candidate.get("pdf_url"),
                    "selection_status": candidate.get("selection_status"),
                    "review_notes": candidate.get("review_notes"),
                }
            )


def build_report(
    query_plan: dict[str, Any],
    raw_candidate_count: int,
    ranked_candidates: list[dict[str, Any]],
    sample_candidates: list[dict[str, Any]],
    output_files: dict[str, str],
) -> dict[str, Any]:
    topic_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    for candidate in ranked_candidates:
        topic_counter.update(str(topic) for topic in candidate.get("topics", []))
        primary_category = str(candidate.get("primary_category") or "")
        if primary_category:
            category_counter[primary_category] += 1
    return {
        "phase": "2A-5A",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_group_count": len(query_plan.get("query_groups", [])),
        "raw_candidate_count": raw_candidate_count,
        "deduped_candidate_count": len(ranked_candidates),
        "sample_candidate_count": len(sample_candidates),
        "counts_by_topic": dict(topic_counter),
        "counts_by_primary_category": dict(category_counter),
        "top_candidates": [
            {
                "rank": index,
                "score": candidate.get("score"),
                "paper_id": candidate.get("paper_id"),
                "arxiv_id": candidate.get("arxiv_id"),
                "title": candidate.get("title"),
            }
            for index, candidate in enumerate(ranked_candidates[:10], start=1)
        ],
        "output_files": output_files,
        "status_taxonomy": STATUS_TAXONOMY,
        "warnings": [
            "This is paper metadata discovery only.",
            "PDFs/full text are not downloaded in Phase 2A-5A.",
            (
                "Final research prompts, KB records, and gold/eval records are deferred "
                "to Phase 2A-5B."
            ),
            (
                "RAG, retrieval, embeddings, prompt assembly, and inference remain deferred "
                "until all five Phase 2A vertical datasets are prepared."
            ),
        ],
        "next_step": (
            "Review candidate_papers_review.csv, approve a 12-20 paper shortlist, then "
            "proceed to Phase 2A-5B to create research AI paper registry, KB/context "
            "samples, prompt/source records, and gold/eval records."
        ),
    }


def _query_max_results(
    query_group: dict[str, Any], query_plan: dict[str, Any], override: int
) -> int:
    if override > 0:
        return override
    return int(query_group.get("max_results") or query_plan.get("max_results_per_query") or 20)


def planned_urls(query_plan: dict[str, Any], max_results_override: int) -> list[dict[str, str]]:
    planned: list[dict[str, str]] = []
    for query_group in query_plan.get("query_groups", []):
        max_results = _query_max_results(query_group, query_plan, max_results_override)
        url = build_arxiv_url(
            api_base_url=str(query_plan["api_base_url"]),
            search_query=str(query_group["search_query"]),
            max_results=max_results,
            sort_by=str(query_group.get("sort_by") or "relevance"),
            sort_order=str(query_group.get("sort_order") or "descending"),
        )
        planned.append(
            {
                "query_id": str(query_group["query_id"]),
                "topic": str(query_group["topic"]),
                "url": url,
            }
        )
    return planned


def discover(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = load_json(args.query_plan)
    raw_candidates: list[dict[str, Any]] = []
    delay_seconds = (
        float(args.delay_seconds)
        if float(args.delay_seconds) > 0
        else float(query_plan.get("delay_seconds") or 3)
    )
    categories_allowed = set(query_plan.get("categories_allowed", [])) or DEFAULT_CATEGORIES_ALLOWED

    for index, query_group in enumerate(query_plan.get("query_groups", [])):
        enriched_group = {**query_group, "categories_allowed": sorted(categories_allowed)}
        max_results = _query_max_results(enriched_group, query_plan, args.max_results_per_query)
        url = build_arxiv_url(
            api_base_url=str(query_plan["api_base_url"]),
            search_query=str(enriched_group["search_query"]),
            max_results=max_results,
            sort_by=str(enriched_group.get("sort_by") or "relevance"),
            sort_order=str(enriched_group.get("sort_order") or "descending"),
        )
        if index > 0:
            time.sleep(delay_seconds)
        xml_text = fetch_arxiv_atom(url)
        parsed_candidates = parse_arxiv_atom(
            xml_text=xml_text,
            query_id=str(enriched_group["query_id"]),
            topic=str(enriched_group["topic"]),
        )
        for candidate in parsed_candidates:
            candidate["matched_keywords"] = matched_keywords_for_candidate(
                candidate,
                enriched_group,
            )
            candidate["score"] = score_candidate(candidate, enriched_group)
        raw_candidates.extend(parsed_candidates)

    deduped_candidates = dedupe_candidates(raw_candidates)
    ranked_candidates = rank_candidates(deduped_candidates)
    sample_candidates = ranked_candidates[: args.sample_size]
    output_files = {
        "candidate_papers_jsonl": str(args.output_candidates),
        "candidate_papers_review_csv": str(args.output_review_csv),
        "candidate_papers_sample_jsonl": str(args.output_sample),
        "discovery_report_json": str(args.output_report),
    }
    report = build_report(
        query_plan=query_plan,
        raw_candidate_count=len(raw_candidates),
        ranked_candidates=ranked_candidates,
        sample_candidates=sample_candidates,
        output_files=output_files,
    )

    write_jsonl(args.output_candidates, ranked_candidates)
    write_review_csv(args.output_review_csv, ranked_candidates)
    write_jsonl(args.output_sample, sample_candidates)
    write_json(args.output_report, report)

    return {
        "mode": "discover",
        "phase": "2A-5A",
        "query_group_count": len(query_plan.get("query_groups", [])),
        "raw_candidate_count": len(raw_candidates),
        "deduped_candidate_count": len(ranked_candidates),
        "sample_candidate_count": len(sample_candidates),
        "output_candidates": str(args.output_candidates),
        "output_review_csv": str(args.output_review_csv),
        "output_sample": str(args.output_sample),
        "output_report": str(args.output_report),
        "warnings": report["warnings"],
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = load_json(args.query_plan)
    urls = planned_urls(query_plan, args.max_results_per_query)
    return {
        "mode": "dry_run",
        "phase": "2A-5A",
        "planned_query_count": len(urls),
        "planned_queries": urls,
        "output_candidates": str(args.output_candidates),
        "output_review_csv": str(args.output_review_csv),
        "output_sample": str(args.output_sample),
        "output_report": str(args.output_report),
        "will_download_pdfs": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--query-plan", type=Path, default=DEFAULT_QUERY_PLAN)
    parser.add_argument("--output-candidates", type=Path, default=DEFAULT_OUTPUT_CANDIDATES)
    parser.add_argument("--output-review-csv", type=Path, default=DEFAULT_OUTPUT_REVIEW_CSV)
    parser.add_argument("--output-sample", type=Path, default=DEFAULT_OUTPUT_SAMPLE)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--max-results-per-query", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.discover and not args.dry_run:
        print("Pass --dry-run or --discover.", file=sys.stderr)
        return 2
    try:
        summary = dry_run(args) if args.dry_run else discover(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
