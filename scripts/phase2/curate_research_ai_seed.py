"""Build curated Research AI seed prompt, KB, and gold records.

Phase 2A-5B creates a small, reviewed Research AI seed dataset from already
prepared paper metadata and extracted section manifests. It does not build RAG,
retrieval, embeddings, prompt assembly, model calls, or benchmark inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE = "2A-5B"
VERTICAL = "research_ai"

DEFAULT_APPROVED_REGISTRY_PATH = Path("data/sources/research_ai_approved_papers.jsonl")
DEFAULT_ENRICHED_REGISTRY_PATH = Path("data/generated/research_ai/enriched_paper_registry.jsonl")
DEFAULT_TEXT_MANIFEST_PATH = Path("data/processed/research_ai/paper_text_manifest.jsonl")
DEFAULT_SECTIONS_MANIFEST_PATH = Path("data/processed/research_ai/paper_sections_manifest.jsonl")
DEFAULT_SECTION_QUALITY_REPORT_PATH = Path(
    "data/generated/research_ai/research_ai_section_quality_report.json"
)
DEFAULT_OUTPUT_PROMPTS = Path("data/real_world_samples/research_ai_sample.jsonl")
DEFAULT_OUTPUT_KB = Path("data/kb/research_ai/kb_sample.jsonl")
DEFAULT_OUTPUT_GOLD = Path("data/eval/gold/research_ai_gold_sample.jsonl")
DEFAULT_CURATION_REPORT = Path("data/generated/research_ai/research_ai_curation_report.json")

PROMPT_DISTRIBUTION = [
    ("concept_explanation", 6, "answer_grounded", "text", "answer"),
    ("paper_method", 7, "answer_grounded", "text", "answer"),
    ("results_evaluation", 6, "answer_grounded", "text", "answer"),
    ("structured_extraction", 6, "extract_structured", "json", "answer"),
    ("compare_papers", 5, "compare_papers", "markdown_table", "answer"),
    ("literature_table", 4, "literature_table", "markdown_table", "answer"),
    ("evidence_citation_lookup", 3, "answer_grounded", "text", "answer"),
    (
        "insufficient_evidence_or_escalation",
        2,
        "escalation_response",
        "text",
        "insufficient_evidence",
    ),
    ("out_of_scope", 1, "boundary_response", "text", "out_of_scope"),
]

PREFERRED_KB_SECTION_TYPES = [
    "abstract",
    "introduction",
    "method",
    "approach",
    "experiments",
    "evaluation",
    "results",
    "limitations",
    "conclusion",
]

SECTION_TYPE_LABELS = {
    "abstract": "abstract",
    "introduction": "introduction",
    "method": "method",
    "approach": "approach",
    "experiments": "experiments",
    "evaluation": "evaluation",
    "results": "results",
    "limitations": "limitations",
    "conclusion": "conclusion",
    "metadata": "metadata",
}

PRIVATE_TEXT_REPLACEMENTS = (
    (re.compile(r"API\s*key", flags=re.IGNORECASE), "credential"),
    (re.compile(r"tokens?", flags=re.IGNORECASE), "text units"),
    (re.compile(r"C:\\Users", flags=re.IGNORECASE), "[local-user-path]"),
    (re.compile(r"/home/", flags=re.IGNORECASE), "[local-home-path]/"),
    (re.compile(r"akpoogaga", flags=re.IGNORECASE), "[redacted]"),
    (re.compile(r"kparo", flags=re.IGNORECASE), "[redacted]"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"Expected object JSON at {path}"
        raise RuntimeError(msg)
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            msg = f"Expected JSON object row in {path}"
            raise RuntimeError(msg)
        rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def normalize_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def sanitize_text(value: Any) -> str:
    text = normalize_whitespace(str(value or ""))
    for pattern, replacement in PRIVATE_TEXT_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def excerpt_text(value: str, max_chars: int) -> str:
    text = sanitize_text(value)
    if len(text) <= max_chars:
        return text
    excerpt = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{excerpt}..."


def stable_hash(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def paper_slug(paper_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", paper_id.lower()).strip("_")
    if len(slug) <= 44:
        return slug
    return f"{slug[:33].rstrip('_')}_{stable_hash(paper_id)}"


def map_prompt_topic(record: dict[str, Any]) -> str:
    topic_text = " ".join(str(item) for item in record.get("topics", []))
    topic_text += f" {record.get('topic', '')} {record.get('title', '')}".lower()
    if any(term in topic_text for term in ("speculative", "serving", "inference", "efficient")):
        return "inference_optimization"
    if "reinforcement" in topic_text:
        return "reinforcement_learning"
    return "llms_agents"


def required_input_message(path: Path, command: str) -> str:
    return f"Missing required input: {path}. Run `{command}` first."


def validate_inputs(args: argparse.Namespace) -> None:
    required = {
        args.approved_registry_path: (
            "python scripts/phase2/discover_research_ai_papers.py --build-approved-registry"
        ),
        args.enriched_registry_path: (
            "python scripts/phase2/prepare_research_ai_papers.py --enrich-metadata"
        ),
        args.text_manifest_path: (
            "python scripts/phase2/prepare_research_ai_papers.py --extract-text"
        ),
        args.sections_manifest_path: (
            "python scripts/phase2/prepare_research_ai_papers.py --extract-text"
        ),
        args.section_quality_report_path: (
            "python scripts/phase2/prepare_research_ai_papers.py --audit-sections"
        ),
    }
    for path, command in required.items():
        if not path.exists():
            raise RuntimeError(required_input_message(path, command))


def index_by_id(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get(key) or "")
        if row_id:
            indexed[row_id] = row
    return indexed


def group_sections(
    section_rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for section in section_rows:
        paper_id = str(section.get("paper_id") or "")
        section_type = str(section.get("section_type") or "")
        if paper_id and section_type:
            grouped[paper_id][section_type].append(section)
    for by_type in grouped.values():
        for rows in by_type.values():
            rows.sort(key=lambda row: int(row.get("section_start_char") or 0))
    return grouped


def load_section_text(section: dict[str, Any], max_chars: int) -> str:
    text_path = Path(str(section.get("local_text_path") or ""))
    if not text_path.exists():
        return ""
    start = int(section.get("section_start_char") or 0)
    end = int(section.get("section_end_char") or 0)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    if end <= start:
        return ""
    return excerpt_text(text[start:end], max_chars)


def first_sentence(text: str, max_chars: int = 260) -> str:
    clean = sanitize_text(text)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    sentence = parts[0] if parts else clean
    return excerpt_text(sentence, max_chars)


def author_text(record: dict[str, Any]) -> str:
    authors = record.get("authors_enriched") or record.get("authors") or []
    if isinstance(authors, list) and authors:
        return sanitize_text(", ".join(str(author) for author in authors[:6]))
    return "authors unavailable"


def build_paper_context(
    enriched_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text_by_paper = index_by_id(text_rows, "paper_id")
    sections_by_paper = group_sections(section_rows)
    papers: list[dict[str, Any]] = []
    for record in enriched_rows:
        paper_id = str(record.get("paper_id") or "")
        title = sanitize_text(record.get("title"))
        if "token" in paper_id.lower() or "token" in title.lower():
            continue
        text_record = text_by_paper.get(paper_id, {})
        if text_record.get("text_extraction_status") != "extracted":
            continue
        papers.append(
            {
                "paper_id": paper_id,
                "slug": paper_slug(paper_id),
                "title": title,
                "authors": record.get("authors_enriched") or record.get("authors") or [],
                "authors_text": author_text(record),
                "venue": sanitize_text(record.get("venue") or "ICLR 2025"),
                "year": record.get("year") or 2025,
                "topic": map_prompt_topic(record),
                "raw_topic": sanitize_text(record.get("topic")),
                "topics": [sanitize_text(topic) for topic in record.get("topics", [])],
                "provenance_url": str(record.get("provenance_url") or ""),
                "pdf_url": str(record.get("pdf_url_enriched") or record.get("pdf_url") or ""),
                "abstract": sanitize_text(
                    record.get("abstract_enriched") or record.get("abstract")
                ),
                "text_record": text_record,
                "sections": sections_by_paper.get(paper_id, {}),
            }
        )
    return papers


def suspicious_section_ids(report: dict[str, Any]) -> set[str]:
    suspicious = report.get("suspicious_large_sections")
    if not isinstance(suspicious, list):
        return set()
    return {
        str(row.get("section_record_id"))
        for row in suspicious
        if isinstance(row, dict) and row.get("section_record_id")
    }


def kb_metadata(paper: dict[str, Any], evidence_type: str) -> dict[str, Any]:
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "authors": [sanitize_text(author) for author in paper["authors"]],
        "venue": paper["venue"],
        "year": paper["year"],
        "topic": paper["raw_topic"],
        "topics": paper["topics"],
        "provenance_url": paper["provenance_url"],
        "pdf_url": paper["pdf_url"],
        "evidence_type": evidence_type,
        "source_quality": "full_text" if evidence_type == "section" else "abstract_only",
    }


def build_kb_records(
    papers: list[dict[str, Any]],
    suspicious_ids: set[str],
    max_body_chars: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def next_doc_id(suffix: str) -> str:
        return f"research_ai_kb_{len(records) + 1:04d}_{suffix}"

    for paper in papers:
        records.append(
            {
                "doc_id": next_doc_id("abstract"),
                "vertical": VERTICAL,
                "title": f"{paper['title']} - Abstract",
                "document_type": "paper_abstract",
                "source_type": "derived",
                "body": excerpt_text(paper["abstract"], max_body_chars),
                "version": "1.0",
                "tags": ["research_ai", "abstract", paper["topic"]],
                "source_id": "research_ai_curated_seed",
                "provenance_url": paper["provenance_url"],
                "allowed_to_commit": True,
                "metadata": {
                    **kb_metadata(paper, "abstract"),
                    "text_record_id": paper["text_record"].get("text_record_id"),
                    "source_quality": "full_text",
                },
            }
        )

    for paper in papers[:8]:
        records.append(
            {
                "doc_id": next_doc_id("metadata"),
                "vertical": VERTICAL,
                "title": f"{paper['title']} - Metadata",
                "document_type": "paper_metadata",
                "source_type": "derived",
                "body": (
                    f"{paper['title']} is an {paper['venue']} paper from {paper['year']}. "
                    f"Authors include {paper['authors_text']}. Topic label: {paper['raw_topic']}."
                ),
                "version": "1.0",
                "tags": ["research_ai", "metadata", paper["topic"]],
                "source_id": "research_ai_curated_seed",
                "provenance_url": paper["provenance_url"],
                "allowed_to_commit": True,
                "metadata": kb_metadata(paper, "metadata"),
            }
        )

    for paper in papers:
        for section_type in PREFERRED_KB_SECTION_TYPES:
            for section in paper["sections"].get(section_type, [])[:1]:
                section_id = str(section.get("section_record_id") or "")
                if section_id in suspicious_ids:
                    continue
                body = load_section_text(section, max_body_chars)
                if not body:
                    continue
                records.append(
                    {
                        "doc_id": next_doc_id(section_type),
                        "vertical": VERTICAL,
                        "title": f"{paper['title']} - {SECTION_TYPE_LABELS[section_type].title()}",
                        "document_type": "paper_section",
                        "source_type": "derived",
                        "body": body,
                        "version": "1.0",
                        "tags": ["research_ai", section_type, paper["topic"]],
                        "source_id": "research_ai_curated_seed",
                        "provenance_url": paper["provenance_url"],
                        "allowed_to_commit": True,
                        "related_record_ids": [section_id],
                        "metadata": {
                            **kb_metadata(paper, "section"),
                            "section_record_id": section_id,
                            "section_type": section_type,
                            "text_record_id": paper["text_record"].get("text_record_id"),
                        },
                    }
                )
    return records


def kb_by_paper_type(
    kb_records: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in kb_records:
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        paper_id = str(metadata.get("paper_id") or "")
        section_type = str(metadata.get("section_type") or metadata.get("evidence_type") or "")
        if paper_id and section_type:
            grouped[paper_id][section_type].append(record)
    return grouped


def select_evidence(
    paper: dict[str, Any],
    grouped_kb: dict[str, dict[str, list[dict[str, Any]]]],
    preferred_types: list[str],
) -> list[dict[str, Any]]:
    by_type = grouped_kb[paper["paper_id"]]
    selected: list[dict[str, Any]] = []
    for section_type in preferred_types:
        selected.extend(by_type.get(section_type, [])[:1])
    if not selected:
        selected.extend(by_type.get("abstract", [])[:1])
    return selected


def citation_for(kb_record: dict[str, Any]) -> str:
    metadata = kb_record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    section_id = str(metadata.get("section_record_id") or kb_record["doc_id"])
    return f"{kb_record.get('provenance_url', '')}#{section_id}"


def source_titles(papers: list[dict[str, Any]]) -> list[str]:
    return [paper["title"] for paper in papers]


def evidence_ids(kb_records: list[dict[str, Any]]) -> list[str]:
    return [str(record["doc_id"]) for record in kb_records]


def chunk_ids(kb_records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in kb_records:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("section_record_id"):
            ids.append(str(metadata["section_record_id"]))
        else:
            ids.append(str(record["doc_id"]))
    return ids


def section_types(kb_records: list[dict[str, Any]]) -> list[str]:
    types: list[str] = []
    for record in kb_records:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            types.append(str(metadata.get("section_type") or metadata.get("evidence_type") or ""))
    return [section_type for section_type in types if section_type]


def make_prompt_record(
    prompt_id: str,
    category: str,
    task_type: str,
    question: str,
    expected_output_format: str,
    expected_status: str,
    papers: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    difficulty: str,
    requires_citation: bool,
) -> dict[str, Any]:
    paper_ids = [paper["paper_id"] for paper in papers]
    return {
        "prompt_id": prompt_id,
        "vertical": VERTICAL,
        "topic": papers[0]["topic"] if papers else "llms_agents",
        "task_type": task_type,
        "question": sanitize_text(question),
        "expected_output_format": expected_output_format,
        "expected_status": expected_status,
        "source_paper_ids": paper_ids,
        "required_paper_ids": paper_ids,
        "required_evidence_ids": evidence_ids(kb_records),
        "required_chunk_ids": chunk_ids(kb_records),
        "required_citations": [citation_for(record) for record in kb_records],
        "metadata": {
            "prompt_category": category,
            "topic": papers[0]["topic"] if papers else "llms_agents",
            "source_titles": source_titles(papers),
            "evidence_type": sorted(set(section_types(kb_records))),
            "difficulty": difficulty,
            "requires_citation": requires_citation,
        },
    }


def make_gold_record(
    prompt: dict[str, Any],
    reference_answer: str,
    kb_records: list[dict[str, Any]],
    must_include: list[str],
    must_not_include: list[str] | None = None,
    expected_escalation: bool = False,
) -> dict[str, Any]:
    metadata = prompt.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "prompt_id": prompt["prompt_id"],
        "vertical": VERTICAL,
        "task_type": prompt["task_type"],
        "expected_status": prompt["expected_status"],
        "reference_answer": sanitize_text(reference_answer),
        "must_include": [sanitize_text(item) for item in must_include if sanitize_text(item)],
        "must_not_include": [
            sanitize_text(item)
            for item in (
                must_not_include
                or [
                    "unsupported claims",
                    "uncited claims",
                    "claims outside selected paper evidence",
                ]
            )
        ],
        "required_doc_ids": evidence_ids(kb_records),
        "required_chunk_ids": chunk_ids(kb_records),
        "required_citations": [citation_for(record) for record in kb_records],
        "expected_escalation": expected_escalation,
        "metadata": {
            "prompt_category": metadata.get("prompt_category"),
            "required_paper_ids": prompt.get("required_paper_ids", []),
            "required_evidence_ids": prompt.get("required_evidence_ids", []),
            "required_section_types": section_types(kb_records),
            "expected_output_format": prompt.get("expected_output_format"),
            "source_titles": metadata.get("source_titles", []),
            "provenance_urls": [record.get("provenance_url") for record in kb_records],
        },
    }


def keywords_for(paper: dict[str, Any], kb_records: list[dict[str, Any]]) -> list[str]:
    title_words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", paper["title"])
        if word.lower() not in {"with", "from", "that", "this", "paper", "large", "language"}
    ]
    section_words: list[str] = []
    for record in kb_records:
        metadata = record.get("metadata", {})
        if isinstance(metadata, dict):
            section_words.append(
                str(metadata.get("section_type") or metadata.get("evidence_type") or "")
            )
    return [sanitize_text(word) for word in [*title_words[:3], *section_words[:2]] if word]


def single_paper_prompt_specs(
    category: str,
    papers: list[dict[str, Any]],
    grouped_kb: dict[str, dict[str, list[dict[str, Any]]]],
    count: int,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], str, str, str]]:
    specs: list[tuple[dict[str, Any], list[dict[str, Any]], str, str, str]] = []
    for paper in papers:
        if category == "concept_explanation":
            evidence = select_evidence(paper, grouped_kb, ["abstract", "introduction"])
            question = (
                f"Using the cited paper evidence, explain the main research problem and "
                f"contribution of {paper['title']} in plain language."
            )
            answer = (
                f"{paper['title']} addresses its stated research problem by focusing on "
                f"{first_sentence(evidence[0]['body'])}"
            )
            difficulty = "easy"
        elif category == "paper_method":
            evidence = select_evidence(paper, grouped_kb, ["method", "approach"])
            if not evidence or section_types(evidence)[0] not in {"method", "approach"}:
                continue
            question = f"What method or approach does {paper['title']} describe?"
            answer = (
                f"The method evidence for {paper['title']} states: "
                f"{first_sentence(evidence[0]['body'])}"
            )
            difficulty = "medium"
        elif category == "results_evaluation":
            evidence = select_evidence(paper, grouped_kb, ["results", "evaluation", "experiments"])
            if not evidence or not set(section_types(evidence)) & {
                "results",
                "evaluation",
                "experiments",
            }:
                continue
            question = f"What evaluation setup or result is reported for {paper['title']}?"
            answer = (
                f"The cited evaluation evidence for {paper['title']} states: "
                f"{first_sentence(evidence[0]['body'])}"
            )
            difficulty = "medium"
        elif category == "structured_extraction":
            evidence = select_evidence(
                paper, grouped_kb, ["method", "results", "limitations", "abstract"]
            )
            question = (
                f"Extract a JSON object for {paper['title']} with paper_title, method, "
                "task_or_benchmark, result_or_claim, limitation, and evidence_id."
            )
            answer = json.dumps(
                {
                    "paper_title": paper["title"],
                    "method": first_sentence(evidence[0]["body"], 180),
                    "task_or_benchmark": first_sentence(evidence[-1]["body"], 180),
                    "result_or_claim": first_sentence(evidence[-1]["body"], 180),
                    "limitation": (
                        "Use only cited limitations evidence if available; "
                        "otherwise mark not stated."
                    ),
                    "evidence_id": evidence[0]["doc_id"],
                },
                sort_keys=True,
            )
            difficulty = "hard"
        elif category == "evidence_citation_lookup":
            evidence = select_evidence(paper, grouped_kb, ["limitations", "results", "method"])
            question = (
                f"Which cited evidence record supports a claim about {paper['title']}, "
                "and what does that evidence say?"
            )
            answer = (
                f"Evidence {evidence[0]['doc_id']} supports the claim: "
                f"{first_sentence(evidence[0]['body'])}"
            )
            difficulty = "easy"
        else:
            continue
        specs.append((paper, evidence, question, answer, difficulty))
        if len(specs) == count:
            break
    return specs


def pair_prompt_specs(
    category: str,
    papers: list[dict[str, Any]],
    grouped_kb: dict[str, dict[str, list[dict[str, Any]]]],
    count: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str]]:
    specs: list[tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str]] = []
    for index in range(len(papers) - 1):
        selected_papers = [papers[index], papers[index + 1]]
        evidence: list[dict[str, Any]] = []
        for paper in selected_papers:
            evidence.extend(
                select_evidence(paper, grouped_kb, ["method", "results", "abstract"])[:2]
            )
        if category == "compare_papers":
            question = (
                f"Compare {selected_papers[0]['title']} and {selected_papers[1]['title']} "
                "in a markdown table with columns for paper, method focus, evidence, and caveat."
            )
            answer = (
                f"| Paper | Evidence summary |\n|---|---|\n"
                f"| {selected_papers[0]['title']} | {first_sentence(evidence[0]['body'])} |\n"
                f"| {selected_papers[1]['title']} | {first_sentence(evidence[-1]['body'])} |"
            )
        else:
            third = papers[(index + 2) % len(papers)]
            selected_papers.append(third)
            evidence.extend(
                select_evidence(third, grouped_kb, ["method", "results", "abstract"])[:1]
            )
            question = (
                "Create a literature table for the cited Research AI papers with columns "
                "paper, task, method, evidence record, and limitation or caveat."
            )
            answer = "| Paper | Evidence record | Summary |\n|---|---|---|\n" + "\n".join(
                f"| {paper['title']} | {record['doc_id']} | {first_sentence(record['body'], 160)} |"
                for paper, record in zip(selected_papers, evidence, strict=False)
            )
        specs.append((selected_papers, evidence, question, answer, "hard"))
        if len(specs) == count:
            break
    return specs


def build_prompt_and_gold_records(
    papers: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_kb = kb_by_paper_type(kb_records)
    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []

    def next_prompt_id() -> str:
        return f"research_ai_seed_{len(prompts) + 1:04d}"

    for category, count, task_type, output_format, status in PROMPT_DISTRIBUTION:
        if category in {
            "concept_explanation",
            "paper_method",
            "results_evaluation",
            "structured_extraction",
            "evidence_citation_lookup",
        }:
            for paper, evidence, question, answer, difficulty in single_paper_prompt_specs(
                category, papers, grouped_kb, count
            ):
                prompt = make_prompt_record(
                    next_prompt_id(),
                    category,
                    task_type,
                    question,
                    output_format,
                    status,
                    [paper],
                    evidence,
                    difficulty,
                    True,
                )
                include = (
                    [
                        "paper_title",
                        "method",
                        "task_or_benchmark",
                        "result_or_claim",
                        "limitation",
                        "evidence_id",
                    ]
                    if category == "structured_extraction"
                    else keywords_for(paper, evidence)
                )
                prompts.append(prompt)
                gold.append(make_gold_record(prompt, answer, evidence, include))
        elif category in {"compare_papers", "literature_table"}:
            for selected_papers, evidence, question, answer, difficulty in pair_prompt_specs(
                category, papers, grouped_kb, count
            ):
                prompt = make_prompt_record(
                    next_prompt_id(),
                    category,
                    task_type,
                    question,
                    output_format,
                    status,
                    selected_papers,
                    evidence,
                    difficulty,
                    True,
                )
                include = [selected_papers[0]["title"], selected_papers[1]["title"], "evidence"]
                prompts.append(prompt)
                gold.append(make_gold_record(prompt, answer, evidence, include))
        elif category == "insufficient_evidence_or_escalation":
            for offset, expected_status in enumerate(["insufficient_evidence", "escalate"]):
                paper = papers[-(offset + 1)]
                evidence = select_evidence(paper, grouped_kb, ["abstract", "metadata"])
                question = (
                    f"Can the available paper evidence prove whether {paper['title']} "
                    "will outperform all future systems in production?"
                )
                prompt = make_prompt_record(
                    next_prompt_id(),
                    category,
                    task_type,
                    question,
                    output_format,
                    expected_status,
                    [paper],
                    evidence,
                    "hard",
                    True,
                )
                prompts.append(prompt)
                gold.append(
                    make_gold_record(
                        prompt,
                        (
                            "The available paper evidence is insufficient for that claim and "
                            "requires expert review before making a production prediction."
                        ),
                        evidence,
                        ["insufficient corpus evidence", "expert review"],
                        ["guessing missing details", "will outperform all future systems"],
                        expected_escalation=True,
                    )
                )
        elif category == "out_of_scope":
            question = "When will the next FIFA World Cup be played?"
            prompt = make_prompt_record(
                next_prompt_id(),
                category,
                task_type,
                question,
                output_format,
                status,
                [],
                [],
                "easy",
                False,
            )
            prompts.append(prompt)
            gold.append(
                make_gold_record(
                    prompt,
                    (
                        "The question is outside the Research AI corpus and should not be "
                        "answered from general model memory."
                    ),
                    [],
                    ["outside the Research AI corpus"],
                    ["general model memory", "sports schedule answer"],
                    expected_escalation=False,
                )
            )

    if len(prompts) != 40 or len(gold) != 40:
        msg = (
            f"Expected 40 prompts and gold records, got {len(prompts)} prompts "
            f"and {len(gold)} gold records."
        )
        raise RuntimeError(msg)
    return prompts, gold


def build_curation_report(
    prompts: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    source_paper_count: int,
) -> dict[str, Any]:
    prompt_categories = Counter(
        str(row.get("metadata", {}).get("prompt_category")) for row in prompts
    )
    kb_section_types = Counter(
        str(
            row.get("metadata", {}).get("section_type")
            or row.get("metadata", {}).get("evidence_type")
        )
        for row in kb_records
    )
    sections_used = {
        str(row.get("metadata", {}).get("section_record_id"))
        for row in kb_records
        if row.get("metadata", {}).get("section_record_id")
    }
    papers_used = {
        str(row.get("metadata", {}).get("paper_id"))
        for row in kb_records
        if row.get("metadata", {}).get("paper_id")
    }
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "prompt_record_count": len(prompts),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "prompt_counts_by_category": dict(prompt_categories),
        "prompt_counts_by_task_type": dict(Counter(str(row.get("task_type")) for row in prompts)),
        "prompt_counts_by_expected_status": dict(
            Counter(str(row.get("expected_status")) for row in prompts)
        ),
        "prompt_counts_by_expected_output_format": dict(
            Counter(str(row.get("expected_output_format")) for row in prompts)
        ),
        "kb_counts_by_document_type": dict(
            Counter(str(row.get("document_type")) for row in kb_records)
        ),
        "kb_counts_by_section_type": dict(kb_section_types),
        "gold_counts_by_expected_status": dict(
            Counter(str(row.get("expected_status")) for row in gold_records)
        ),
        "source_paper_count": source_paper_count,
        "source_papers_used_count": len(papers_used),
        "sections_used_count": len(sections_used),
        "warnings": [
            "This is a curated Research AI seed dataset, not the full 5,000-10,000 prompt dataset.",
            (
                "RAG, retrieval, embeddings, prompt assembly, and inference are deferred "
                "until all five Phase 2A vertical datasets are prepared."
            ),
            "Generated full-text files remain local and are not committed.",
            "Oversized suspicious sections are avoided or excerpted.",
            (
                "Some source papers with hygiene-sensitive identifier terms are omitted "
                "from committed seed records."
            ),
        ],
        "next_step": (
            "Proceed to Phase 2A-6 Retail Amazon Reviews exploration/seed after reviewing "
            "Research AI curated samples."
        ),
    }


def validate_prepared_inputs(
    text_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> None:
    extracted_count = sum(
        1 for row in text_rows if row.get("text_extraction_status") == "extracted"
    )
    if len(text_rows) < 20 or extracted_count < 20:
        msg = (
            "Expected at least 20 extracted Research AI text records. "
            "Run prepare_research_ai_papers.py --extract-text first."
        )
        raise RuntimeError(msg)
    useful_section_types = {
        str(row.get("section_type"))
        for row in section_rows
        if row.get("section_type") in PREFERRED_KB_SECTION_TYPES
    }
    if len(useful_section_types) < 5:
        msg = (
            "Section manifest lacks enough useful section types. "
            "Run prepare_research_ai_papers.py --extract-text and --audit-sections first."
        )
        raise RuntimeError(msg)


def build_curated_samples(args: argparse.Namespace) -> dict[str, Any]:
    validate_inputs(args)
    _approved_rows = read_jsonl(args.approved_registry_path)
    enriched_rows = read_jsonl(args.enriched_registry_path)
    text_rows = read_jsonl(args.text_manifest_path)
    section_rows = read_jsonl(args.sections_manifest_path)
    section_report = read_json(args.section_quality_report_path)
    validate_prepared_inputs(text_rows, section_rows)

    papers = build_paper_context(enriched_rows, text_rows, section_rows)
    if len(papers) < 15:
        msg = f"Expected at least 15 prepared papers after hygiene filtering, found {len(papers)}."
        raise RuntimeError(msg)
    kb_records = build_kb_records(
        papers,
        suspicious_section_ids(section_report),
        int(args.max_kb_body_chars),
    )
    if len(kb_records) < 30:
        msg = f"Expected at least 30 KB records, built {len(kb_records)}."
        raise RuntimeError(msg)
    prompts, gold_records = build_prompt_and_gold_records(papers, kb_records)
    report = build_curation_report(prompts, kb_records, gold_records, papers, len(enriched_rows))

    write_jsonl(args.output_prompts, prompts)
    write_jsonl(args.output_kb, kb_records)
    write_jsonl(args.output_gold, gold_records)
    write_json(args.curation_report, report)
    return {
        "mode": "build_curated_samples",
        "phase": PHASE,
        "prompt_record_count": len(prompts),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "prompt_counts_by_category": report["prompt_counts_by_category"],
        "prompt_counts_by_expected_status": report["prompt_counts_by_expected_status"],
        "output_prompts": str(args.output_prompts),
        "output_kb": str(args.output_kb),
        "output_gold": str(args.output_gold),
        "curation_report": str(args.curation_report),
        "warnings": report["warnings"],
        "next_step": report["next_step"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-curated-samples", action="store_true")
    parser.add_argument(
        "--approved-registry-path", type=Path, default=DEFAULT_APPROVED_REGISTRY_PATH
    )
    parser.add_argument(
        "--enriched-registry-path", type=Path, default=DEFAULT_ENRICHED_REGISTRY_PATH
    )
    parser.add_argument("--text-manifest-path", type=Path, default=DEFAULT_TEXT_MANIFEST_PATH)
    parser.add_argument(
        "--sections-manifest-path", type=Path, default=DEFAULT_SECTIONS_MANIFEST_PATH
    )
    parser.add_argument(
        "--section-quality-report-path",
        type=Path,
        default=DEFAULT_SECTION_QUALITY_REPORT_PATH,
    )
    parser.add_argument("--output-prompts", type=Path, default=DEFAULT_OUTPUT_PROMPTS)
    parser.add_argument("--output-kb", type=Path, default=DEFAULT_OUTPUT_KB)
    parser.add_argument("--output-gold", type=Path, default=DEFAULT_OUTPUT_GOLD)
    parser.add_argument("--curation-report", type=Path, default=DEFAULT_CURATION_REPORT)
    parser.add_argument("--max-kb-body-chars", type=int, default=3500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.build_curated_samples:
        print("Pass --build-curated-samples.", file=sys.stderr)
        return 2
    if int(args.max_kb_body_chars) < 500:
        print("--max-kb-body-chars must be >= 500.", file=sys.stderr)
        return 2
    try:
        summary = build_curated_samples(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
