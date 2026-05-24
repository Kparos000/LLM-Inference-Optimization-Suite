"""Prepare approved Research AI papers with metadata, PDFs, and extracted text.

This Phase 2A-5A-Text script enriches the approved paper registry and prepares
local text artifacts where possible. It does not build RAG indexes, embeddings,
prompts, gold records, or run model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

PHASE = "2A-5A-Text"
EXPANSION_PHASE = "2A-12C"
USER_AGENT = "LLM-Inference-Optimization-Suite research-paper-prep"
PDF_EXTRACTION_DEPENDENCY_WARNING = (
    "PDF text extraction dependency is missing. Install pypdf or PyPDF2, then rerun --extract-text."
)

DEFAULT_APPROVED_REGISTRY_PATH = Path("data/sources/research_ai_approved_papers.jsonl")
DEFAULT_ENRICHED_REGISTRY_PATH = Path("data/generated/research_ai/enriched_paper_registry.jsonl")
DEFAULT_RAW_PAPER_DIR = Path("data/raw/research_ai/papers/")
DEFAULT_PAPER_TEXT_DIR = Path("data/processed/research_ai/paper_text/")
DEFAULT_TEXT_MANIFEST_PATH = Path("data/processed/research_ai/paper_text_manifest.jsonl")
DEFAULT_SECTIONS_MANIFEST_PATH = Path("data/processed/research_ai/paper_sections_manifest.jsonl")
DEFAULT_REPORT_PATH = Path("data/generated/research_ai/research_ai_paper_preparation_report.json")
DEFAULT_SECTION_QUALITY_REPORT_PATH = Path(
    "data/generated/research_ai/research_ai_section_quality_report.json"
)
DEFAULT_40_PAPER_EXPANSION_REPORT_PATH = Path(
    "data/generated/research_ai/research_ai_40_paper_expansion_report.json"
)
DEFAULT_40_PAPER_REVIEW_CSV_PATH = Path(
    "data/generated/research_ai/research_ai_40_paper_review.csv"
)
DEFAULT_EXPANDED_SECTION_QUALITY_REPORT_PATH = Path(
    "data/generated/research_ai/research_ai_expanded_section_quality_report.json"
)
DEFAULT_1000_SCALE_CANDIDATE_TEMPLATE_PATH = Path(
    "data/sources/research_ai_1000_scale_candidate_papers_template.jsonl"
)
TARGET_RESEARCH_AI_APPROVED_PAPER_COUNT = 40
TARGET_RESEARCH_AI_SECTION_COUNT_MIN = 800
TARGET_RESEARCH_AI_SECTION_COUNT_MAX = 1200

PAPER_SECTION_HEADINGS = {
    "abstract": ("Abstract",),
    "introduction": ("Introduction",),
    "related_work": ("Related Work", "Related Works"),
    "background": ("Background", "Preliminaries", "Preliminary"),
    "method": (
        "Method",
        "Methods",
        "Methodology",
        "Model",
        "Framework",
        "Algorithm",
        "Training",
        "Data",
        "Dataset",
        "Datasets",
    ),
    "approach": ("Approach",),
    "experiments": ("Experiments", "Experimental Setup", "Experiment Setup"),
    "evaluation": ("Evaluation",),
    "results": ("Results", "Main Results", "Additional Results", "Experimental Results"),
    "analysis": ("Analysis", "Ablation", "Ablations", "Ablation Study"),
    "discussion": ("Discussion",),
    "limitations": ("Limitations", "Limitation"),
    "conclusion": ("Conclusion", "Conclusions"),
    "references": ("References", "Bibliography"),
    "appendix": ("Appendix", "Appendices", "Supplementary Material", "Supplemental Material"),
}
NOISY_ABSTRACT_MARKERS = (
    "Show more",
    "Video",
    "Chat is not available",
    "Successful Page Load",
    "ICLR uses cookies",
    "Useful links",
    "About ICLR",
    "Sponsor / Exhibitor Information",
    "ICLR Proceedings at OpenReview",
    "Privacy Policy",
    "Contact",
)
PDF_TYPE_PRIORITY = {
    "openreview_pdf": 0,
    "full_paper_pdf": 1,
    "unknown_pdf": 2,
    "supplementary_pdf": 3,
    "poster_pdf": 4,
    "slides_pdf": 5,
    "missing": 6,
}
PAPER_BODY_PDF_TYPES = {"openreview_pdf", "full_paper_pdf", "unknown_pdf"}
MAX_LOCAL_PAPER_DIR_CHARS = 72


class LinkExtractor(HTMLParser):
    """Extract links and visible link text from a small HTML document."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[dict[str, str]] = []
        self._active_links: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = {name.lower(): value or "" for name, value in attrs}
        href = attr_map.get("href", "").strip()
        if not href:
            return
        self._active_links.append({"href": href, "text_parts": []})

    def handle_data(self, data: str) -> None:
        if self._active_links:
            self._active_links[-1]["text_parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._active_links:
            return
        active = self._active_links.pop()
        href = str(active["href"])
        text = normalize_whitespace(" ".join(str(part) for part in active["text_parts"]))
        self.links.append(
            {
                "text": text,
                "href": html.unescape(href),
                "absolute_url": urllib.parse.urljoin(self.base_url, html.unescape(href)),
            }
        )


class PageMetadataParser(HTMLParser):
    """Collect visible text, metadata tags, and the HTML title."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.title_parts: list[str] = []
        self.visible_parts: list[str] = []
        self._capture_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag_lower == "title":
            self._capture_title = True
            return
        if tag_lower == "meta":
            attr_map = {name.lower(): value or "" for name, value in attrs}
            key = attr_map.get("name") or attr_map.get("property")
            content = normalize_whitespace(attr_map.get("content"))
            if key and content:
                self.meta.setdefault(key.lower(), []).append(content)
        if tag_lower in {"br", "p", "div", "section", "h1", "h2", "h3", "li"}:
            self.visible_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._capture_title:
            self.title_parts.append(data)
        self.visible_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag_lower == "title":
            self._capture_title = False
        if tag_lower in {"p", "div", "section", "h1", "h2", "h3", "li"}:
            self.visible_parts.append("\n")

    @property
    def title(self) -> str:
        return normalize_whitespace(" ".join(self.title_parts))

    @property
    def visible_text(self) -> str:
        text = "\n".join(part.strip() for part in self.visible_parts if part.strip())
        return normalize_text_block(text)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def normalize_text_block(value: str | None) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean_iclr_abstract(raw_text: str) -> str:
    text = normalize_whitespace(raw_text)
    text = re.sub(r"^Abstract\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return ""
    lower_text = text.lower()
    marker_positions = [
        lower_text.find(marker.lower())
        for marker in NOISY_ABSTRACT_MARKERS
        if lower_text.find(marker.lower()) >= 0
    ]
    if marker_positions:
        text = text[: min(marker_positions)].strip()
    text = re.sub(
        r"\b(?:Terms of Service|Code of Conduct|Accessibility|Copyright)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    word_count = len(re.findall(r"\b[A-Za-z][A-Za-z0-9\-]*\b", text))
    if word_count < 3:
        return ""
    return text


def is_noisy_abstract(text: str) -> bool:
    lowered = normalize_whitespace(text).lower()
    return any(marker.lower() in lowered for marker in NOISY_ABSTRACT_MARKERS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        msg = f"Missing JSONL input: {path}"
        raise RuntimeError(msg)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {path} on line {line_number}: {exc.msg}"
            raise RuntimeError(msg) from exc
        if not isinstance(parsed, dict):
            msg = f"Expected JSON object in {path} on line {line_number}"
            raise RuntimeError(msg)
        rows.append(parsed)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def apply_limit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def fetch_html(url: str, timeout_seconds: int, delay_seconds: float) -> str:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        msg = f"Failed to fetch HTML from {url}: {exc}"
        raise RuntimeError(msg) from exc
    return raw.decode("utf-8", errors="replace")


def extract_links_from_html(html_text: str, base_url: str) -> list[dict[str, str]]:
    parser = LinkExtractor(base_url)
    parser.feed(html_text)
    return parser.links


def parse_page_metadata(html_text: str) -> PageMetadataParser:
    parser = PageMetadataParser()
    parser.feed(html_text)
    return parser


def first_meta_value(parser: PageMetadataParser, names: tuple[str, ...]) -> str | None:
    for name in names:
        values = parser.meta.get(name.lower())
        if values:
            return values[0]
    return None


def meta_values(parser: PageMetadataParser, names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for name in names:
        values.extend(parser.meta.get(name.lower(), []))
    return dedupe_preserve_order(values)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = normalize_whitespace(value)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def first_link_url(
    links: list[dict[str, str]],
    *,
    url_contains: tuple[str, ...] = (),
    text_contains: tuple[str, ...] = (),
    url_endswith: tuple[str, ...] = (),
) -> str | None:
    for link in links:
        absolute_url = link["absolute_url"]
        url_lower = absolute_url.lower()
        text_lower = link["text"].lower()
        if url_endswith and not any(url_lower.split("?", 1)[0].endswith(s) for s in url_endswith):
            continue
        if url_contains and not any(term in url_lower for term in url_contains):
            continue
        if text_contains and not any(term in text_lower for term in text_contains):
            continue
        return absolute_url
    return None


def is_paper_specific_openreview_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if "openreview.net" not in parsed.netloc.lower():
        return False
    path = parsed.path.lower()
    query = urllib.parse.parse_qs(parsed.query)
    paper_id = (query.get("id") or [""])[0].strip()
    if path.startswith("/group") or "group?id=" in url.lower():
        return False
    if paper_id.lower().startswith("iclr.cc") or paper_id.lower() == "iclr.cc":
        return False
    return bool(paper_id) and path in {"/forum", "/pdf", "/attachment"}


def is_pdf_like_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    lowered = url.lower()
    return path.endswith(".pdf") or "openreview.net/pdf" in lowered or path == "/pdf"


def classify_pdf_url(url: str | None) -> str:
    if not url:
        return "missing"
    lowered = urllib.parse.unquote(url).lower()
    path = urllib.parse.urlparse(url).path.lower()
    if "openreview.net/pdf" in lowered or (
        "openreview.net" in lowered and path == "/pdf" and "id=" in lowered
    ):
        return "openreview_pdf"
    if "/slides/" in lowered or "slides" in lowered:
        return "slides_pdf"
    if "/posters/" in lowered or "poster" in lowered:
        return "poster_pdf"
    if "supplement" in lowered or "supp" in lowered:
        return "supplementary_pdf"
    if "full_paper" in lowered or "full-paper" in lowered or "fulltext" in lowered:
        return "full_paper_pdf"
    if path.endswith(".pdf"):
        return "unknown_pdf"
    return "missing"


def pdf_candidates_from_urls(urls: list[str]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for url in urls:
        normalized_url = url.strip()
        if not normalized_url or normalized_url in seen:
            continue
        pdf_link_type = classify_pdf_url(normalized_url)
        if pdf_link_type == "missing":
            continue
        seen.add(normalized_url)
        candidates.append({"url": normalized_url, "pdf_link_type": pdf_link_type})
    return candidates


def select_preferred_pdf_url(candidate_urls: list[str]) -> tuple[str | None, str]:
    candidates = pdf_candidates_from_urls(candidate_urls)
    if not candidates:
        return None, "missing"
    selected = min(
        candidates,
        key=lambda candidate: (
            PDF_TYPE_PRIORITY.get(candidate["pdf_link_type"], PDF_TYPE_PRIORITY["missing"]),
            candidate["url"],
        ),
    )
    return selected["url"], selected["pdf_link_type"]


def pdf_candidates_from_links(links: list[dict[str, str]]) -> list[dict[str, str]]:
    return pdf_candidates_from_urls(
        [link["absolute_url"] for link in links if is_pdf_like_url(link["absolute_url"])]
    )


def select_openreview_forum_url(links: list[dict[str, str]]) -> tuple[str | None, bool]:
    openreview_urls = [
        link["absolute_url"] for link in links if "openreview.net" in link["absolute_url"].lower()
    ]
    paper_specific_urls = [url for url in openreview_urls if is_paper_specific_openreview_url(url)]
    generic_rejected = bool(openreview_urls) and len(paper_specific_urls) < len(openreview_urls)
    forum_urls = [
        url for url in paper_specific_urls if urllib.parse.urlparse(url).path.lower() == "/forum"
    ]
    if forum_urls:
        return forum_urls[0], generic_rejected
    return None, generic_rejected


def abstract_quality_fields(abstract: str | None) -> dict[str, Any]:
    text = abstract or ""
    if not text:
        status = "missing"
    elif is_noisy_abstract(text):
        status = "noisy"
    else:
        status = "clean"
    return {
        "abstract_quality_status": status,
        "abstract_char_count": len(text),
        "abstract_word_count": len(re.findall(r"\b\w+\b", text)),
    }


def pdf_availability_fields(
    pdf_candidates: list[dict[str, str]], pdf_url: str | None
) -> dict[str, Any]:
    pdf_link_type = classify_pdf_url(pdf_url)
    candidate_types = {candidate["pdf_link_type"] for candidate in pdf_candidates}
    paper_body_available = bool(pdf_url) and pdf_link_type in PAPER_BODY_PDF_TYPES
    return {
        "pdf_url_candidates": pdf_candidates,
        "pdf_link_type": pdf_link_type,
        "paper_body_available": paper_body_available,
        "slides_available": "slides_pdf" in candidate_types,
        "poster_available": "poster_pdf" in candidate_types,
        "supplementary_available": "supplementary_pdf" in candidate_types,
    }


def ready_for_text_extraction(record: dict[str, Any]) -> bool:
    pdf_url = record.get("pdf_url_enriched") or record.get("pdf_url")
    abstract_status = str(record.get("abstract_quality_status") or "missing")
    return bool(record.get("paper_body_available") and pdf_url) and abstract_status in {
        "clean",
        "missing",
    }


def extract_title_from_metadata(parser: PageMetadataParser) -> str | None:
    title = first_meta_value(
        parser,
        (
            "citation_title",
            "og:title",
            "twitter:title",
            "dc.title",
        ),
    )
    if title:
        return strip_site_suffix(title)
    if parser.title:
        return strip_site_suffix(parser.title)
    match = re.search(r"(?im)^\s*Title\s*[:\n]\s*(.+)$", parser.visible_text)
    if match:
        return normalize_whitespace(match.group(1))
    return None


def strip_site_suffix(title: str) -> str:
    return re.sub(r"\s*[\-|]\s*(OpenReview|ICLR.*)$", "", normalize_whitespace(title)).strip()


def split_authors(value: str) -> list[str]:
    text = normalize_whitespace(value)
    if not text:
        return []
    text = re.sub(r"^(Authors?|By)\s*:\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"\s*;\s*|\s*,\s*|\s+\band\b\s+", text)
    return [
        part.strip()
        for part in parts
        if part.strip()
        and len(part.strip()) <= 120
        and not part.strip().lower().startswith("abstract")
    ]


def extract_authors_from_metadata(parser: PageMetadataParser) -> list[str]:
    authors = meta_values(
        parser,
        (
            "citation_author",
            "dc.creator",
            "author",
        ),
    )
    if authors:
        return authors
    text = parser.visible_text
    match = re.search(
        r"(?is)(?:^|\n)\s*Authors?\s*[:\n]\s*(.+?)(?:\n\s*(?:Abstract|Keywords|TL;DR|PDF)\b|$)",
        text,
    )
    if match:
        return split_authors(match.group(1))
    return []


def extract_abstract_from_metadata(parser: PageMetadataParser) -> str | None:
    abstract = first_meta_value(
        parser,
        (
            "citation_abstract",
            "description",
            "og:description",
            "twitter:description",
            "dc.description",
        ),
    )
    if abstract and not looks_like_navigation_text(abstract):
        return abstract
    text = parser.visible_text
    match = re.search(
        (
            r"(?is)(?:^|\n)\s*Abstract\s*[:\n]\s*(.+?)"
            r"(?=\n\s*(?:1\s+)?(?:Introduction|Keywords|Related Work|Background|Method|PDF)\b|$)"
        ),
        text,
    )
    if match:
        return normalize_text_block(match.group(1))
    return None


def looks_like_navigation_text(value: str) -> bool:
    text = value.lower()
    return any(term in text for term in ("openreview", "sign in", "conference management"))


def parse_keywords_from_metadata(parser: PageMetadataParser) -> list[str]:
    keywords = meta_values(parser, ("citation_keywords", "keywords"))
    parsed_keywords: list[str] = []
    for value in keywords:
        parsed_keywords.extend(part.strip() for part in re.split(r"[,;]", value) if part.strip())
    return dedupe_preserve_order(parsed_keywords)


def parse_iclr_poster_page(html_text: str, source_url: str) -> dict[str, Any]:
    links = extract_links_from_html(html_text, source_url)
    parser = parse_page_metadata(html_text)
    openreview_url, generic_openreview_rejected = select_openreview_forum_url(links)
    pdf_candidates = pdf_candidates_from_links(links)
    pdf_url, pdf_link_type = select_preferred_pdf_url(
        [candidate["url"] for candidate in pdf_candidates]
    )
    raw_abstract = extract_abstract_from_metadata(parser)
    abstract = clean_iclr_abstract(raw_abstract or "")
    supplementary_url = first_link_url(
        links,
        url_contains=("supplement", "attachment"),
    ) or first_link_url(links, text_contains=("supplement", "appendix"))
    presentation_url = first_link_url(links, text_contains=("presentation", "video", "slides"))
    poster_url = first_link_url(links, text_contains=("poster",)) or (
        source_url if "/poster/" in source_url else None
    )
    return {
        "source_url": source_url,
        "title": extract_title_from_metadata(parser),
        "authors": extract_authors_from_metadata(parser),
        "abstract": abstract,
        **abstract_quality_fields(abstract),
        "openreview_url": openreview_url,
        "paper_specific_openreview_found": bool(openreview_url),
        "generic_openreview_rejected": generic_openreview_rejected,
        "pdf_url": pdf_url,
        **pdf_availability_fields(pdf_candidates, pdf_url),
        "ready_for_text_extraction": bool(
            pdf_url and pdf_link_type in PAPER_BODY_PDF_TYPES and not is_noisy_abstract(abstract)
        ),
        "supplementary_url": supplementary_url,
        "presentation_url": presentation_url,
        "poster_url": poster_url,
    }


def parse_openreview_page(html_text: str, source_url: str) -> dict[str, Any]:
    links = extract_links_from_html(html_text, source_url)
    parser = parse_page_metadata(html_text)
    pdf_candidates = pdf_candidates_from_links(links)
    pdf_url, pdf_link_type = select_preferred_pdf_url(
        [candidate["url"] for candidate in pdf_candidates]
    )
    forum_url = (
        source_url
        if is_paper_specific_openreview_url(source_url)
        and urllib.parse.urlparse(source_url).path.lower() == "/forum"
        else select_openreview_forum_url(links)[0]
    )
    venue = first_meta_value(parser, ("citation_conference_title", "citation_journal_title"))
    if not venue:
        venue_match = re.search(r"(?i)\bICLR\s+20\d{2}\b", parser.visible_text)
        venue = venue_match.group(0) if venue_match else None
    abstract = clean_iclr_abstract(extract_abstract_from_metadata(parser) or "")
    return {
        "source_url": source_url,
        "title": extract_title_from_metadata(parser),
        "authors": extract_authors_from_metadata(parser),
        "abstract": abstract,
        **abstract_quality_fields(abstract),
        "pdf_url": pdf_url,
        **pdf_availability_fields(pdf_candidates, pdf_url),
        "ready_for_text_extraction": bool(
            pdf_url and pdf_link_type in PAPER_BODY_PDF_TYPES and not is_noisy_abstract(abstract)
        ),
        "forum_url": forum_url,
        "venue": venue,
        "keywords": parse_keywords_from_metadata(parser),
    }


def choose_first_text(*values: Any) -> str | None:
    for value in values:
        text = normalize_text_block(str(value)) if value not in (None, "", []) else ""
        if text:
            return text
    return None


def choose_first_list(*values: Any) -> list[str]:
    for value in values:
        if isinstance(value, list):
            parsed = [str(item).strip() for item in value if str(item).strip()]
        elif value:
            parsed = split_authors(str(value))
        else:
            parsed = []
        if parsed:
            return parsed
    return []


def enrich_paper_record(
    record: dict[str, Any], fetched_pages: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    iclr_page = fetched_pages.get("iclr", {})
    openreview_page = fetched_pages.get("openreview", {})
    original_authors = record.get("authors")
    original_abstract = record.get("abstract") or record.get("abstract_enriched")
    original_pdf_url = record.get("pdf_url")
    original_openreview_url = record.get("openreview_url")

    authors = choose_first_list(
        openreview_page.get("authors"),
        iclr_page.get("authors"),
        original_authors,
    )
    raw_abstract = choose_first_text(
        openreview_page.get("abstract"),
        iclr_page.get("abstract"),
        original_abstract,
    )
    abstract = clean_iclr_abstract(raw_abstract or "")
    pdf_candidates = pdf_candidates_from_urls(
        [
            str(candidate.get("url") or "")
            for page in (openreview_page, iclr_page)
            for candidate in page.get("pdf_url_candidates", [])
            if isinstance(candidate, dict)
        ]
    )
    page_pdf_urls = [
        str(page.get("pdf_url") or "")
        for page in (openreview_page, iclr_page)
        if page.get("pdf_url")
    ]
    if page_pdf_urls:
        pdf_candidates = pdf_candidates_from_urls(
            [*(candidate["url"] for candidate in pdf_candidates), *page_pdf_urls]
        )
    if original_pdf_url:
        pdf_candidates = pdf_candidates_from_urls(
            [*(candidate["url"] for candidate in pdf_candidates), str(original_pdf_url)]
        )
    pdf_url, pdf_link_type = select_preferred_pdf_url(
        [candidate["url"] for candidate in pdf_candidates]
    )
    openreview_candidate = choose_first_text(
        iclr_page.get("openreview_url"),
        openreview_page.get("forum_url"),
        original_openreview_url,
    )
    openreview_url = (
        openreview_candidate
        if openreview_candidate and is_paper_specific_openreview_url(openreview_candidate)
        else None
    )
    paper_specific_openreview_found = bool(openreview_url)
    generic_openreview_rejected = bool(
        iclr_page.get("generic_openreview_rejected")
        or openreview_page.get("generic_openreview_rejected")
        or (
            openreview_candidate and not openreview_url and "openreview.net" in openreview_candidate
        )
    )
    source_pages_checked = [
        str(page.get("source_url"))
        for page in (iclr_page, openreview_page)
        if page.get("source_url")
    ]
    errors = [str(page.get("error")) for page in (iclr_page, openreview_page) if page.get("error")]
    missing_fields_after_enrichment = [
        field
        for field, value in (
            ("authors", authors),
            ("abstract", abstract),
            ("pdf_url", pdf_url),
            ("openreview_url", openreview_url),
        )
        if value in (None, "", [])
    ]
    quality_notes: list[str] = []
    abstract_quality = abstract_quality_fields(abstract)
    pdf_availability = pdf_availability_fields(pdf_candidates, pdf_url)
    if abstract_quality["abstract_quality_status"] == "noisy":
        quality_notes.append("Abstract still contains boilerplate and needs review.")
    if not paper_specific_openreview_found:
        quality_notes.append("No paper-specific OpenReview forum URL was found.")
    if generic_openreview_rejected:
        quality_notes.append("Generic OpenReview group links were rejected.")
    if not pdf_availability["paper_body_available"]:
        quality_notes.append("No full-paper PDF source is available yet.")
    useful_fields_found = sum(
        1 for value in (authors, abstract, pdf_url, openreview_url) if value not in (None, "", [])
    )
    if not missing_fields_after_enrichment and not errors:
        status = "success"
    elif useful_fields_found > 0 or source_pages_checked:
        status = "partial"
    else:
        status = "failed"
    record_ready_for_text_extraction = bool(
        pdf_availability["paper_body_available"]
        and pdf_url
        and abstract_quality["abstract_quality_status"] in {"clean", "missing"}
    )

    enriched = dict(record)
    enriched.update(
        {
            "enriched_metadata_status": status,
            "enriched_at_utc": utc_now(),
            "authors_enriched": authors,
            "abstract_enriched": abstract,
            "pdf_url_enriched": pdf_url,
            "openreview_url": openreview_url,
            "paper_specific_openreview_found": paper_specific_openreview_found,
            "generic_openreview_rejected": generic_openreview_rejected,
            "source_pages_checked": source_pages_checked,
            "missing_fields_after_enrichment": missing_fields_after_enrichment,
            "enrichment_errors": errors,
            **abstract_quality,
            **pdf_availability,
            "ready_for_text_extraction": record_ready_for_text_extraction,
            "enrichment_quality_notes": quality_notes,
        }
    )
    if authors and not enriched.get("authors"):
        enriched["authors"] = authors
    if pdf_url and not enriched.get("pdf_url"):
        enriched["pdf_url"] = pdf_url
    if abstract and not enriched.get("abstract"):
        enriched["abstract"] = abstract
    return enriched


def stable_short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def safe_local_paper_dir_name(
    record: dict[str, Any], max_chars: int = MAX_LOCAL_PAPER_DIR_CHARS
) -> str:
    paper_id = str(record.get("paper_id") or "unknown_paper").strip() or "unknown_paper"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", paper_id.lower()).strip("_") or "unknown_paper"
    if len(slug) <= max_chars:
        return slug

    suffix = stable_short_hash(paper_id)
    prefix_chars = max(1, max_chars - len(suffix) - 1)
    prefix = slug[:prefix_chars].rstrip("_") or "paper"
    return f"{prefix}_{suffix}"


def build_local_pdf_path(record: dict[str, Any], raw_paper_dir: Path) -> Path:
    return raw_paper_dir / safe_local_paper_dir_name(record) / "paper.pdf"


def parse_retry_after_seconds(headers: Any) -> float | None:
    if not headers:
        return None
    retry_after = headers.get("Retry-After")
    if retry_after is None:
        return None
    retry_after_text = str(retry_after).strip()
    if not retry_after_text:
        return None
    try:
        return max(0.0, float(retry_after_text))
    except ValueError:
        return None


def download_failure_result(
    url: str,
    destination: Path,
    attempts: int,
    error_message: str,
    status_code: int | None = None,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "download_status": "failed",
        "url": url,
        "source_url": url,
        "destination": str(destination),
        "local_pdf_path": str(destination),
        "bytes_written": 0,
        "file_size_bytes": 0,
        "sha256": None,
        "content_type": None,
        "downloaded_at_utc": utc_now(),
        "attempts": attempts,
        "status_code": status_code,
        "error_message": error_message,
    }


def download_binary_with_retries(
    url: str,
    destination: Path,
    timeout_seconds: int,
    request_delay_seconds: float,
    max_retries: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_attempts = max(0, int(max_retries)) + 1
    last_error_message = ""
    last_status_code: int | None = None
    attempts_used = 0
    for attempt in range(1, total_attempts + 1):
        attempts_used = attempt
        if request_delay_seconds > 0:
            time.sleep(request_delay_seconds)
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/pdf,application/octet-stream,*/*",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
                headers = getattr(response, "headers", None)
                content_type = headers.get("Content-Type") if headers else None
        except urllib.error.HTTPError as exc:
            last_status_code = int(exc.code)
            last_error_message = str(exc)
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt >= total_attempts:
                break
            sleep_seconds = parse_retry_after_seconds(exc.headers) if exc.code == 429 else None
            if sleep_seconds is None:
                sleep_seconds = backoff_seconds * attempt
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error_message = str(exc)
            if attempt >= total_attempts:
                break
            sleep_seconds = backoff_seconds * attempt
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            continue

        try:
            destination.write_bytes(payload)
        except OSError as exc:
            return download_failure_result(
                url,
                destination,
                attempts=attempt,
                error_message=f"Failed to write PDF to {destination}: {exc}",
            )
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        return {
            "status": "downloaded",
            "download_status": "downloaded",
            "url": url,
            "source_url": url,
            "destination": str(destination),
            "local_pdf_path": str(destination),
            "bytes_written": len(payload),
            "file_size_bytes": len(payload),
            "sha256": payload_sha256,
            "content_type": content_type,
            "downloaded_at_utc": utc_now(),
            "attempts": attempt,
            "status_code": None,
            "error_message": "",
        }
    return download_failure_result(
        url,
        destination,
        attempts=attempts_used or total_attempts,
        status_code=last_status_code,
        error_message=last_error_message or "Download failed.",
    )


def download_binary(
    url: str,
    destination: Path,
    timeout_seconds: int,
    delay_seconds: float,
) -> dict[str, Any]:
    result = download_binary_with_retries(
        url,
        destination,
        timeout_seconds=timeout_seconds,
        request_delay_seconds=delay_seconds,
        max_retries=0,
        backoff_seconds=0,
    )
    if result.get("download_status") == "failed":
        return result
    payload_sha256 = str(result.get("sha256") or "")
    return {
        "status": "downloaded",
        "download_status": "downloaded",
        "url": url,
        "source_url": url,
        "destination": str(destination),
        "local_pdf_path": str(destination),
        "bytes_written": result.get("bytes_written"),
        "file_size_bytes": result.get("file_size_bytes"),
        "sha256": payload_sha256,
        "content_type": result.get("content_type"),
        "downloaded_at_utc": result.get("downloaded_at_utc"),
        "attempts": result.get("attempts"),
        "status_code": result.get("status_code"),
        "error_message": "",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_pdf_text_extraction_backend() -> str:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        reader = getattr(module, "PdfReader", None)
        if reader is not None:
            return module_name
    return "missing"


def load_pdf_reader() -> tuple[Any | None, str]:
    backend = get_pdf_text_extraction_backend()
    if backend == "missing":
        return None, "skipped_missing_dependency"
    module = importlib.import_module(backend)
    return module.PdfReader, backend


def extract_text_from_pdf_with_error(pdf_path: Path) -> tuple[str, str, str]:
    reader_class, method = load_pdf_reader()
    if reader_class is None:
        return (
            "",
            method,
            PDF_EXTRACTION_DEPENDENCY_WARNING,
        )
    try:
        reader = reader_class(str(pdf_path))
        pages = getattr(reader, "pages", [])
        page_text = []
        for index, page in enumerate(pages, start=1):
            extracted = str(page.extract_text() or "").strip()
            if extracted:
                page_text.append(f"--- Page {index} ---\n{extracted}")
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop preparation.
        return "", f"failed_{method}", str(exc)
    text = normalize_text_block("\n\n".join(page_text))
    if not text:
        return "", f"failed_{method}_empty_text", "No text was extracted from local PDF."
    return text, method, ""


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, str]:
    extracted_text, method, _error_message = extract_text_from_pdf_with_error(pdf_path)
    return extracted_text, method


def write_text_file(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "local_text_path": str(path),
        "text_char_count": len(text),
        "text_word_count": len(re.findall(r"\b\w+\b", text)),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def repair_spaced_heading_words(value: str) -> str:
    text = value
    for _ in range(4):
        repaired = re.sub(r"\b([A-Z])\s+([A-Z][A-Z]+)\b", r"\1\2", text)
        if repaired == text:
            break
        text = repaired
    return text


def normalized_heading_key(value: str) -> str:
    value = repair_spaced_heading_words(value)
    text = normalize_whitespace(value).lower()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_section_type(section_title: str) -> str:
    lowered = normalized_heading_key(section_title)
    if lowered.startswith("appendix"):
        return "appendix"
    for section_type, headings in PAPER_SECTION_HEADINGS.items():
        if lowered in {normalized_heading_key(heading) for heading in headings}:
            return section_type
    return "other"


def section_heading_aliases() -> list[tuple[str, str, str]]:
    aliases: list[tuple[str, str, str]] = []
    for section_type, headings in PAPER_SECTION_HEADINGS.items():
        for heading in headings:
            aliases.append((normalized_heading_key(heading), section_type, heading))
    return sorted(aliases, key=lambda item: len(item[0]), reverse=True)


def match_section_heading_alias(
    value: str,
    *,
    allow_prefix: bool,
) -> tuple[str, str] | None:
    normalized = normalized_heading_key(value)
    if not normalized:
        return None
    if normalized.startswith("appendix"):
        return "Appendix", "appendix"
    for alias, section_type, canonical_title in section_heading_aliases():
        if normalized == alias:
            return canonical_title, section_type
        if allow_prefix and normalized.startswith(f"{alias} "):
            return canonical_title, section_type
    if normalized.endswith(" results") and len(normalized.split()) <= 5:
        return "Results", "results"
    return None


def is_false_positive_section_heading(line: str, *, numbered: bool) -> bool:
    text = normalize_whitespace(line)
    if not text:
        return True
    if re.match(r"^-+\s*Page\s+\d+\s*-+$", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^(fig\.?|figure|table)\s+\d+", text, flags=re.IGNORECASE):
        return True
    if re.match(r"^\[\d+\]", text):
        return True
    if len(text) > (240 if numbered else 120):
        return True
    if not numbered and text.endswith("."):
        return True
    return False


def parse_section_heading_line(line: str) -> tuple[str, str] | None:
    text = normalize_whitespace(line)
    if not text:
        return None
    numbered_match = re.match(
        r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.?|[A-Z]\.?)\s+(?P<title>.+?)\s*$",
        text,
    )
    if numbered_match:
        candidate = numbered_match.group("title").strip(" :-\u2013\u2014")
        if is_false_positive_section_heading(candidate, numbered=True):
            return None
        return match_section_heading_alias(candidate, allow_prefix=True)
    if is_false_positive_section_heading(text, numbered=False):
        return None
    stripped = text.strip(" :-\u2013\u2014")
    exact_match = match_section_heading_alias(stripped, allow_prefix=False)
    if exact_match is not None:
        return exact_match
    inline_heading_match = re.match(r"^(?P<title>[^.:]{3,60})[.:]\s+\S+", stripped)
    if inline_heading_match:
        return match_section_heading_alias(
            inline_heading_match.group("title"),
            allow_prefix=True,
        )
    return None


def iter_section_heading_matches(text: str) -> list[tuple[int, str, str]]:
    matches: list[tuple[int, str, str]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        parsed = parse_section_heading_line(line)
        if parsed is not None:
            section_title, section_type = parsed
            matches.append(
                (offset + len(raw_line) - len(raw_line.lstrip()), section_title, section_type)
            )
        offset += len(raw_line)
    return matches


def detect_research_paper_sections(text: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    matches = iter_section_heading_matches(text)
    if not matches:
        return []
    paper_id = str(record.get("paper_id") or "unknown_paper")
    title = str(record.get("title") or "")
    local_text_path = str(record.get("local_text_path") or "")
    extraction_method = str(record.get("extraction_method") or "")
    sections: list[dict[str, Any]] = []
    for index, (section_start, section_title, section_type) in enumerate(matches):
        section_end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end].strip()
        sections.append(
            {
                "section_record_id": (
                    f"research_ai_section_{paper_id}_{index + 1:03d}_{section_type}"
                ),
                "paper_id": paper_id,
                "title": title,
                "section_type": section_type,
                "section_title": section_title,
                "section_start_char": section_start,
                "section_end_char": section_end,
                "char_count": len(section_text),
                "word_count": len(re.findall(r"\b\w+\b", section_text)),
                "extraction_method": extraction_method,
                "local_text_path": local_text_path,
            }
        )
    return sections


def counts_by_topic(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        topics = record.get("topics")
        if isinstance(topics, list):
            counter.update(str(topic) for topic in topics)
        elif record.get("topic"):
            counter[str(record["topic"])] += 1
    return dict(counter)


def counts_by_source(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("source") or "") for record in records))


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def section_rows_by_paper(section_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for section in section_rows:
        paper_id = str(section.get("paper_id") or "")
        if not paper_id:
            continue
        grouped.setdefault(paper_id, []).append(section)
    return grouped


def paper_titles_by_id(
    text_manifest_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> dict[str, str]:
    titles: dict[str, str] = {}
    for row in [*text_manifest_rows, *section_rows]:
        paper_id = str(row.get("paper_id") or "")
        title = str(row.get("title") or "")
        if paper_id and title and paper_id not in titles:
            titles[paper_id] = title
    return titles


def build_section_quality_records(
    section_rows: list[dict[str, Any]],
    text_manifest_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    text_rows = text_manifest_rows or []
    grouped_sections = section_rows_by_paper(section_rows)
    titles = paper_titles_by_id(text_rows, section_rows)
    paper_ids = set(grouped_sections)
    paper_ids.update(
        str(row.get("paper_id"))
        for row in text_rows
        if row.get("paper_id") and str(row.get("text_extraction_status") or "") == "extracted"
    )
    quality_records: list[dict[str, Any]] = []
    for paper_id in sorted(paper_ids):
        rows = grouped_sections.get(paper_id, [])
        section_types = {str(row.get("section_type") or "other") for row in rows}
        useful_types = section_types - {"references", "appendix"}
        has_abstract = "abstract" in section_types
        has_introduction = "introduction" in section_types
        has_method_or_approach = bool(section_types & {"method", "approach"})
        has_experiments_or_results = bool(
            section_types & {"experiments", "evaluation", "results", "analysis"}
        )
        has_limitations = "limitations" in section_types
        has_conclusion = "conclusion" in section_types
        has_references = "references" in section_types
        largest_section_word_count = max(
            (safe_int(row.get("word_count")) for row in rows), default=0
        )
        if (
            (has_abstract or has_introduction)
            and has_method_or_approach
            and (has_experiments_or_results or has_conclusion)
        ):
            quality_status = "good"
        elif len(useful_types) >= 2:
            quality_status = "partial"
        else:
            quality_status = "poor"
        quality_records.append(
            {
                "paper_id": paper_id,
                "title": titles.get(paper_id, ""),
                "sections_detected_count": len(rows),
                "has_abstract": has_abstract,
                "has_introduction": has_introduction,
                "has_method_or_approach": has_method_or_approach,
                "has_experiments_or_results": has_experiments_or_results,
                "has_limitations": has_limitations,
                "has_conclusion": has_conclusion,
                "has_references": has_references,
                "largest_section_word_count": largest_section_word_count,
                "section_quality_status": quality_status,
            }
        )
    return quality_records


def aggregate_section_quality_metrics(
    section_rows: list[dict[str, Any]],
    text_manifest_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    quality_records = build_section_quality_records(section_rows, text_manifest_rows)
    quality_statuses = Counter(str(record["section_quality_status"]) for record in quality_records)
    section_counts_by_type = Counter(
        str(row.get("section_type") or "other") for row in section_rows
    )
    return {
        "section_quality_records": quality_records,
        "section_counts_by_type": dict(section_counts_by_type),
        "papers_with_good_sections_count": quality_statuses.get("good", 0),
        "papers_with_partial_sections_count": quality_statuses.get("partial", 0),
        "papers_with_poor_sections_count": quality_statuses.get("poor", 0),
        "papers_with_method_sections_count": sum(
            1 for record in quality_records if record["has_method_or_approach"]
        ),
        "papers_with_results_sections_count": sum(
            1 for record in quality_records if record["has_experiments_or_results"]
        ),
        "papers_with_limitations_sections_count": sum(
            1 for record in quality_records if record["has_limitations"]
        ),
        "no_sections_detected_count": sum(
            1 for record in quality_records if record["sections_detected_count"] == 0
        ),
        "poor_section_quality_titles": [
            record["title"] or record["paper_id"]
            for record in quality_records
            if record["section_quality_status"] == "poor"
        ],
    }


def build_paper_preparation_report(
    approved_records: list[dict[str, Any]] | None = None,
    enriched_records: list[dict[str, Any]] | None = None,
    text_manifest_rows: list[dict[str, Any]] | None = None,
    section_rows: list[dict[str, Any]] | None = None,
    output_files: dict[str, str] | None = None,
    download_policy: dict[str, Any] | None = None,
    pdf_text_backend: str | None = None,
) -> dict[str, Any]:
    approved = approved_records or []
    enriched = enriched_records or []
    text_rows = text_manifest_rows or []
    sections = section_rows or []
    records_for_counts = enriched or approved
    enrichment_statuses = Counter(
        str(record.get("enriched_metadata_status") or "") for record in enriched
    )
    pdf_download_statuses = Counter(
        str(record.get("pdf_download_status") or "") for record in enriched
    )
    text_statuses = Counter(str(row.get("text_extraction_status") or "") for row in text_rows)
    backend = pdf_text_backend or get_pdf_text_extraction_backend()
    abstract_quality_counter = Counter(
        str(record.get("abstract_quality_status") or "missing") for record in enriched
    )
    pdf_candidate_type_counter: Counter[str] = Counter()
    selected_pdf_type_counter = Counter(
        str(
            record.get("pdf_link_type")
            or classify_pdf_url(record.get("pdf_url_enriched") or record.get("pdf_url"))
        )
        for record in enriched
    )
    for record in enriched:
        candidates = record.get("pdf_url_candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    pdf_candidate_type_counter[
                        str(candidate.get("pdf_link_type") or "missing")
                    ] += 1
        elif record.get("pdf_url_enriched") or record.get("pdf_url"):
            pdf_candidate_type_counter[
                classify_pdf_url(str(record.get("pdf_url_enriched") or record.get("pdf_url")))
            ] += 1
    paper_body_available_count = sum(1 for record in enriched if record.get("paper_body_available"))
    records_ready_for_text_extraction = sum(
        1
        for record in enriched
        if bool(record.get("ready_for_text_extraction")) or ready_for_text_extraction(record)
    )
    warnings = [
        "This is paper detail acquisition and text preparation only.",
        "No RAG, retrieval, embeddings, prompt assembly, or inference is performed.",
        "Some conference records may not expose PDFs or abstracts on the listing page.",
        (
            "Phase 2A-5B should create curated Research AI prompts/KB/gold only "
            "from records with enough metadata/text evidence."
        ),
    ]
    if abstract_quality_counter.get("noisy", 0) > 0:
        warnings.append("Some enriched abstracts still look noisy and need review.")
    if enriched and paper_body_available_count == 0:
        warnings.append(
            "No full-paper PDF bodies are available; do not generate method/result prompts yet."
        )
    slide_or_poster_count = pdf_candidate_type_counter.get(
        "slides_pdf", 0
    ) + pdf_candidate_type_counter.get("poster_pdf", 0)
    if slide_or_poster_count > paper_body_available_count:
        warnings.append("Many PDF links are slides or posters and are not full paper text sources.")
    if text_rows and all(
        str(row.get("text_extraction_status") or "") == "skipped_missing_pdf" for row in text_rows
    ):
        warnings.append(
            "No local PDFs were found. Run --download-pdfs --skip-existing before --extract-text."
        )
    if backend == "missing":
        warnings.append(PDF_EXTRACTION_DEPENDENCY_WARNING)
    local_pdf_count = sum(
        1
        for row in text_rows
        if row.get("local_pdf_path") and Path(str(row["local_pdf_path"])).exists()
    )
    section_metrics = aggregate_section_quality_metrics(sections, text_rows)
    no_sections_detected_count = int(section_metrics["no_sections_detected_count"])
    if no_sections_detected_count > 0:
        warnings.append("Some extracted texts did not yield recognized section headings.")
    if int(section_metrics["papers_with_poor_sections_count"]) > 0:
        warnings.append("Some papers have poor section coverage and need review before 2A-5B.")
    pdf_download_failure_details = [
        {
            "paper_id": record.get("paper_id"),
            "title": record.get("title"),
            "pdf_url": record.get("pdf_url_enriched") or record.get("pdf_url"),
            "status_code": record.get("pdf_download_status_code"),
            "attempts": record.get("pdf_download_attempts"),
            "error_message": record.get("pdf_download_error"),
        }
        for record in enriched
        if record.get("pdf_download_status") == "failed"
    ]
    policy = download_policy or {}
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "approved_record_count": len(approved),
        "enriched_record_count": len(enriched),
        "enrichment_success_count": enrichment_statuses.get("success", 0),
        "enrichment_partial_count": enrichment_statuses.get("partial", 0),
        "enrichment_failed_count": enrichment_statuses.get("failed", 0),
        "pdf_urls_found": sum(
            1 for record in enriched if record.get("pdf_url_enriched") or record.get("pdf_url")
        ),
        "pdfs_downloaded": (
            pdf_download_statuses.get("downloaded", 0) + pdf_download_statuses.get("existing", 0)
        ),
        "pdf_download_failures": pdf_download_statuses.get("failed", 0),
        "pdf_download_failure_details": pdf_download_failure_details,
        "download_max_retries": policy.get("download_max_retries"),
        "download_backoff_seconds": policy.get("download_backoff_seconds"),
        "request_delay_seconds": policy.get("request_delay_seconds"),
        "download_timeout_seconds": policy.get("download_timeout_seconds"),
        "failed_only": policy.get("failed_only"),
        "paper_id_filter": policy.get("paper_id_filter"),
        "pdf_text_backend": backend,
        "local_pdf_count": local_pdf_count,
        "clean_abstract_count": abstract_quality_counter.get("clean", 0),
        "noisy_abstract_count": abstract_quality_counter.get("noisy", 0),
        "missing_abstract_count": abstract_quality_counter.get("missing", 0),
        "paper_body_available_count": paper_body_available_count,
        "openreview_pdf_count": pdf_candidate_type_counter.get("openreview_pdf", 0),
        "full_paper_pdf_count": pdf_candidate_type_counter.get("full_paper_pdf", 0),
        "unknown_pdf_count": pdf_candidate_type_counter.get("unknown_pdf", 0),
        "slides_pdf_count": pdf_candidate_type_counter.get("slides_pdf", 0),
        "poster_pdf_count": pdf_candidate_type_counter.get("poster_pdf", 0),
        "supplementary_pdf_count": pdf_candidate_type_counter.get("supplementary_pdf", 0),
        "missing_pdf_count": selected_pdf_type_counter.get("missing", 0),
        "paper_specific_openreview_count": sum(
            1 for record in enriched if record.get("paper_specific_openreview_found")
        ),
        "generic_openreview_rejected_count": sum(
            1 for record in enriched if record.get("generic_openreview_rejected")
        ),
        "records_ready_for_text_extraction": records_ready_for_text_extraction,
        "text_extracted_count": text_statuses.get("extracted", 0),
        "text_extraction_skipped_count": sum(
            count for status, count in text_statuses.items() if status.startswith("skipped")
        ),
        "text_extraction_failed_count": sum(
            count for status, count in text_statuses.items() if status.startswith("failed")
        ),
        "sections_extracted_count": len(sections),
        "no_sections_detected_count": no_sections_detected_count,
        "papers_with_good_sections_count": section_metrics["papers_with_good_sections_count"],
        "papers_with_partial_sections_count": section_metrics["papers_with_partial_sections_count"],
        "papers_with_poor_sections_count": section_metrics["papers_with_poor_sections_count"],
        "papers_with_method_sections_count": section_metrics["papers_with_method_sections_count"],
        "papers_with_results_sections_count": section_metrics["papers_with_results_sections_count"],
        "papers_with_limitations_sections_count": section_metrics[
            "papers_with_limitations_sections_count"
        ],
        "section_counts_by_type": section_metrics["section_counts_by_type"],
        "poor_section_quality_titles": section_metrics["poor_section_quality_titles"],
        "counts_by_topic": counts_by_topic(records_for_counts),
        "counts_by_source": counts_by_source(records_for_counts),
        "missing_authors_count": sum(
            1
            for record in records_for_counts
            if not record.get("authors_enriched") and not record.get("authors")
        ),
        "missing_pdf_url_count": sum(
            1
            for record in records_for_counts
            if not record.get("pdf_url_enriched") and not record.get("pdf_url")
        ),
        "output_files": output_files or {},
        "warnings": warnings,
        "next_step": (
            "Proceed to --download-pdfs only for records with paper_body_available=true. "
            "If coverage is low, enrich from OpenReview/arXiv manually before 2A-5B."
        ),
    }


def output_files_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "approved_registry_path": str(args.approved_registry_path),
        "enriched_registry_path": str(args.enriched_registry_path),
        "raw_paper_dir": str(args.raw_paper_dir),
        "paper_text_dir": str(args.paper_text_dir),
        "text_manifest_path": str(args.text_manifest_path),
        "sections_manifest_path": str(args.sections_manifest_path),
        "report_path": str(args.report_path),
        "section_quality_report_path": str(
            getattr(args, "section_quality_report_path", DEFAULT_SECTION_QUALITY_REPORT_PATH)
        ),
    }


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    approved_records = apply_limit(read_jsonl(args.approved_registry_path), int(args.limit))
    report = build_paper_preparation_report(
        approved_records=approved_records,
        output_files=output_files_from_args(args),
    )
    return {
        "mode": "dry_run",
        "phase": PHASE,
        "approved_record_count": len(approved_records),
        "planned_provenance_fetches": sum(
            1 for record in approved_records if record.get("provenance_url")
        ),
        "will_download_pdfs": False,
        "will_extract_text": False,
        "output_files": output_files_from_args(args),
        "warnings": report["warnings"],
        "next_step": "Run --enrich-metadata before PDF download or text extraction.",
    }


def enrich_metadata(args: argparse.Namespace) -> dict[str, Any]:
    approved_records = apply_limit(read_jsonl(args.approved_registry_path), int(args.limit))
    enriched_records: list[dict[str, Any]] = []
    for record in approved_records:
        fetched_pages: dict[str, dict[str, Any]] = {}
        provenance_url = str(record.get("provenance_url") or "")
        if provenance_url:
            try:
                iclr_html = fetch_html(
                    provenance_url,
                    timeout_seconds=int(args.timeout_seconds),
                    delay_seconds=float(args.request_delay_seconds),
                )
                fetched_pages["iclr"] = parse_iclr_poster_page(iclr_html, provenance_url)
            except RuntimeError as exc:
                fetched_pages["iclr"] = {"source_url": provenance_url, "error": str(exc)}
        else:
            fetched_pages["iclr"] = {"error": "missing provenance_url"}

        openreview_url = str(fetched_pages.get("iclr", {}).get("openreview_url") or "")
        if openreview_url:
            try:
                openreview_html = fetch_html(
                    openreview_url,
                    timeout_seconds=int(args.timeout_seconds),
                    delay_seconds=float(args.request_delay_seconds),
                )
                fetched_pages["openreview"] = parse_openreview_page(
                    openreview_html,
                    openreview_url,
                )
            except RuntimeError as exc:
                fetched_pages["openreview"] = {
                    "source_url": openreview_url,
                    "error": str(exc),
                }
        enriched_records.append(enrich_paper_record(record, fetched_pages))

    write_jsonl(args.enriched_registry_path, enriched_records)
    report = build_paper_preparation_report(
        approved_records=approved_records,
        enriched_records=enriched_records,
        output_files=output_files_from_args(args),
    )
    write_json(args.report_path, report)
    status_counter = Counter(str(row.get("enriched_metadata_status")) for row in enriched_records)
    return {
        "mode": "enrich_metadata",
        "phase": PHASE,
        "records_attempted": len(approved_records),
        "records_enriched_success": status_counter.get("success", 0),
        "records_enriched_partial": status_counter.get("partial", 0),
        "records_failed": status_counter.get("failed", 0),
        "pdf_urls_found": int(report["pdf_urls_found"]),
        "abstracts_found": sum(1 for record in enriched_records if record.get("abstract_enriched")),
        "authors_found": sum(1 for record in enriched_records if record.get("authors_enriched")),
        "enriched_registry_path": str(args.enriched_registry_path),
        "report_path": str(args.report_path),
        "warnings": report["warnings"],
    }


def existing_pdf_download_result(url: str, destination: Path, pdf_link_type: str) -> dict[str, Any]:
    return {
        "status": "existing",
        "download_status": "existing",
        "url": url,
        "source_url": url,
        "destination": str(destination),
        "local_pdf_path": str(destination),
        "pdf_link_type": pdf_link_type,
        "bytes_written": destination.stat().st_size,
        "file_size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
        "content_type": "application/pdf",
        "downloaded_at_utc": utc_now(),
        "attempts": 0,
        "status_code": None,
        "error_message": "",
    }


def apply_download_result(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated.update(
        {
            "local_pdf_path": result.get("local_pdf_path"),
            "pdf_download_status": result.get("download_status"),
            "pdf_download_type": result.get("pdf_link_type") or record.get("pdf_link_type"),
            "pdf_download_error": result.get("error_message"),
            "pdf_sha256": result.get("sha256"),
            "pdf_file_size_bytes": result.get("file_size_bytes"),
            "pdf_downloaded_at_utc": result.get("downloaded_at_utc"),
            "pdf_content_type": result.get("content_type"),
            "pdf_download_attempts": result.get("attempts"),
            "pdf_download_status_code": result.get("status_code"),
        }
    )
    return updated


def effective_download_timeout_seconds(args: argparse.Namespace) -> int:
    explicit_timeout = getattr(args, "download_timeout_seconds", None)
    if explicit_timeout is not None:
        return int(explicit_timeout)
    return int(args.timeout_seconds)


def effective_download_request_delay_seconds(args: argparse.Namespace) -> float:
    explicit_delay = getattr(args, "download_request_delay_seconds", None)
    if explicit_delay is not None:
        return float(explicit_delay)
    return float(args.request_delay_seconds)


def download_policy_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "download_max_retries": int(getattr(args, "download_max_retries", 3)),
        "download_backoff_seconds": float(getattr(args, "download_backoff_seconds", 30.0)),
        "request_delay_seconds": effective_download_request_delay_seconds(args),
        "download_timeout_seconds": effective_download_timeout_seconds(args),
        "failed_only": bool(getattr(args, "failed_only", False)),
        "paper_id_filter": str(getattr(args, "paper_id", "") or ""),
    }


def read_failed_download_paper_ids(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    failed_details = report.get("pdf_download_failure_details")
    if not isinstance(failed_details, list):
        return set()
    return {
        str(detail.get("paper_id"))
        for detail in failed_details
        if isinstance(detail, dict) and detail.get("paper_id")
    }


def download_pdfs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.enriched_registry_path.exists():
        msg = f"Missing enriched registry: {args.enriched_registry_path}"
        raise RuntimeError(msg)
    all_records = read_jsonl(args.enriched_registry_path)
    scoped_records = apply_limit(all_records, int(args.limit))
    scoped_paper_ids = {str(record.get("paper_id") or "") for record in scoped_records}
    failed_only = bool(getattr(args, "failed_only", False))
    paper_id_filter = str(getattr(args, "paper_id", "") or "")
    failed_paper_ids = read_failed_download_paper_ids(args.report_path) if failed_only else set()
    download_policy = download_policy_from_args(args)
    updated_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []
    selected_record_count = 0
    eligible_count = 0
    attempted_downloads = 0
    for record in all_records:
        paper_id = str(record.get("paper_id") or "")
        selected = paper_id in scoped_paper_ids
        if paper_id_filter:
            selected = selected and paper_id == paper_id_filter
        if failed_only:
            selected = selected and paper_id in failed_paper_ids
        if not selected:
            updated_records.append(record)
            continue

        selected_record_count += 1
        pdf_url = str(record.get("pdf_url_enriched") or record.get("pdf_url") or "")
        pdf_link_type = str(record.get("pdf_link_type") or classify_pdf_url(pdf_url))
        destination = build_local_pdf_path(record, args.raw_paper_dir)
        if not pdf_url:
            result = {
                "status": "skipped_no_pdf_url",
                "download_status": "skipped_no_pdf_url",
                "url": "",
                "source_url": "",
                "destination": str(destination),
                "local_pdf_path": str(destination),
                "pdf_link_type": pdf_link_type,
                "bytes_written": 0,
                "file_size_bytes": 0,
                "sha256": None,
                "content_type": None,
                "downloaded_at_utc": None,
                "attempts": 0,
                "status_code": None,
                "error_message": "",
            }
        elif not bool(args.include_non_paper_pdfs) and not ready_for_text_extraction(record):
            result = {
                "status": "skipped_not_full_paper",
                "download_status": "skipped_not_full_paper",
                "url": pdf_url,
                "source_url": pdf_url,
                "destination": str(destination),
                "local_pdf_path": str(destination),
                "pdf_link_type": pdf_link_type,
                "bytes_written": 0,
                "file_size_bytes": 0,
                "sha256": None,
                "content_type": None,
                "downloaded_at_utc": None,
                "attempts": 0,
                "status_code": None,
                "error_message": "PDF is not marked ready for full-paper text extraction.",
            }
        else:
            eligible_count += 1
            if bool(args.skip_existing) and destination.exists():
                result = existing_pdf_download_result(pdf_url, destination, pdf_link_type)
            else:
                attempted_downloads += 1
                result = download_binary_with_retries(
                    pdf_url,
                    destination,
                    timeout_seconds=int(download_policy["download_timeout_seconds"]),
                    request_delay_seconds=float(download_policy["request_delay_seconds"]),
                    max_retries=int(download_policy["download_max_retries"]),
                    backoff_seconds=float(download_policy["download_backoff_seconds"]),
                )
            result["pdf_link_type"] = pdf_link_type
        run_records.append(record)
        run_results.append(result)
        updated_records.append(apply_download_result(record, result))

    write_jsonl(args.enriched_registry_path, updated_records)
    approved_records = read_jsonl(args.approved_registry_path)
    report = build_paper_preparation_report(
        approved_records=approved_records,
        enriched_records=updated_records,
        output_files=output_files_from_args(args),
        download_policy=download_policy,
    )
    write_json(args.report_path, report)
    download_statuses = Counter(str(result.get("download_status") or "") for result in run_results)
    downloads_by_type = Counter(
        str(result.get("pdf_link_type") or "missing")
        for result in run_results
        if result.get("download_status") in {"downloaded", "existing"}
    )
    failure_details = [
        {
            "paper_id": record.get("paper_id"),
            "title": record.get("title"),
            "pdf_url": result.get("url") or result.get("source_url"),
            "status_code": result.get("status_code"),
            "attempts": result.get("attempts"),
            "error_message": result.get("error_message"),
        }
        for record, result in zip(run_records, run_results, strict=False)
        if result.get("download_status") == "failed"
    ]
    summary = {
        "mode": "download_pdfs",
        "phase": PHASE,
        "records_attempted": selected_record_count,
        "pdf_urls_found": sum(1 for result in run_results if result.get("url")),
        "pdfs_downloaded": download_statuses.get("downloaded", 0)
        + download_statuses.get("existing", 0),
        "pdf_download_failures": download_statuses.get("failed", 0),
        "pdfs_skipped_existing": download_statuses.get("existing", 0),
        "pdfs_skipped_no_url": download_statuses.get("skipped_no_pdf_url", 0),
        "pdfs_skipped_not_full_paper": download_statuses.get("skipped_not_full_paper", 0),
        "pdfs_downloaded_by_type": dict(downloads_by_type),
        "download_max_retries": download_policy["download_max_retries"],
        "download_backoff_seconds": download_policy["download_backoff_seconds"],
        "request_delay_seconds": download_policy["request_delay_seconds"],
        "download_timeout_seconds": download_policy["download_timeout_seconds"],
        "failed_only": failed_only,
        "paper_id_filter": paper_id_filter,
        "pdf_download_failure_details": failure_details,
        "enriched_registry_path": str(args.enriched_registry_path),
        "raw_paper_dir": str(args.raw_paper_dir),
        "report_path": str(args.report_path),
        "warnings": report["warnings"],
    }
    if eligible_count == 0:
        msg = "No records are eligible for PDF download."
        raise RuntimeError(msg)
    if attempted_downloads > 0 and download_statuses.get("failed", 0) == attempted_downloads:
        msg = "All attempted PDF downloads failed."
        raise RuntimeError(msg)
    return summary


def build_text_manifest_row(
    record: dict[str, Any],
    local_pdf_path: Path,
    local_text_path: Path,
    status: str,
    method: str,
    text_metadata: dict[str, Any] | None = None,
    error_message: str = "",
) -> dict[str, Any]:
    metadata = text_metadata or {}
    paper_id = str(record.get("paper_id") or "")
    return {
        "text_record_id": f"research_ai_text_{paper_id}",
        "paper_id": paper_id,
        "title": record.get("title"),
        "source": record.get("source"),
        "venue": record.get("venue"),
        "provenance_url": record.get("provenance_url"),
        "pdf_url": record.get("pdf_url_enriched") or record.get("pdf_url"),
        "local_pdf_path": str(local_pdf_path),
        "local_text_path": metadata.get("local_text_path") or str(local_text_path),
        "text_extraction_status": status,
        "extraction_method": method,
        "text_char_count": metadata.get("text_char_count", 0),
        "text_word_count": metadata.get("text_word_count", 0),
        "text_sha256": metadata.get("text_sha256"),
        "extracted_at_utc": utc_now(),
        "error_message": error_message,
    }


def extract_text(args: argparse.Namespace) -> dict[str, Any]:
    records = apply_limit(read_jsonl(args.enriched_registry_path), int(args.limit))
    text_manifest_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []
    backend = get_pdf_text_extraction_backend()
    for record in records:
        local_pdf_value = str(record.get("local_pdf_path") or "").strip()
        local_pdf_path = (
            Path(local_pdf_value)
            if local_pdf_value
            else build_local_pdf_path(record, args.raw_paper_dir)
        )
        local_text_path = args.paper_text_dir / f"{record.get('paper_id')}.txt"
        if not local_pdf_path.exists():
            text_manifest_rows.append(
                build_text_manifest_row(
                    record,
                    local_pdf_path,
                    local_text_path,
                    "skipped_missing_pdf",
                    "not_attempted",
                    error_message="Local PDF file is missing.",
                )
            )
            continue
        extracted_text, method, error_message = extract_text_from_pdf_with_error(local_pdf_path)
        if extracted_text:
            text_metadata = write_text_file(local_text_path, extracted_text)
            row = build_text_manifest_row(
                record,
                local_pdf_path,
                local_text_path,
                "extracted",
                method,
                text_metadata,
            )
            text_manifest_rows.append(row)
            section_record = {
                **record,
                "local_text_path": row["local_text_path"],
                "extraction_method": method,
            }
            section_rows.extend(detect_research_paper_sections(extracted_text, section_record))
        else:
            text_manifest_rows.append(
                build_text_manifest_row(
                    record,
                    local_pdf_path,
                    local_text_path,
                    method,
                    method,
                    error_message=error_message or "No text was extracted from local PDF.",
                )
            )

    quality_by_paper = {
        str(record["paper_id"]): record
        for record in build_section_quality_records(section_rows, text_manifest_rows)
    }
    for row in text_manifest_rows:
        paper_id = str(row.get("paper_id") or "")
        quality = quality_by_paper.get(paper_id)
        row["sections_detected_count"] = (
            quality["sections_detected_count"] if quality is not None else 0
        )
        row["section_quality_status"] = (
            quality["section_quality_status"] if quality is not None else "not_available"
        )

    write_jsonl(args.text_manifest_path, text_manifest_rows)
    write_jsonl(args.sections_manifest_path, section_rows)
    approved_records = read_jsonl(args.approved_registry_path)
    report = build_paper_preparation_report(
        approved_records=approved_records,
        enriched_records=records,
        text_manifest_rows=text_manifest_rows,
        section_rows=section_rows,
        output_files=output_files_from_args(args),
        pdf_text_backend=backend,
    )
    write_json(args.report_path, report)
    return {
        "mode": "extract_text",
        "phase": PHASE,
        "records_attempted": len(records),
        "pdf_text_backend": report["pdf_text_backend"],
        "local_pdf_count": int(report["local_pdf_count"]),
        "text_extracted_count": int(report["text_extracted_count"]),
        "text_extraction_skipped_count": int(report["text_extraction_skipped_count"]),
        "text_extraction_failed_count": int(report["text_extraction_failed_count"]),
        "sections_extracted_count": int(report["sections_extracted_count"]),
        "no_sections_detected_count": int(report["no_sections_detected_count"]),
        "text_manifest_path": str(args.text_manifest_path),
        "sections_manifest_path": str(args.sections_manifest_path),
        "report_path": str(args.report_path),
        "warnings": report["warnings"],
    }


def suspicious_large_sections(section_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suspicious: list[dict[str, Any]] = []
    for section in section_rows:
        section_type = str(section.get("section_type") or "other")
        word_count = safe_int(section.get("word_count"))
        is_large_abstract = section_type == "abstract" and word_count > 1500
        is_large_non_reference = section_type != "references" and word_count > 8000
        if not is_large_abstract and not is_large_non_reference:
            continue
        suspicious.append(
            {
                "section_record_id": section.get("section_record_id"),
                "paper_id": section.get("paper_id"),
                "title": section.get("title"),
                "section_type": section_type,
                "section_title": section.get("section_title"),
                "word_count": word_count,
                "reason": (
                    "abstract_above_1500_words"
                    if is_large_abstract
                    else "non_reference_section_above_8000_words"
                ),
            }
        )
    return suspicious


def build_section_quality_audit_report(
    text_manifest_rows: list[dict[str, Any]],
    section_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    section_metrics = aggregate_section_quality_metrics(section_rows, text_manifest_rows)
    quality_records = section_metrics["section_quality_records"]
    suspicious_sections = suspicious_large_sections(section_rows)
    recommendations = [
        "Review poor section coverage before creating Research AI KB/gold records.",
        (
            "Use method, experiments, results, limitations, and conclusion sections "
            "for 2A-5B curation."
        ),
        "Do not treat oversized abstracts as reliable abstract-only evidence without inspection.",
    ]
    if not suspicious_sections:
        recommendations.append("No oversized abstract or non-reference sections were detected.")
    return {
        "phase": "2A-5A-Text-Section-QA",
        "generated_at_utc": utc_now(),
        "total_papers": len(quality_records),
        "total_sections": len(section_rows),
        "sections_by_type": section_metrics["section_counts_by_type"],
        "papers_with_good_sections_count": section_metrics["papers_with_good_sections_count"],
        "papers_with_partial_sections_count": section_metrics["papers_with_partial_sections_count"],
        "papers_with_poor_sections_count": section_metrics["papers_with_poor_sections_count"],
        "poor_section_quality_titles": section_metrics["poor_section_quality_titles"],
        "suspicious_large_sections": suspicious_sections,
        "recommendations": recommendations,
        "next_step": (
            "Review section quality before Phase 2A-5B curated Research AI KB/gold creation."
        ),
    }


def audit_sections(args: argparse.Namespace) -> dict[str, Any]:
    text_rows = read_jsonl(args.text_manifest_path) if args.text_manifest_path.exists() else []
    section_rows = (
        read_jsonl(args.sections_manifest_path) if args.sections_manifest_path.exists() else []
    )
    report = build_section_quality_audit_report(text_rows, section_rows)
    write_json(args.section_quality_report_path, report)
    return {
        "mode": "audit_sections",
        "phase": report["phase"],
        "total_papers": report["total_papers"],
        "total_sections": report["total_sections"],
        "sections_by_type": report["sections_by_type"],
        "papers_with_good_sections_count": report["papers_with_good_sections_count"],
        "papers_with_partial_sections_count": report["papers_with_partial_sections_count"],
        "papers_with_poor_sections_count": report["papers_with_poor_sections_count"],
        "suspicious_large_sections_count": len(report["suspicious_large_sections"]),
        "report_path": str(args.section_quality_report_path),
        "recommendations": report["recommendations"],
        "next_step": report["next_step"],
    }


def is_approved_research_ai_paper(record: dict[str, Any]) -> bool:
    if record.get("not_for_benchmark_claims") is True:
        return False
    if record.get("missing_pdf_or_section_text") is True:
        return False
    approval_status = str(record.get("approval_status") or "").lower()
    selection_status = str(record.get("selection_status") or "").lower()
    return approval_status == "approved" or selection_status == "approved"


def approved_research_ai_papers(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    approved: list[dict[str, Any]] = []
    seen_paper_ids: set[str] = set()
    for record in records:
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id or paper_id in seen_paper_ids:
            continue
        if not is_approved_research_ai_paper(record):
            continue
        approved.append(record)
        seen_paper_ids.add(paper_id)
    return approved


def build_research_ai_candidate_slots(additional_papers_needed: int) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for index in range(1, additional_papers_needed + 1):
        slots.append(
            {
                "approval_status": "needs_review",
                "candidate_slot": index,
                "missing_pdf_or_section_text": True,
                "not_for_benchmark_claims": True,
                "paper_id": f"research_ai_1000_candidate_slot_{index:02d}",
                "required_actions": [
                    "replace this placeholder with real paper metadata",
                    "verify the paper is in scope for Research AI benchmark evidence",
                    "download or otherwise validate the full paper source",
                    "extract text and section evidence before approval",
                ],
                "selection_status": "needs_review",
                "title": f"Needs review Research AI paper slot {index:02d}",
            }
        )
    return slots


def build_research_ai_expansion_review_rows(
    approved_records: list[dict[str, Any]], candidate_slots: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in approved_records:
        rows.append(
            {
                "approval_status": "approved",
                "missing_pdf_or_section_text": False,
                "not_for_benchmark_claims": False,
                "paper_id": record.get("paper_id", ""),
                "slot_type": "current_approved",
                "title": record.get("title", ""),
                "venue": record.get("venue", ""),
                "year": record.get("year", ""),
            }
        )
    for slot in candidate_slots:
        rows.append(
            {
                "approval_status": slot["approval_status"],
                "missing_pdf_or_section_text": slot["missing_pdf_or_section_text"],
                "not_for_benchmark_claims": slot["not_for_benchmark_claims"],
                "paper_id": slot["paper_id"],
                "slot_type": "needs_review_placeholder",
                "title": slot["title"],
                "venue": "",
                "year": "",
            }
        )
    return rows


def build_research_ai_expanded_section_quality_report(
    *,
    approved_count: int,
    section_rows: list[dict[str, Any]],
    expansion_ready_for_1000: bool,
    missing_requirements: list[str],
) -> dict[str, Any]:
    sections_by_type = Counter(str(row.get("section_type") or "unknown") for row in section_rows)
    paper_ids_with_sections = {
        str(row.get("paper_id") or "") for row in section_rows if row.get("paper_id")
    }
    return {
        "phase": EXPANSION_PHASE,
        "generated_at_utc": utc_now(),
        "approved_paper_count": approved_count,
        "papers_with_sections_count": len(paper_ids_with_sections),
        "section_count": len(section_rows),
        "sections_by_type": dict(sorted(sections_by_type.items())),
        "target_section_count_range": {
            "min": TARGET_RESEARCH_AI_SECTION_COUNT_MIN,
            "max": TARGET_RESEARCH_AI_SECTION_COUNT_MAX,
        },
        "missing_requirements": missing_requirements,
        "expansion_ready_for_1000": expansion_ready_for_1000,
    }


def build_research_ai_40_paper_expansion(args: argparse.Namespace) -> dict[str, Any]:
    approved_registry = read_jsonl(args.approved_registry_path)
    section_rows = (
        read_jsonl(args.sections_manifest_path) if args.sections_manifest_path.exists() else []
    )
    approved_records = approved_research_ai_papers(approved_registry)
    approved_count = len(approved_records)
    additional_papers_needed = max(0, TARGET_RESEARCH_AI_APPROVED_PAPER_COUNT - approved_count)
    candidate_slots = build_research_ai_candidate_slots(additional_papers_needed)
    missing_requirements: list[str] = []
    if additional_papers_needed:
        missing_requirements.append(f"additional_approved_papers_needed:{additional_papers_needed}")
    if len(section_rows) < TARGET_RESEARCH_AI_SECTION_COUNT_MIN:
        missing_requirements.append(
            "section_coverage_below_target_min:"
            f"{TARGET_RESEARCH_AI_SECTION_COUNT_MIN - len(section_rows)}"
        )
    expansion_ready_for_1000 = (
        approved_count >= TARGET_RESEARCH_AI_APPROVED_PAPER_COUNT
        and len(section_rows) >= TARGET_RESEARCH_AI_SECTION_COUNT_MIN
    )
    if candidate_slots:
        write_jsonl(args.candidate_template_path, candidate_slots)
    review_rows = build_research_ai_expansion_review_rows(approved_records, candidate_slots)
    write_csv(
        args.expansion_review_csv_path,
        review_rows,
        [
            "slot_type",
            "paper_id",
            "title",
            "venue",
            "year",
            "approval_status",
            "not_for_benchmark_claims",
            "missing_pdf_or_section_text",
        ],
    )
    expanded_section_quality_report = build_research_ai_expanded_section_quality_report(
        approved_count=approved_count,
        section_rows=section_rows,
        expansion_ready_for_1000=expansion_ready_for_1000,
        missing_requirements=missing_requirements,
    )
    write_json(args.expanded_section_quality_report_path, expanded_section_quality_report)
    report = {
        "phase": EXPANSION_PHASE,
        "generated_at_utc": utc_now(),
        "current_approved_paper_count": approved_count,
        "target_approved_paper_count": TARGET_RESEARCH_AI_APPROVED_PAPER_COUNT,
        "additional_papers_needed": additional_papers_needed,
        "current_section_count": len(section_rows),
        "target_section_count_range": {
            "min": TARGET_RESEARCH_AI_SECTION_COUNT_MIN,
            "max": TARGET_RESEARCH_AI_SECTION_COUNT_MAX,
        },
        "expansion_ready_for_1000": expansion_ready_for_1000,
        "missing_requirements": missing_requirements,
        "candidate_template_path": str(args.candidate_template_path),
        "candidate_template_placeholder_count": len(candidate_slots),
        "placeholders_counted_as_approved": False,
        "expansion_review_csv_path": str(args.expansion_review_csv_path),
        "expanded_section_quality_report_path": str(args.expanded_section_quality_report_path),
        "recommended_next_step": (
            "Proceed to Research AI 1,000-scale generator implementation."
            if expansion_ready_for_1000
            else "Approve real additional papers and extract section text; do not use "
            "placeholder slots as benchmark evidence."
        ),
    }
    write_json(args.expansion_report_path, report)
    return {
        "mode": "build_40_paper_expansion",
        "phase": EXPANSION_PHASE,
        "current_approved_paper_count": approved_count,
        "target_approved_paper_count": TARGET_RESEARCH_AI_APPROVED_PAPER_COUNT,
        "additional_papers_needed": additional_papers_needed,
        "current_section_count": len(section_rows),
        "expansion_ready_for_1000": expansion_ready_for_1000,
        "missing_requirements": missing_requirements,
        "expansion_report_path": str(args.expansion_report_path),
        "candidate_template_path": str(args.candidate_template_path),
        "expansion_review_csv_path": str(args.expansion_review_csv_path),
        "expanded_section_quality_report_path": str(args.expanded_section_quality_report_path),
        "recommended_next_step": report["recommended_next_step"],
    }


def summarize_local(args: argparse.Namespace) -> dict[str, Any]:
    approved_records = read_jsonl(args.approved_registry_path)
    enriched_records = (
        read_jsonl(args.enriched_registry_path) if args.enriched_registry_path.exists() else []
    )
    text_rows = read_jsonl(args.text_manifest_path) if args.text_manifest_path.exists() else []
    section_rows = (
        read_jsonl(args.sections_manifest_path) if args.sections_manifest_path.exists() else []
    )
    report = build_paper_preparation_report(
        approved_records=approved_records,
        enriched_records=enriched_records,
        text_manifest_rows=text_rows,
        section_rows=section_rows,
        output_files=output_files_from_args(args),
    )
    write_json(args.report_path, report)
    return {
        "mode": "summarize_local",
        "phase": PHASE,
        "approved_record_count": len(approved_records),
        "enriched_record_count": len(enriched_records),
        "text_manifest_record_count": len(text_rows),
        "section_record_count": len(section_rows),
        "report_path": str(args.report_path),
        "warnings": report["warnings"],
        "next_step": report["next_step"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enrich-metadata", action="store_true")
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--extract-text", action="store_true")
    parser.add_argument("--audit-sections", action="store_true")
    parser.add_argument("--summarize-local", action="store_true")
    parser.add_argument("--build-40-paper-expansion", action="store_true")
    parser.add_argument(
        "--approved-registry-path",
        type=Path,
        default=DEFAULT_APPROVED_REGISTRY_PATH,
    )
    parser.add_argument(
        "--enriched-registry-path",
        type=Path,
        default=DEFAULT_ENRICHED_REGISTRY_PATH,
    )
    parser.add_argument("--raw-paper-dir", type=Path, default=DEFAULT_RAW_PAPER_DIR)
    parser.add_argument("--paper-text-dir", type=Path, default=DEFAULT_PAPER_TEXT_DIR)
    parser.add_argument("--text-manifest-path", type=Path, default=DEFAULT_TEXT_MANIFEST_PATH)
    parser.add_argument(
        "--sections-manifest-path", type=Path, default=DEFAULT_SECTIONS_MANIFEST_PATH
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--section-quality-report-path",
        type=Path,
        default=DEFAULT_SECTION_QUALITY_REPORT_PATH,
    )
    parser.add_argument(
        "--expansion-report-path",
        type=Path,
        default=DEFAULT_40_PAPER_EXPANSION_REPORT_PATH,
    )
    parser.add_argument(
        "--expansion-review-csv-path",
        type=Path,
        default=DEFAULT_40_PAPER_REVIEW_CSV_PATH,
    )
    parser.add_argument(
        "--expanded-section-quality-report-path",
        type=Path,
        default=DEFAULT_EXPANDED_SECTION_QUALITY_REPORT_PATH,
    )
    parser.add_argument(
        "--candidate-template-path",
        type=Path,
        default=DEFAULT_1000_SCALE_CANDIDATE_TEMPLATE_PATH,
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-non-paper-pdfs", action="store_true")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--download-max-retries", type=int, default=3)
    parser.add_argument("--download-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--download-timeout-seconds", type=int, default=None)
    parser.add_argument("--download-request-delay-seconds", type=float, default=None)
    parser.add_argument("--paper-id", default="")
    parser.add_argument("--failed-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode_count = sum(
        bool(mode)
        for mode in (
            args.dry_run,
            args.enrich_metadata,
            args.download_pdfs,
            args.extract_text,
            args.audit_sections,
            args.summarize_local,
            args.build_40_paper_expansion,
        )
    )
    if mode_count != 1:
        print(
            (
                "Pass exactly one mode: --dry-run, --enrich-metadata, --download-pdfs, "
                "--extract-text, --audit-sections, --summarize-local, or "
                "--build-40-paper-expansion."
            ),
            file=sys.stderr,
        )
        return 2
    if int(args.limit) < 0:
        print("--limit must be >= 0.", file=sys.stderr)
        return 2
    if int(args.download_max_retries) < 0:
        print("--download-max-retries must be >= 0.", file=sys.stderr)
        return 2
    if float(args.download_backoff_seconds) < 0:
        print("--download-backoff-seconds must be >= 0.", file=sys.stderr)
        return 2
    if effective_download_timeout_seconds(args) <= 0:
        print("Download timeout must be > 0.", file=sys.stderr)
        return 2
    if effective_download_request_delay_seconds(args) < 0:
        print("Download request delay must be >= 0.", file=sys.stderr)
        return 2
    try:
        if args.dry_run:
            summary = dry_run(args)
        elif args.enrich_metadata:
            summary = enrich_metadata(args)
        elif args.download_pdfs:
            summary = download_pdfs(args)
        elif args.extract_text:
            summary = extract_text(args)
        elif args.audit_sections:
            summary = audit_sections(args)
        elif args.build_40_paper_expansion:
            summary = build_research_ai_40_paper_expansion(args)
        else:
            summary = summarize_local(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
