"""Discover AI Research Assistant candidate papers from arXiv metadata."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
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
DEFAULT_RUN_LOG = Path("data/generated/research_ai/research_ai_discovery_run_log.jsonl")
DEFAULT_MANUAL_TEMPLATE = Path("data/generated/research_ai/manual_paper_registry_template.csv")
DEFAULT_MANUAL_REGISTRY = Path("data/sources/research_ai_approved_papers.jsonl")
DEFAULT_MANUAL_VALIDATION_REPORT = Path(
    "data/generated/research_ai/manual_registry_validation_report.json"
)
DEFAULT_ICLR_URL = "https://iclr.cc/virtual/2025/papers.html"
DEFAULT_HUGGINGFACE_URL = "https://huggingface.co/papers"
DEFAULT_HUGGINGFACE_SEARCH_QUERIES = (
    "LLM inference,vLLM,speculative decoding,RAG,LLM routing,LLM agents,small language models"
)

ARXIV_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_EXTENSION_NAMESPACE = "http://arxiv.org/schemas/atom"
ATOM = {"atom": ARXIV_NAMESPACE, "arxiv": ARXIV_EXTENSION_NAMESPACE}
DEFAULT_CATEGORIES_ALLOWED = {"cs.CL", "cs.LG", "cs.AI", "cs.DC", "cs.SE", "cs.IR"}
USER_AGENT = "LLM-Inference-Optimization-Suite research-discovery"
SIMPLE_SEARCH_QUERIES = {
    "llm_serving_inference_optimization": 'all:"LLM inference"',
    "vllm_pagedattention_continuous_batching": 'all:"vLLM"',
    "speculative_decoding_kv_cache": 'all:"speculative decoding"',
    "rag_context_engineering": 'all:"retrieval augmented generation"',
    "llm_routing_model_selection": 'all:"LLM routing"',
    "agentic_workflows_tool_use": 'all:"LLM agents"',
    "small_language_models_efficient_llms": 'all:"small language models"',
}
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
TOPIC_KEYWORDS = {
    "llm_serving_inference_optimization": (
        "inference",
        "serving",
        "latency",
        "throughput",
        "batching",
        "gpu",
        "memory",
    ),
    "vllm_pagedattention_continuous_batching": (
        "vllm",
        "pagedattention",
        "continuous batching",
    ),
    "speculative_decoding_kv_cache": (
        "speculative decoding",
        "kv cache",
        "prefix caching",
        "decoding",
    ),
    "rag_context_engineering": (
        "retrieval augmented generation",
        "rag",
        "grounding",
        "citation",
        "citations",
        "context",
    ),
    "llm_routing_model_selection": (
        "routing",
        "router",
        "model selection",
        "cost",
    ),
    "agentic_workflows_tool_use": (
        "agent",
        "agents",
        "agentic",
        "tool use",
        "workflow",
    ),
    "small_language_models_efficient_llms": (
        "small language model",
        "small language models",
        "efficient language model",
        "efficient language models",
        "distillation",
    ),
}
HTML_PREFERRED_KEYWORDS = (
    "llm",
    "large language model",
    "inference",
    "serving",
    "latency",
    "throughput",
    "vllm",
    "pagedattention",
    "speculative decoding",
    "kv cache",
    "rag",
    "retrieval",
    "agent",
    "agentic",
    "routing",
    "small language model",
    "efficient language model",
    "distillation",
)
SOURCE_PRIORITY = {
    "ICLR": 0,
    "Hugging Face Papers": 1,
    "arXiv": 2,
}
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
MANUAL_REGISTRY_FIELDS = [
    "selection_status",
    "topic",
    "paper_id",
    "arxiv_id",
    "title",
    "authors",
    "published",
    "primary_category",
    "abstract_url",
    "pdf_url",
    "reason_for_inclusion",
    "notes",
]
MANUAL_REGISTRY_REQUIRED_FIELDS = [
    "paper_id",
    "arxiv_id",
    "title",
    "authors",
    "published",
    "primary_category",
    "abstract_url",
    "pdf_url",
    "topic",
    "reason_for_inclusion",
    "selection_status",
    "provenance_url",
]
MANUAL_REGISTRY_ALLOWED_STATUSES = {
    "approved",
    "candidate",
    "rejected",
    "example_not_approved",
}


class ArxivFetchError(RuntimeError):
    """Raised when arXiv metadata fetch fails after retry handling."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        msg = f"Missing required JSON file: {path}"
        raise RuntimeError(msg)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        msg = f"Expected JSON object in {path}"
        raise RuntimeError(msg)
    return parsed


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        msg = f"Missing required JSONL file: {path}"
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


def write_run_log_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON event to the Research AI discovery run log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": utc_now(),
        "phase": "2A-5A",
        **event,
    }
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def log_event(
    run_log_path: Path | None,
    mode: str,
    event_type: str,
    message: str,
    **kwargs: Any,
) -> None:
    if run_log_path is None:
        return
    write_run_log_event(
        run_log_path,
        {
            "mode": mode,
            "event_type": event_type,
            "message": message,
            **{key: value for key, value in kwargs.items() if value is not None},
        },
    )


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


def slugify(value: str, max_chars: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        slug = "untitled"
    return slug[:max_chars].strip("_")


def paper_id_from_title(source_id: str, title: str) -> str:
    return f"research_ai_{slugify(source_id, 32)}_{slugify(title, 72)}"


def normalize_url(url: str, base_url: str) -> str:
    return urllib.parse.urljoin(base_url, html.unescape(url.strip()))


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


def format_arxiv_submitted_date(date_string: str, end_of_day: bool = False) -> str:
    """Format a YYYY-MM-DD date for arXiv submittedDate range queries."""

    try:
        parsed = datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError as exc:
        msg = f"Expected date in YYYY-MM-DD format, got: {date_string}"
        raise RuntimeError(msg) from exc
    suffix = "2359" if end_of_day else "0000"
    return f"{parsed:%Y%m%d}{suffix}"


def apply_arxiv_date_filter(search_query: str, start_date: str, end_date: str) -> str:
    """Add the approved submittedDate range to an arXiv search query."""

    start = format_arxiv_submitted_date(start_date)
    end = format_arxiv_submitted_date(end_date, end_of_day=True)
    date_filter = f"submittedDate:[{start} TO {end}]"
    stripped_query = search_query.strip()
    if " OR " in stripped_query and not (
        stripped_query.startswith("(") and stripped_query.endswith(")")
    ):
        stripped_query = f"({stripped_query})"
    return f"{stripped_query} AND {date_filter}"


def paper_window_from_args(query_plan: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    configured_window = query_plan.get("paper_window", {})
    start_date = str(args.start_date or configured_window.get("start_date") or "2024-01-01")
    end_date = str(args.end_date or configured_window.get("end_date") or "2026-05-30")
    return {
        "start_date": start_date,
        "end_date": end_date,
        "arxiv_submitted_date_start": format_arxiv_submitted_date(start_date),
        "arxiv_submitted_date_end": format_arxiv_submitted_date(end_date, end_of_day=True),
        "notes": str(
            configured_window.get("notes")
            or "Approved Research AI paper discovery window for Phase 2A seed curation."
        ),
    }


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after and retry_after.isdigit():
        return float(retry_after)
    return None


def _backoff_delay(
    attempt_number: int,
    backoff_seconds: float,
    retry_after: float | None = None,
) -> float:
    if retry_after is not None:
        return retry_after
    deterministic_jitter = min(1.0, attempt_number * 0.25)
    return backoff_seconds * attempt_number + deterministic_jitter


def _error_body_snippet(error: urllib.error.HTTPError, max_chars: int = 500) -> str:
    if error.fp is None:
        return ""
    try:
        body = error.fp.read(max_chars)
    except Exception:  # noqa: BLE001 - body snippets are diagnostic best effort.
        return ""
    if isinstance(body, str):
        return body[:max_chars]
    return body.decode("utf-8", errors="replace")[:max_chars]


def _base_error_details(
    *,
    url: str,
    attempt_number: int,
    exception_type: str,
    message: str,
    elapsed_seconds: float,
    query_id: str | None = None,
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "url": url,
        "attempt_number": attempt_number,
        "exception_type": exception_type,
        "error_message": message,
        "elapsed_seconds": round(elapsed_seconds, 4),
    }


def fetch_url_with_retries(
    url: str,
    user_agent: str,
    timeout_seconds: int,
    max_retries: int,
    backoff_seconds: float,
    run_log_path: Path | None = None,
    mode: str = "discover",
    query_id: str | None = None,
) -> str:
    """Fetch arXiv metadata with conservative retry handling for transient failures."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/atom+xml, application/xml, text/xml",
        },
    )
    attempts = max_retries + 1
    last_error = ""
    last_details: dict[str, Any] = {}
    for attempt_index in range(attempts):
        attempt_number = attempt_index + 1
        started = time.monotonic()
        log_event(
            run_log_path,
            mode,
            "request_attempt",
            "Starting arXiv metadata request.",
            query_id=query_id,
            attempt_number=attempt_number,
            url=url,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                text = response.read().decode("utf-8")
                elapsed_seconds = time.monotonic() - started
                log_event(
                    run_log_path,
                    mode,
                    "request_success",
                    "arXiv metadata request succeeded.",
                    query_id=query_id,
                    attempt_number=attempt_number,
                    url=url,
                    elapsed_seconds=round(elapsed_seconds, 4),
                )
                return text
        except urllib.error.HTTPError as exc:
            elapsed_seconds = time.monotonic() - started
            status = int(exc.code)
            last_error = f"HTTP {status}: {exc.reason}"
            retry_after = _retry_after_seconds(exc)
            last_details = {
                **_base_error_details(
                    url=url,
                    attempt_number=attempt_number,
                    exception_type=type(exc).__name__,
                    message=last_error,
                    elapsed_seconds=elapsed_seconds,
                    query_id=query_id,
                ),
                "status_code": status,
                "reason": str(exc.reason),
                "retry_after": retry_after,
                "response_body_snippet": _error_body_snippet(exc),
            }
            if status == 429:
                if attempt_index >= max_retries:
                    break
                delay = _backoff_delay(attempt_number, backoff_seconds, retry_after)
                log_event(
                    run_log_path,
                    mode,
                    "request_retry",
                    "arXiv returned HTTP 429; retrying after backoff.",
                    **last_details,
                )
                time.sleep(delay)
                continue
            if 500 <= status <= 599:
                if attempt_index >= max_retries:
                    break
                log_event(
                    run_log_path,
                    mode,
                    "request_retry",
                    "arXiv returned HTTP 5xx; retrying after backoff.",
                    **last_details,
                )
                time.sleep(_backoff_delay(attempt_number, backoff_seconds))
                continue
            msg = f"Failed to fetch arXiv metadata from {url}: HTTP {status}: {exc.reason}"
            log_event(
                run_log_path,
                mode,
                "request_failed",
                "arXiv request failed with non-retryable HTTP status.",
                **last_details,
            )
            raise ArxivFetchError(msg, last_details) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            elapsed_seconds = time.monotonic() - started
            last_error = str(exc)
            last_details = _base_error_details(
                url=url,
                attempt_number=attempt_number,
                exception_type=type(exc).__name__,
                message=last_error,
                elapsed_seconds=elapsed_seconds,
                query_id=query_id,
            )
            if attempt_index >= max_retries:
                break
            log_event(
                run_log_path,
                mode,
                "request_retry",
                "arXiv request raised a transient network error; retrying.",
                **last_details,
            )
            time.sleep(_backoff_delay(attempt_number, backoff_seconds))
    msg = f"Failed to fetch arXiv metadata from {url} after {attempts} attempts: {last_error}"
    last_details = {
        **last_details,
        "error_message": msg,
    }
    log_event(
        run_log_path,
        mode,
        "request_failed",
        "arXiv request failed after all retry attempts.",
        **last_details,
    )
    raise ArxivFetchError(msg, last_details)


def fetch_arxiv_atom(
    url: str,
    timeout_seconds: int = 30,
    max_retries: int = 3,
    backoff_seconds: float = 10,
    run_log_path: Path | None = None,
    mode: str = "discover",
    query_id: str | None = None,
) -> str:
    return fetch_url_with_retries(
        url=url,
        user_agent=USER_AGENT,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
        run_log_path=run_log_path,
        mode=mode,
        query_id=query_id,
    )


def fetch_html(
    url: str,
    user_agent: str,
    timeout_seconds: int,
    run_log_path: Path | None = None,
    mode: str = "discover",
    source_id: str | None = None,
) -> str:
    """Fetch one public HTML metadata page without downloading PDFs."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml,text/plain,*/*",
        },
    )
    started = time.monotonic()
    log_event(
        run_log_path,
        mode,
        "request_started",
        "Starting public HTML metadata request.",
        query_id=source_id,
        url=url,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            text = body.decode("utf-8", errors="replace")
            status_code = getattr(response, "status", None) or response.getcode()
            elapsed_seconds = time.monotonic() - started
            log_event(
                run_log_path,
                mode,
                "request_success",
                "Public HTML metadata request succeeded.",
                query_id=source_id,
                url=url,
                status_code=status_code,
                elapsed_seconds=round(elapsed_seconds, 4),
            )
            return text
    except urllib.error.HTTPError as exc:
        elapsed_seconds = time.monotonic() - started
        snippet = _error_body_snippet(exc)
        details = {
            "source_id": source_id,
            "url": url,
            "status_code": int(exc.code),
            "reason": str(exc.reason),
            "exception_type": type(exc).__name__,
            "response_body_snippet": snippet,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "error_message": f"HTTP {exc.code}: {exc.reason}",
        }
        log_event(
            run_log_path,
            mode,
            "request_failed",
            "Public HTML metadata request failed.",
            **details,
        )
        msg = f"Failed to fetch HTML metadata from {url}: HTTP {exc.code}: {exc.reason}"
        raise ArxivFetchError(msg, details) from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        elapsed_seconds = time.monotonic() - started
        details = {
            "source_id": source_id,
            "url": url,
            "exception_type": type(exc).__name__,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "error_message": str(exc),
        }
        log_event(
            run_log_path,
            mode,
            "request_failed",
            "Public HTML metadata request failed.",
            **details,
        )
        msg = f"Failed to fetch HTML metadata from {url}: {exc}"
        raise ArxivFetchError(msg, details) from exc


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


def _anchor_records(html_text: str) -> list[dict[str, str]]:
    anchors: list[dict[str, str]] = []
    for match in re.finditer(
        r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        attrs = match.group("attrs")
        href_match = re.search(r"href=[\"'](?P<href>[^\"']+)[\"']", attrs, flags=re.IGNORECASE)
        if href_match is None:
            continue
        text = normalize_text(re.sub(r"<[^>]+>", " ", match.group("body")))
        href = html.unescape(href_match.group("href"))
        if text:
            anchors.append({"href": href, "text": text})
    return anchors


def _looks_like_paper_title(title: str) -> bool:
    if len(title) < 8:
        return False
    lowered = title.lower()
    if lowered in {"paper", "poster", "pdf", "openreview", "abstract", "video", "details"}:
        return False
    if len(title.split()) < 2:
        return False
    return bool(re.search(r"[A-Za-z]", title))


def infer_research_topics(title: str, abstract: str = "") -> list[str]:
    haystack = f"{title} {abstract}".lower()
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword in haystack for keyword in keywords)
    ]
    return topics or ["research_ai_candidate"]


def _matched_html_keywords(title: str, abstract: str = "") -> list[str]:
    haystack = f"{title} {abstract}".lower()
    return sorted(
        {keyword for keyword in HTML_PREFERRED_KEYWORDS if keyword in haystack},
        key=str.lower,
    )


def _extract_arxiv_id_from_url(url: str | None) -> tuple[str, str]:
    if not url:
        return "", ""
    if "arxiv.org/abs/" in url or "arxiv.org/pdf/" in url:
        cleaned = url.replace("/pdf/", "/abs/")
        arxiv_id, version = normalize_arxiv_id(cleaned)
        return arxiv_id, version
    return "", ""


def _candidate_from_html(
    *,
    title: str,
    source: str,
    source_id: str,
    source_url: str,
    provenance_url: str,
    query_id: str,
    venue: str,
    year: int | None,
    abstract: str = "",
    authors: list[str] | None = None,
    published: str | None = None,
    primary_category: str = "papers_html",
    categories: list[str] | None = None,
    pdf_url: str | None = None,
) -> dict[str, Any]:
    arxiv_id, arxiv_id_version = _extract_arxiv_id_from_url(provenance_url)
    topics = infer_research_topics(title, abstract)
    matched_keywords = _matched_html_keywords(title, abstract)
    paper_id = (
        paper_id_from_arxiv_id(arxiv_id) if arxiv_id else paper_id_from_title(source_id, title)
    )
    return {
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "arxiv_id_version": arxiv_id_version,
        "title": title,
        "abstract": abstract,
        "authors": authors or [],
        "published": published or (str(year) if year else ""),
        "updated": "",
        "primary_category": primary_category,
        "categories": categories or [],
        "abstract_url": provenance_url,
        "pdf_url": pdf_url,
        "source": source,
        "source_id": source_id,
        "source_ids": [source_id],
        "source_url": source_url,
        "query_ids": [query_id],
        "topics": topics,
        "matched_keywords": matched_keywords,
        "score": 0,
        "selection_status": "candidate",
        "review_notes": "",
        "license": None,
        "provenance_url": provenance_url,
        "venue": venue,
        "year": year,
    }


def parse_iclr_2025_papers_html(html_text: str, source_url: str) -> list[dict[str, Any]]:
    """Extract candidate title/link metadata from the public ICLR 2025 papers page."""

    candidates: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for anchor in _anchor_records(html_text):
        href = anchor["href"]
        title = anchor["text"]
        if not _looks_like_paper_title(title):
            continue
        href_lower = href.lower()
        title_lower = title.lower()
        if not (
            "paper" in href_lower
            or "poster" in href_lower
            or "virtual/2025" in href_lower
            or any(keyword in title_lower for keyword in HTML_PREFERRED_KEYWORDS)
        ):
            continue
        title_key = title_lower
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        provenance_url = normalize_url(href, source_url)
        candidates.append(
            _candidate_from_html(
                title=title,
                source="ICLR",
                source_id="iclr_2025_virtual",
                source_url=source_url,
                provenance_url=provenance_url,
                query_id="iclr_2025_virtual",
                venue="ICLR 2025",
                year=2025,
                published="2025",
                primary_category="conference",
                categories=["ICLR 2025"],
            )
        )
    return candidates


def parse_search_queries(value: str) -> list[str]:
    return [query.strip() for query in value.split(",") if query.strip()]


def build_huggingface_search_urls(base_url: str, search_queries: list[str]) -> list[dict[str, str]]:
    return [
        {
            "query": query,
            "url": f"{base_url.rstrip('/')}?{urllib.parse.urlencode({'q': query})}",
        }
        for query in search_queries
    ]


def parse_huggingface_papers_html(
    html_text: str, source_url: str, query: str
) -> list[dict[str, Any]]:
    """Extract candidate title/link metadata from Hugging Face Papers HTML."""

    candidates: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for anchor in _anchor_records(html_text):
        href = anchor["href"]
        title = anchor["text"]
        if not _looks_like_paper_title(title):
            continue
        if "/papers/" not in href and not href.startswith("/papers"):
            continue
        if title.lower() in {"papers", "daily papers"}:
            continue
        title_key = title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        provenance_url = normalize_url(href, source_url)
        candidates.append(
            _candidate_from_html(
                title=title,
                source="Hugging Face Papers",
                source_id="huggingface_papers",
                source_url=source_url,
                provenance_url=provenance_url,
                query_id=f"huggingface_{slugify(query, 48)}",
                venue="Hugging Face Papers",
                year=None,
                primary_category="papers_html",
                categories=["Hugging Face Papers"],
            )
        )
    return candidates


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


def score_html_candidate(candidate: dict[str, Any]) -> int:
    title = str(candidate.get("title") or "")
    abstract = str(candidate.get("abstract") or "")
    haystack = f"{title} {abstract}".lower()
    title_lower = title.lower()
    score = 0

    for keywords in TOPIC_KEYWORDS.values():
        if any(keyword in title_lower for keyword in keywords):
            score += 5
            break
    for keyword in HTML_PREFERRED_KEYWORDS:
        if keyword in haystack:
            score += 3
    if candidate.get("source") == "ICLR":
        score += 2
    if candidate.get("source") == "Hugging Face Papers" and candidate.get("topics"):
        score += 2
    year = candidate.get("year")
    if year in {2024, 2025, 2026}:
        score += 1
    published = str(candidate.get("published") or "")
    year_match = re.match(r"(\d{4})", published)
    if year_match and int(year_match.group(1)) in {2024, 2025, 2026}:
        score += 1
    if not any(term in haystack for term in RELATED_TERMS):
        score -= 3
    return score


def _version_sort_key(candidate: dict[str, Any]) -> tuple[int, str]:
    version = str(candidate.get("arxiv_id_version") or "")
    match = re.search(r"v(\d+)$", version)
    version_number = int(match.group(1)) if match else 0
    return version_number, str(candidate.get("updated") or "")


def _normalized_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _candidate_dedupe_key(candidate: dict[str, Any]) -> str:
    arxiv_id = str(candidate.get("arxiv_id") or "").strip()
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    title_key = _normalized_title_key(str(candidate.get("title") or ""))
    if title_key:
        return f"title:{title_key}"
    provenance_url = str(candidate.get("provenance_url") or "").strip()
    return f"url:{provenance_url}"


def _merge_candidate_metadata(
    existing: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    merged = {**existing}
    for field in ("query_ids", "topics", "matched_keywords", "source_ids"):
        merged[field] = sorted(
            set(existing.get(field, [])) | set(candidate.get(field, [])),
            key=str.lower,
        )
    merged["score"] = max(int(existing.get("score") or 0), int(candidate.get("score") or 0))
    for field, value in candidate.items():
        if field in {"query_ids", "topics", "matched_keywords", "source_ids", "score"}:
            continue
        existing_value = merged.get(field)
        if existing_value in (None, "", [], {}) and value not in (None, "", [], {}):
            merged[field] = value
    if _version_sort_key(candidate) > _version_sort_key(existing):
        for field, value in candidate.items():
            if field not in {"query_ids", "topics", "matched_keywords", "source_ids", "score"}:
                merged[field] = value if value not in (None, "", [], {}) else merged.get(field)
    return merged


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by arXiv ID, normalized title, or provenance URL."""

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = _candidate_dedupe_key(candidate)
        if key == "url:":
            continue
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {**candidate}
            continue
        deduped[key] = _merge_candidate_metadata(existing, candidate)
    return list(deduped.values())


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ranked = sorted(
        candidates,
        key=lambda candidate: (
            -int(candidate.get("score") or 0),
            SOURCE_PRIORITY.get(str(candidate.get("source") or ""), 99),
            str(candidate.get("title") or "").lower(),
        ),
    )
    return source_ranked


def write_review_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "score",
        "paper_id",
        "arxiv_id",
        "title",
        "source",
        "venue",
        "year",
        "topics",
        "authors",
        "published",
        "primary_category",
        "categories",
        "provenance_url",
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
                    "source": candidate.get("source"),
                    "venue": candidate.get("venue"),
                    "year": candidate.get("year"),
                    "topics": "; ".join(candidate.get("topics", [])),
                    "authors": "; ".join(candidate.get("authors", [])),
                    "published": candidate.get("published"),
                    "primary_category": candidate.get("primary_category"),
                    "categories": "; ".join(candidate.get("categories", [])),
                    "provenance_url": candidate.get("provenance_url"),
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
    discovery_status: str = "success",
    selected_query_ids: list[str] | None = None,
    failed_query_groups: list[str] | None = None,
    successful_query_groups: list[str] | None = None,
    errors: list[dict[str, Any]] | None = None,
    retry_policy: dict[str, Any] | None = None,
    simple_query_mode: bool = False,
    paper_window: dict[str, Any] | None = None,
    date_filter_enabled: bool = True,
    sources_attempted: list[str] | None = None,
    sources_succeeded: list[str] | None = None,
    sources_failed: list[str] | None = None,
) -> dict[str, Any]:
    topic_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for candidate in ranked_candidates:
        topic_counter.update(str(topic) for topic in candidate.get("topics", []))
        primary_category = str(candidate.get("primary_category") or "")
        if primary_category:
            category_counter[primary_category] += 1
        source = str(candidate.get("source") or "")
        if source:
            source_counter[source] += 1
    if discovery_status == "failed":
        next_step = (
            "Retry later with --simple-query-mode, --query-id, --max-results-per-query 3, "
            "--delay-seconds 30; inspect the run log; use --write-manual-template and "
            "--validate-manual-registry if arXiv remains unavailable."
        )
    else:
        next_step = (
            "Review candidate_papers_review.csv, approve a 12-20 paper shortlist, then "
            "proceed to Phase 2A-5B to create research AI paper registry, KB/context "
            "samples, prompt/source records, and gold/eval records."
        )
    return {
        "phase": "2A-5A",
        "generated_at_utc": utc_now(),
        "discovery_status": discovery_status,
        "selected_query_ids": selected_query_ids or [],
        "query_group_count": len(selected_query_ids or query_plan.get("query_groups", [])),
        "raw_candidate_count": raw_candidate_count,
        "deduped_candidate_count": len(ranked_candidates),
        "sample_candidate_count": len(sample_candidates),
        "failed_query_groups": failed_query_groups or [],
        "successful_query_groups": successful_query_groups or [],
        "sources_attempted": sources_attempted or [],
        "sources_succeeded": sources_succeeded or [],
        "sources_failed": sources_failed or [],
        "errors": errors or [],
        "retry_policy": retry_policy or {},
        "simple_query_mode": simple_query_mode,
        "paper_window": paper_window or query_plan.get("paper_window", {}),
        "date_filter_enabled": date_filter_enabled,
        "arxiv_submitted_date_start": (paper_window or {}).get("arxiv_submitted_date_start")
        or query_plan.get("paper_window", {}).get("arxiv_submitted_date_start"),
        "arxiv_submitted_date_end": (paper_window or {}).get("arxiv_submitted_date_end")
        or query_plan.get("paper_window", {}).get("arxiv_submitted_date_end"),
        "counts_by_topic": dict(topic_counter),
        "counts_by_primary_category": dict(category_counter),
        "counts_by_source": dict(source_counter),
        "top_candidates": [
            {
                "rank": index,
                "score": candidate.get("score"),
                "paper_id": candidate.get("paper_id"),
                "arxiv_id": candidate.get("arxiv_id"),
                "title": candidate.get("title"),
                "source": candidate.get("source"),
                "provenance_url": candidate.get("provenance_url"),
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
        "next_step": next_step,
    }


def _query_max_results(
    query_group: dict[str, Any], query_plan: dict[str, Any], override: int
) -> int:
    if override > 0:
        return override
    return int(query_group.get("max_results") or query_plan.get("max_results_per_query") or 20)


def query_search_query(query_group: dict[str, Any], simple_query_mode: bool = False) -> str:
    query_id = str(query_group.get("query_id") or "")
    if simple_query_mode and query_id in SIMPLE_SEARCH_QUERIES:
        return SIMPLE_SEARCH_QUERIES[query_id]
    return str(query_group["search_query"])


def select_query_groups(query_plan: dict[str, Any], query_id: str = "all") -> list[dict[str, Any]]:
    """Return requested arXiv query groups from a query plan."""

    query_groups = list(query_plan.get("query_groups", []))
    if query_id == "all":
        return query_groups
    selected = [
        query_group
        for query_group in query_groups
        if str(query_group.get("query_id") or "") == query_id
    ]
    if not selected:
        known_ids = ", ".join(str(query_group.get("query_id")) for query_group in query_groups)
        msg = f"Unknown query_id '{query_id}'. Known query IDs: {known_ids}"
        raise RuntimeError(msg)
    return selected


def planned_urls(
    query_plan: dict[str, Any],
    max_results_override: int,
    query_id: str = "all",
    simple_query_mode: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    date_filter_enabled: bool = True,
) -> list[dict[str, str]]:
    planned: list[dict[str, str]] = []
    for query_group in select_query_groups(query_plan, query_id):
        max_results = _query_max_results(query_group, query_plan, max_results_override)
        search_query = query_search_query(query_group, simple_query_mode)
        if date_filter_enabled:
            search_query = apply_arxiv_date_filter(
                search_query,
                start_date or str(query_plan.get("paper_window", {}).get("start_date")),
                end_date or str(query_plan.get("paper_window", {}).get("end_date")),
            )
        url = build_arxiv_url(
            api_base_url=str(query_plan["api_base_url"]),
            search_query=search_query,
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


def enabled_discovery_source_ids(query_plan: dict[str, Any]) -> list[str]:
    source_ids = []
    for source in query_plan.get("discovery_sources", []):
        if source.get("enabled") and source.get("source_id") != "arxiv_api":
            source_ids.append(str(source["source_id"]))
    return source_ids or ["iclr_2025_virtual", "huggingface_papers"]


def source_ids_for_filter(source_filter: str, query_plan: dict[str, Any]) -> list[str]:
    if source_filter == "all":
        return enabled_discovery_source_ids(query_plan)
    if source_filter == "iclr":
        return ["iclr_2025_virtual"]
    if source_filter == "huggingface":
        return ["huggingface_papers"]
    if source_filter == "arxiv":
        return ["arxiv_api"]
    msg = "Unknown source. Use one of: all, arxiv, iclr, huggingface."
    raise RuntimeError(msg)


def output_files_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "candidate_papers_jsonl": str(args.output_candidates),
        "candidate_papers_review_csv": str(args.output_review_csv),
        "candidate_papers_sample_jsonl": str(args.output_sample),
        "discovery_report_json": str(args.output_report),
        "run_log_jsonl": str(args.run_log_path),
        "manual_template_csv": str(args.output_manual_template),
        "manual_validation_report_json": str(args.manual_validation_report),
    }


def planned_html_sources(
    args: argparse.Namespace, query_plan: dict[str, Any]
) -> list[dict[str, Any]]:
    source_ids = source_ids_for_filter(args.source, query_plan)
    plans: list[dict[str, Any]] = []
    if "iclr_2025_virtual" in source_ids:
        plans.append(
            {
                "source_id": "iclr_2025_virtual",
                "source": "ICLR",
                "url": args.iclr_url,
            }
        )
    if "huggingface_papers" in source_ids:
        plans.append(
            {
                "source_id": "huggingface_papers",
                "source": "Hugging Face Papers",
                "search_urls": build_huggingface_search_urls(
                    args.huggingface_url,
                    parse_search_queries(args.huggingface_search_queries),
                ),
            }
        )
    return plans


def discover_arxiv(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = load_json(args.query_plan)
    paper_window = paper_window_from_args(query_plan, args)
    date_filter_enabled = not args.disable_date_filter
    raw_candidates: list[dict[str, Any]] = []
    delay_seconds = (
        float(args.delay_seconds)
        if float(args.delay_seconds) > 0
        else float(query_plan.get("delay_seconds") or 3)
    )
    retry_policy = {
        "max_retries": args.max_retries,
        "backoff_seconds": args.backoff_seconds,
        "timeout_seconds": args.timeout_seconds,
        "delay_seconds": delay_seconds,
    }
    categories_allowed = set(query_plan.get("categories_allowed", [])) or DEFAULT_CATEGORIES_ALLOWED
    query_groups = select_query_groups(query_plan, args.query_id)
    selected_query_ids = [str(query_group["query_id"]) for query_group in query_groups]
    successful_query_groups: list[str] = []
    failed_query_groups: list[str] = []
    errors: list[dict[str, Any]] = []
    log_event(
        args.run_log_path,
        "discover",
        "run_started",
        "Research AI arXiv discovery started.",
        selected_query_ids=selected_query_ids,
        simple_query_mode=args.simple_query_mode,
        date_filter_enabled=date_filter_enabled,
        paper_window=paper_window,
    )

    for index, query_group in enumerate(query_groups):
        enriched_group = {**query_group, "categories_allowed": sorted(categories_allowed)}
        max_results = _query_max_results(enriched_group, query_plan, args.max_results_per_query)
        search_query = query_search_query(enriched_group, args.simple_query_mode)
        if date_filter_enabled:
            search_query = apply_arxiv_date_filter(
                search_query,
                paper_window["start_date"],
                paper_window["end_date"],
            )
        url = build_arxiv_url(
            api_base_url=str(query_plan["api_base_url"]),
            search_query=search_query,
            max_results=max_results,
            sort_by=str(enriched_group.get("sort_by") or "relevance"),
            sort_order=str(enriched_group.get("sort_order") or "descending"),
        )
        if index > 0:
            time.sleep(delay_seconds)
        query_id = str(enriched_group["query_id"])
        log_event(
            args.run_log_path,
            "discover",
            "query_started",
            "Starting arXiv query group.",
            query_id=query_id,
            url=url,
        )
        try:
            xml_text = fetch_arxiv_atom(
                url,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                backoff_seconds=args.backoff_seconds,
                run_log_path=args.run_log_path,
                mode="discover",
                query_id=query_id,
            )
        except ArxivFetchError as exc:
            failed_query_groups.append(query_id)
            error_details = {
                **exc.details,
                "query_id": query_id,
                "url": url,
                "error_message": str(exc),
            }
            errors.append(error_details)
            log_event(
                args.run_log_path,
                "discover",
                "query_failed",
                "arXiv query group failed.",
                **error_details,
            )
            if not args.continue_on_error:
                break
            continue
        parsed_candidates = parse_arxiv_atom(
            xml_text=xml_text,
            query_id=query_id,
            topic=str(enriched_group["topic"]),
        )
        for candidate in parsed_candidates:
            candidate["matched_keywords"] = matched_keywords_for_candidate(
                candidate,
                enriched_group,
            )
            candidate["score"] = score_candidate(candidate, enriched_group)
        raw_candidates.extend(parsed_candidates)
        successful_query_groups.append(query_id)
        log_event(
            args.run_log_path,
            "discover",
            "query_succeeded",
            "arXiv query group succeeded.",
            query_id=query_id,
            url=url,
            raw_candidate_count=len(parsed_candidates),
        )

    deduped_candidates = dedupe_candidates(raw_candidates)
    ranked_candidates = rank_candidates(deduped_candidates)
    sample_candidates = ranked_candidates[: args.sample_size]
    output_files = output_files_from_args(args)
    if not successful_query_groups:
        discovery_status = "failed"
    elif failed_query_groups:
        discovery_status = "partial"
    else:
        discovery_status = "success"
    report = build_report(
        query_plan=query_plan,
        raw_candidate_count=len(raw_candidates),
        ranked_candidates=ranked_candidates,
        sample_candidates=sample_candidates,
        output_files=output_files,
        discovery_status=discovery_status,
        selected_query_ids=selected_query_ids,
        failed_query_groups=failed_query_groups,
        successful_query_groups=successful_query_groups,
        errors=errors,
        retry_policy=retry_policy,
        simple_query_mode=args.simple_query_mode,
        paper_window=paper_window,
        date_filter_enabled=date_filter_enabled,
        sources_attempted=["arxiv_api"],
        sources_succeeded=["arxiv_api"] if successful_query_groups else [],
        sources_failed=["arxiv_api"] if failed_query_groups and not successful_query_groups else [],
    )

    if successful_query_groups:
        write_jsonl(args.output_candidates, ranked_candidates)
        write_review_csv(args.output_review_csv, ranked_candidates)
        if sample_candidates:
            write_jsonl(args.output_sample, sample_candidates)
    write_json(args.output_report, report)
    log_event(
        args.run_log_path,
        "discover",
        "outputs_written",
        "Research AI discovery outputs written.",
        discovery_status=discovery_status,
        output_report=str(args.output_report),
        output_run_log_path=str(args.run_log_path),
    )

    summary = {
        "mode": "discover",
        "phase": "2A-5A",
        "discovery_status": discovery_status,
        "query_group_count": len(query_groups),
        "raw_candidate_count": len(raw_candidates),
        "deduped_candidate_count": len(ranked_candidates),
        "sample_candidate_count": len(sample_candidates),
        "successful_query_groups": successful_query_groups,
        "failed_query_groups": failed_query_groups,
        "errors": errors,
        "output_candidates": str(args.output_candidates),
        "output_review_csv": str(args.output_review_csv),
        "output_sample": str(args.output_sample),
        "output_report": str(args.output_report),
        "date_filter_enabled": date_filter_enabled,
        "paper_window": paper_window,
        "warnings": report["warnings"],
    }
    if discovery_status == "failed":
        log_event(
            args.run_log_path,
            "discover",
            "run_failed",
            "All selected arXiv query groups failed.",
            output_report=str(args.output_report),
            output_run_log_path=str(args.run_log_path),
        )
        msg = (
            "All selected arXiv query groups failed. See:\n"
            f"- {args.output_report}\n"
            f"- {args.run_log_path}"
        )
        raise RuntimeError(msg)
    if discovery_status == "partial" and not (args.allow_partial or args.continue_on_error):
        msg = "arXiv discovery was partial; rerun with --allow-partial to accept partial outputs."
        raise RuntimeError(msg)
    log_event(
        args.run_log_path,
        "discover",
        "run_completed",
        "Research AI arXiv discovery completed.",
        discovery_status=discovery_status,
    )
    return summary


def discover_html(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = load_json(args.query_plan)
    paper_window = paper_window_from_args(query_plan, args)
    source_ids = source_ids_for_filter(args.source, query_plan)
    if source_ids == ["arxiv_api"]:
        return discover_arxiv(args)

    raw_candidates: list[dict[str, Any]] = []
    sources_attempted: list[str] = []
    sources_succeeded: list[str] = []
    sources_failed: list[str] = []
    errors: list[dict[str, Any]] = []
    log_event(
        args.run_log_path,
        "discover",
        "run_started",
        "Research AI multi-source HTML discovery started.",
        sources=source_ids,
    )

    if "iclr_2025_virtual" in source_ids:
        sources_attempted.append("iclr_2025_virtual")
        try:
            html_text = fetch_html(
                args.iclr_url,
                USER_AGENT,
                args.html_timeout_seconds,
                args.run_log_path,
                mode="discover",
                source_id="iclr_2025_virtual",
            )
            parsed = parse_iclr_2025_papers_html(html_text, args.iclr_url)
            for candidate in parsed[: args.max_candidates_per_source]:
                candidate["score"] = score_html_candidate(candidate)
            raw_candidates.extend(parsed[: args.max_candidates_per_source])
            sources_succeeded.append("iclr_2025_virtual")
            log_event(
                args.run_log_path,
                "discover",
                "query_succeeded",
                "ICLR 2025 HTML discovery succeeded.",
                query_id="iclr_2025_virtual",
                raw_candidate_count=len(parsed),
            )
        except ArxivFetchError as exc:
            sources_failed.append("iclr_2025_virtual")
            errors.append(
                {**exc.details, "source_id": "iclr_2025_virtual", "error_message": str(exc)}
            )
            log_event(
                args.run_log_path,
                "discover",
                "query_failed",
                "ICLR 2025 HTML discovery failed.",
                **errors[-1],
            )

    if "huggingface_papers" in source_ids:
        sources_attempted.append("huggingface_papers")
        source_candidates: list[dict[str, Any]] = []
        source_errors: list[dict[str, Any]] = []
        for planned in build_huggingface_search_urls(
            args.huggingface_url,
            parse_search_queries(args.huggingface_search_queries),
        ):
            try:
                html_text = fetch_html(
                    planned["url"],
                    USER_AGENT,
                    args.html_timeout_seconds,
                    args.run_log_path,
                    mode="discover",
                    source_id="huggingface_papers",
                )
                parsed = parse_huggingface_papers_html(
                    html_text,
                    planned["url"],
                    planned["query"],
                )
                for candidate in parsed:
                    candidate["score"] = score_html_candidate(candidate)
                source_candidates.extend(parsed)
            except ArxivFetchError as exc:
                source_errors.append(
                    {
                        **exc.details,
                        "source_id": "huggingface_papers",
                        "query": planned["query"],
                        "error_message": str(exc),
                    }
                )
        if source_candidates:
            raw_candidates.extend(source_candidates[: args.max_candidates_per_source])
            sources_succeeded.append("huggingface_papers")
            log_event(
                args.run_log_path,
                "discover",
                "query_succeeded",
                "Hugging Face Papers HTML discovery succeeded.",
                query_id="huggingface_papers",
                raw_candidate_count=len(source_candidates),
            )
        else:
            sources_failed.append("huggingface_papers")
            errors.extend(source_errors)
            log_event(
                args.run_log_path,
                "discover",
                "query_failed",
                "Hugging Face Papers HTML discovery failed.",
                query_id="huggingface_papers",
                error_count=len(source_errors),
            )

    deduped_candidates = dedupe_candidates(raw_candidates)
    ranked_candidates = rank_candidates(deduped_candidates)
    sample_candidates = ranked_candidates[: args.sample_size]
    if not sources_succeeded:
        discovery_status = "failed"
    elif sources_failed:
        discovery_status = "partial"
    else:
        discovery_status = "success"

    output_files = output_files_from_args(args)
    report = build_report(
        query_plan=query_plan,
        raw_candidate_count=len(raw_candidates),
        ranked_candidates=ranked_candidates,
        sample_candidates=sample_candidates,
        output_files=output_files,
        discovery_status=discovery_status,
        errors=errors,
        retry_policy={
            "html_timeout_seconds": args.html_timeout_seconds,
            "max_candidates_per_source": args.max_candidates_per_source,
        },
        simple_query_mode=args.simple_query_mode,
        paper_window=paper_window,
        date_filter_enabled=False,
        sources_attempted=sources_attempted,
        sources_succeeded=sources_succeeded,
        sources_failed=sources_failed,
    )
    if sources_succeeded:
        write_jsonl(args.output_candidates, ranked_candidates)
        write_review_csv(args.output_review_csv, ranked_candidates)
        if sample_candidates:
            write_jsonl(args.output_sample, sample_candidates)
    write_json(args.output_report, report)
    log_event(
        args.run_log_path,
        "discover",
        "outputs_written",
        "Research AI multi-source discovery outputs written.",
        discovery_status=discovery_status,
        output_report=str(args.output_report),
        output_run_log_path=str(args.run_log_path),
    )

    summary = {
        "mode": "discover",
        "phase": "2A-5A",
        "source": args.source,
        "discovery_status": discovery_status,
        "sources_attempted": sources_attempted,
        "sources_succeeded": sources_succeeded,
        "sources_failed": sources_failed,
        "raw_candidate_count": len(raw_candidates),
        "deduped_candidate_count": len(ranked_candidates),
        "sample_candidate_count": len(sample_candidates),
        "output_candidates": str(args.output_candidates),
        "output_review_csv": str(args.output_review_csv),
        "output_sample": str(args.output_sample),
        "output_report": str(args.output_report),
        "warnings": report["warnings"],
    }
    if discovery_status == "failed":
        log_event(
            args.run_log_path,
            "discover",
            "run_failed",
            "All selected HTML discovery sources failed.",
            output_report=str(args.output_report),
            output_run_log_path=str(args.run_log_path),
        )
        msg = (
            "All selected Research AI discovery sources failed. See:\n"
            f"- {args.output_report}\n"
            f"- {args.run_log_path}"
        )
        raise RuntimeError(msg)
    log_event(
        args.run_log_path,
        "discover",
        "run_completed",
        "Research AI multi-source HTML discovery completed.",
        discovery_status=discovery_status,
    )
    return summary


def discover(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "source", "arxiv") == "arxiv":
        return discover_arxiv(args)
    return discover_html(args)


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = load_json(args.query_plan)
    paper_window = paper_window_from_args(query_plan, args)
    date_filter_enabled = not args.disable_date_filter
    source_ids = source_ids_for_filter(args.source, query_plan)
    if source_ids != ["arxiv_api"]:
        source_plans = planned_html_sources(args, query_plan)
        log_event(
            args.run_log_path,
            "dry_run",
            "run_started",
            "Research AI multi-source discovery dry-run started.",
            sources=source_ids,
        )
        for planned in source_plans:
            log_event(
                args.run_log_path,
                "dry_run",
                "dry_run_planned_query",
                "Planned public HTML metadata source.",
                query_id=planned["source_id"],
                url=planned.get("url"),
            )
        output_files = output_files_from_args(args)
        report = build_report(
            query_plan=query_plan,
            raw_candidate_count=0,
            ranked_candidates=[],
            sample_candidates=[],
            output_files=output_files,
            discovery_status="dry_run",
            selected_query_ids=[],
            retry_policy={
                "html_timeout_seconds": args.html_timeout_seconds,
                "max_candidates_per_source": args.max_candidates_per_source,
            },
            simple_query_mode=args.simple_query_mode,
            paper_window=paper_window,
            date_filter_enabled=False,
            sources_attempted=source_ids,
        )
        write_json(args.output_report, report)
        log_event(
            args.run_log_path,
            "dry_run",
            "outputs_written",
            "Research AI multi-source discovery dry-run report written.",
            output_report=str(args.output_report),
            output_run_log_path=str(args.run_log_path),
        )
        log_event(
            args.run_log_path,
            "dry_run",
            "run_completed",
            "Research AI multi-source discovery dry-run completed.",
        )
        return {
            "mode": "dry_run",
            "phase": "2A-5A",
            "source": args.source,
            "discovery_status": "dry_run",
            "planned_source_count": len(source_plans),
            "planned_sources": source_plans,
            "planned_query_count": sum(
                len(plan.get("search_urls", [plan.get("url")])) for plan in source_plans
            ),
            "output_candidates": str(args.output_candidates),
            "output_review_csv": str(args.output_review_csv),
            "output_sample": str(args.output_sample),
            "output_report": str(args.output_report),
            "will_download_pdfs": False,
        }
    urls = planned_urls(
        query_plan,
        args.max_results_per_query,
        args.query_id,
        args.simple_query_mode,
        paper_window["start_date"],
        paper_window["end_date"],
        date_filter_enabled,
    )
    selected_query_ids = [planned["query_id"] for planned in urls]
    log_event(
        args.run_log_path,
        "dry_run",
        "run_started",
        "Research AI arXiv discovery dry-run started.",
        selected_query_ids=selected_query_ids,
        simple_query_mode=args.simple_query_mode,
        date_filter_enabled=date_filter_enabled,
        paper_window=paper_window,
    )
    for planned in urls:
        log_event(
            args.run_log_path,
            "dry_run",
            "dry_run_planned_query",
            "Planned arXiv metadata query.",
            query_id=planned["query_id"],
            url=planned["url"],
        )
    output_files = output_files_from_args(args)
    report = build_report(
        query_plan=query_plan,
        raw_candidate_count=0,
        ranked_candidates=[],
        sample_candidates=[],
        output_files=output_files,
        discovery_status="dry_run",
        selected_query_ids=selected_query_ids,
        retry_policy={
            "max_retries": args.max_retries,
            "backoff_seconds": args.backoff_seconds,
            "timeout_seconds": args.timeout_seconds,
            "delay_seconds": (
                float(args.delay_seconds)
                if float(args.delay_seconds) > 0
                else float(query_plan.get("delay_seconds") or 3)
            ),
        },
        simple_query_mode=args.simple_query_mode,
        paper_window=paper_window,
        date_filter_enabled=date_filter_enabled,
        sources_attempted=["arxiv_api"],
    )
    write_json(args.output_report, report)
    log_event(
        args.run_log_path,
        "dry_run",
        "outputs_written",
        "Research AI discovery dry-run report written.",
        output_report=str(args.output_report),
        output_run_log_path=str(args.run_log_path),
    )
    log_event(
        args.run_log_path,
        "dry_run",
        "run_completed",
        "Research AI arXiv discovery dry-run completed.",
    )
    return {
        "mode": "dry_run",
        "phase": "2A-5A",
        "source": args.source,
        "discovery_status": "dry_run",
        "planned_query_count": len(urls),
        "planned_queries": urls,
        "output_candidates": str(args.output_candidates),
        "output_review_csv": str(args.output_review_csv),
        "output_sample": str(args.output_sample),
        "output_report": str(args.output_report),
        "date_filter_enabled": date_filter_enabled,
        "paper_window": paper_window,
        "will_download_pdfs": False,
    }


def write_manual_template(args: argparse.Namespace) -> dict[str, Any]:
    log_event(
        args.run_log_path,
        "write_manual_template",
        "run_started",
        "Writing manual Research AI paper registry template.",
    )
    args.output_manual_template.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manual_template.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANUAL_REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "selection_status": "example_not_approved",
                "topic": "example_topic",
                "paper_id": "research_ai_example_001",
                "arxiv_id": "0000.00000",
                "title": "Example only - replace with approved paper metadata",
                "authors": "Example Author",
                "published": "2024-01-01",
                "primary_category": "cs.CL",
                "abstract_url": "https://arxiv.org/abs/0000.00000",
                "pdf_url": "https://arxiv.org/pdf/0000.00000",
                "reason_for_inclusion": "Example row only; not approved for Phase 2A-5B.",
                "notes": "Replace this row before manual approval.",
            }
        )
    log_event(
        args.run_log_path,
        "write_manual_template",
        "outputs_written",
        "Manual Research AI paper registry template written.",
        output_manual_template=str(args.output_manual_template),
    )
    log_event(
        args.run_log_path,
        "write_manual_template",
        "run_completed",
        "Manual Research AI paper registry template completed.",
    )
    return {
        "mode": "write_manual_template",
        "phase": "2A-5A",
        "output_manual_template": str(args.output_manual_template),
        "run_log_path": str(args.run_log_path),
        "columns": MANUAL_REGISTRY_FIELDS,
        "warnings": [
            (
                "Manual fallback templates must preserve paper provenance and must not "
                "fabricate metadata."
            ),
            "Phase 2A-5B can use a manually approved registry only after review.",
        ],
    }


def _parse_date_prefix(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None


def validate_manual_registry_records(
    records: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_paper_ids: set[str] = set()
    window_start = datetime.strptime(start_date, "%Y-%m-%d")
    window_end = datetime.strptime(end_date, "%Y-%m-%d")
    for index, record in enumerate(records, start=1):
        paper_id = str(record.get("paper_id") or "").strip()
        missing_fields = [
            field
            for field in MANUAL_REGISTRY_REQUIRED_FIELDS
            if record.get(field) in (None, "", [])
        ]
        if missing_fields:
            errors.append(
                {
                    "line_number": index,
                    "paper_id": paper_id,
                    "error_type": "missing_required_fields",
                    "fields": missing_fields,
                }
            )
        if paper_id:
            if paper_id in seen_paper_ids:
                errors.append(
                    {
                        "line_number": index,
                        "paper_id": paper_id,
                        "error_type": "duplicate_paper_id",
                    }
                )
            seen_paper_ids.add(paper_id)
        if str(record.get("selection_status") or "") not in MANUAL_REGISTRY_ALLOWED_STATUSES:
            errors.append(
                {
                    "line_number": index,
                    "paper_id": paper_id,
                    "error_type": "invalid_selection_status",
                    "selection_status": record.get("selection_status"),
                }
            )
        published = _parse_date_prefix(record.get("published"))
        if published is not None and not (window_start <= published <= window_end):
            errors.append(
                {
                    "line_number": index,
                    "paper_id": paper_id,
                    "error_type": "published_date_out_of_window",
                    "published": record.get("published"),
                    "approved_start_date": start_date,
                    "approved_end_date": end_date,
                }
            )
        elif published is None and record.get("published"):
            warnings.append(
                f"Could not parse published date for paper_id={paper_id}; date window not checked."
            )
        abstract_url = str(record.get("abstract_url") or "")
        if abstract_url and not abstract_url.startswith("https://arxiv.org/abs/"):
            errors.append(
                {
                    "line_number": index,
                    "paper_id": paper_id,
                    "error_type": "invalid_abstract_url",
                    "abstract_url": abstract_url,
                }
            )
        pdf_url = str(record.get("pdf_url") or "")
        if pdf_url and not pdf_url.startswith("https://arxiv.org/pdf/"):
            errors.append(
                {
                    "line_number": index,
                    "paper_id": paper_id,
                    "error_type": "invalid_pdf_url",
                    "pdf_url": pdf_url,
                }
            )
        if not str(record.get("arxiv_id") or "").strip():
            errors.append(
                {
                    "line_number": index,
                    "paper_id": paper_id,
                    "error_type": "empty_arxiv_id",
                }
            )
    return errors, warnings


def build_manual_validation_report(
    records: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    warnings: list[str],
    registry_path: Path,
    paper_window: dict[str, Any],
) -> dict[str, Any]:
    status_counter = Counter(str(record.get("selection_status") or "") for record in records)
    return {
        "phase": "2A-5A",
        "generated_at_utc": utc_now(),
        "validation_status": "failed" if errors else "passed",
        "manual_registry_path": str(registry_path),
        "record_count": len(records),
        "valid_record_count": len(records) - len({error.get("line_number") for error in errors}),
        "invalid_record_count": len({error.get("line_number") for error in errors}),
        "counts_by_selection_status": dict(status_counter),
        "paper_window": paper_window,
        "required_fields": MANUAL_REGISTRY_REQUIRED_FIELDS,
        "allowed_selection_statuses": sorted(MANUAL_REGISTRY_ALLOWED_STATUSES),
        "errors": errors,
        "warnings": warnings,
        "next_step": (
            "Proceed to Phase 2A-5B only after the manual registry validates and all "
            "approved records contain real paper provenance. Do not use example_not_approved "
            "records for curation."
        ),
    }


def validate_manual_registry(args: argparse.Namespace) -> dict[str, Any]:
    query_plan = load_json(args.query_plan)
    paper_window = paper_window_from_args(query_plan, args)
    if not args.manual_registry_path.exists():
        msg = (
            "Manual approved registry not found. Create "
            "data/sources/research_ai_approved_papers.jsonl from the template or rerun "
            "arXiv discovery later."
        )
        raise RuntimeError(msg)

    records = read_jsonl(args.manual_registry_path)
    errors, warnings = validate_manual_registry_records(
        records,
        paper_window["start_date"],
        paper_window["end_date"],
    )
    report = build_manual_validation_report(
        records=records,
        errors=errors,
        warnings=warnings,
        registry_path=args.manual_registry_path,
        paper_window=paper_window,
    )
    write_json(args.manual_validation_report, report)
    summary = {
        "mode": "validate_manual_registry",
        "phase": "2A-5A",
        "validation_status": report["validation_status"],
        "manual_registry_path": str(args.manual_registry_path),
        "manual_validation_report": str(args.manual_validation_report),
        "record_count": len(records),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "paper_window": paper_window,
    }
    if errors:
        msg = f"Manual registry validation failed. See {args.manual_validation_report} for details."
        raise RuntimeError(msg)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--write-manual-template", action="store_true")
    parser.add_argument("--validate-manual-registry", action="store_true")
    parser.add_argument("--query-plan", type=Path, default=DEFAULT_QUERY_PLAN)
    parser.add_argument("--query-id", default="all")
    parser.add_argument("--source", default="all", choices=["all", "arxiv", "iclr", "huggingface"])
    parser.add_argument("--iclr-url", default=DEFAULT_ICLR_URL)
    parser.add_argument("--huggingface-url", default=DEFAULT_HUGGINGFACE_URL)
    parser.add_argument(
        "--huggingface-search-queries",
        default=DEFAULT_HUGGINGFACE_SEARCH_QUERIES,
    )
    parser.add_argument("--output-candidates", type=Path, default=DEFAULT_OUTPUT_CANDIDATES)
    parser.add_argument("--output-review-csv", type=Path, default=DEFAULT_OUTPUT_REVIEW_CSV)
    parser.add_argument("--output-sample", type=Path, default=DEFAULT_OUTPUT_SAMPLE)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--run-log-path", type=Path, default=DEFAULT_RUN_LOG)
    parser.add_argument("--output-manual-template", type=Path, default=DEFAULT_MANUAL_TEMPLATE)
    parser.add_argument("--manual-registry-path", type=Path, default=DEFAULT_MANUAL_REGISTRY)
    parser.add_argument(
        "--manual-validation-report",
        type=Path,
        default=DEFAULT_MANUAL_VALIDATION_REPORT,
    )
    parser.add_argument("--max-results-per-query", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--max-candidates-per-source", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--html-timeout-seconds", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--simple-query-mode", action="store_true")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--disable-date-filter", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode_count = sum(
        bool(mode)
        for mode in (
            args.discover,
            args.dry_run,
            args.write_manual_template,
            args.validate_manual_registry,
        )
    )
    if mode_count != 1:
        print(
            (
                "Pass exactly one mode: --dry-run, --discover, --write-manual-template, "
                "or --validate-manual-registry."
            ),
            file=sys.stderr,
        )
        return 2
    try:
        if args.dry_run:
            summary = dry_run(args)
        elif args.write_manual_template:
            summary = write_manual_template(args)
        elif args.validate_manual_registry:
            summary = validate_manual_registry(args)
        else:
            summary = discover(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
