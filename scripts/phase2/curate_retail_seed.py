"""Build curated Retail / E-commerce Support Phase 2A-6C seed records.

This script consumes controlled local Amazon Reviews 2023 samples produced by
Phase 2A-6B and creates a small curated seed dataset. It does not build RAG,
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
from statistics import mean
from typing import Any

PHASE = "2A-6C"
VERTICAL = "retail"
DEFAULT_REVIEWS_INPUT = Path("data/generated/retail/amazon_reviews_sample.jsonl")
DEFAULT_METADATA_INPUT = Path("data/generated/retail/amazon_metadata_sample.jsonl")
DEFAULT_EXPLORATION_REPORT = Path("data/generated/retail/amazon_reviews_exploration_report.json")
DEFAULT_QUALITY_REPORT = Path("data/generated/retail/amazon_reviews_quality_report.json")
DEFAULT_OUTPUT_PROMPTS = Path("data/real_world_samples/retail_sample.jsonl")
DEFAULT_OUTPUT_KB = Path("data/kb/retail/kb_sample.jsonl")
DEFAULT_OUTPUT_GOLD = Path("data/eval/gold/retail_gold_sample.jsonl")
DEFAULT_CURATION_REPORT = Path("data/generated/retail/retail_curation_report.json")

RUN_2A6B_MESSAGE = (
    "Run Phase 2A-6B first, for example:\n"
    "python scripts/phase2/explore_retail_amazon_reviews.py --load-from-huggingface "
    "--category All_Beauty --sample-limit 1000 --metadata-limit 1000"
)

PROMPT_DISTRIBUTION = [
    ("review_summary", 6, "answer_grounded", "text", "answer"),
    ("issue_identification", 7, "answer_grounded", "text", "answer"),
    ("compare_products", 5, "compare_products", "markdown_table", "answer"),
    ("structured_extraction", 6, "extract_structured", "json", "answer"),
    ("support_policy_reasoning", 5, "policy_reasoning", "text", "answer"),
    ("evidence_citation_lookup", 4, "answer_grounded", "text", "answer"),
    ("spam_or_low_quality_review", 3, "quality_boundary", "text", "spam_or_low_quality"),
    (
        "insufficient_evidence_or_escalation",
        3,
        "escalation_response",
        "text",
        "insufficient_evidence",
    ),
    ("out_of_scope", 1, "boundary_response", "text", "out_of_scope"),
]

ISSUE_TERMS = [
    "broken",
    "damaged",
    "defective",
    "cheap",
    "quality",
    "return",
    "refund",
    "size",
    "fit",
    "battery",
    "charger",
    "smell",
    "leak",
    "missing",
    "late",
    "fake",
    "works",
    "recommend",
]

SUPPORT_POLICIES = [
    (
        "return_refund_triage",
        "Synthetic benchmark policy: return and refund triage",
        (
            "This is synthetic benchmark policy, not Amazon policy. For return or refund "
            "requests, use the cited product evidence to identify the reported issue, then "
            "ask for order-specific eligibility details before promising a refund."
        ),
        ["return", "refund", "triage"],
    ),
    (
        "damaged_item_handling",
        "Synthetic benchmark policy: damaged item handling",
        (
            "This is synthetic benchmark policy, not Amazon policy. If review evidence says "
            "an item arrived damaged, the assistant may recommend support follow-up for "
            "replacement or refund review, but should not guarantee an outcome."
        ),
        ["damaged", "replacement", "refund"],
    ),
    (
        "missing_item_handling",
        "Synthetic benchmark policy: missing item handling",
        (
            "This is synthetic benchmark policy, not Amazon policy. Missing item claims need "
            "order, shipment, and package evidence before the assistant recommends a final "
            "resolution."
        ),
        ["missing", "shipment", "escalation"],
    ),
    (
        "wrong_item_handling",
        "Synthetic benchmark policy: wrong item handling",
        (
            "This is synthetic benchmark policy, not Amazon policy. Wrong item reports should "
            "be routed to support review with product identifiers and customer-provided order "
            "evidence."
        ),
        ["wrong_item", "support_review"],
    ),
    (
        "quality_complaint_handling",
        "Synthetic benchmark policy: quality complaint handling",
        (
            "This is synthetic benchmark policy, not Amazon policy. Quality complaints should "
            "be summarized from cited reviews, separated from one-off comments, and escalated "
            "when safety or defect claims require review."
        ),
        ["quality", "complaint", "safety"],
    ),
    (
        "low_quality_review_handling",
        "Synthetic benchmark policy: low-quality review handling",
        (
            "This is synthetic benchmark policy, not Amazon policy. Very short, repetitive, "
            "or unclear reviews should not be treated as strong product evidence; flag them "
            "as low-quality or escalate if moderation judgment is needed."
        ),
        ["low_quality", "review_quality"],
    ),
    (
        "escalation_rules",
        "Synthetic benchmark policy: escalation rules",
        (
            "This is synthetic benchmark policy, not Amazon policy. Escalate when the evidence "
            "is missing, ambiguous, safety-related, or requires account, payment, or order "
            "data that is not present in the benchmark context."
        ),
        ["escalation", "insufficient_evidence"],
    ),
    (
        "out_of_scope_rules",
        "Synthetic benchmark policy: out-of-scope rules",
        (
            "This is synthetic benchmark policy, not Amazon policy. Questions unrelated to "
            "the selected retail product, review, metadata, or support-policy evidence should "
            "be marked out_of_scope and should not be answered from general memory."
        ),
        ["out_of_scope", "boundary"],
    ),
]

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\s().-]*){7,}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
FORBIDDEN_PUBLIC_RE = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"C:\\Users",
        r"/home/",
        r"akpoogaga",
        r"kparo",
        r"\btoken\b",
        r"API\s*key",
    )
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: str, length: int = 10) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return parsed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object row in {path}")
        rows.append(parsed)
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = HTML_TAG_RE.sub(" ", text)
    replacements = {
        "&amp;": "&",
        "&quot;": '"',
        "&#34;": '"',
        "&#39;": "'",
        "&apos;": "'",
        "&lt;": "<",
        "&gt;": ">",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = normalize_whitespace(text)
    text = re.sub(r"API\s*key", "credential", text, flags=re.IGNORECASE)
    text = re.sub(r"\btokens?\b", "text units", text, flags=re.IGNORECASE)
    return text


def has_forbidden_public_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in FORBIDDEN_PUBLIC_RE)


def has_pii_like_text(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text))


def truncate_words(text: str, max_chars: int) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}."


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def issue_terms_for_text(text: str) -> list[str]:
    tokens = set(tokenize(text))
    return [term for term in ISSUE_TERMS if term in tokens]


def rating_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def short_product_title(title: str, fallback: str) -> str:
    cleaned = clean_text(title)
    if not cleaned:
        return fallback
    return truncate_words(cleaned, 90)


def is_generic_retail_title(title: str, category: str) -> bool:
    cleaned = clean_text(title)
    if not cleaned or cleaned.lower() in {"none", "null", "n/a"}:
        return True
    escaped_category = re.escape(category).replace("_", r"[_ ]")
    generic_patterns = [
        rf"{escaped_category}\s+product\s+B[0-9A-Z]{{7,12}}",
        r"retail\s+product\s+B[0-9A-Z]{7,12}",
        r"product\s+B[0-9A-Z]{7,12}",
    ]
    return any(re.fullmatch(pattern, cleaned, flags=re.IGNORECASE) for pattern in generic_patterns)


def metadata_details(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_asin_values(value: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"B[0-9A-Z]{7,12}", stripped, flags=re.IGNORECASE):
            values.add(stripped.upper())
        return values
    if isinstance(value, list | tuple | set):
        for item in value:
            values.update(_extract_asin_values(item))
        return values
    if isinstance(value, dict):
        for nested_value in value.values():
            values.update(_extract_asin_values(nested_value))
    return values


def _metadata_index_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("parent_asin", "asin", "parentAsin", "parentASIN"):
        keys.update(_extract_asin_values(row.get(field)))
    for key, value in row.items():
        normalized_key = key.lower().replace("-", "_")
        if normalized_key == "asin" or normalized_key.endswith("_asin"):
            keys.update(_extract_asin_values(value))
    details = metadata_details(row.get("details"))
    for key, value in details.items():
        normalized_key = str(key).lower().replace("-", "_")
        if normalized_key == "asin" or normalized_key.endswith("_asin"):
            keys.update(_extract_asin_values(value))
    return keys


def metadata_row_quality_score(row: dict[str, Any], category: str = "All_Beauty") -> int:
    score = 0
    title = clean_text(row.get("title"))
    if title and not is_generic_retail_title(title, category):
        score += 10
    for field, weight in (
        ("features", 2),
        ("description", 2),
        ("categories", 1),
        ("store", 1),
        ("average_rating", 1),
        ("rating_number", 1),
    ):
        value = row.get(field)
        if value not in (None, "", [], {}, "None"):
            score += weight
    return score


def build_product_metadata_index(
    metadata_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        for key in _metadata_index_keys(row):
            current = index.get(key)
            if current is None or metadata_row_quality_score(row) > metadata_row_quality_score(
                current
            ):
                index[key] = row
    return index


def build_metadata_index(metadata_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return build_product_metadata_index(metadata_rows)


def resolve_product_title(
    parent_asin: str,
    asin: str | None,
    metadata_index: dict[str, dict[str, Any]],
    category: str,
) -> tuple[str, dict[str, Any]]:
    fallback = f"{category} product {parent_asin}"
    source_keys = [parent_asin]
    if asin:
        source_keys.append(asin)
    for source_key in dict.fromkeys(key.upper() for key in source_keys if key):
        metadata_row = metadata_index.get(source_key)
        if not metadata_row:
            continue
        title = clean_text(metadata_row.get("title"))
        if title and not is_generic_retail_title(title, category):
            return (
                short_product_title(title, fallback),
                {
                    "title_resolution": "metadata_title",
                    "title_source_key": source_key,
                    "metadata_found": True,
                    "parent_asin": parent_asin,
                },
            )
        if any(
            metadata_row.get(field) not in (None, "", [], {}, "None")
            for field in ("features", "description", "store", "categories", "main_category")
        ):
            return (
                fallback,
                {
                    "title_resolution": "metadata_partial",
                    "title_source_key": source_key,
                    "metadata_found": True,
                    "parent_asin": parent_asin,
                },
            )
    return (
        fallback,
        {
            "title_resolution": "generic_fallback",
            "title_source_key": parent_asin,
            "metadata_found": False,
            "parent_asin": parent_asin,
        },
    )


def validate_inputs(args: argparse.Namespace) -> None:
    required_paths = [
        Path(args.reviews_input),
        Path(args.metadata_input),
        Path(args.exploration_report),
        Path(args.quality_report),
    ]
    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"Missing required input: {path}. {RUN_2A6B_MESSAGE}")


def validate_reviews(reviews: list[dict[str, Any]]) -> None:
    if not reviews:
        raise RuntimeError(f"Review sample is empty. {RUN_2A6B_MESSAGE}")
    raw_identifier_rows = [index for index, row in enumerate(reviews) if "user_id" in row]
    if raw_identifier_rows:
        raise RuntimeError(
            "Review sample contains raw customer identifiers. Rerun Phase 2A-6B "
            "sanitization before curation."
        )


def is_low_quality_review(text: str) -> bool:
    tokens = tokenize(text)
    if len(tokens) < 5:
        return True
    if len(tokens) >= 8:
        common_count = Counter(tokens).most_common(1)[0][1]
        if common_count / len(tokens) > 0.55:
            return True
    return False


def build_review_candidates(
    reviews: list[dict[str, Any]],
    metadata_index: dict[str, dict[str, Any]],
    max_review_body_chars: int,
    category: str = "All_Beauty",
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(reviews, start=1):
        parent_asin = str(row.get("parent_asin") or "").strip()
        asin = str(row.get("asin") or parent_asin).strip()
        title = clean_text(row.get("title"))
        text = clean_text(row.get("text"))
        combined_text = f"{title} {text}"
        if not parent_asin or not text or "user_id" in row:
            continue
        if has_pii_like_text(combined_text) or has_forbidden_public_text(combined_text):
            continue
        rating = rating_value(row.get("rating"))
        if rating is None or rating < 1 or rating > 5:
            continue
        product_title, title_resolution = resolve_product_title(
            parent_asin,
            asin,
            metadata_index,
            category,
        )
        review_id = f"retail_review_{index:04d}_{stable_hash(parent_asin + asin + title, 8)}"
        terms = issue_terms_for_text(combined_text)
        low_quality = is_low_quality_review(text)
        candidates.append(
            {
                "review_id": review_id,
                "row_index": index,
                "asin": asin,
                "parent_asin": parent_asin,
                "product_title": product_title,
                "title_resolution": title_resolution["title_resolution"],
                "title_source_key": title_resolution["title_source_key"],
                "metadata_found": title_resolution["metadata_found"],
                "rating": rating,
                "review_title": title,
                "review_text": truncate_words(text, max_review_body_chars),
                "verified_purchase": bool(row.get("verified_purchase")),
                "helpful_vote": int(row.get("helpful_vote") or 0),
                "issue_terms": terms,
                "low_quality": low_quality,
                "source_quality": (
                    "low_quality_review" if low_quality else "real_sample_sanitized"
                ),
            }
        )
    return candidates


def build_metadata_candidates(metadata_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(metadata_rows, start=1):
        parent_asin = str(row.get("parent_asin") or "").strip()
        title = clean_text(row.get("title"))
        if not parent_asin or not title:
            continue
        candidate_text = " ".join(
            clean_text(value)
            for value in (
                row.get("title"),
                row.get("main_category"),
                row.get("store"),
                row.get("price"),
                row.get("details"),
            )
        )
        if has_pii_like_text(candidate_text) or has_forbidden_public_text(candidate_text):
            continue
        details = metadata_details(row.get("details"))
        candidates.append(
            {
                "metadata_id": f"retail_metadata_{index:04d}_{stable_hash(parent_asin, 8)}",
                "parent_asin": parent_asin,
                "title": short_product_title(title, f"All_Beauty product {parent_asin}"),
                "main_category": clean_text(row.get("main_category")) or "All_Beauty",
                "average_rating": row.get("average_rating"),
                "rating_number": row.get("rating_number"),
                "features": row.get("features") if isinstance(row.get("features"), list) else [],
                "description": (
                    row.get("description") if isinstance(row.get("description"), list) else []
                ),
                "price": clean_text(row.get("price")),
                "categories": row.get("categories")
                if isinstance(row.get("categories"), list)
                else [],
                "details": details,
                "store": clean_text(row.get("store")),
            }
        )
    return candidates


def metadata_body(candidate: dict[str, Any], max_chars: int = 1000) -> str:
    features = "; ".join(clean_text(item) for item in candidate.get("features", [])[:4] if item)
    description = "; ".join(
        clean_text(item) for item in candidate.get("description", [])[:2] if item
    )
    details = candidate.get("details", {})
    detail_bits = []
    if isinstance(details, dict):
        for key, value in list(details.items())[:4]:
            detail_bits.append(f"{clean_text(key)}: {clean_text(value)}")
    parts = [
        f"Product metadata for {candidate['title']}.",
        f"Parent ASIN: {candidate['parent_asin']}.",
        f"Category: {candidate.get('main_category') or 'All_Beauty'}.",
    ]
    if candidate.get("average_rating") is not None:
        parts.append(f"Average rating: {candidate['average_rating']}.")
    if candidate.get("rating_number") is not None:
        parts.append(f"Rating count: {candidate['rating_number']}.")
    if candidate.get("price") and candidate.get("price") != "None":
        parts.append(f"Price field: {candidate['price']}.")
    if features:
        parts.append(f"Features: {features}.")
    if description:
        parts.append(f"Description: {description}.")
    if detail_bits:
        parts.append(f"Details: {'; '.join(detail_bits)}.")
    return truncate_words(" ".join(parts), max_chars)


def review_body(candidate: dict[str, Any]) -> str:
    terms = ", ".join(candidate["issue_terms"]) if candidate["issue_terms"] else "none detected"
    return (
        f"Review evidence for {candidate['product_title']} ({candidate['parent_asin']}). "
        f"Rating: {candidate['rating']}. Verified purchase: {candidate['verified_purchase']}. "
        f"Helpful votes: {candidate['helpful_vote']}. Review title: {candidate['review_title']}. "
        f"Review text: {candidate['review_text']} Issue/use terms detected: {terms}."
    )


def build_kb_records(
    review_candidates: list[dict[str, Any]],
    metadata_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    context: dict[str, dict[str, Any]] = {}

    def add_record(record: dict[str, Any]) -> None:
        records.append(record)
        context[str(record["doc_id"])] = record

    for key, title, body, tags in SUPPORT_POLICIES:
        doc_id = f"retail_policy_{key}"
        add_record(
            {
                "doc_id": doc_id,
                "vertical": VERTICAL,
                "title": title,
                "document_type": "support_policy",
                "source_type": "derived",
                "body": body,
                "version": "1.0",
                "tags": ["retail", "synthetic_benchmark_policy", *tags],
                "source_id": "retail_phase2a6c_policy",
                "allowed_to_commit": True,
                "metadata": {
                    "source_quality": "synthetic_policy",
                    "category": "All_Beauty",
                    "evidence_type": "policy",
                    "policy_key": key,
                    "synthetic_benchmark_policy": True,
                    "not_amazon_policy": True,
                },
            }
        )

    selected_metadata = metadata_candidates[:18]
    for index, candidate in enumerate(selected_metadata, start=1):
        doc_id = f"retail_metadata_{index:04d}_{stable_hash(candidate['parent_asin'], 6)}"
        add_record(
            {
                "doc_id": doc_id,
                "vertical": VERTICAL,
                "title": f"{candidate['title']} - Product Metadata",
                "document_type": "product_metadata",
                "source_type": "derived",
                "body": metadata_body(candidate),
                "version": "1.0",
                "tags": ["retail", "product_metadata", "All_Beauty"],
                "source_id": "amazon_reviews_2023_controlled_sample",
                "allowed_to_commit": True,
                "metadata": {
                    "parent_asin": candidate["parent_asin"],
                    "product_title": candidate["title"],
                    "title_resolution": "metadata_title",
                    "title_source_key": candidate["parent_asin"],
                    "metadata_found": True,
                    "category": "All_Beauty",
                    "evidence_type": "metadata",
                    "source_quality": "real_sample_sanitized",
                    "average_rating": candidate.get("average_rating"),
                    "rating_number": candidate.get("rating_number"),
                },
            }
        )

    selected_reviews = review_candidates
    for index, candidate in enumerate(selected_reviews, start=1):
        doc_id = f"retail_review_{index:04d}_{stable_hash(candidate['review_id'], 6)}"
        add_record(
            {
                "doc_id": doc_id,
                "vertical": VERTICAL,
                "title": f"{candidate['product_title']} - Review Evidence",
                "document_type": "review_evidence",
                "source_type": "derived",
                "body": review_body(candidate),
                "version": "1.0",
                "tags": ["retail", "review_evidence", "All_Beauty"],
                "source_id": "amazon_reviews_2023_controlled_sample",
                "allowed_to_commit": True,
                "metadata": {
                    "parent_asin": candidate["parent_asin"],
                    "asin": candidate["asin"],
                    "product_title": candidate["product_title"],
                    "title_resolution": candidate["title_resolution"],
                    "title_source_key": candidate["title_source_key"],
                    "metadata_found": candidate["metadata_found"],
                    "rating": candidate["rating"],
                    "verified_purchase": candidate["verified_purchase"],
                    "helpful_vote": candidate["helpful_vote"],
                    "category": "All_Beauty",
                    "evidence_type": "review",
                    "source_quality": candidate["source_quality"],
                    "issue_terms": candidate["issue_terms"],
                    "review_evidence_id": candidate["review_id"],
                },
            }
        )
        candidate["review_doc_id"] = doc_id

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in selected_reviews:
        grouped[candidate["parent_asin"]].append(candidate)
    for index, (parent_asin, rows) in enumerate(list(grouped.items())[:14], start=1):
        ratings = [float(row["rating"]) for row in rows]
        terms = sorted({term for row in rows for term in row["issue_terms"]})
        title = str(rows[0]["product_title"])
        summary_terms = ", ".join(terms) if terms else "general product experience"
        body = (
            f"Review summary for {title} ({parent_asin}) from {len(rows)} sanitized "
            f"controlled-sample review(s). Average cited rating: {round(mean(ratings), 2)}. "
            f"Detected themes: {summary_terms}. This summary is derived only from the cited "
            "sanitized review evidence."
        )
        doc_id = f"retail_summary_{index:04d}_{stable_hash(parent_asin, 6)}"
        add_record(
            {
                "doc_id": doc_id,
                "vertical": VERTICAL,
                "title": f"{title} - Review Summary",
                "document_type": "review_summary",
                "source_type": "derived",
                "body": body,
                "version": "1.0",
                "tags": ["retail", "review_summary", "All_Beauty"],
                "source_id": "amazon_reviews_2023_controlled_sample",
                "allowed_to_commit": True,
                "metadata": {
                    "parent_asin": parent_asin,
                    "product_title": title,
                    "title_resolution": rows[0]["title_resolution"],
                    "title_source_key": rows[0]["title_source_key"],
                    "metadata_found": rows[0]["metadata_found"],
                    "category": "All_Beauty",
                    "evidence_type": "summary",
                    "source_quality": "real_sample_sanitized",
                    "review_doc_ids": [row["review_doc_id"] for row in rows],
                    "rating": round(mean(ratings), 2),
                    "issue_terms": terms,
                },
            }
        )
        for row in rows:
            row.setdefault("summary_doc_id", doc_id)

    return records, context


def doc_ref(doc_id: str, kb_by_id: dict[str, dict[str, Any]]) -> str:
    record = kb_by_id[doc_id]
    parent_asin = record.get("metadata", {}).get("parent_asin", "policy")
    return f"retail://All_Beauty/{parent_asin}#{doc_id}"


def must_include_for_context(context: dict[str, Any], extra: list[str] | None = None) -> list[str]:
    terms = [str(context["parent_asin"]), str(context["product_title"]).split()[0]]
    terms.extend(context.get("issue_terms") or [])
    if extra:
        terms.extend(extra)
    return [term for term in dict.fromkeys(terms) if term]


def make_prompt(
    *,
    prompt_id: str,
    prompt_category: str,
    task_type: str,
    question: str,
    expected_output_format: str,
    expected_status: str,
    evidence_ids: list[str],
    parent_asins: list[str],
    titles: list[str],
    difficulty: str,
    requires_citation: bool,
    issue_type: str,
    title_resolutions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    product_title = " vs ".join(titles[:2]) if titles else "Retail evidence"
    resolution_rows = title_resolutions or [
        {
            "parent_asin": parent_asin,
            "product_title": title,
            "title_resolution": (
                "policy_context"
                if str(parent_asin).startswith("retail_")
                else (
                    "generic_fallback"
                    if is_generic_retail_title(title, "All_Beauty")
                    else "metadata_title"
                )
            ),
            "title_source_key": parent_asin,
            "metadata_found": (
                False
                if str(parent_asin).startswith("retail_")
                else not is_generic_retail_title(title, "All_Beauty")
            ),
        }
        for parent_asin, title in zip(parent_asins, titles, strict=False)
    ]
    return {
        "prompt_id": prompt_id,
        "vertical": VERTICAL,
        "task_type": task_type,
        "question": question,
        "expected_output_format": expected_output_format,
        "expected_status": expected_status,
        "required_evidence_ids": evidence_ids,
        "required_doc_ids": evidence_ids,
        "source_parent_asins": parent_asins,
        "source_product_ids": parent_asins,
        "product_id": parent_asins[0] if parent_asins else "retail_boundary",
        "product_title": product_title,
        "issue_type": issue_type,
        "expected_action": "escalate" if expected_status == "escalate" else "answer",
        "category": "All_Beauty",
        "metadata": {
            "prompt_category": prompt_category,
            "category": "All_Beauty",
            "source_parent_asins": parent_asins,
            "source_titles": titles,
            "title_resolution": resolution_rows,
            "evidence_type": "retail_curated_seed",
            "difficulty": difficulty,
            "requires_citation": requires_citation,
        },
    }


def make_gold(
    prompt: dict[str, Any],
    *,
    reference_answer: str,
    must_include: list[str],
    must_not_include: list[str],
    kb_by_id: dict[str, dict[str, Any]],
    expected_escalation: bool = False,
) -> dict[str, Any]:
    evidence_ids = [str(doc_id) for doc_id in prompt["required_evidence_ids"]]
    return {
        "prompt_id": prompt["prompt_id"],
        "vertical": VERTICAL,
        "task_type": prompt["task_type"],
        "expected_status": prompt["expected_status"],
        "reference_answer": reference_answer,
        "must_include": [term for term in dict.fromkeys(must_include) if term],
        "must_not_include": [term for term in dict.fromkeys(must_not_include) if term],
        "required_doc_ids": evidence_ids,
        "required_chunk_ids": evidence_ids,
        "required_citations": [
            doc_ref(doc_id, kb_by_id) for doc_id in evidence_ids if doc_id in kb_by_id
        ],
        "expected_escalation": expected_escalation,
        "metadata": {
            "prompt_category": prompt["metadata"]["prompt_category"],
            "required_parent_asins": prompt["source_parent_asins"],
            "required_evidence_ids": evidence_ids,
            "expected_output_format": prompt["expected_output_format"],
            "source_titles": prompt["metadata"]["source_titles"],
            "title_resolution": prompt["metadata"].get("title_resolution", []),
            "evidence_types": [
                kb_by_id[doc_id].get("metadata", {}).get("evidence_type", "unknown")
                for doc_id in evidence_ids
                if doc_id in kb_by_id
            ],
        },
    }


def context_from_review(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": candidate["parent_asin"],
        "asin": candidate["asin"],
        "product_title": candidate["product_title"],
        "title_resolution": candidate["title_resolution"],
        "title_source_key": candidate["title_source_key"],
        "metadata_found": candidate["metadata_found"],
        "rating": candidate["rating"],
        "review_title": candidate["review_title"],
        "review_text": candidate["review_text"],
        "issue_terms": candidate["issue_terms"],
        "review_doc_id": candidate["review_doc_id"],
        "summary_doc_id": candidate.get("summary_doc_id"),
    }


def title_resolution_from_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": context["parent_asin"],
        "product_title": context["product_title"],
        "title_resolution": context.get("title_resolution", "generic_fallback"),
        "title_source_key": context.get("title_source_key", context["parent_asin"]),
        "metadata_found": bool(context.get("metadata_found")),
    }


def review_selection_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    resolution_priority = {
        "metadata_title": 0,
        "metadata_partial": 1,
        "generic_fallback": 2,
    }
    return (
        resolution_priority.get(str(row.get("title_resolution")), 3),
        -len(row.get("issue_terms") or []),
        int(row.get("row_index") or 0),
    )


def build_prompt_and_gold_records(
    review_candidates: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    kb_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    contexts = [
        context_from_review(row)
        for row in review_candidates
        if row.get("review_doc_id") and not row.get("low_quality")
    ]
    issue_contexts = [row for row in contexts if row["issue_terms"]]
    low_quality = [
        context_from_review(row)
        for row in review_candidates
        if row.get("review_doc_id") and row.get("low_quality")
    ]
    summary_contexts = [row for row in contexts if row.get("summary_doc_id")]

    if len(contexts) < 28 or len(issue_contexts) < 18 or len(low_quality) < 3:
        raise RuntimeError("Not enough safe Retail review evidence to build the curated seed.")

    def next_id() -> str:
        return f"retail_seed_{len(prompts) + 1:04d}"

    def add(prompt: dict[str, Any], gold_record: dict[str, Any]) -> None:
        prompts.append(prompt)
        gold.append(gold_record)

    for ctx in summary_contexts[:6]:
        evidence = [ctx["summary_doc_id"], ctx["review_doc_id"]]
        question = (
            f"Summarize the available review evidence for {ctx['product_title']} "
            f"({ctx['parent_asin']}) in two or three support-ready sentences."
        )
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="review_summary",
            task_type="answer_grounded",
            question=question,
            expected_output_format="text",
            expected_status="answer",
            evidence_ids=evidence,
            parent_asins=[ctx["parent_asin"]],
            titles=[ctx["product_title"]],
            difficulty="easy",
            requires_citation=True,
            issue_type="review_summary",
            title_resolutions=[title_resolution_from_context(ctx)],
        )
        answer = (
            f"The cited evidence for {ctx['product_title']} ({ctx['parent_asin']}) "
            f"comes from a sanitized review and derived summary. The review rating is "
            f"{ctx['rating']} and the main visible themes are "
            f"{', '.join(ctx['issue_terms']) if ctx['issue_terms'] else 'general experience'}. "
            "A grounded support answer should summarize only those cited review signals."
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=must_include_for_context(ctx, ["review", "rating"]),
                must_not_include=[
                    "unsupported claims",
                    "customer identifiers",
                    "outside product evidence",
                ],
                kb_by_id=kb_by_id,
            ),
        )

    for ctx in issue_contexts[:7]:
        issue_text = ", ".join(ctx["issue_terms"][:3])
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="issue_identification",
            task_type="answer_grounded",
            question=(
                f"What issue or usage theme is supported by the cited review for "
                f"{ctx['product_title']} ({ctx['parent_asin']})?"
            ),
            expected_output_format="text",
            expected_status="answer",
            evidence_ids=[ctx["review_doc_id"]],
            parent_asins=[ctx["parent_asin"]],
            titles=[ctx["product_title"]],
            difficulty="easy",
            requires_citation=True,
            issue_type="quality_complaint",
            title_resolutions=[title_resolution_from_context(ctx)],
        )
        answer = (
            f"The cited review supports the theme {issue_text} for {ctx['product_title']} "
            f"({ctx['parent_asin']}). The review has rating {ctx['rating']} and should be "
            "treated as one piece of sampled review evidence, not as a claim about all buyers."
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=must_include_for_context(ctx, ctx["issue_terms"][:2]),
                must_not_include=[
                    "all customers",
                    "unsupported defect rate",
                    "customer identifiers",
                ],
                kb_by_id=kb_by_id,
            ),
        )

    compare_pairs = [(contexts[i], contexts[i + 1]) for i in range(7, 17, 2)]
    for left, right in compare_pairs[:5]:
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="compare_products",
            task_type="compare_products",
            question=(
                f"Compare the cited review evidence for {left['product_title']} "
                f"({left['parent_asin']}) and {right['product_title']} ({right['parent_asin']})."
            ),
            expected_output_format="markdown_table",
            expected_status="answer",
            evidence_ids=[left["review_doc_id"], right["review_doc_id"]],
            parent_asins=[left["parent_asin"], right["parent_asin"]],
            titles=[left["product_title"], right["product_title"]],
            difficulty="medium",
            requires_citation=True,
            issue_type="recommendation",
            title_resolutions=[
                title_resolution_from_context(left),
                title_resolution_from_context(right),
            ],
        )
        answer = (
            "| Product | Cited rating | Evidence theme |\n"
            "| --- | ---: | --- |\n"
            f"| {left['product_title']} ({left['parent_asin']}) | {left['rating']} | "
            f"{', '.join(left['issue_terms']) if left['issue_terms'] else 'general review'} |\n"
            f"| {right['product_title']} ({right['parent_asin']}) | {right['rating']} | "
            f"{', '.join(right['issue_terms']) if right['issue_terms'] else 'general review'} |"
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=[
                    left["parent_asin"],
                    right["parent_asin"],
                    "rating",
                    "Evidence theme",
                ],
                must_not_include=["uncited comparison", "overall marketplace ranking"],
                kb_by_id=kb_by_id,
            ),
        )

    for ctx in issue_contexts[7:13]:
        issue_type = ctx["issue_terms"][0] if ctx["issue_terms"] else "review_theme"
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="structured_extraction",
            task_type="extract_structured",
            question=(
                f"Extract a JSON support triage object from the cited review for "
                f"{ctx['product_title']} ({ctx['parent_asin']})."
            ),
            expected_output_format="json",
            expected_status="answer",
            evidence_ids=[ctx["review_doc_id"]],
            parent_asins=[ctx["parent_asin"]],
            titles=[ctx["product_title"]],
            difficulty="medium",
            requires_citation=True,
            issue_type="catalog_metadata",
            title_resolutions=[title_resolution_from_context(ctx)],
        )
        answer_obj = {
            "product_id": ctx["parent_asin"],
            "product_title": ctx["product_title"],
            "issue_type": issue_type,
            "rating": ctx["rating"],
            "evidence_summary": f"Sanitized review title: {ctx['review_title']}",
            "recommended_action": "summarize_review_evidence",
            "evidence_id": ctx["review_doc_id"],
        }
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=json.dumps(answer_obj, sort_keys=True),
                must_include=[
                    "product_id",
                    "product_title",
                    "issue_type",
                    "rating",
                    "evidence_summary",
                    "recommended_action",
                    "evidence_id",
                ],
                must_not_include=["customer identifiers", "unsupported refund approval"],
                kb_by_id=kb_by_id,
            ),
        )

    policy_sequence = [
        ("retail_policy_return_refund_triage", "return_refund"),
        ("retail_policy_damaged_item_handling", "return_refund"),
        ("retail_policy_quality_complaint_handling", "quality_complaint"),
        ("retail_policy_escalation_rules", "return_refund"),
        ("retail_policy_missing_item_handling", "shipping"),
    ]
    for ctx, (policy_id, issue_type) in zip(issue_contexts[13:18], policy_sequence, strict=False):
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="support_policy_reasoning",
            task_type="policy_reasoning",
            question=(
                f"Using the synthetic support policy and cited review, what support action is "
                f"appropriate for {ctx['product_title']} ({ctx['parent_asin']})?"
            ),
            expected_output_format="text",
            expected_status="answer",
            evidence_ids=[policy_id, ctx["review_doc_id"]],
            parent_asins=[ctx["parent_asin"]],
            titles=[ctx["product_title"]],
            difficulty="medium",
            requires_citation=True,
            issue_type=issue_type,
            title_resolutions=[title_resolution_from_context(ctx)],
        )
        answer = (
            f"Under the synthetic benchmark policy, the assistant should summarize the cited "
            f"review issue for {ctx['product_title']} ({ctx['parent_asin']}) and route it for "
            "support review rather than promising a final refund or replacement. The policy is "
            "benchmark guidance, not Amazon policy."
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=must_include_for_context(ctx, ["synthetic benchmark policy"]),
                must_not_include=[
                    "Amazon policy claim",
                    "guaranteed refund",
                    "customer identifiers",
                ],
                kb_by_id=kb_by_id,
            ),
        )

    for ctx in contexts[18:22]:
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="evidence_citation_lookup",
            task_type="answer_grounded",
            question=(
                f"Which cited evidence record supports the review claim for "
                f"{ctx['product_title']} ({ctx['parent_asin']})?"
            ),
            expected_output_format="text",
            expected_status="answer",
            evidence_ids=[ctx["review_doc_id"]],
            parent_asins=[ctx["parent_asin"]],
            titles=[ctx["product_title"]],
            difficulty="easy",
            requires_citation=True,
            issue_type="product_question",
            title_resolutions=[title_resolution_from_context(ctx)],
        )
        answer = (
            f"The supporting evidence is {ctx['review_doc_id']} for {ctx['product_title']} "
            f"({ctx['parent_asin']}). It contains the sanitized review title, rating "
            f"{ctx['rating']}, and the cited review text."
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=[ctx["review_doc_id"], ctx["parent_asin"], "rating"],
                must_not_include=["uncited review", "customer identifiers"],
                kb_by_id=kb_by_id,
            ),
        )

    spam_statuses = ["spam_or_low_quality", "spam_or_low_quality", "escalate"]
    for ctx, status in zip(low_quality[:3], spam_statuses, strict=False):
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="spam_or_low_quality_review",
            task_type="quality_boundary",
            question=(
                f"Should the cited review for {ctx['product_title']} ({ctx['parent_asin']}) "
                "be used as strong product evidence?"
            ),
            expected_output_format="text",
            expected_status=status,
            evidence_ids=["retail_policy_low_quality_review_handling", ctx["review_doc_id"]],
            parent_asins=[ctx["parent_asin"]],
            titles=[ctx["product_title"]],
            difficulty="medium",
            requires_citation=True,
            issue_type="suspicious_review",
            title_resolutions=[title_resolution_from_context(ctx)],
        )
        answer = (
            "The cited review is too short or unclear to use as strong product evidence. "
            "Apply the synthetic low-quality review policy: flag it as low-quality or "
            "escalate if moderation judgment is required, and do not generalize from it."
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=["low-quality", ctx["parent_asin"], "do not generalize"],
                must_not_include=["strong product evidence", "unsupported product conclusion"],
                kb_by_id=kb_by_id,
                expected_escalation=status == "escalate",
            ),
        )

    insufficient_specs = [
        (
            "insufficient_evidence",
            "Can you confirm this product is eligible for a refund without order details?",
            (
                "The question is in scope, but the cited benchmark evidence does not include "
                "order details or eligibility data. The answer should ask for more information "
                "or mark insufficient_evidence rather than guessing."
            ),
        ),
        (
            "insufficient_evidence",
            "Can you verify whether this product is safe for a specific medical condition?",
            (
                "The selected Retail corpus does not include medical suitability evidence. "
                "A grounded support answer should state that the available product and review "
                "evidence is insufficient."
            ),
        ),
        (
            "escalate",
            "Can you decide a final replacement outcome from the review evidence alone?",
            (
                "The review evidence alone is not enough to decide a final replacement outcome. "
                "The assistant should escalate for order-specific support review and avoid "
                "guessing."
            ),
        ),
    ]
    for status, question, answer in insufficient_specs:
        policy_id = "retail_policy_escalation_rules"
        prompt = make_prompt(
            prompt_id=next_id(),
            prompt_category="insufficient_evidence_or_escalation",
            task_type="escalation_response",
            question=question,
            expected_output_format="text",
            expected_status=status,
            evidence_ids=[policy_id],
            parent_asins=["retail_policy_only"],
            titles=["Synthetic Retail escalation policy"],
            difficulty="hard",
            requires_citation=True,
            issue_type="return_refund",
        )
        add(
            prompt,
            make_gold(
                prompt,
                reference_answer=answer,
                must_include=["insufficient", "order", "evidence"],
                must_not_include=["guessing", "guaranteed refund", "general model memory"],
                kb_by_id=kb_by_id,
                expected_escalation=status == "escalate",
            ),
        )

    prompt = make_prompt(
        prompt_id=next_id(),
        prompt_category="out_of_scope",
        task_type="boundary_response",
        question="Who won the UEFA Champions League in 2024?",
        expected_output_format="text",
        expected_status="out_of_scope",
        evidence_ids=["retail_policy_out_of_scope_rules"],
        parent_asins=["retail_boundary"],
        titles=["Synthetic Retail out-of-scope policy"],
        difficulty="easy",
        requires_citation=False,
        issue_type="product_question",
    )
    add(
        prompt,
        make_gold(
            prompt,
            reference_answer=(
                "This question is outside the Retail support corpus. A grounded system "
                "should not answer from general model memory or fabricate retail citations."
            ),
            must_include=["outside the Retail support corpus"],
            must_not_include=["general model memory", "sports answer", "fabricated citation"],
            kb_by_id=kb_by_id,
        ),
    )

    if len(prompts) != 40 or len(gold) != 40:
        raise RuntimeError(
            f"Expected 40 prompts and gold records, got {len(prompts)} and {len(gold)}."
        )
    return prompts, gold


def collect_product_title_resolution_rows(
    prompts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    priority = {"metadata_title": 0, "metadata_partial": 1, "generic_fallback": 2}
    by_parent: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        for row in prompt.get("metadata", {}).get("title_resolution", []):
            parent_asin = str(row.get("parent_asin") or "").strip()
            if not parent_asin or parent_asin.startswith("retail_"):
                continue
            current = by_parent.get(parent_asin)
            if current is None or priority.get(str(row.get("title_resolution")), 9) < priority.get(
                str(current.get("title_resolution")), 9
            ):
                by_parent[parent_asin] = row
    return list(by_parent.values())


def build_curation_report(
    prompts: list[dict[str, Any]],
    kb_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    source_review_count: int,
    source_metadata_count: int,
) -> dict[str, Any]:
    prompt_categories = Counter(row.get("metadata", {}).get("prompt_category") for row in prompts)
    products_used = {
        parent
        for prompt in prompts
        for parent in prompt.get("source_parent_asins", [])
        if parent and not str(parent).startswith("retail_")
    }
    review_evidence_used = {
        doc_id
        for prompt in prompts
        for doc_id in prompt.get("required_evidence_ids", [])
        if str(doc_id).startswith("retail_review_")
    }
    title_resolution_rows = collect_product_title_resolution_rows(prompts)
    title_resolution_counts = Counter(
        row.get("title_resolution", "unknown") for row in title_resolution_rows
    )
    metadata_title_count = int(title_resolution_counts.get("metadata_title", 0))
    metadata_partial_title_count = int(title_resolution_counts.get("metadata_partial", 0))
    generic_product_title_count = int(title_resolution_counts.get("generic_fallback", 0))
    title_resolution_total = max(len(title_resolution_rows), 1)
    product_metadata_join_rate = round(
        (metadata_title_count + metadata_partial_title_count) / title_resolution_total,
        3,
    )
    products_with_generic_titles = [
        {
            "parent_asin": row.get("parent_asin"),
            "product_title": row.get("product_title"),
            "title_source_key": row.get("title_source_key"),
        }
        for row in title_resolution_rows
        if row.get("title_resolution") == "generic_fallback"
    ]
    warnings = [
        "This is a curated Retail seed dataset, not the full 5,000-10,000 prompt dataset.",
        (
            "RAG, retrieval, embeddings, prompt assembly, and inference are deferred "
            "until all five Phase 2A vertical datasets are prepared."
        ),
        "Generated real Amazon samples remain local and are not committed.",
        "Synthetic support policy records are benchmark policies, not Amazon policy claims.",
        "Raw customer identifiers are not included in committed records.",
    ]
    if generic_product_title_count:
        warnings.append(
            "Some curated products still use generic fallback titles; run a larger metadata "
            "sample or targeted metadata retrieval for selected parent_asins before scaling."
        )
    if product_metadata_join_rate < 0.8:
        warnings.append(
            "Product metadata join coverage is weak; improve targeted metadata coverage before "
            "Phase 2A scale-up."
        )
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "prompt_record_count": len(prompts),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "prompt_counts_by_category": dict(prompt_categories),
        "prompt_counts_by_task_type": dict(Counter(row.get("task_type") for row in prompts)),
        "prompt_counts_by_expected_status": dict(
            Counter(row.get("expected_status") for row in prompts)
        ),
        "prompt_counts_by_expected_output_format": dict(
            Counter(row.get("expected_output_format") for row in prompts)
        ),
        "kb_counts_by_document_type": dict(Counter(row.get("document_type") for row in kb_records)),
        "gold_counts_by_expected_status": dict(
            Counter(row.get("expected_status") for row in gold_records)
        ),
        "source_review_count": source_review_count,
        "source_metadata_count": source_metadata_count,
        "source_products_used_count": len(products_used),
        "review_evidence_used_count": len(review_evidence_used),
        "product_title_resolution_counts": dict(title_resolution_counts),
        "generic_product_title_count": generic_product_title_count,
        "metadata_title_count": metadata_title_count,
        "metadata_partial_title_count": metadata_partial_title_count,
        "products_with_generic_titles": products_with_generic_titles,
        "product_metadata_join_rate": product_metadata_join_rate,
        "policy_record_count": sum(
            1 for row in kb_records if row.get("document_type") == "support_policy"
        ),
        "warnings": warnings,
        "next_step": (
            "Proceed to Phase 2A-7 cross-vertical data QA and scale-up planning after "
            "reviewing Retail curated samples."
        ),
    }


def assert_public_hygiene(paths: list[Path]) -> None:
    forbidden_terms = [
        "raw user_id",
        "C:\\Users",
        "/home/",
        "akpoogaga",
        "kparo",
        "token",
        "API key",
    ]
    for path in paths:
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        for term in forbidden_terms:
            if term.lower() in lowered:
                raise RuntimeError(f"Public hygiene check failed for {path}: found {term!r}")
        if EMAIL_RE.search(content) or PHONE_RE.search(content):
            raise RuntimeError(f"Public hygiene check failed for {path}: possible PII-like text.")


def build_curated_samples(args: argparse.Namespace) -> dict[str, Any]:
    validate_inputs(args)
    reviews = read_jsonl(Path(args.reviews_input))
    metadata_rows = read_jsonl(Path(args.metadata_input))
    _exploration_report = read_json(Path(args.exploration_report))
    _quality_report = read_json(Path(args.quality_report))
    validate_reviews(reviews)

    metadata_by_parent = build_metadata_index(metadata_rows)
    review_candidates = build_review_candidates(
        reviews,
        metadata_by_parent,
        int(args.max_review_body_chars),
    )
    metadata_candidates = build_metadata_candidates(metadata_rows)
    if len(review_candidates) < 40 or len(metadata_candidates) < 10:
        raise RuntimeError("Not enough sanitized Retail evidence to build the curated seed.")

    issue_reviews = sorted(
        [row for row in review_candidates if row["issue_terms"] and not row["low_quality"]],
        key=review_selection_sort_key,
    )
    non_issue_safe_reviews = sorted(
        [row for row in review_candidates if not row["low_quality"] and not row["issue_terms"]],
        key=review_selection_sort_key,
    )
    low_quality_reviews = sorted(
        [row for row in review_candidates if row["low_quality"]],
        key=review_selection_sort_key,
    )
    selected_reviews = []
    seen_reviews: set[str] = set()
    for pool in (issue_reviews[:24], non_issue_safe_reviews[:8], low_quality_reviews[:3]):
        for row in pool:
            if row["review_id"] not in seen_reviews:
                selected_reviews.append(row)
                seen_reviews.add(row["review_id"])
    kb_records, kb_by_id = build_kb_records(selected_reviews, metadata_candidates)
    if len(kb_records) < 40:
        raise RuntimeError(f"Expected at least 40 KB records, built {len(kb_records)}.")

    prompts, gold_records = build_prompt_and_gold_records(selected_reviews, kb_records, kb_by_id)
    report = build_curation_report(
        prompts,
        kb_records,
        gold_records,
        len(reviews),
        len(metadata_rows),
    )

    write_jsonl(Path(args.output_prompts), prompts)
    write_jsonl(Path(args.output_kb), kb_records)
    write_jsonl(Path(args.output_gold), gold_records)
    write_json(Path(args.curation_report), report)
    assert_public_hygiene([Path(args.output_prompts), Path(args.output_kb), Path(args.output_gold)])
    return {
        "mode": "build_curated_samples",
        "phase": PHASE,
        "prompt_record_count": len(prompts),
        "kb_record_count": len(kb_records),
        "gold_record_count": len(gold_records),
        "prompt_counts_by_category": report["prompt_counts_by_category"],
        "prompt_counts_by_expected_status": report["prompt_counts_by_expected_status"],
        "product_title_resolution_counts": report["product_title_resolution_counts"],
        "generic_product_title_count": report["generic_product_title_count"],
        "product_metadata_join_rate": report["product_metadata_join_rate"],
        "output_prompts": str(args.output_prompts),
        "output_kb": str(args.output_kb),
        "output_gold": str(args.output_gold),
        "curation_report": str(args.curation_report),
        "warnings": report["warnings"],
        "next_step": report["next_step"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-curated-samples", action="store_true")
    parser.add_argument("--reviews-input", type=Path, default=DEFAULT_REVIEWS_INPUT)
    parser.add_argument("--metadata-input", type=Path, default=DEFAULT_METADATA_INPUT)
    parser.add_argument("--exploration-report", type=Path, default=DEFAULT_EXPLORATION_REPORT)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--output-prompts", type=Path, default=DEFAULT_OUTPUT_PROMPTS)
    parser.add_argument("--output-kb", type=Path, default=DEFAULT_OUTPUT_KB)
    parser.add_argument("--output-gold", type=Path, default=DEFAULT_OUTPUT_GOLD)
    parser.add_argument("--curation-report", type=Path, default=DEFAULT_CURATION_REPORT)
    parser.add_argument("--max-review-body-chars", type=int, default=1200)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.build_curated_samples:
        parser.error("Pass --build-curated-samples to create Retail Phase 2A-6C seed records.")
    try:
        summary = build_curated_samples(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
