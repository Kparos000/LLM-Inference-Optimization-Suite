"""Explore controlled samples from Amazon Reviews 2023 for Phase 2A-6B."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

PHASE = "2A-6B"
SOURCE_NAME = "Amazon Reviews 2023"
SOURCE_OWNER = "McAuley Lab"
SOURCE_LOCATION = "Hugging Face dataset McAuley-Lab/Amazon-Reviews-2023"
HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

DEFAULT_REVIEWS_INPUT = Path("data/generated/retail/amazon_reviews_sample.jsonl")
DEFAULT_METADATA_INPUT = Path("data/generated/retail/amazon_metadata_sample.jsonl")
DEFAULT_OUTPUT_REPORT = Path("data/generated/retail/amazon_reviews_exploration_report.json")
DEFAULT_FIELD_PROFILE_OUTPUT = Path("data/generated/retail/amazon_reviews_field_profile.json")
DEFAULT_TEXT_PROFILE_OUTPUT = Path("data/generated/retail/amazon_reviews_text_profile.json")
DEFAULT_QUALITY_REPORT_OUTPUT = Path("data/generated/retail/amazon_reviews_quality_report.json")
DEFAULT_PLOTS_DIR = Path("data/generated/retail/plots")
DEFAULT_WORD_VIEWS_DIR = Path("data/generated/retail/word_views")
DEFAULT_OUTPUT_REVIEWS_SAMPLE = Path("data/generated/retail/amazon_reviews_sample.jsonl")
DEFAULT_OUTPUT_METADATA_SAMPLE = Path("data/generated/retail/amazon_metadata_sample.jsonl")
REVIEW_SCHEMA_SAMPLE_PATH = Path(
    "data/real_world_samples/retail_amazon_reviews_schema_sample.jsonl"
)
METADATA_SCHEMA_SAMPLE_PATH = Path(
    "data/real_world_samples/retail_amazon_metadata_schema_sample.jsonl"
)

EXPECTED_REVIEW_FIELDS = [
    "rating",
    "title",
    "text",
    "images",
    "asin",
    "parent_asin",
    "user_id",
    "timestamp",
    "verified_purchase",
    "helpful_vote",
]

EXPECTED_METADATA_FIELDS = [
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "features",
    "description",
    "price",
    "images",
    "videos",
    "store",
    "categories",
    "details",
    "parent_asin",
    "bought_together",
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

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "so",
    "the",
    "this",
    "to",
    "too",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d[\s().-]*){7,}\b")
HTML_RE = re.compile(r"<[^>]+>")
WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s]+")
UNIX_HOME_PATH_RE = re.compile(r"/home/[^\s]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def read_jsonl_limited(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit > 0 and len(rows) >= limit:
                break
            if not line.strip():
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def build_hf_dataset_names(category: str) -> dict[str, str]:
    normalized_category = category.strip() or "All_Beauty"
    return {
        "dataset_name": HF_DATASET_NAME,
        "reviews_config": f"raw_review_{normalized_category}",
        "metadata_config": f"raw_meta_{normalized_category}",
        "split": "full",
    }


def _hash_user_id(user_id: Any) -> str | None:
    if user_id is None or user_id == "":
        return None
    return hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()[:16]


def sanitize_review_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {
        field: row.get(field)
        for field in EXPECTED_REVIEW_FIELDS
        if field != "user_id" and field in row
    }
    user_id_hash = _hash_user_id(row.get("user_id"))
    if user_id_hash:
        sanitized["user_id_hash"] = user_id_hash
    sanitized["source_type"] = "real_sample"
    sanitized["sanitized"] = True
    return sanitized


def sanitize_metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = {field: row.get(field) for field in EXPECTED_METADATA_FIELDS if field in row}
    sanitized["source_type"] = "real_sample"
    sanitized["sanitized"] = True
    return sanitized


def _import_hf_load_dataset() -> Any:
    try:
        datasets_module = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError(
            "Install datasets to use --load-from-huggingface, or provide local JSONL files "
            "to --explore-local."
        ) from exc
    load_dataset = getattr(datasets_module, "load_dataset", None)
    if load_dataset is None:
        raise RuntimeError(
            "Install datasets to use --load-from-huggingface, or provide local JSONL files "
            "to --explore-local."
        )
    return load_dataset


def _load_hf_rows(
    *,
    load_dataset_func: Any,
    config_name: str,
    limit: int,
    streaming: bool,
) -> list[dict[str, Any]]:
    if limit < 1:
        return []
    try:
        dataset = load_dataset_func(
            HF_DATASET_NAME,
            config_name,
            split="full",
            streaming=streaming,
        )
    except Exception as first_error:
        try:
            dataset = load_dataset_func(
                HF_DATASET_NAME,
                config_name,
                split="train",
                streaming=streaming,
            )
        except Exception as second_error:
            raise RuntimeError(
                f"Failed to load Hugging Face config {config_name!r}. Inspect available "
                "Amazon Reviews 2023 config names and rerun with --reviews-config or "
                f"--metadata-config. Original errors: {first_error}; {second_error}"
            ) from second_error

    rows: list[dict[str, Any]] = []
    for row in dataset:
        if isinstance(row, dict):
            rows.append(dict(row))
        else:
            rows.append(dict(row))
        if len(rows) >= limit:
            break
    return rows


def load_hf_review_sample(
    *,
    load_dataset_func: Any,
    category: str,
    sample_limit: int,
    reviews_config: str | None,
    streaming: bool,
    seed: int,
) -> list[dict[str, Any]]:
    _ = seed
    names = build_hf_dataset_names(category)
    config_name = reviews_config or names["reviews_config"]
    rows = _load_hf_rows(
        load_dataset_func=load_dataset_func,
        config_name=config_name,
        limit=sample_limit,
        streaming=streaming,
    )
    return [sanitize_review_row(row) for row in rows]


def load_hf_metadata_sample(
    *,
    load_dataset_func: Any,
    category: str,
    metadata_limit: int,
    metadata_config: str | None,
    streaming: bool,
    seed: int,
) -> list[dict[str, Any]]:
    _ = seed
    names = build_hf_dataset_names(category)
    config_name = metadata_config or names["metadata_config"]
    rows = _load_hf_rows(
        load_dataset_func=load_dataset_func,
        config_name=config_name,
        limit=metadata_limit,
        streaming=streaming,
    )
    return [sanitize_metadata_row(row) for row in rows]


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def _jsonable_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): count for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(mean(values), 3),
        "median": round(median(values), 3),
        "max": max(values),
    }


def _field_example(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 160:
        return value[:157] + "..."
    if isinstance(value, list) and len(value) > 3:
        return value[:3]
    if isinstance(value, dict) and len(value) > 5:
        return {key: value[key] for key in list(value)[:5]}
    return value


def profile_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields_seen = sorted({field for row in rows for field in row})
    missing_counts: dict[str, int] = {}
    type_counts_by_field: dict[str, dict[str, int]] = {}
    example_values: dict[str, Any] = {}

    for field in fields_seen:
        missing_counts[field] = sum(1 for row in rows if _is_missing(row.get(field)))
        type_counter: Counter[str] = Counter(
            _type_name(row.get(field)) for row in rows if field in row
        )
        type_counts_by_field[field] = dict(sorted(type_counter.items()))
        for row in rows:
            value = row.get(field)
            if not _is_missing(value):
                example_values[field] = _field_example(value)
                break

    selected_unique_fields = [
        "asin",
        "parent_asin",
        "rating",
        "verified_purchase",
        "main_category",
    ]
    unique_counts: dict[str, int] = {}
    for field in selected_unique_fields:
        if field in fields_seen:
            unique_counts[field] = len(
                {
                    json.dumps(row.get(field), sort_keys=True)
                    for row in rows
                    if not _is_missing(row.get(field))
                }
            )

    return {
        "row_count": len(rows),
        "fields_seen": fields_seen,
        "missing_counts": missing_counts,
        "type_counts_by_field": type_counts_by_field,
        "example_values": example_values,
        "unique_counts": unique_counts,
    }


def _text_value(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    return value if isinstance(value, str) else ""


def _tokens(text: str) -> list[str]:
    lowered = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [
        token for token in TOKEN_RE.findall(lowered) if len(token) > 2 and token not in STOPWORDS
    ]


def _top_items(counter: Counter[str], limit: int = 25) -> list[dict[str, int | str]]:
    return [{"term": term, "count": count} for term, count in counter.most_common(limit)]


def _coerce_float(value: Any) -> float | None:
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


def _word_count(text: str) -> int:
    return len(_tokens(text))


def profile_text_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    text_values = [_text_value(row, "text") for row in rows]
    title_values = [_text_value(row, "title") for row in rows]
    text_lengths_chars = [len(text) for text in text_values if text.strip()]
    text_lengths_words = [_word_count(text) for text in text_values if text.strip()]
    title_lengths_words = [_word_count(title) for title in title_values if title.strip()]

    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    title_terms: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    for row in rows:
        tokens = _tokens(_text_value(row, "text"))
        unigrams.update(tokens)
        bigrams.update(f"{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False))
        title_terms.update(_tokens(_text_value(row, "title")))
        token_set = set(tokens)
        for term in ISSUE_TERMS:
            if term in token_set:
                issue_counts[term] += 1

    rating_distribution: Counter[str] = Counter()
    verified_distribution: Counter[str] = Counter()
    helpful_votes: list[float] = []
    ratings_by_verified: dict[str, list[float]] = {}
    category_counts: Counter[str] = Counter()
    low_rating_count = 0
    high_rating_count = 0
    for row in rows:
        rating = row.get("rating")
        if not _is_missing(rating):
            rating_distribution[str(rating)] += 1
        rating_value = _coerce_float(rating)
        if rating_value is not None:
            if rating_value <= 2:
                low_rating_count += 1
            if rating_value >= 4:
                high_rating_count += 1
        verified = row.get("verified_purchase")
        if not _is_missing(verified):
            verified_distribution[str(verified)] += 1
            if rating_value is not None:
                ratings_by_verified.setdefault(str(verified), []).append(rating_value)
        helpful_vote = _coerce_float(row.get("helpful_vote"))
        if helpful_vote is not None:
            helpful_votes.append(helpful_vote)
        category = row.get("main_category") or row.get("category")
        if not _is_missing(category):
            category_counts[str(category)] += 1

    return {
        "row_count": len(rows),
        "text_field_present_count": sum(1 for text in text_values if text.strip()),
        "title_field_present_count": sum(1 for title in title_values if title.strip()),
        "empty_text_count": sum(1 for text in text_values if not text.strip()),
        "empty_title_count": sum(1 for title in title_values if not title.strip()),
        "text_length_chars_min": _numeric_summary([float(value) for value in text_lengths_chars])[
            "min"
        ],
        "text_length_chars_mean": _numeric_summary([float(value) for value in text_lengths_chars])[
            "mean"
        ],
        "text_length_chars_median": _numeric_summary(
            [float(value) for value in text_lengths_chars]
        )["median"],
        "text_length_chars_max": _numeric_summary([float(value) for value in text_lengths_chars])[
            "max"
        ],
        "text_length_words_min": _numeric_summary([float(value) for value in text_lengths_words])[
            "min"
        ],
        "text_length_words_mean": _numeric_summary([float(value) for value in text_lengths_words])[
            "mean"
        ],
        "text_length_words_median": _numeric_summary(
            [float(value) for value in text_lengths_words]
        )["median"],
        "text_length_words_max": _numeric_summary([float(value) for value in text_lengths_words])[
            "max"
        ],
        "title_length_words_min": _numeric_summary([float(value) for value in title_lengths_words])[
            "min"
        ],
        "title_length_words_mean": _numeric_summary(
            [float(value) for value in title_lengths_words]
        )["mean"],
        "title_length_words_median": _numeric_summary(
            [float(value) for value in title_lengths_words]
        )["median"],
        "title_length_words_max": _numeric_summary([float(value) for value in title_lengths_words])[
            "max"
        ],
        "rating_distribution": dict(sorted(rating_distribution.items())),
        "verified_purchase_distribution": dict(sorted(verified_distribution.items())),
        "helpful_vote_summary": _numeric_summary(helpful_votes),
        "top_unigrams": _top_items(unigrams),
        "top_bigrams": _top_items(bigrams),
        "top_review_title_terms": _top_items(title_terms),
        "frequent_issue_terms": _top_items(issue_counts, limit=len(ISSUE_TERMS)),
        "average_rating_by_verified_purchase": {
            bucket: round(mean(values), 3) for bucket, values in sorted(ratings_by_verified.items())
        },
        "low_rating_count": low_rating_count,
        "high_rating_count": high_rating_count,
        "issue_term_counts": dict(sorted(issue_counts.items())),
        "category": dict(sorted(category_counts.items())),
    }


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for char in text if ord(char) > 127)
    return non_ascii / len(text)


def _looks_low_quality(text: str) -> bool:
    tokens = _tokens(text)
    if not tokens:
        return True
    if len(tokens) <= 3:
        return True
    if len(tokens) >= 8:
        most_common_count = Counter(tokens).most_common(1)[0][1]
        if most_common_count / len(tokens) > 0.6:
            return True
    if text.count("!") >= 6 or text.count("?") >= 6:
        return True
    return False


def profile_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    duplicate_key_counter: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        asin = row.get("asin")
        user_id = row.get("user_id_hash") or row.get("user_id")
        timestamp = row.get("timestamp")
        if not _is_missing(asin) and not _is_missing(user_id) and not _is_missing(timestamp):
            duplicate_key_counter[(str(asin), str(user_id), str(timestamp))] += 1
    duplicate_review_key_count = sum(
        count - 1 for count in duplicate_key_counter.values() if count > 1
    )

    invalid_rating_count = 0
    non_ascii_ratios: list[float] = []
    pii_like_pattern_count = 0
    html_like_text_count = 0
    very_short_review_count = 0
    very_long_review_count = 0
    possible_spam_or_low_quality_count = 0
    for row in rows:
        rating = _coerce_float(row.get("rating"))
        if rating is None or rating < 1 or rating > 5:
            invalid_rating_count += 1
        text = _text_value(row, "text")
        word_count = _word_count(text)
        if word_count < 5:
            very_short_review_count += 1
        if word_count > 1000 or len(text) > 8000:
            very_long_review_count += 1
        if _looks_low_quality(text):
            possible_spam_or_low_quality_count += 1
        if EMAIL_RE.search(text) or PHONE_RE.search(text):
            pii_like_pattern_count += 1
        if HTML_RE.search(text):
            html_like_text_count += 1
        non_ascii_ratios.append(_non_ascii_ratio(text))

    return {
        "row_count": len(rows),
        "duplicate_review_key_count": duplicate_review_key_count,
        "raw_user_id_present_count": sum(1 for row in rows if not _is_missing(row.get("user_id"))),
        "sanitized_user_id_hash_present_count": sum(
            1 for row in rows if not _is_missing(row.get("user_id_hash"))
        ),
        "missing_text_count": sum(1 for row in rows if not _text_value(row, "text").strip()),
        "missing_title_count": sum(1 for row in rows if not _text_value(row, "title").strip()),
        "missing_parent_asin_count": sum(1 for row in rows if _is_missing(row.get("parent_asin"))),
        "missing_rating_count": sum(1 for row in rows if _is_missing(row.get("rating"))),
        "invalid_rating_count": invalid_rating_count,
        "very_short_review_count": very_short_review_count,
        "very_long_review_count": very_long_review_count,
        "possible_spam_or_low_quality_count": possible_spam_or_low_quality_count,
        "pii_like_pattern_count": pii_like_pattern_count,
        "html_like_text_count": html_like_text_count,
        "non_ascii_ratio_summary": _numeric_summary(non_ascii_ratios),
    }


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _redact_preview_text(text: str, limit: int = 300) -> str:
    sanitized = EMAIL_RE.sub("[redacted_email]", text)
    sanitized = PHONE_RE.sub("[redacted_number]", sanitized)
    sanitized = WINDOWS_PATH_RE.sub("[redacted_path]", sanitized)
    sanitized = UNIX_HOME_PATH_RE.sub("[redacted_path]", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if len(sanitized) > limit:
        return sanitized[: limit - 3] + "..."
    return sanitized


def write_word_views(
    rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    text_profile: dict[str, Any],
    word_views_dir: Path,
) -> None:
    word_views_dir.mkdir(parents=True, exist_ok=True)

    def term_lines(key: str) -> list[str]:
        terms = text_profile.get(key, [])
        if not isinstance(terms, list):
            return []
        return [
            f"{item.get('term')},{item.get('count')}" for item in terms if isinstance(item, dict)
        ]

    _write_text_lines(word_views_dir / "top_unigrams.txt", term_lines("top_unigrams"))
    _write_text_lines(word_views_dir / "top_bigrams.txt", term_lines("top_bigrams"))
    _write_text_lines(word_views_dir / "issue_terms.txt", term_lines("frequent_issue_terms"))

    rows_with_text = [row for row in rows if _text_value(row, "text").strip()]
    longest = sorted(rows_with_text, key=lambda row: len(_text_value(row, "text")), reverse=True)[
        :5
    ]
    shortest = sorted(rows_with_text, key=lambda row: len(_text_value(row, "text")))[:5]
    low_rating = [
        row
        for row in rows_with_text
        if (
            _coerce_float(row.get("rating")) is not None
            and (_coerce_float(row.get("rating")) or 0) <= 2
        )
    ][:5]
    high_rating = [
        row
        for row in rows_with_text
        if (
            _coerce_float(row.get("rating")) is not None
            and (_coerce_float(row.get("rating")) or 0) >= 4
        )
    ][:5]

    def preview_lines(preview_rows: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for row in preview_rows:
            title = _redact_preview_text(_text_value(row, "title"), limit=100)
            text = _redact_preview_text(_text_value(row, "text"))
            lines.append(
                json.dumps(
                    {
                        "asin": row.get("asin"),
                        "parent_asin": row.get("parent_asin"),
                        "rating": row.get("rating"),
                        "verified_purchase": row.get("verified_purchase"),
                        "helpful_vote": row.get("helpful_vote"),
                        "title_preview": title,
                        "text_preview": text,
                    },
                    sort_keys=True,
                )
            )
        return lines

    _write_text_lines(word_views_dir / "longest_reviews_preview.txt", preview_lines(longest))
    _write_text_lines(word_views_dir / "shortest_reviews_preview.txt", preview_lines(shortest))
    _write_text_lines(word_views_dir / "low_rating_issue_preview.txt", preview_lines(low_rating))
    _write_text_lines(
        word_views_dir / "high_rating_positive_preview.txt", preview_lines(high_rating)
    )

    metadata_title_lines: list[str] = []
    metadata_feature_lines: list[str] = []
    for row in metadata_rows[:10]:
        metadata_title_lines.append(
            json.dumps(
                {
                    "parent_asin": row.get("parent_asin"),
                    "main_category": row.get("main_category"),
                    "average_rating": row.get("average_rating"),
                    "rating_number": row.get("rating_number"),
                    "title_preview": _redact_preview_text(_text_value(row, "title"), limit=140),
                },
                sort_keys=True,
            )
        )
        features = row.get("features")
        if isinstance(features, list):
            feature_preview = "; ".join(
                _redact_preview_text(str(item), 120) for item in features[:5]
            )
        else:
            feature_preview = _redact_preview_text(str(features or ""), 300)
        metadata_feature_lines.append(
            json.dumps(
                {
                    "parent_asin": row.get("parent_asin"),
                    "title_preview": _redact_preview_text(_text_value(row, "title"), limit=100),
                    "features_preview": feature_preview,
                },
                sort_keys=True,
            )
        )
    _write_text_lines(
        word_views_dir / "metadata_product_titles_preview.txt",
        metadata_title_lines,
    )
    _write_text_lines(
        word_views_dir / "metadata_features_preview.txt",
        metadata_feature_lines,
    )


def write_plots_or_summaries(
    rows: list[dict[str, Any]],
    text_profile: dict[str, Any],
    plots_dir: Path,
) -> dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rating_distribution = text_profile.get("rating_distribution", {})
    verified_distribution = text_profile.get("verified_purchase_distribution", {})
    top_terms = text_profile.get("top_unigrams", [])
    issue_terms = text_profile.get("frequent_issue_terms", [])
    helpful_votes = [
        value for row in rows if (value := _coerce_float(row.get("helpful_vote"))) is not None
    ]
    text_lengths = [
        len(_text_value(row, "text")) for row in rows if _text_value(row, "text").strip()
    ]

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        _write_text_lines(
            plots_dir / "rating_distribution.txt",
            [f"{key},{value}" for key, value in dict(rating_distribution).items()],
        )
        _write_text_lines(
            plots_dir / "verified_purchase_distribution.txt",
            [f"{key},{value}" for key, value in dict(verified_distribution).items()],
        )
        _write_text_lines(
            plots_dir / "helpful_vote_distribution.txt",
            [str(value) for value in helpful_votes[:100]],
        )
        _write_text_lines(
            plots_dir / "review_text_length_histogram.txt",
            [str(value) for value in text_lengths[:100]],
        )
        _write_text_lines(
            plots_dir / "top_terms_bar.txt",
            [
                f"{item.get('term')},{item.get('count')}"
                for item in top_terms[:20]
                if isinstance(item, dict)
            ],
        )
        _write_text_lines(
            plots_dir / "issue_terms_bar.txt",
            [
                f"{item.get('term')},{item.get('count')}"
                for item in issue_terms[:20]
                if isinstance(item, dict)
            ],
        )
        return {"plot_status": "fallback_text_summaries", "plots_dir": str(plots_dir)}

    def save_bar(path: Path, title: str, labels: list[str], values: list[int]) -> None:
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    save_bar(
        plots_dir / "rating_distribution.png",
        "Rating Distribution",
        list(dict(rating_distribution).keys()),
        [int(value) for value in dict(rating_distribution).values()],
    )
    save_bar(
        plots_dir / "verified_purchase_distribution.png",
        "Verified Purchase Distribution",
        list(dict(verified_distribution).keys()),
        [int(value) for value in dict(verified_distribution).values()],
    )
    save_bar(
        plots_dir / "top_terms_bar.png",
        "Top Review Terms",
        [str(item.get("term")) for item in top_terms[:15] if isinstance(item, dict)],
        [int(item.get("count", 0)) for item in top_terms[:15] if isinstance(item, dict)],
    )
    save_bar(
        plots_dir / "issue_terms_bar.png",
        "Issue Terms",
        [str(item.get("term")) for item in issue_terms[:15] if isinstance(item, dict)],
        [int(item.get("count", 0)) for item in issue_terms[:15] if isinstance(item, dict)],
    )

    plt.figure(figsize=(8, 4))
    plt.hist(text_lengths, bins=min(20, max(1, len(text_lengths))))
    plt.title("Review Text Lengths")
    plt.tight_layout()
    plt.savefig(plots_dir / "review_text_length_histogram.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(helpful_votes, bins=min(20, max(1, len(helpful_votes))))
    plt.title("Helpful Vote Distribution")
    plt.tight_layout()
    plt.savefig(plots_dir / "helpful_vote_distribution.png")
    plt.close()

    return {"plot_status": "matplotlib_png", "plots_dir": str(plots_dir)}


def build_source_plan() -> dict[str, Any]:
    return {
        "source_name": SOURCE_NAME,
        "source_owner": SOURCE_OWNER,
        "source_location": SOURCE_LOCATION,
        "vertical": "retail",
        "phase": PHASE,
        "purpose": (
            "Explore Amazon review and product metadata schemas with controlled local "
            "samples before creating Retail support seed prompts, KB records, and gold records."
        ),
        "do_not_download_full_dataset": True,
        "controlled_sampling_required": True,
        "recommended_categories": [
            "All_Beauty",
            "Home_and_Kitchen",
            "Electronics",
            "Clothing_Shoes_and_Jewelry",
            "Sports_and_Outdoors",
            "Toys_and_Games",
        ],
        "initial_category_priority": [
            "All_Beauty",
            "Home_and_Kitchen",
            "Electronics",
        ],
        "target_exploration_sample_size": 1000,
        "target_metadata_sample_size": 1000,
        "final_seed_prompt_target": 40,
        "future_scale_targets": [250, 1000, 5000, 10000],
        "expected_review_fields": EXPECTED_REVIEW_FIELDS,
        "expected_metadata_fields": EXPECTED_METADATA_FIELDS,
        "retail_support_use_cases": [
            "product QA from reviews",
            "review summarization",
            "product comparison",
            "return/refund policy reasoning",
            "defect/quality issue identification",
            "review evidence lookup",
            "metadata extraction",
            "escalation/out-of-scope handling",
        ],
    }


def build_review_schema_samples() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "schema_example",
            "not_real_customer_data": True,
            "not_for_benchmark_claims": True,
            "rating": 5.0,
            "title": "Works well in the example scenario",
            "text": (
                "This fake review shows the expected schema shape for controlled "
                "Retail exploration. It is not real customer data."
            ),
            "images": [],
            "asin": "B000EXAMPLE1",
            "parent_asin": "B000PARENT1",
            "user_id": "schema_user_001",
            "timestamp": 1704067200000,
            "verified_purchase": True,
            "helpful_vote": 2,
        },
        {
            "source_type": "schema_example",
            "not_real_customer_data": True,
            "not_for_benchmark_claims": True,
            "rating": 2.0,
            "title": "Fake example with a quality issue",
            "text": (
                "This synthetic schema example mentions a damaged package and return "
                "workflow only to exercise issue-term profiling."
            ),
            "images": [],
            "asin": "B000EXAMPLE2",
            "parent_asin": "B000PARENT2",
            "user_id": "schema_user_002",
            "timestamp": 1704153600000,
            "verified_purchase": False,
            "helpful_vote": 0,
        },
    ]


def build_metadata_schema_samples() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "schema_example",
            "not_real_customer_data": True,
            "not_for_benchmark_claims": True,
            "main_category": "All_Beauty",
            "title": "Example Retail Product",
            "average_rating": 4.4,
            "rating_number": 123,
            "features": ["schema-only feature", "fake example field"],
            "description": ["Synthetic product metadata used only for schema validation."],
            "price": "19.99",
            "images": [],
            "videos": [],
            "store": "Example Store",
            "categories": ["All_Beauty", "Example Category"],
            "details": {"Material": "Example material", "Package Quantity": "1"},
            "parent_asin": "B000PARENT1",
            "bought_together": [],
        }
    ]


def write_schema_samples() -> dict[str, Any]:
    review_rows = build_review_schema_samples()
    metadata_rows = build_metadata_schema_samples()
    write_jsonl(REVIEW_SCHEMA_SAMPLE_PATH, review_rows)
    write_jsonl(METADATA_SCHEMA_SAMPLE_PATH, metadata_rows)
    return {
        "mode": "write_schema_samples",
        "phase": PHASE,
        "review_schema_sample_path": str(REVIEW_SCHEMA_SAMPLE_PATH),
        "metadata_schema_sample_path": str(METADATA_SCHEMA_SAMPLE_PATH),
        "review_schema_sample_count": len(review_rows),
        "metadata_schema_sample_count": len(metadata_rows),
        "warnings": [
            "Schema samples are fake examples only.",
            "Schema samples are not real customer data and are not for benchmark claims.",
        ],
    }


def _report_warnings() -> list[str]:
    return [
        "This is exploration only.",
        "Do not use this exploration output to make benchmark claims.",
        "RAG, retrieval, embeddings, prompt assembly, and inference are deferred.",
        "The full Amazon Reviews 2023 dataset is very large; use controlled sampling.",
    ]


def build_exploration_report(
    *,
    mode: str,
    reviews_row_count: int,
    metadata_row_count: int,
    category: str,
    field_profile_path: Path,
    text_profile_path: Path,
    quality_report_path: Path,
    plots_dir: Path,
    word_views_dir: Path,
    text_profile: dict[str, Any],
    quality_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "mode": mode,
        "reviews_row_count": reviews_row_count,
        "metadata_row_count": metadata_row_count,
        "category": category,
        "field_profile_path": str(field_profile_path),
        "text_profile_path": str(text_profile_path),
        "quality_report_path": str(quality_report_path),
        "plots_dir": str(plots_dir),
        "word_views_dir": str(word_views_dir),
        "rating_distribution": text_profile.get("rating_distribution", {}),
        "top_issue_terms": text_profile.get("frequent_issue_terms", [])[:10],
        "review_length_summary": {
            "text_length_chars_min": text_profile.get("text_length_chars_min"),
            "text_length_chars_mean": text_profile.get("text_length_chars_mean"),
            "text_length_chars_median": text_profile.get("text_length_chars_median"),
            "text_length_chars_max": text_profile.get("text_length_chars_max"),
            "text_length_words_min": text_profile.get("text_length_words_min"),
            "text_length_words_mean": text_profile.get("text_length_words_mean"),
            "text_length_words_median": text_profile.get("text_length_words_median"),
            "text_length_words_max": text_profile.get("text_length_words_max"),
        },
        "quality_flags": {
            "duplicate_review_key_count": quality_report.get("duplicate_review_key_count", 0),
            "raw_user_id_present_count": quality_report.get("raw_user_id_present_count", 0),
            "sanitized_user_id_hash_present_count": quality_report.get(
                "sanitized_user_id_hash_present_count",
                0,
            ),
            "missing_text_count": quality_report.get("missing_text_count", 0),
            "missing_title_count": quality_report.get("missing_title_count", 0),
            "missing_parent_asin_count": quality_report.get("missing_parent_asin_count", 0),
            "missing_rating_count": quality_report.get("missing_rating_count", 0),
            "invalid_rating_count": quality_report.get("invalid_rating_count", 0),
            "very_short_review_count": quality_report.get("very_short_review_count", 0),
            "very_long_review_count": quality_report.get("very_long_review_count", 0),
            "possible_spam_or_low_quality_count": quality_report.get(
                "possible_spam_or_low_quality_count",
                0,
            ),
            "pii_like_pattern_count": quality_report.get("pii_like_pattern_count", 0),
            "html_like_text_count": quality_report.get("html_like_text_count", 0),
        },
        "recommended_seed_strategy": [
            "Use product metadata plus review-derived summaries as KB/context.",
            "Create 40 Retail seed prompts only after reviewing EDA quality outputs.",
            (
                "Include answer, insufficient_evidence, escalate, out_of_scope, "
                "and spam_or_fraud behaviors."
            ),
            "Avoid raw user IDs in committed records.",
            "Avoid committing bulk raw reviews.",
        ],
        "warnings": _report_warnings(),
        "next_step": "Proceed to Phase 2A-6C Retail curated seed creation after reviewing EDA.",
    }


def run_local_exploration(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    reviews_input = Path(args.reviews_input)
    metadata_input = Path(args.metadata_input)
    if not reviews_input.exists():
        raise FileNotFoundError(
            f"Reviews sample not found: {reviews_input}. Provide --reviews-input "
            "or create a controlled sample."
        )
    if not metadata_input.exists():
        raise FileNotFoundError(
            f"Metadata sample not found: {metadata_input}. Provide --metadata-input "
            "or create a controlled sample."
        )

    reviews = read_jsonl_limited(reviews_input, int(args.sample_limit))
    metadata = read_jsonl_limited(metadata_input, int(args.sample_limit))
    field_profile = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "reviews": profile_fields(reviews),
        "metadata": profile_fields(metadata),
    }
    text_profile = profile_text_fields(reviews)
    quality_report = profile_quality(reviews)

    field_profile_path = Path(args.field_profile_output)
    text_profile_path = Path(args.text_profile_output)
    quality_report_path = Path(args.quality_report_output)
    plots_dir = Path(args.plots_dir)
    word_views_dir = Path(args.word_views_dir)
    report_path = Path(args.output_report)

    write_json(field_profile_path, field_profile)
    write_json(text_profile_path, text_profile)
    write_json(quality_report_path, quality_report)
    plot_status = write_plots_or_summaries(reviews, text_profile, plots_dir)
    write_word_views(reviews, metadata, text_profile, word_views_dir)
    report = build_exploration_report(
        mode=mode,
        reviews_row_count=len(reviews),
        metadata_row_count=len(metadata),
        category=str(args.category),
        field_profile_path=field_profile_path,
        text_profile_path=text_profile_path,
        quality_report_path=quality_report_path,
        plots_dir=plots_dir,
        word_views_dir=word_views_dir,
        text_profile=text_profile,
        quality_report=quality_report,
    )
    report["plot_status"] = plot_status
    if plot_status.get("plot_status") == "fallback_text_summaries":
        report["warnings"].append(
            "matplotlib is unavailable; wrote text plot summaries instead of PNG plots."
        )
    write_json(report_path, report)

    return {
        "mode": mode,
        "phase": PHASE,
        "reviews_row_count": len(reviews),
        "metadata_row_count": len(metadata),
        "category": str(args.category),
        "output_report": str(report_path),
        "field_profile_output": str(field_profile_path),
        "text_profile_output": str(text_profile_path),
        "quality_report_output": str(quality_report_path),
        "plots_dir": str(plots_dir),
        "word_views_dir": str(word_views_dir),
        "warnings": report["warnings"],
    }


def run_huggingface_load(args: argparse.Namespace) -> dict[str, Any]:
    load_dataset_func = _import_hf_load_dataset()
    output_reviews_sample = Path(args.output_reviews_sample)
    output_metadata_sample = Path(args.output_metadata_sample)
    reviews = load_hf_review_sample(
        load_dataset_func=load_dataset_func,
        category=str(args.category),
        sample_limit=int(args.sample_limit),
        reviews_config=args.reviews_config,
        streaming=bool(args.streaming),
        seed=int(args.seed),
    )
    metadata = load_hf_metadata_sample(
        load_dataset_func=load_dataset_func,
        category=str(args.category),
        metadata_limit=int(args.metadata_limit),
        metadata_config=args.metadata_config,
        streaming=bool(args.streaming),
        seed=int(args.seed),
    )
    write_jsonl(output_reviews_sample, reviews)
    write_jsonl(output_metadata_sample, metadata)

    local_args = argparse.Namespace(**vars(args))
    local_args.reviews_input = str(output_reviews_sample)
    local_args.metadata_input = str(output_metadata_sample)
    exploration_summary = run_local_exploration(local_args, mode="load_from_huggingface")
    text_profile = json.loads(Path(args.text_profile_output).read_text(encoding="utf-8"))
    quality_report = json.loads(Path(args.quality_report_output).read_text(encoding="utf-8"))
    report = json.loads(Path(args.output_report).read_text(encoding="utf-8"))

    return {
        "mode": "load_from_huggingface",
        "phase": PHASE,
        "category": str(args.category),
        "reviews_config": args.reviews_config
        or build_hf_dataset_names(str(args.category))["reviews_config"],
        "metadata_config": args.metadata_config
        or build_hf_dataset_names(str(args.category))["metadata_config"],
        "streaming": bool(args.streaming),
        "seed": int(args.seed),
        "reviews_sample_count": len(reviews),
        "metadata_sample_count": len(metadata),
        "output_reviews_sample": str(output_reviews_sample),
        "output_metadata_sample": str(output_metadata_sample),
        "output_report": exploration_summary["output_report"],
        "field_profile_output": exploration_summary["field_profile_output"],
        "text_profile_output": exploration_summary["text_profile_output"],
        "quality_report_output": exploration_summary["quality_report_output"],
        "plots_dir": exploration_summary["plots_dir"],
        "word_views_dir": exploration_summary["word_views_dir"],
        "rating_distribution": text_profile.get("rating_distribution", {}),
        "top_issue_terms": text_profile.get("frequent_issue_terms", [])[:10],
        "quality_flags": report.get("quality_flags", quality_report),
        "warnings": report.get("warnings", _report_warnings()),
        "next_step": report.get(
            "next_step",
            "Proceed to Phase 2A-6C Retail curated seed creation after reviewing EDA.",
        ),
    }


def build_dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    dataset_names = build_hf_dataset_names(str(args.category))
    return {
        "mode": "dry_run",
        "phase": PHASE,
        "source_name": SOURCE_NAME,
        "source_owner": SOURCE_OWNER,
        "source_location": SOURCE_LOCATION,
        "category": str(args.category),
        "sample_limit": int(args.sample_limit),
        "planned_inputs": {
            "reviews_input": str(args.reviews_input),
            "metadata_input": str(args.metadata_input),
        },
        "planned_outputs": {
            "output_report": str(args.output_report),
            "output_reviews_sample": str(args.output_reviews_sample),
            "output_metadata_sample": str(args.output_metadata_sample),
            "field_profile_output": str(args.field_profile_output),
            "text_profile_output": str(args.text_profile_output),
            "quality_report_output": str(args.quality_report_output),
            "plots_dir": str(args.plots_dir),
            "word_views_dir": str(args.word_views_dir),
        },
        "sampling_plan": {
            "do_not_download_full_dataset": True,
            "controlled_sampling_required": True,
            "target_exploration_sample_size": 1000,
            "target_metadata_sample_size": 1000,
            "initial_category_priority": ["All_Beauty", "Home_and_Kitchen", "Electronics"],
            "default_reviews_config": dataset_names["reviews_config"],
            "default_metadata_config": dataset_names["metadata_config"],
        },
        "warnings": _report_warnings(),
        "next_step": (
            "Use --write-schema-samples or provide local JSONL samples for --explore-local."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-schema-samples", action="store_true")
    parser.add_argument("--explore-local", action="store_true")
    parser.add_argument("--summarize-local", action="store_true")
    parser.add_argument("--load-from-huggingface", action="store_true")
    parser.add_argument("--reviews-input", default=str(DEFAULT_REVIEWS_INPUT))
    parser.add_argument("--metadata-input", default=str(DEFAULT_METADATA_INPUT))
    parser.add_argument("--output-reviews-sample", default=str(DEFAULT_OUTPUT_REVIEWS_SAMPLE))
    parser.add_argument("--output-metadata-sample", default=str(DEFAULT_OUTPUT_METADATA_SAMPLE))
    parser.add_argument("--output-report", default=str(DEFAULT_OUTPUT_REPORT))
    parser.add_argument("--field-profile-output", default=str(DEFAULT_FIELD_PROFILE_OUTPUT))
    parser.add_argument("--text-profile-output", default=str(DEFAULT_TEXT_PROFILE_OUTPUT))
    parser.add_argument("--quality-report-output", default=str(DEFAULT_QUALITY_REPORT_OUTPUT))
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR))
    parser.add_argument("--word-views-dir", default=str(DEFAULT_WORD_VIEWS_DIR))
    parser.add_argument("--sample-limit", type=int, default=1000)
    parser.add_argument("--metadata-limit", type=int, default=1000)
    parser.add_argument("--category", default="All_Beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", dest="streaming", action="store_true", default=True)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--reviews-config", default=None)
    parser.add_argument("--metadata-config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = [
        args.dry_run,
        args.write_schema_samples,
        args.explore_local,
        args.summarize_local,
        args.load_from_huggingface,
    ]
    if sum(1 for enabled in modes if enabled) != 1:
        parser.error(
            "Exactly one mode is required: --dry-run, --write-schema-samples, "
            "--explore-local, --summarize-local, or --load-from-huggingface."
        )

    try:
        if args.dry_run:
            summary = build_dry_run_summary(args)
        elif args.write_schema_samples:
            summary = write_schema_samples()
        elif args.load_from_huggingface:
            summary = run_huggingface_load(args)
        elif args.explore_local:
            summary = run_local_exploration(args, mode="explore_local")
        else:
            summary = run_local_exploration(args, mode="summarize_local")
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
