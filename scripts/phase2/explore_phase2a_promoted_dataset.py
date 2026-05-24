"""Build the Phase 2A-16R EDA layer for the promoted 10,000-record dataset.

This is data exploration only. It reads committed benchmark JSONL files and
writes local generated reports, static figures, Plotly HTML dashboards, and
cleaned word views. It does not run inference, build RAG, create embeddings,
call model APIs, or create vector indexes.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PHASE = "2A-16R"
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
VERTICAL_LABELS = {
    "airline": "Airline",
    "healthcare_admin": "Healthcare Admin",
    "retail": "Retail",
    "finance": "Finance",
    "research_ai": "Research AI",
}
FILE_KINDS = ["prompts", "gold", "kb"]

DEFAULT_DATASET_ROOT = Path("data/scaleup_2000_full")
DEFAULT_OUTPUT_DIR = Path("data/generated/phase2a/eda")
DEFAULT_RESEARCH_AI_CORPUS = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_full_sections_corpus.jsonl"
)
DEFAULT_RESEARCH_AI_MANIFEST = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_full_sections_manifest.json"
)
DEFAULT_RESEARCH_AI_QUALITY = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/"
    "research_ai_retrieval_corpus_quality_report.json"
)

REQUIRED_INTERACTIVE_FILES = [
    "inventory_prompts_gold_kb_by_vertical.html",
    "status_distribution_by_vertical.html",
    "output_format_by_vertical.html",
    "task_type_mix_by_vertical.html",
    "prompt_length_boxplot.html",
    "gold_length_boxplot.html",
    "kb_length_boxplot.html",
    "workload_shape_by_vertical.html",
    "evidence_reuse_by_vertical.html",
    "vertical_task_heatmap.html",
    "vertical_status_heatmap.html",
]

REQUIRED_STATIC_PLOTS = [
    "inventory_prompts_gold_kb_by_vertical.png",
    "kb_rows_by_vertical.png",
    "prompts_by_vertical.png",
    "gold_by_vertical.png",
    "status_distribution_by_vertical.png",
    "output_format_by_vertical.png",
    "task_type_mix_by_vertical.png",
    "prompt_length_by_vertical.png",
    "gold_length_by_vertical.png",
    "kb_length_by_vertical.png",
    "workload_shape_by_vertical.png",
    "evidence_reuse_by_vertical.png",
]

COMMON_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "they",
    "can",
    "should",
    "what",
    "how",
    "using",
    "only",
    "cited",
    "evidence",
    "records",
    "record",
    "scenario",
    "selected",
    "answer",
    "question",
    "request",
    "help",
    "prompt",
    "gold",
    "kb",
    "scaleup",
    "use",
    "based",
    "about",
    "after",
    "against",
    "also",
    "any",
    "are",
    "before",
    "being",
    "between",
    "but",
    "does",
    "each",
    "given",
    "has",
    "have",
    "into",
    "its",
    "may",
    "must",
    "needs",
    "not",
    "per",
    "provide",
    "say",
    "says",
    "show",
    "stay",
    "than",
    "then",
    "there",
    "these",
    "those",
    "to",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "will",
    "within",
    "without",
    "write",
}

VERTICAL_BOILERPLATE = {
    "airline": {"canada", "air", "ca", "pol", "traveler", "passenger", "route"},
    "healthcare_admin": {
        "maplecare",
        "health",
        "patient",
        "staff",
        "admin",
        "administrative",
        "clinic",
        "mch",
    },
    "retail": {"retail", "support", "agent", "product", "review", "item"},
    "finance": {"finance", "filing", "sec", "inc", "company"},
    "research_ai": {"research", "ai", "paper", "language", "model", "models"},
}

CRITICAL_SAFETY_PATTERNS = {
    "private_windows_path": r"C:\\Users",
    "private_unix_path": r"/home/",
    "private_username_akpoogaga": r"\bakpoogaga\b",
    "private_username_kparo": r"\bkparo\b",
    "api_key_reference": r"\bAPI key\b",
    "token_reference": r"\btoken\b",
    "secret_reference": r"\bsecret\b",
    "password_reference": r"\bpassword\b",
    "raw_user_id_reference": r"\braw user_id\b",
}

WARNING_SAFETY_PATTERNS = {
    "buy_recommendation": r"\bbuy recommendation\b",
    "sell_recommendation": r"\bsell recommendation\b",
    "hold_recommendation": r"\bhold recommendation\b",
    "price_target": r"\bprice target\b",
    "diagnose": r"\bdiagnose\b",
    "treatment_advice": r"\btreatment advice\b",
    "medical_diagnosis": r"\bmedical diagnosis\b",
    "guaranteed_compensation": r"\bguaranteed compensation\b",
    "bypass_verification": r"\bbypass verification\b",
    "fabricated_citation": r"\bfabricated citation\b",
    "general_model_memory": r"\bgeneral model memory\b",
}

PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Expected JSON object in {path} line {line_number}.")
        rows.append(parsed)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int | float | bool):
        return str(value)
    if isinstance(value, dict):
        return " ".join(flatten_text(item) for item in value.values())
    if isinstance(value, list | tuple | set):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_count(text: str) -> int:
    return len(words(text))


def estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, math.ceil(len(text) / 4))


def percentiles(values: list[int] | list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0, "mean": 0}
    sorted_values = sorted(float(value) for value in values)

    def pct(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = (p / 100) * (len(sorted_values) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return sorted_values[lower]
        return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)

    return {
        "min": round(sorted_values[0], 3),
        "p25": round(pct(25), 3),
        "median": round(pct(50), 3),
        "p75": round(pct(75), 3),
        "p95": round(pct(95), 3),
        "max": round(sorted_values[-1], 3),
        "mean": round(mean(sorted_values), 3),
    }


def evidence_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in ["required_doc_ids", "required_evidence_ids", "source_doc_ids"]:
        value = row.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    value = metadata(row).get("required_evidence_ids")
    if isinstance(value, list):
        ids.extend(str(item) for item in value if item)
    return list(dict.fromkeys(ids))


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def normalized_template(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    normalized = re.sub(r"\b[a-z]{2,}[_-][a-z0-9_-]+\b", "<id>", normalized)
    normalized = re.sub(r"\b[a-z]{2,3}-[a-z]{2,3}\b", "<route>", normalized)
    normalized = re.sub(r"\b[A-Z]{1,5}\b", "<ticker>", normalized)
    normalized = re.sub(r"\d+", "<num>", normalized)
    return " ".join(normalized.split())


def prompt_text(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("issue") or flatten_text(row))


def gold_text(row: dict[str, Any]) -> str:
    return str(row.get("reference_answer") or flatten_text(row))


def kb_text(row: dict[str, Any]) -> str:
    return str(row.get("body") or row.get("text") or flatten_text(row))


def text_for_word_views(row: dict[str, Any]) -> str:
    fields = [
        "question",
        "issue",
        "reference_answer",
        "body",
        "text",
        "title",
        "product_title",
        "company",
        "category",
        "support_type",
        "task_type",
        "document_type",
        "issue_type",
        "topic",
        "filing_form",
        "ticker",
    ]
    pieces = []
    for field in fields:
        if not row.get(field):
            continue
        value = str(row[field])
        if field in {"body", "text", "reference_answer"}:
            value = value[:4000]
        pieces.append(value)
    row_metadata = metadata(row)
    for field in [
        "topic",
        "topics",
        "title",
        "company_name",
        "section_type",
        "evidence_type",
        "form",
        "ticker",
        "category",
    ]:
        if row_metadata.get(field):
            pieces.append(flatten_text(row_metadata[field]))
    return " ".join(pieces)


def dataset_files(dataset_root: Path) -> dict[str, dict[str, Path]]:
    return {
        vertical: {
            kind: dataset_root / vertical / f"{vertical}_{kind}_2000.jsonl" for kind in FILE_KINDS
        }
        for vertical in VERTICALS
    }


def load_dataset(
    dataset_root: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, str]], list[str]]:
    paths = dataset_files(dataset_root)
    dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    file_paths: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for vertical, kind_paths in paths.items():
        dataset[vertical] = {}
        file_paths[vertical] = {}
        for kind, path in kind_paths.items():
            file_paths[vertical][kind] = str(path)
            if not path.exists():
                missing.append(str(path))
            dataset[vertical][kind] = read_jsonl(path)
    return dataset, file_paths, missing


def tokenize_for_eda(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+", " ", text.lower())
    cleaned = re.sub(r"[_/\\#.:;,(){}\[\]|+-]+", " ", cleaned)
    tokens: list[str] = []
    for token in re.findall(r"[a-z][a-z0-9']{1,}", cleaned):
        token = token.strip("'")
        if len(token) < 2 or any(char.isdigit() for char in token):
            continue
        tokens.append(token)
    return tokens


def clean_tokens(text: str, vertical: str, remove_vertical_boilerplate: bool) -> list[str]:
    stopwords = set(COMMON_STOPWORDS)
    if remove_vertical_boilerplate:
        stopwords.update(VERTICAL_BOILERPLATE[vertical])
    return [token for token in tokenize_for_eda(text) if token not in stopwords]


def ngram_counter(
    texts: list[str], vertical: str, n: int, remove_boilerplate: bool
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = clean_tokens(text, vertical, remove_boilerplate)
        for index in range(max(0, len(tokens) - n + 1)):
            gram = " ".join(tokens[index : index + n])
            if gram:
                counter[gram] += 1
    return counter


def ngram_counter_from_tokens(token_lists: list[list[str]], n: int) -> Counter[str]:
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        for index in range(max(0, len(tokens) - n + 1)):
            gram = " ".join(tokens[index : index + n])
            if gram:
                counter[gram] += 1
    return counter


def top_counter_rows(counter: Counter[str], limit: int = 30) -> list[dict[str, Any]]:
    return [{"term": term, "count": count} for term, count in counter.most_common(limit)]


def distinctive_terms(
    vertical_texts: dict[str, str], limit: int = 40
) -> dict[str, list[dict[str, Any]]]:
    cleaned_docs = {
        vertical: " ".join(clean_tokens(text, vertical, remove_vertical_boilerplate=False))
        for vertical, text in vertical_texts.items()
    }
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

        vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=6000)
        matrix = vectorizer.fit_transform(cleaned_docs[vertical] for vertical in VERTICALS)
        features = vectorizer.get_feature_names_out()
        output: dict[str, list[dict[str, Any]]] = {}
        for row_index, vertical in enumerate(VERTICALS):
            scores = matrix.getrow(row_index).toarray()[0]
            ranked_indices = scores.argsort()[::-1][:limit]
            output[vertical] = [
                {"term": str(features[index]), "score": round(float(scores[index]), 6)}
                for index in ranked_indices
                if scores[index] > 0
            ]
        return output
    except Exception:
        document_frequency: Counter[str] = Counter()
        term_counts: dict[str, Counter[str]] = {}
        for vertical, doc in cleaned_docs.items():
            counter = Counter(doc.split())
            term_counts[vertical] = counter
            document_frequency.update(counter.keys())
        output = {}
        vertical_count = len(cleaned_docs)
        for vertical, counter in term_counts.items():
            weighted = Counter()
            total = sum(counter.values()) or 1
            for term, count in counter.items():
                idf = math.log((vertical_count + 1) / (document_frequency[term] + 1)) + 1
                weighted[term] = (count / total) * idf
            output[vertical] = [
                {"term": term, "score": round(float(score), 6)}
                for term, score in weighted.most_common(limit)
            ]
        return output


def build_metric_rows(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    kb_rows: list[dict[str, Any]] = []
    for vertical, records in dataset.items():
        for row in records["prompts"]:
            text = prompt_text(row)
            prompt_rows.append(
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "prompt_id": str(row.get("prompt_id") or ""),
                    "text": text,
                    "char_count": len(text),
                    "word_count": word_count(text),
                    "estimated_tokens": estimate_tokens(text),
                    "task_type": str(row.get("task_type") or "unknown"),
                    "expected_status": str(row.get("expected_status") or "unknown"),
                    "expected_output_format": str(row.get("expected_output_format") or "unknown"),
                    "difficulty": str(metadata(row).get("difficulty") or "unknown"),
                    "template": normalized_template(text),
                    "evidence_count": len(evidence_ids(row)),
                }
            )
        for row in records["gold"]:
            text = gold_text(row)
            must_include = row.get("must_include")
            must_not_include = row.get("must_not_include")
            gold_rows.append(
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "prompt_id": str(row.get("prompt_id") or ""),
                    "text": text,
                    "char_count": len(text),
                    "word_count": word_count(text),
                    "estimated_tokens": estimate_tokens(text),
                    "expected_status": str(row.get("expected_status") or "unknown"),
                    "expected_output_format": str(
                        metadata(row).get("expected_output_format") or "unknown"
                    ),
                    "task_type": str(row.get("task_type") or "unknown"),
                    "must_include_count": len(must_include)
                    if isinstance(must_include, list)
                    else 0,
                    "must_not_include_count": len(must_not_include)
                    if isinstance(must_not_include, list)
                    else 0,
                    "evidence_count": len(evidence_ids(row)),
                }
            )
        for row in records["kb"]:
            text = kb_text(row)
            kb_rows.append(
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "doc_id": str(row.get("doc_id") or ""),
                    "text": text,
                    "char_count": len(text),
                    "word_count": word_count(text),
                    "estimated_tokens": estimate_tokens(text),
                    "document_type": str(row.get("document_type") or "unknown"),
                    "source_type": str(row.get("source_type") or "unknown"),
                }
            )
    return prompt_rows, gold_rows, kb_rows


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for vertical in VERTICALS:
        output[vertical] = dict(
            Counter(str(row[field]) for row in rows if row["vertical"] == vertical)
        )
    return output


def long_count_rows(rows: list[dict[str, Any]], field: str, label: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        counter = Counter(str(row[field]) for row in rows if row["vertical"] == vertical)
        for value, count in sorted(counter.items()):
            output.append(
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    label: value or "unknown",
                    "count": count,
                }
            )
    return output


def inventory_report(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    file_paths: dict[str, dict[str, str]],
    missing_files: list[str],
    dataset_root: Path,
) -> dict[str, Any]:
    manifest_path = dataset_root / "phase2a_2000_full_manifest.json"
    manifest = read_json(manifest_path)
    per_vertical = {
        vertical: {
            "prompt_count": len(records["prompts"]),
            "gold_count": len(records["gold"]),
            "kb_count": len(records["kb"]),
            "files": file_paths[vertical],
        }
        for vertical, records in dataset.items()
    }
    totals = {
        "total_prompt_count": sum(row["prompt_count"] for row in per_vertical.values()),
        "total_gold_count": sum(row["gold_count"] for row in per_vertical.values()),
        "total_kb_count": sum(row["kb_count"] for row in per_vertical.values()),
    }
    expected_totals = {
        "total_prompt_count": manifest.get("total_prompt_count"),
        "total_gold_count": manifest.get("total_gold_count"),
        "total_kb_count": manifest.get("total_kb_count"),
    }
    manifest_count_validation = {
        key: totals[key] == expected
        for key, expected in expected_totals.items()
        if expected is not None
    }
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "dataset_name": "phase2a_10000_promoted_2000_full",
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "manifest_loaded": bool(manifest),
        "manifest_expected_counts": expected_totals,
        "manifest_count_validation": manifest_count_validation,
        "all_manifest_counts_match": all(manifest_count_validation.values())
        if manifest_count_validation
        else False,
        "prompt_count_by_vertical": {v: row["prompt_count"] for v, row in per_vertical.items()},
        "gold_count_by_vertical": {v: row["gold_count"] for v, row in per_vertical.items()},
        "kb_count_by_vertical": {v: row["kb_count"] for v, row in per_vertical.items()},
        **totals,
        "vertical_count": len(VERTICALS),
        "file_paths": file_paths,
        "missing_files": missing_files,
        "per_vertical": per_vertical,
    }


def prompt_profile(prompt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    for vertical in VERTICALS:
        rows = [row for row in prompt_rows if row["vertical"] == vertical]
        texts = [str(row["text"]) for row in rows]
        exact_duplicates = sum(count - 1 for count in Counter(texts).values() if count > 1)
        template_counter = Counter(str(row["template"]) for row in rows)
        near_duplicates = sum(count - 1 for count in template_counter.values() if count > 1)
        dominant_template_count = max(template_counter.values()) if template_counter else 0
        by_vertical[vertical] = {
            "prompt_count": len(rows),
            "raw_text_profile": {
                "word_count_distribution": percentiles([row["word_count"] for row in rows]),
                "character_count_distribution": percentiles([row["char_count"] for row in rows]),
                "estimated_token_count_distribution": percentiles(
                    [row["estimated_tokens"] for row in rows]
                ),
            },
            "prompt_word_count_distribution": percentiles([row["word_count"] for row in rows]),
            "prompt_character_count_distribution": percentiles([row["char_count"] for row in rows]),
            "estimated_prompt_token_count_distribution": percentiles(
                [row["estimated_tokens"] for row in rows]
            ),
            "task_type_distribution": dict(Counter(str(row["task_type"]) for row in rows)),
            "expected_status_distribution": dict(
                Counter(str(row["expected_status"]) for row in rows)
            ),
            "expected_output_format_distribution": dict(
                Counter(str(row["expected_output_format"]) for row in rows)
            ),
            "difficulty_distribution": dict(Counter(str(row["difficulty"]) for row in rows)),
            "duplicate_prompt_text_count": exact_duplicates,
            "near_duplicate_template_count": near_duplicates,
            "most_common_prompt_templates": [
                {"template": template, "count": count}
                for template, count in template_counter.most_common(10)
            ],
            "linguistic_variation_summary": {
                "unique_template_count": len(template_counter),
                "unique_template_share": safe_ratio(len(template_counter), len(rows)),
                "dominant_template_share": safe_ratio(dominant_template_count, len(rows)),
            },
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def kb_profile(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    kb_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    for vertical in VERTICALS:
        rows = [row for row in kb_rows if row["vertical"] == vertical]
        doc_ids = [str(row["doc_id"]) for row in rows if row["doc_id"]]
        doc_hashes = [
            hashlib.sha256(str(row["text"]).strip().encode("utf-8")).hexdigest() for row in rows
        ]
        referenced = Counter()
        for gold in dataset[vertical]["gold"]:
            referenced.update(evidence_ids(gold))
        unused_ids = sorted(set(doc_ids) - set(referenced))
        by_vertical[vertical] = {
            "kb_count": len(rows),
            "kb_word_count_distribution": percentiles([row["word_count"] for row in rows]),
            "kb_character_count_distribution": percentiles([row["char_count"] for row in rows]),
            "estimated_kb_token_count_distribution": percentiles(
                [row["estimated_tokens"] for row in rows]
            ),
            "document_type_distribution": dict(Counter(str(row["document_type"]) for row in rows)),
            "source_type_distribution": dict(Counter(str(row["source_type"]) for row in rows)),
            "unique_evidence_id_count": len(set(doc_ids)),
            "duplicate_evidence_id_count": sum(
                count - 1 for count in Counter(doc_ids).values() if count > 1
            ),
            "duplicate_kb_row_count": sum(
                count - 1 for count in Counter(doc_hashes).values() if count > 1
            ),
            "referenced_kb_count": len(set(referenced) & set(doc_ids)),
            "unreferenced_required_evidence_id_count": len(set(referenced) - set(doc_ids)),
            "unused_kb_count": len(unused_ids),
            "unused_kb_share": safe_ratio(len(unused_ids), len(doc_ids)),
            "kb_records_referenced_by_gold": sorted(set(referenced) & set(doc_ids))[:50],
            "kb_records_never_referenced_by_gold": unused_ids[:50],
            "top_reused_kb_evidence_ids": [
                {"evidence_id": evidence_id, "reference_count": count}
                for evidence_id, count in referenced.most_common(20)
            ],
            "largest_kb_rows": [
                {"doc_id": str(row["doc_id"]), "word_count": row["word_count"]}
                for row in sorted(rows, key=lambda item: int(item["word_count"]), reverse=True)[:10]
            ],
            "shortest_kb_rows": [
                {"doc_id": str(row["doc_id"]), "word_count": row["word_count"]}
                for row in sorted(rows, key=lambda item: int(item["word_count"]))[:10]
            ],
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def gold_profile(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    mechanical_phrases = [
        "the cited",
        "the response should",
        "avoid projections",
        "stay within",
        "using only",
    ]
    for vertical in VERTICALS:
        rows = [row for row in gold_rows if row["vertical"] == vertical]
        raw_gold = dataset[vertical]["gold"]
        answerable_missing_evidence = [
            str(row.get("prompt_id") or "")
            for row in raw_gold
            if str(row.get("expected_status") or "") == "answer" and not evidence_ids(row)
        ]
        negative_missing_must_not = [
            str(row.get("prompt_id") or "")
            for row in raw_gold
            if str(row.get("expected_status") or "") != "answer" and not row.get("must_not_include")
        ]
        by_vertical[vertical] = {
            "gold_count": len(rows),
            "reference_answer_word_count_distribution": percentiles(
                [row["word_count"] for row in rows]
            ),
            "reference_answer_character_count_distribution": percentiles(
                [row["char_count"] for row in rows]
            ),
            "estimated_reference_answer_token_distribution": percentiles(
                [row["estimated_tokens"] for row in rows]
            ),
            "must_include_count_distribution": percentiles(
                [row["must_include_count"] for row in rows]
            ),
            "must_not_include_count_distribution": percentiles(
                [row["must_not_include_count"] for row in rows]
            ),
            "required_evidence_id_count_distribution": percentiles(
                [row["evidence_count"] for row in rows]
            ),
            "expected_status_distribution": dict(
                Counter(str(row["expected_status"]) for row in rows)
            ),
            "empty_reference_answer_count": sum(1 for row in rows if not str(row["text"]).strip()),
            "answerable_gold_with_missing_evidence_count": len(answerable_missing_evidence),
            "negative_gold_with_missing_must_not_include_count": len(negative_missing_must_not),
            "mechanical_phrase_detection": {
                phrase: sum(1 for row in rows if phrase in str(row["text"]).lower())
                for phrase in mechanical_phrases
            },
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def alignment_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    for vertical, records in dataset.items():
        prompt_ids = [str(row.get("prompt_id") or "") for row in records["prompts"]]
        gold_ids = [str(row.get("prompt_id") or "") for row in records["gold"]]
        prompt_set = set(prompt_ids)
        gold_set = set(gold_ids)
        prompt_format = {
            str(row.get("prompt_id") or ""): str(row.get("expected_output_format") or "")
            for row in records["prompts"]
        }
        gold_format = {
            str(row.get("prompt_id") or ""): str(metadata(row).get("expected_output_format") or "")
            for row in records["gold"]
        }
        answerable_without_evidence = [
            str(row.get("prompt_id") or "")
            for row in records["gold"]
            if str(row.get("expected_status") or "") == "answer" and not evidence_ids(row)
        ]
        negative_without_must_not = [
            str(row.get("prompt_id") or "")
            for row in records["gold"]
            if str(row.get("expected_status") or "") != "answer" and not row.get("must_not_include")
        ]
        output_format_mismatches = [
            prompt_id
            for prompt_id, expected_format in prompt_format.items()
            if gold_format.get(prompt_id)
            and expected_format
            and gold_format[prompt_id] != expected_format
        ]
        row = {
            "missing_gold_for_prompts": sorted(prompt_set - gold_set),
            "orphan_gold_without_prompt": sorted(gold_set - prompt_set),
            "duplicate_prompt_ids": [
                prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1
            ],
            "duplicate_gold_prompt_ids": [
                prompt_id for prompt_id, count in Counter(gold_ids).items() if count > 1
            ],
            "answerable_prompts_without_evidence": answerable_without_evidence,
            "negative_prompts_without_must_not_include": negative_without_must_not,
            "prompt_gold_output_format_mismatch": output_format_mismatches,
        }
        by_vertical[vertical] = row
        for key, values in row.items():
            if not values:
                continue
            severity = (
                "warning" if key == "negative_prompts_without_must_not_include" else "critical"
            )
            issues.append(
                {
                    "severity": severity,
                    "vertical": vertical,
                    "issue_type": key,
                    "count": len(values),
                    "examples": values[:10],
                }
            )
    critical_count = sum(issue["count"] for issue in issues if issue["severity"] == "critical")
    warning_count = sum(issue["count"] for issue in issues if issue["severity"] == "warning")
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "alignment_clean": critical_count == 0,
        "issue_list": issues,
        "by_vertical": by_vertical,
    }


def evidence_reuse_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    for vertical, records in dataset.items():
        kb_ids = {str(row.get("doc_id") or "") for row in records["kb"] if row.get("doc_id")}
        referenced = Counter()
        evidence_count_per_gold: list[int] = []
        for gold in records["gold"]:
            ids = evidence_ids(gold)
            evidence_count_per_gold.append(len(ids))
            referenced.update(ids)
        max_reuse = max(referenced.values()) if referenced else 0
        max_reuse_share = safe_ratio(max_reuse, len(records["gold"]))
        unused_count = len(kb_ids - set(referenced))
        label = (
            "high" if max_reuse_share >= 0.15 else "medium" if max_reuse_share >= 0.05 else "low"
        )
        by_vertical[vertical] = {
            "gold_count": len(records["gold"]),
            "kb_count": len(kb_ids),
            "evidence_coverage_rate": safe_ratio(
                sum(1 for count in evidence_count_per_gold if count > 0), len(records["gold"])
            ),
            "average_evidence_ids_per_prompt": round(mean(evidence_count_per_gold), 3)
            if evidence_count_per_gold
            else 0,
            "single_evidence_prompt_share": safe_ratio(
                sum(1 for count in evidence_count_per_gold if count == 1),
                len(evidence_count_per_gold),
            ),
            "multi_evidence_prompt_share": safe_ratio(
                sum(1 for count in evidence_count_per_gold if count > 1),
                len(evidence_count_per_gold),
            ),
            "top_20_reused_evidence_ids": [
                {"evidence_id": evidence_id, "reference_count": count}
                for evidence_id, count in referenced.most_common(20)
            ],
            "top_reused_evidence_ids": [
                {"evidence_id": evidence_id, "reference_count": count}
                for evidence_id, count in referenced.most_common(20)
            ],
            "max_evidence_reuse_share": max_reuse_share,
            "unused_kb_count": unused_count,
            "unused_kb_share": safe_ratio(unused_count, len(kb_ids)),
            "referenced_kb_count": len(set(referenced) & kb_ids),
            "evidence_reuse_concentration_label": label,
            "evidence_reuse_concentration": label,
        }
    return {"phase": PHASE, "generated_at_utc": utc_now(), "by_vertical": by_vertical}


def safety_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    for vertical, records in dataset.items():
        combined = "\n".join(flatten_text(records[kind]) for kind in FILE_KINDS)
        critical_hits = {
            label: len(re.findall(pattern, combined, flags=re.IGNORECASE))
            for label, pattern in CRITICAL_SAFETY_PATTERNS.items()
        }
        warning_hits = {
            label: len(re.findall(pattern, combined, flags=re.IGNORECASE))
            for label, pattern in WARNING_SAFETY_PATTERNS.items()
        }
        critical_hits = {key: value for key, value in critical_hits.items() if value > 0}
        warning_hits = {key: value for key, value in warning_hits.items() if value > 0}
        for label, count in critical_hits.items():
            issues.append(
                {"severity": "critical", "vertical": vertical, "flag": label, "count": count}
            )
        for label, count in warning_hits.items():
            issues.append(
                {"severity": "warning", "vertical": vertical, "flag": label, "count": count}
            )
        by_vertical[vertical] = {
            "critical_flag_counts": critical_hits,
            "warning_flag_counts": warning_hits,
            "safety_flag_count": sum(critical_hits.values()) + sum(warning_hits.values()),
            "critical_issue_count": sum(critical_hits.values()),
            "warning_count": sum(warning_hits.values()),
        }
    critical_count = sum(issue["count"] for issue in issues if issue["severity"] == "critical")
    warning_count = sum(issue["count"] for issue in issues if issue["severity"] == "warning")
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "critical_issue_count": critical_count,
        "warning_count": warning_count,
        "safety_clean": critical_count == 0,
        "flag_counts_by_vertical": by_vertical,
        "issue_list": issues,
        "notes": [
            "Domain-boundary terms may appear in guardrail text such as must_not_include lists.",
            "Critical counts are reserved for private paths, usernames, secrets, tokens, and raw IDs.",
        ],
        "by_vertical": by_vertical,
    }


def workload_shape_report(
    prompt_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    kb_token_lookup = {
        str(row["doc_id"]): int(row["estimated_tokens"]) for row in kb_rows if row.get("doc_id")
    }
    prompt_token_lookup = {
        str(row["prompt_id"]): int(row["estimated_tokens"])
        for row in prompt_rows
        if row.get("prompt_id")
    }
    output_token_lookup = {
        str(row["prompt_id"]): int(row["estimated_tokens"])
        for row in gold_rows
        if row.get("prompt_id")
    }
    by_vertical: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        prompts = [row for row in prompt_rows if row["vertical"] == vertical]
        gold = [row for row in gold_rows if row["vertical"] == vertical]
        kb = [row for row in kb_rows if row["vertical"] == vertical]
        referenced_kb_tokens_per_prompt: list[int] = []
        total_prompt_input_tokens = 0
        total_expected_output_tokens = 0
        evidence_counts: list[int] = []
        for raw_gold in dataset[vertical]["gold"]:
            prompt_id = str(raw_gold.get("prompt_id") or "")
            ids = evidence_ids(raw_gold)
            evidence_counts.append(len(ids))
            total_prompt_input_tokens += prompt_token_lookup.get(prompt_id, 0)
            total_expected_output_tokens += output_token_lookup.get(prompt_id, 0)
            referenced_kb_tokens_per_prompt.append(
                sum(kb_token_lookup.get(item, 0) for item in ids)
            )
        total_referenced_kb_tokens = sum(referenced_kb_tokens_per_prompt)
        total_estimated_input_tokens = total_prompt_input_tokens + total_referenced_kb_tokens
        average_input_tokens = (
            total_estimated_input_tokens / len(dataset[vertical]["gold"])
            if dataset[vertical]["gold"]
            else 0
        )
        average_output_tokens = (
            total_expected_output_tokens / len(dataset[vertical]["gold"])
            if dataset[vertical]["gold"]
            else 0
        )
        pressure_score = round(average_input_tokens + average_output_tokens, 3)
        row = {
            "estimated_prompt_tokens": {
                "total": sum(int(item["estimated_tokens"]) for item in prompts),
                "distribution": percentiles([item["estimated_tokens"] for item in prompts]),
            },
            "estimated_kb_tokens": {
                "total_corpus_tokens": sum(int(item["estimated_tokens"]) for item in kb),
                "total_referenced_kb_tokens": total_referenced_kb_tokens,
                "referenced_per_prompt_distribution": percentiles(referenced_kb_tokens_per_prompt),
                "corpus_row_distribution": percentiles([item["estimated_tokens"] for item in kb]),
            },
            "estimated_expected_output_tokens": {
                "total": total_expected_output_tokens,
                "distribution": percentiles([item["estimated_tokens"] for item in gold]),
            },
            "total_estimated_input_tokens": total_estimated_input_tokens,
            "average_estimated_input_tokens_per_prompt": round(average_input_tokens, 3),
            "expected_output_tokens_per_prompt": round(average_output_tokens, 3),
            "output_format_mix": dict(
                Counter(str(row["expected_output_format"]) for row in prompts)
            ),
            "single_evidence_prompt_share": safe_ratio(
                sum(1 for count in evidence_counts if count == 1), len(evidence_counts)
            ),
            "multi_evidence_prompt_share": safe_ratio(
                sum(1 for count in evidence_counts if count > 1), len(evidence_counts)
            ),
            "workload_pressure_score": pressure_score,
        }
        by_vertical[vertical] = row
        ranking.append(
            {
                "vertical": vertical,
                "average_estimated_input_tokens_per_prompt": row[
                    "average_estimated_input_tokens_per_prompt"
                ],
                "workload_pressure_score": pressure_score,
            }
        )
    ranked = sorted(ranking, key=lambda item: item["workload_pressure_score"], reverse=True)
    for index, row in enumerate(ranked):
        label = "high" if index < 2 else "medium" if index < 4 else "low"
        row["likely_inference_cost_pressure"] = label
        by_vertical[row["vertical"]]["likely_inference_cost_pressure"] = label
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "by_vertical": by_vertical,
        "context_heavy_vertical_ranking": ranked,
    }


def build_word_profile(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    vertical_text_lists: dict[str, list[str]] = {}
    vertical_texts: dict[str, str] = {}
    for vertical, records in dataset.items():
        texts: list[str] = []
        for kind in FILE_KINDS:
            texts.extend(text_for_word_views(row) for row in records[kind])
        vertical_text_lists[vertical] = texts
        domain_tokens = [
            token
            for text in texts
            for token in clean_tokens(text, vertical, remove_vertical_boilerplate=False)
        ]
        vertical_texts[vertical] = " ".join(domain_tokens)

    tfidf_terms = distinctive_terms(vertical_texts)
    by_vertical: dict[str, Any] = {}
    for vertical in VERTICALS:
        texts = vertical_text_lists[vertical]
        clean_token_lists = [
            clean_tokens(text, vertical, remove_vertical_boilerplate=True) for text in texts
        ]
        domain_token_lists = [
            clean_tokens(text, vertical, remove_vertical_boilerplate=False) for text in texts
        ]
        by_vertical[vertical] = {
            "raw_text_profile": {
                "record_count": len(texts),
                "word_count_distribution": percentiles([word_count(text) for text in texts]),
            },
            "clean_terms_without_boilerplate": {
                "unigrams": top_counter_rows(
                    ngram_counter_from_tokens(clean_token_lists, 1), limit=50
                ),
                "bigrams": top_counter_rows(
                    ngram_counter_from_tokens(clean_token_lists, 2), limit=50
                ),
                "trigrams": top_counter_rows(
                    ngram_counter_from_tokens(clean_token_lists, 3), limit=50
                ),
            },
            "domain_terms_with_domain_words": {
                "unigrams": top_counter_rows(
                    ngram_counter_from_tokens(domain_token_lists, 1), limit=50
                ),
                "bigrams": top_counter_rows(
                    ngram_counter_from_tokens(domain_token_lists, 2), limit=50
                ),
                "trigrams": top_counter_rows(
                    ngram_counter_from_tokens(domain_token_lists, 3), limit=50
                ),
            },
            "tfidf_distinctive_terms": tfidf_terms.get(vertical, []),
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def frequency_dict(items: list[dict[str, Any]], key: str, limit: int = 20) -> dict[str, int]:
    counter = Counter(str(item.get(key) or "") for item in items)
    return {name: count for name, count in counter.most_common(limit) if name}


def frequency_from_metadata(
    rows: list[dict[str, Any]], key: str, limit: int = 20
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = metadata(row).get(key)
        if isinstance(value, list):
            counter.update(str(item) for item in value if item)
        elif value:
            counter[str(value)] += 1
    return {name: count for name, count in counter.most_common(limit) if name}


def vertical_specific_reports(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    research_ai_corpus_path: Path,
    research_ai_manifest_path: Path,
    research_ai_quality_path: Path,
) -> dict[str, Any]:
    finance_prompts = dataset["finance"]["prompts"]
    finance_kb = dataset["finance"]["kb"]
    research_kb = dataset["research_ai"]["kb"]
    research_corpus = (
        read_jsonl(research_ai_corpus_path) if research_ai_corpus_path.exists() else []
    )
    research_manifest = read_json(research_ai_manifest_path)
    research_quality = read_json(research_ai_quality_path)

    finance_evidence_types = Counter()
    task_by_evidence_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in finance_prompts:
        evidence_type = flatten_text(metadata(row).get("evidence_type") or "unknown")
        finance_evidence_types[evidence_type] += 1
        task_by_evidence_type[evidence_type][str(row.get("task_type") or "unknown")] += 1

    retail_prompts = dataset["retail"]["prompts"]
    retail_gold = dataset["retail"]["gold"]
    retail_negative_count = sum(
        1
        for row in retail_gold
        if str(row.get("expected_status") or "").lower()
        in {"spam_or_low_quality", "refuse", "negative"}
    )

    airline_prompts = dataset["airline"]["prompts"]
    healthcare_prompts = dataset["healthcare_admin"]["prompts"]

    return {
        "finance": {
            "ticker_coverage": frequency_dict(finance_prompts, "ticker"),
            "company_coverage": frequency_dict(finance_prompts, "company"),
            "filing_form_coverage": frequency_dict(finance_prompts, "filing_form"),
            "kb_ticker_coverage": frequency_from_metadata(finance_kb, "ticker"),
            "kb_filing_form_coverage": frequency_from_metadata(finance_kb, "form"),
            "xbrl_concept_coverage": frequency_from_metadata(finance_kb, "concept"),
            "evidence_type_coverage": dict(finance_evidence_types),
            "task_type_by_financial_evidence_type": {
                evidence_type: dict(counter)
                for evidence_type, counter in task_by_evidence_type.items()
            },
            "warnings": [
                "XBRL concept metadata is unavailable in the promoted KB."
                if not frequency_from_metadata(finance_kb, "concept")
                else ""
            ],
        },
        "research_ai": {
            "promoted_benchmark_kb_count": len(research_kb),
            "full_retrieval_corpus_path": str(research_ai_corpus_path),
            "full_retrieval_corpus_exists": research_ai_corpus_path.exists(),
            "full_retrieval_corpus_count": len(research_corpus),
            "full_retrieval_corpus_manifest_exists": bool(research_manifest),
            "full_retrieval_corpus_quality_report_exists": bool(research_quality),
            "paper_coverage": frequency_from_metadata(research_kb, "paper_id", limit=50),
            "paper_coverage_count": len(
                frequency_from_metadata(research_kb, "paper_id", limit=5000)
            ),
            "section_type_distribution": frequency_from_metadata(research_kb, "section_type"),
            "evidence_type_distribution": frequency_from_metadata(research_kb, "evidence_type"),
            "topic_distribution": frequency_from_metadata(research_kb, "topic"),
            "benchmark_kb_vs_retrieval_corpus_note": (
                "Promoted benchmark KB rows are the evidence records used by the 10,000-record "
                "benchmark. The optional full Research AI retrieval corpus contains broader "
                "paper sections for later retrieval experiments and is not used by this EDA as RAG."
            ),
            "warnings": []
            if research_ai_corpus_path.exists()
            else [
                "Full Research AI retrieval-corpus comparison skipped because the corpus file is missing."
            ],
        },
        "retail": {
            "category_coverage": frequency_dict(retail_prompts, "category"),
            "product_title_coverage_count": sum(
                1 for row in retail_prompts if row.get("product_title")
            ),
            "issue_type_coverage": frequency_dict(retail_prompts, "issue_type"),
            "spam_or_low_quality_status_share": safe_ratio(retail_negative_count, len(retail_gold)),
        },
        "airline": {
            "policy_category_coverage": frequency_dict(airline_prompts, "support_type"),
            "escalation_share": safe_ratio(
                sum(1 for row in airline_prompts if row.get("expected_status") == "escalate"),
                len(airline_prompts),
            ),
            "fraud_share": safe_ratio(
                sum(1 for row in airline_prompts if row.get("expected_status") == "spam_or_fraud"),
                len(airline_prompts),
            ),
        },
        "healthcare_admin": {
            "admin_category_coverage": frequency_dict(healthcare_prompts, "support_type"),
            "privacy_sensitive_share": safe_ratio(
                sum(1 for row in healthcare_prompts if row.get("privacy_sensitive")),
                len(healthcare_prompts),
            ),
            "identity_or_privacy_boundary_share": safe_ratio(
                sum(
                    1
                    for row in healthcare_prompts
                    if row.get("privacy_sensitive")
                    or "identity" in str(row.get("support_type") or "").lower()
                    or "privacy" in str(row.get("support_type") or "").lower()
                ),
                len(healthcare_prompts),
            ),
            "safety_boundary_share": safe_ratio(
                sum(
                    1
                    for row in healthcare_prompts
                    if row.get("expected_status") == "safety_boundary"
                ),
                len(healthcare_prompts),
            ),
        },
    }


def summary_rows(
    inventory: dict[str, Any],
    evidence: dict[str, Any],
    workload: dict[str, Any],
    prompt: dict[str, Any],
    gold: dict[str, Any],
    kb: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        rows.append(
            {
                "vertical": vertical,
                "prompt_count": inventory["prompt_count_by_vertical"][vertical],
                "gold_count": inventory["gold_count_by_vertical"][vertical],
                "kb_count": inventory["kb_count_by_vertical"][vertical],
                "evidence_coverage_rate": evidence["by_vertical"][vertical][
                    "evidence_coverage_rate"
                ],
                "average_evidence_ids_per_prompt": evidence["by_vertical"][vertical][
                    "average_evidence_ids_per_prompt"
                ],
                "unused_kb_share": evidence["by_vertical"][vertical]["unused_kb_share"],
                "prompt_words_mean": prompt["by_vertical"][vertical][
                    "prompt_word_count_distribution"
                ]["mean"],
                "gold_words_mean": gold["by_vertical"][vertical][
                    "reference_answer_word_count_distribution"
                ]["mean"],
                "kb_words_mean": kb["by_vertical"][vertical]["kb_word_count_distribution"]["mean"],
                "workload_pressure_score": workload["by_vertical"][vertical][
                    "workload_pressure_score"
                ],
                "likely_inference_cost_pressure": workload["by_vertical"][vertical][
                    "likely_inference_cost_pressure"
                ],
            }
        )
    return rows


def inventory_chart_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = {
        "prompt_count_by_vertical": "Prompts",
        "gold_count_by_vertical": "Gold/Evals",
        "kb_count_by_vertical": "KB rows",
    }
    for source, label in labels.items():
        for vertical, count in inventory[source].items():
            rows.append(
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "record_type": label,
                    "count": count,
                }
            )
    return rows


def workload_chart_rows(workload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        item = workload["by_vertical"][vertical]
        rows.extend(
            [
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "token_component": "Prompt tokens",
                    "tokens": item["estimated_prompt_tokens"]["total"],
                },
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "token_component": "Referenced KB tokens",
                    "tokens": item["estimated_kb_tokens"]["total_referenced_kb_tokens"],
                },
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "token_component": "Expected output tokens",
                    "tokens": item["estimated_expected_output_tokens"]["total"],
                },
            ]
        )
    return rows


def evidence_chart_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        item = evidence["by_vertical"][vertical]
        rows.extend(
            [
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "metric": "Evidence coverage rate",
                    "value": item["evidence_coverage_rate"],
                },
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "metric": "Max reuse share",
                    "value": item["max_evidence_reuse_share"],
                },
                {
                    "vertical": vertical,
                    "vertical_label": VERTICAL_LABELS[vertical],
                    "metric": "Unused KB share",
                    "value": item["unused_kb_share"],
                },
            ]
        )
    return rows


def fallback_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    pre {{ background: #f3f4f6; padding: 16px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
</body>
</html>
"""


def write_plotly_html(path: Path, fig: Any, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        html_text = fig.to_html(
            full_html=True,
            include_plotlyjs="cdn",
            config={"responsive": True, "displaylogo": False},
        )
    except Exception:
        html_text = fallback_html(title, "<p>Plotly rendering failed for this chart.</p>")
    path.write_text(html_text, encoding="utf-8")


def build_plotly_figures(
    inventory: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    workload: dict[str, Any],
) -> dict[str, Any]:
    try:
        import plotly.express as px  # type: ignore[import-untyped]
        import plotly.graph_objects as go  # type: ignore[import-untyped]
    except Exception:
        return {}

    prompt_plot_rows = [
        {key: value for key, value in row.items() if key != "text"} for row in prompt_rows
    ]
    gold_plot_rows = [
        {key: value for key, value in row.items() if key != "text"} for row in gold_rows
    ]
    kb_plot_rows = [{key: value for key, value in row.items() if key != "text"} for row in kb_rows]
    inventory_rows = inventory_chart_rows(inventory)
    status_rows = long_count_rows(prompt_plot_rows, "expected_status", "expected_status")
    output_rows = long_count_rows(
        prompt_plot_rows, "expected_output_format", "expected_output_format"
    )
    task_rows = long_count_rows(prompt_plot_rows, "task_type", "task_type")
    workload_rows = workload_chart_rows(workload)
    reuse_rows = evidence_chart_rows(evidence)

    figures: dict[str, Any] = {}
    figures["inventory_prompts_gold_kb_by_vertical"] = px.bar(
        inventory_rows,
        x="vertical_label",
        y="count",
        color="record_type",
        barmode="group",
        title="Prompts, Gold/Evals, and KB Rows by Vertical",
        labels={"vertical_label": "Vertical", "count": "Rows", "record_type": "Record type"},
        template="plotly_white",
    )
    figures["status_distribution_by_vertical"] = px.bar(
        status_rows,
        x="vertical_label",
        y="count",
        color="expected_status",
        barmode="stack",
        title="Expected Status Distribution by Vertical",
        labels={"vertical_label": "Vertical", "count": "Prompts", "expected_status": "Status"},
        template="plotly_white",
    )
    figures["output_format_by_vertical"] = px.bar(
        output_rows,
        x="vertical_label",
        y="count",
        color="expected_output_format",
        barmode="stack",
        title="Expected Output Format by Vertical",
        labels={
            "vertical_label": "Vertical",
            "count": "Prompts",
            "expected_output_format": "Output format",
        },
        template="plotly_white",
    )
    figures["task_type_mix_by_vertical"] = px.bar(
        task_rows,
        x="vertical_label",
        y="count",
        color="task_type",
        barmode="stack",
        title="Task Type Mix by Vertical",
        labels={"vertical_label": "Vertical", "count": "Prompts", "task_type": "Task type"},
        template="plotly_white",
    )
    figures["prompt_length_boxplot"] = px.box(
        prompt_plot_rows,
        x="vertical_label",
        y="word_count",
        color="vertical_label",
        points=False,
        title="Prompt Length Distribution by Vertical",
        labels={"vertical_label": "Vertical", "word_count": "Prompt words"},
        template="plotly_white",
    )
    figures["gold_length_boxplot"] = px.box(
        gold_plot_rows,
        x="vertical_label",
        y="word_count",
        color="vertical_label",
        points=False,
        title="Reference Answer Length Distribution by Vertical",
        labels={"vertical_label": "Vertical", "word_count": "Reference answer words"},
        template="plotly_white",
    )
    figures["kb_length_boxplot"] = px.box(
        kb_plot_rows,
        x="vertical_label",
        y="word_count",
        color="vertical_label",
        points=False,
        title="KB Row Length Distribution by Vertical",
        labels={"vertical_label": "Vertical", "word_count": "KB row words"},
        template="plotly_white",
    )
    figures["workload_shape_by_vertical"] = px.bar(
        workload_rows,
        x="vertical_label",
        y="tokens",
        color="token_component",
        barmode="group",
        title="Estimated Workload Shape by Vertical",
        labels={"vertical_label": "Vertical", "tokens": "Estimated tokens"},
        template="plotly_white",
    )
    figures["evidence_reuse_by_vertical"] = px.bar(
        reuse_rows,
        x="vertical_label",
        y="value",
        color="metric",
        barmode="group",
        title="Evidence Coverage, Reuse, and Unused KB Share",
        labels={"vertical_label": "Vertical", "value": "Share", "metric": "Metric"},
        template="plotly_white",
    )

    task_values = sorted({str(row["task_type"]) for row in prompt_plot_rows})
    task_z = []
    for vertical in VERTICALS:
        counter = Counter(
            str(row["task_type"]) for row in prompt_plot_rows if row["vertical"] == vertical
        )
        task_z.append([counter.get(task, 0) for task in task_values])
    figures["vertical_task_heatmap"] = go.Figure(
        data=go.Heatmap(
            z=task_z,
            x=task_values,
            y=[VERTICAL_LABELS[vertical] for vertical in VERTICALS],
            colorscale="Viridis",
            hovertemplate="Vertical=%{y}<br>Task=%{x}<br>Count=%{z}<extra></extra>",
        )
    )
    figures["vertical_task_heatmap"].update_layout(
        title="Vertical x Task Type Heatmap",
        xaxis_title="Task type",
        yaxis_title="Vertical",
        template="plotly_white",
    )

    status_values = sorted({str(row["expected_status"]) for row in prompt_plot_rows})
    status_z = []
    for vertical in VERTICALS:
        counter = Counter(
            str(row["expected_status"]) for row in prompt_plot_rows if row["vertical"] == vertical
        )
        status_z.append([counter.get(status, 0) for status in status_values])
    figures["vertical_status_heatmap"] = go.Figure(
        data=go.Heatmap(
            z=status_z,
            x=status_values,
            y=[VERTICAL_LABELS[vertical] for vertical in VERTICALS],
            colorscale="Cividis",
            hovertemplate="Vertical=%{y}<br>Status=%{x}<br>Count=%{z}<extra></extra>",
        )
    )
    figures["vertical_status_heatmap"].update_layout(
        title="Vertical x Expected Status Heatmap",
        xaxis_title="Expected status",
        yaxis_title="Vertical",
        template="plotly_white",
    )

    for fig in figures.values():
        fig.update_layout(font={"family": "Arial", "size": 13}, legend_title_text="")
    return figures


def write_interactive_outputs(figures: dict[str, Any], output_dir: Path) -> None:
    interactive_dir = output_dir / "interactive"
    interactive_dir.mkdir(parents=True, exist_ok=True)
    if not figures:
        for filename in REQUIRED_INTERACTIVE_FILES:
            write_text(
                interactive_dir / filename,
                fallback_html(
                    filename.replace("_", " ").replace(".html", "").title(),
                    "<p>Plotly is unavailable; interactive chart generation was skipped.</p>",
                ),
            )
        return
    for name in REQUIRED_INTERACTIVE_FILES:
        stem = name.removesuffix(".html")
        write_plotly_html(interactive_dir / name, figures[stem], stem.replace("_", " ").title())


def write_dashboard(
    output_dir: Path,
    inventory: dict[str, Any],
    alignment: dict[str, Any],
    safety: dict[str, Any],
    figures: dict[str, Any],
    research_ai: dict[str, Any],
) -> None:
    dashboard_dir = output_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    cards = [
        ("Total prompts", inventory["total_prompt_count"]),
        ("Total gold/evals", inventory["total_gold_count"]),
        ("Total KB rows", inventory["total_kb_count"]),
        ("Verticals", inventory["vertical_count"]),
        ("Critical issues", alignment["critical_issue_count"] + safety["critical_issue_count"]),
        (
            "Warnings",
            alignment["warning_count"] + safety["warning_count"] + len(inventory["missing_files"]),
        ),
    ]
    chart_order = [
        (
            "Inventory",
            "Prompts and gold records are balanced at 2,000 per vertical; KB shape varies by source.",
            "inventory_prompts_gold_kb_by_vertical",
        ),
        (
            "Expected Status",
            "Status mix shows the answerable, refusal, escalation, and boundary cases before inference.",
            "status_distribution_by_vertical",
        ),
        (
            "Output Format",
            "Output format mix previews evaluation and rendering requirements for later benchmark runs.",
            "output_format_by_vertical",
        ),
        (
            "Task Type",
            "Task mix shows which verticals emphasize policy lookup, grounded QA, summaries, or boundaries.",
            "task_type_mix_by_vertical",
        ),
        (
            "Prompt Length",
            "Prompt length distributions estimate front-end prompt assembly pressure.",
            "prompt_length_boxplot",
        ),
        (
            "Gold Length",
            "Reference answer lengths estimate expected output size and evaluation surface.",
            "gold_length_boxplot",
        ),
        (
            "KB Length",
            "KB row lengths show which verticals will be context-heavy once retrieval is introduced.",
            "kb_length_boxplot",
        ),
        (
            "Evidence Reuse",
            "Evidence reuse and unused KB share reveal concentration and coverage in promoted evidence.",
            "evidence_reuse_by_vertical",
        ),
        (
            "Workload Shape",
            "Token estimates separate prompt, referenced KB, and expected output pressure by vertical.",
            "workload_shape_by_vertical",
        ),
        (
            "Task Heatmap",
            "Heatmaps make cross-vertical task imbalances visible for paper-ready screenshots.",
            "vertical_task_heatmap",
        ),
        (
            "Status Heatmap",
            "Status heatmaps clarify answerable and boundary coverage across verticals.",
            "vertical_status_heatmap",
        ),
    ]
    if figures:
        figure_html = "\n".join(
            f"<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>"
            f"{figures[name].to_html(full_html=False, include_plotlyjs=False)}</section>"
            for title, note, name in chart_order
        )
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    else:
        figure_html = "<p>Plotly is unavailable; see JSON, CSV, and PNG outputs.</p>"
        plotly_script = ""

    corpus_note = (
        f"Research AI full retrieval corpus rows: {research_ai['full_retrieval_corpus_count']}."
        if research_ai["full_retrieval_corpus_exists"]
        else "Research AI full retrieval-corpus comparison skipped because the corpus file is missing."
    )
    card_html = "\n".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in cards
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phase 2A 10,000-Record EDA Dashboard</title>
  {plotly_script}
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f7f8fb; }}
    header {{ background: #172033; color: white; padding: 28px 36px; }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 28px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #dde2ea; border-radius: 8px; padding: 16px; }}
    .card span {{ display: block; font-size: 13px; color: #536273; margin-bottom: 8px; }}
    .card strong {{ font-size: 28px; color: #172033; }}
    section {{ background: white; border: 1px solid #dde2ea; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    h1, h2 {{ margin-top: 0; }}
    p {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <header>
    <h1>Phase 2A promoted 10,000-record dataset EDA</h1>
    <p>Interactive, paper-ready analytics for prompts, gold/evals, KB shape, evidence reuse, workload pressure, and clean term views.</p>
  </header>
  <main>
    <div class="cards">{card_html}</div>
    <section>
      <h2>Scope</h2>
      <p>This dashboard explores committed benchmark data before inference. It does not run RAG, embeddings, vector indexing, model APIs, or GPU experiments.</p>
      <p>{html.escape(corpus_note)}</p>
    </section>
    {figure_html}
  </main>
</body>
</html>
"""
    write_text(dashboard_dir / "phase2a_10000_overview.html", html_text)
    write_text(
        dashboard_dir / "phase2a_10000_overview.md",
        "\n".join(
            [
                "# Phase 2A 10,000-Record EDA Overview",
                "",
                f"- Total prompts: {inventory['total_prompt_count']}",
                f"- Total gold/evals: {inventory['total_gold_count']}",
                f"- Total KB rows: {inventory['total_kb_count']}",
                f"- Critical issues: {cards[4][1]}",
                f"- Warnings: {cards[5][1]}",
                "",
                "Open `phase2a_10000_overview.html` for the interactive Plotly dashboard.",
                "",
                "The dashboard separates inventory, task/status/output mix, prompt/gold/KB length, "
                "evidence reuse, and estimated workload pressure so the promoted dataset can be "
                "reviewed before inference.",
                "",
                corpus_note,
            ]
        )
        + "\n",
    )


def import_matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]

        return plt
    except Exception:
        return None


def static_bar(
    plt: Any,
    path: Path,
    labels: list[str],
    values: list[int | float],
    title: str,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color=PLOT_COLORS[: len(labels)])
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def static_grouped_bar(
    plt: Any,
    path: Path,
    labels: list[str],
    series: dict[str, list[int | float]],
    title: str,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(series))
    x_values = list(range(len(labels)))
    for index, (name, values) in enumerate(series.items()):
        offsets = [x + (index - (len(series) - 1) / 2) * width for x in x_values]
        axis.bar(
            offsets, values, width=width, label=name, color=PLOT_COLORS[index % len(PLOT_COLORS)]
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=20)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def static_stacked_bar(
    plt: Any,
    path: Path,
    rows: list[dict[str, Any]],
    category_field: str,
    title: str,
) -> None:
    labels = [VERTICAL_LABELS[vertical] for vertical in VERTICALS]
    categories = sorted({str(row[category_field]) for row in rows})
    fig, axis = plt.subplots(figsize=(10, 5))
    bottoms = [0] * len(labels)
    for index, category in enumerate(categories):
        values = [
            sum(
                int(row["count"])
                for row in rows
                if row["vertical"] == vertical and str(row[category_field]) == category
            )
            for vertical in VERTICALS
        ]
        axis.bar(
            labels,
            values,
            bottom=bottoms,
            label=category,
            color=PLOT_COLORS[index % len(PLOT_COLORS)],
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    axis.set_title(title)
    axis.set_ylabel("Prompts")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def static_boxplot(
    plt: Any,
    path: Path,
    rows: list[dict[str, Any]],
    title: str,
    ylabel: str,
) -> None:
    labels = [VERTICAL_LABELS[vertical] for vertical in VERTICALS]
    values = [
        [row["word_count"] for row in rows if row["vertical"] == vertical] for vertical in VERTICALS
    ]
    fig, axis = plt.subplots(figsize=(10, 5))
    try:
        axis.boxplot(values, tick_labels=labels, patch_artist=True)
    except TypeError:
        axis.boxplot(values, labels=labels, patch_artist=True)
    for patch, color in zip(axis.patches, PLOT_COLORS, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_static_plots(
    output_dir: Path,
    inventory: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    workload: dict[str, Any],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt = import_matplotlib()
    if plt is None:
        write_text(
            plot_dir / "plots_skipped.md",
            "Matplotlib is unavailable; static PNG plots were skipped. JSON, CSV, and HTML outputs still exist.\n",
        )
        return

    labels = [VERTICAL_LABELS[vertical] for vertical in VERTICALS]
    static_grouped_bar(
        plt,
        plot_dir / "inventory_prompts_gold_kb_by_vertical.png",
        labels,
        {
            "Prompts": [inventory["prompt_count_by_vertical"][vertical] for vertical in VERTICALS],
            "Gold/Evals": [inventory["gold_count_by_vertical"][vertical] for vertical in VERTICALS],
            "KB rows": [inventory["kb_count_by_vertical"][vertical] for vertical in VERTICALS],
        },
        "Phase 2A Inventory by Vertical",
        "Rows",
    )
    static_bar(
        plt,
        plot_dir / "kb_rows_by_vertical.png",
        labels,
        [inventory["kb_count_by_vertical"][vertical] for vertical in VERTICALS],
        "KB Rows by Vertical",
        "KB rows",
    )
    static_bar(
        plt,
        plot_dir / "prompts_by_vertical.png",
        labels,
        [inventory["prompt_count_by_vertical"][vertical] for vertical in VERTICALS],
        "Prompts by Vertical",
        "Prompts",
    )
    static_bar(
        plt,
        plot_dir / "gold_by_vertical.png",
        labels,
        [inventory["gold_count_by_vertical"][vertical] for vertical in VERTICALS],
        "Gold/Eval Records by Vertical",
        "Gold/eval records",
    )
    static_stacked_bar(
        plt,
        plot_dir / "status_distribution_by_vertical.png",
        long_count_rows(prompt_rows, "expected_status", "expected_status"),
        "expected_status",
        "Expected Status by Vertical",
    )
    static_stacked_bar(
        plt,
        plot_dir / "output_format_by_vertical.png",
        long_count_rows(prompt_rows, "expected_output_format", "expected_output_format"),
        "expected_output_format",
        "Expected Output Format by Vertical",
    )
    static_stacked_bar(
        plt,
        plot_dir / "task_type_mix_by_vertical.png",
        long_count_rows(prompt_rows, "task_type", "task_type"),
        "task_type",
        "Task Type Mix by Vertical",
    )
    static_boxplot(
        plt,
        plot_dir / "prompt_length_by_vertical.png",
        prompt_rows,
        "Prompt Length by Vertical",
        "Prompt words",
    )
    static_boxplot(
        plt,
        plot_dir / "gold_length_by_vertical.png",
        gold_rows,
        "Reference Answer Length by Vertical",
        "Reference answer words",
    )
    static_boxplot(
        plt,
        plot_dir / "kb_length_by_vertical.png",
        kb_rows,
        "KB Row Length by Vertical",
        "KB row words",
    )
    static_grouped_bar(
        plt,
        plot_dir / "workload_shape_by_vertical.png",
        labels,
        {
            "Prompt tokens": [
                workload["by_vertical"][vertical]["estimated_prompt_tokens"]["total"]
                for vertical in VERTICALS
            ],
            "Referenced KB tokens": [
                workload["by_vertical"][vertical]["estimated_kb_tokens"][
                    "total_referenced_kb_tokens"
                ]
                for vertical in VERTICALS
            ],
            "Expected output tokens": [
                workload["by_vertical"][vertical]["estimated_expected_output_tokens"]["total"]
                for vertical in VERTICALS
            ],
        },
        "Estimated Workload Shape by Vertical",
        "Estimated tokens",
    )
    static_grouped_bar(
        plt,
        plot_dir / "evidence_reuse_by_vertical.png",
        labels,
        {
            "Coverage": [
                evidence["by_vertical"][vertical]["evidence_coverage_rate"]
                for vertical in VERTICALS
            ],
            "Max reuse": [
                evidence["by_vertical"][vertical]["max_evidence_reuse_share"]
                for vertical in VERTICALS
            ],
            "Unused KB": [
                evidence["by_vertical"][vertical]["unused_kb_share"] for vertical in VERTICALS
            ],
        },
        "Evidence Reuse by Vertical",
        "Share",
    )


def write_word_cloud_image(
    path: Path,
    terms: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frequencies = {str(item["term"]): float(item["count"]) for item in terms if item.get("term")}
    if not frequencies:
        write_text(path.with_suffix(".txt"), "No terms available for word cloud.\n")
        return
    try:
        from wordcloud import WordCloud  # type: ignore[import-untyped]

        cloud = WordCloud(width=1200, height=700, background_color="white", colormap="viridis")
        cloud.generate_from_frequencies(frequencies)
        cloud.to_file(str(path))
        return
    except Exception:
        pass

    plt = import_matplotlib()
    if plt is None:
        write_text(
            path.with_suffix(".txt"),
            "wordcloud and matplotlib are unavailable; word cloud PNG was skipped.\n",
        )
        return

    ranked = list(frequencies.items())[:40]
    max_count = max(frequencies.values()) or 1
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.axis("off")
    columns = 4
    for index, (term, count) in enumerate(ranked):
        row = index // columns
        column = index % columns
        x = 0.08 + column * 0.24
        y = 0.9 - row * 0.08
        size = 10 + 26 * (count / max_count)
        axis.text(
            x,
            y,
            term,
            fontsize=size,
            color=PLOT_COLORS[index % len(PLOT_COLORS)],
            transform=axis.transAxes,
            alpha=0.9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_word_views(output_dir: Path, word_profile: dict[str, Any]) -> None:
    word_dir = output_dir / "word_views"
    cloud_dir = output_dir / "word_clouds"
    word_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir.mkdir(parents=True, exist_ok=True)
    for vertical in VERTICALS:
        profile = word_profile["by_vertical"][vertical]
        clean = profile["clean_terms_without_boilerplate"]
        domain = profile["domain_terms_with_domain_words"]
        tfidf_terms = profile["tfidf_distinctive_terms"]
        lines = [
            f"# {VERTICAL_LABELS[vertical]} Clean Terms",
            "",
            "## clean_terms_without_boilerplate",
            "",
            "### top_clean_unigrams",
            *[f"{item['term']}\t{item['count']}" for item in clean["unigrams"][:30]],
            "",
            "### top_clean_bigrams",
            *[f"{item['term']}\t{item['count']}" for item in clean["bigrams"][:30]],
            "",
            "### top_clean_trigrams",
            *[f"{item['term']}\t{item['count']}" for item in clean["trigrams"][:30]],
            "",
            "## domain_terms_with_domain_words",
            "",
            "### top_domain_unigrams",
            *[f"{item['term']}\t{item['count']}" for item in domain["unigrams"][:30]],
            "",
            "### top_domain_bigrams",
            *[f"{item['term']}\t{item['count']}" for item in domain["bigrams"][:30]],
        ]
        write_text(word_dir / f"{vertical}_clean_terms.txt", "\n".join(lines) + "\n")
        tfidf_lines = [
            f"# {VERTICAL_LABELS[vertical]} TF-IDF Distinctive Terms",
            "",
            "term\tscore",
            *[f"{item['term']}\t{item['score']}" for item in tfidf_terms[:50]],
        ]
        write_text(word_dir / f"{vertical}_tfidf_terms.txt", "\n".join(tfidf_lines) + "\n")
        write_word_cloud_image(cloud_dir / f"{vertical}_wordcloud.png", clean["unigrams"])


def table_html(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows[:limit]:
        cells = "".join(
            f"<td>{html.escape(str(row.get(column, ''))[:500])}</td>" for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def key_value_table(values: dict[str, Any], limit: int = 20) -> str:
    rows = [
        {"key": key, "value": value} for key, value in list(values.items())[:limit] if value != ""
    ]
    return table_html(rows, ["key", "value"], limit=limit)


def vertical_analysis_text(vertical: str, specific: dict[str, Any]) -> str:
    if vertical == "finance":
        warnings = [warning for warning in specific["warnings"] if warning]
        return (
            "<p>Finance coverage is organized around ticker, company, filing form, and financial "
            "evidence type. The page highlights whether XBRL concept fields are populated and how "
            "task types map to SEC/XBRL evidence.</p>"
            + (
                "<p><strong>Warning:</strong> " + html.escape("; ".join(warnings)) + "</p>"
                if warnings
                else ""
            )
        )
    if vertical == "research_ai":
        warnings = specific["warnings"]
        note = html.escape(specific["benchmark_kb_vs_retrieval_corpus_note"])
        return f"<p>{note}</p>" + (
            "<p><strong>Warning:</strong> " + html.escape("; ".join(warnings)) + "</p>"
            if warnings
            else ""
        )
    if vertical == "retail":
        return (
            "<p>Retail coverage emphasizes product/review signals, category coverage, product-title "
            "coverage, issue-type mix, and any spam or low-quality status share.</p>"
        )
    if vertical == "airline":
        return (
            "<p>Airline coverage emphasizes policy categories, escalation share, fraud share, and "
            "policy-grounded support tasks.</p>"
        )
    return (
        "<p>Healthcare Admin coverage emphasizes administrative categories, privacy/identity "
        "boundaries, and safety-boundary routing for non-clinical support.</p>"
    )


def write_vertical_pages(
    output_dir: Path,
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    prompt_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    kb_rows: list[dict[str, Any]],
    evidence: dict[str, Any],
    word_profile: dict[str, Any],
    vertical_specific: dict[str, Any],
) -> None:
    try:
        import plotly.express as px  # type: ignore[import-untyped]
    except Exception:
        px = None

    for vertical in VERTICALS:
        page_dir = output_dir / "verticals" / vertical
        page_dir.mkdir(parents=True, exist_ok=True)
        prompt_subset = [row for row in prompt_rows if row["vertical"] == vertical]
        gold_subset = [row for row in gold_rows if row["vertical"] == vertical]
        kb_subset = [row for row in kb_rows if row["vertical"] == vertical]
        prompt_plot_subset = [
            {key: value for key, value in row.items() if key != "text"} for row in prompt_subset
        ]
        gold_plot_subset = [
            {key: value for key, value in row.items() if key != "text"} for row in gold_subset
        ]
        kb_plot_subset = [
            {key: value for key, value in row.items() if key != "text"} for row in kb_subset
        ]
        clean_terms = word_profile["by_vertical"][vertical]["clean_terms_without_boilerplate"][
            "unigrams"
        ]
        chart_html = ""
        if px is not None:
            charts = [
                px.histogram(
                    prompt_plot_subset,
                    x="word_count",
                    nbins=30,
                    title="Prompt Length",
                    labels={"word_count": "Prompt words"},
                    template="plotly_white",
                ),
                px.histogram(
                    kb_plot_subset,
                    x="word_count",
                    nbins=30,
                    title="KB Row Length",
                    labels={"word_count": "KB row words"},
                    template="plotly_white",
                ),
                px.histogram(
                    gold_plot_subset,
                    x="word_count",
                    nbins=30,
                    title="Reference Answer Length",
                    labels={"word_count": "Reference answer words"},
                    template="plotly_white",
                ),
                px.bar(
                    long_count_rows(prompt_plot_subset, "task_type", "task_type"),
                    x="task_type",
                    y="count",
                    title="Task Type Mix",
                    template="plotly_white",
                ),
                px.bar(
                    long_count_rows(prompt_plot_subset, "expected_status", "expected_status"),
                    x="expected_status",
                    y="count",
                    title="Status Mix",
                    template="plotly_white",
                ),
                px.bar(
                    long_count_rows(
                        prompt_plot_subset, "expected_output_format", "expected_output_format"
                    ),
                    x="expected_output_format",
                    y="count",
                    title="Output Format Mix",
                    template="plotly_white",
                ),
            ]
            chart_html = "\n".join(
                chart.to_html(full_html=False, include_plotlyjs=False) for chart in charts
            )
            plotly_script = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        else:
            plotly_script = ""
            chart_html = "<p>Plotly is unavailable; see JSON reports and static plots.</p>"

        prompt_samples = [
            {
                "prompt_id": row.get("prompt_id"),
                "task_type": row.get("task_type"),
                "expected_status": row.get("expected_status"),
                "question": row.get("question"),
            }
            for row in dataset[vertical]["prompts"][:5]
        ]
        gold_samples = [
            {
                "prompt_id": row.get("prompt_id"),
                "expected_status": row.get("expected_status"),
                "reference_answer": row.get("reference_answer"),
            }
            for row in dataset[vertical]["gold"][:5]
        ]
        evidence_summary = evidence["by_vertical"][vertical]
        specific = vertical_specific[vertical]
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(VERTICAL_LABELS[vertical])} EDA</title>
  {plotly_script}
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #172033; background: #f7f8fb; }}
    header {{ background: #26364d; color: white; padding: 24px 34px; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #dde2ea; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #dde2ea; border-radius: 8px; padding: 14px; }}
    .card span {{ display: block; color: #536273; font-size: 13px; }}
    .card strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    img {{ max-width: 100%; border: 1px solid #dde2ea; border-radius: 8px; background: white; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(VERTICAL_LABELS[vertical])} EDA</h1></header>
  <main>
    <div class="cards">
      <div class="card"><span>Prompts</span><strong>{len(prompt_subset)}</strong></div>
      <div class="card"><span>Gold/Evals</span><strong>{len(gold_subset)}</strong></div>
      <div class="card"><span>KB rows</span><strong>{len(kb_subset)}</strong></div>
      <div class="card"><span>Evidence coverage</span><strong>{evidence_summary["evidence_coverage_rate"]}</strong></div>
      <div class="card"><span>Unused KB share</span><strong>{evidence_summary["unused_kb_share"]}</strong></div>
    </div>
    <section>
      <h2>Vertical-Specific Analysis</h2>
      {vertical_analysis_text(vertical, specific)}
      {key_value_table(specific, limit=25)}
    </section>
    <section>
      <h2>Interactive Distributions</h2>
      {chart_html}
    </section>
    <section>
      <h2>Top Cleaned Terms</h2>
      {table_html(clean_terms, ["term", "count"], limit=25)}
      <p><img alt="{html.escape(VERTICAL_LABELS[vertical])} word cloud" src="../../word_clouds/{vertical}_wordcloud.png"></p>
    </section>
    <section>
      <h2>Evidence Reuse Summary</h2>
      {key_value_table(evidence_summary, limit=20)}
    </section>
    <section>
      <h2>Representative Prompts</h2>
      {table_html(prompt_samples, ["prompt_id", "task_type", "expected_status", "question"], limit=5)}
    </section>
    <section>
      <h2>Representative Gold/Eval Records</h2>
      {table_html(gold_samples, ["prompt_id", "expected_status", "reference_answer"], limit=5)}
    </section>
  </main>
</body>
</html>
"""
        write_text(page_dir / f"{vertical}_eda.html", page)


def write_eda_readme(output_dir: Path) -> None:
    write_text(
        output_dir / "README.md",
        """# Phase 2A EDA Outputs

This directory is generated by:

```powershell
python scripts/phase2/explore_phase2a_promoted_dataset.py --dataset-root data/scaleup_2000_full --write-report
```

Layout:

- `dashboard/` contains the executive Plotly overview HTML and markdown summary.
- `interactive/` contains one standalone Plotly HTML file per major chart.
- `plots/` contains static PNG figures for papers and documentation.
- `word_clouds/` contains one generated word-cloud-style PNG per vertical.
- `word_views/` contains cleaned and TF-IDF-style term tables.
- `verticals/` contains one interactive HTML EDA page per vertical.
- `phase2a_*_profile.json` and `phase2a_*_report.json` contain machine-readable EDA.

The EDA does not run inference, build RAG, create embeddings, call model APIs, or create
vector indexes. Generated artifacts are local outputs and are not the benchmark source data.
""",
    )


def write_top_level_summary(
    output_dir: Path,
    inventory: dict[str, Any],
    alignment: dict[str, Any],
    evidence: dict[str, Any],
    safety: dict[str, Any],
    workload: dict[str, Any],
    vertical_specific: dict[str, Any],
) -> None:
    lines = [
        "# Phase 2A-16R EDA Summary",
        "",
        "## Dataset Totals",
        "",
        f"- Prompts: {inventory['total_prompt_count']}",
        f"- Gold/eval records: {inventory['total_gold_count']}",
        f"- KB rows: {inventory['total_kb_count']}",
        f"- Verticals: {inventory['vertical_count']}",
        "",
        "## Quality Signals",
        "",
        f"- Alignment critical issues: {alignment['critical_issue_count']}",
        f"- Alignment warnings: {alignment['warning_count']}",
        f"- Safety critical issues: {safety['critical_issue_count']}",
        f"- Safety warnings: {safety['warning_count']}",
        "",
        "## Evidence Reuse",
        "",
    ]
    for vertical in VERTICALS:
        row = evidence["by_vertical"][vertical]
        lines.append(
            f"- {VERTICAL_LABELS[vertical]}: coverage {row['evidence_coverage_rate']}, "
            f"unused KB share {row['unused_kb_share']}, concentration "
            f"{row['evidence_reuse_concentration_label']}"
        )
    lines.extend(["", "## Workload Pressure", ""])
    for row in workload["context_heavy_vertical_ranking"]:
        lines.append(
            f"- {VERTICAL_LABELS[row['vertical']]}: score {row['workload_pressure_score']}, "
            f"{row['likely_inference_cost_pressure']} pressure"
        )
    research = vertical_specific["research_ai"]
    lines.extend(
        [
            "",
            "## Research AI Retrieval Corpus",
            "",
            f"- Promoted benchmark KB rows: {research['promoted_benchmark_kb_count']}",
            f"- Full retrieval corpus available: {research['full_retrieval_corpus_exists']}",
            f"- Full retrieval corpus rows: {research['full_retrieval_corpus_count']}",
            "",
            "This is EDA only: no RAG, no embeddings, no vector indexes, no model APIs, and no inference.",
        ]
    )
    write_text(output_dir / "phase2a_eda_summary.md", "\n".join(lines) + "\n")


def remove_legacy_eda_outputs(output_dir: Path) -> None:
    legacy_paths = [
        output_dir / "phase2a_vertical_specific_report.json",
        output_dir / "plots/phase2a_eda_overview.png",
    ]
    legacy_paths.extend((output_dir / "word_views").glob("*_prompt_terms.txt"))
    for path in legacy_paths:
        if path.exists() and path.is_file():
            path.unlink()


def write_reports(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    remove_legacy_eda_outputs(output_dir)
    dataset, file_paths, missing_files = load_dataset(dataset_root)
    prompt_rows, gold_rows, kb_rows = build_metric_rows(dataset)

    inventory = inventory_report(dataset, file_paths, missing_files, dataset_root)
    prompt = prompt_profile(prompt_rows)
    kb = kb_profile(dataset, kb_rows)
    gold = gold_profile(dataset, gold_rows)
    alignment = alignment_report(dataset)
    evidence = evidence_reuse_report(dataset)
    safety = safety_report(dataset)
    workload = workload_shape_report(prompt_rows, gold_rows, kb_rows, dataset)
    word_profile = build_word_profile(dataset)
    vertical_specific = vertical_specific_reports(
        dataset,
        Path(args.research_ai_retrieval_corpus),
        Path(args.research_ai_retrieval_manifest),
        Path(args.research_ai_retrieval_quality_report),
    )

    inventory["critical_issue_count"] = (
        alignment["critical_issue_count"] + safety["critical_issue_count"]
    )
    inventory["warning_count"] = (
        alignment["warning_count"] + safety["warning_count"] + len(missing_files)
    )
    inventory["vertical_specific"] = vertical_specific

    rows = summary_rows(inventory, evidence, workload, prompt, gold, kb)
    write_json(output_dir / "phase2a_10000_dataset_inventory.json", inventory)
    write_csv(
        output_dir / "phase2a_10000_dataset_summary.csv",
        rows,
        [
            "vertical",
            "prompt_count",
            "gold_count",
            "kb_count",
            "evidence_coverage_rate",
            "average_evidence_ids_per_prompt",
            "unused_kb_share",
            "prompt_words_mean",
            "gold_words_mean",
            "kb_words_mean",
            "workload_pressure_score",
            "likely_inference_cost_pressure",
        ],
    )
    write_json(output_dir / "phase2a_prompt_profile.json", prompt)
    write_json(output_dir / "phase2a_kb_profile.json", kb)
    write_json(output_dir / "phase2a_gold_profile.json", gold)
    write_json(output_dir / "phase2a_alignment_report.json", alignment)
    write_json(output_dir / "phase2a_evidence_reuse_report.json", evidence)
    write_json(output_dir / "phase2a_safety_report.json", safety)
    write_json(output_dir / "phase2a_workload_shape_report.json", workload)

    figures = build_plotly_figures(inventory, prompt_rows, gold_rows, kb_rows, evidence, workload)
    write_interactive_outputs(figures, output_dir)
    write_static_plots(output_dir, inventory, prompt_rows, gold_rows, kb_rows, evidence, workload)
    write_word_views(output_dir, word_profile)
    write_dashboard(
        output_dir,
        inventory,
        alignment,
        safety,
        figures,
        vertical_specific["research_ai"],
    )
    write_vertical_pages(
        output_dir,
        dataset,
        prompt_rows,
        gold_rows,
        kb_rows,
        evidence,
        word_profile,
        vertical_specific,
    )
    write_top_level_summary(
        output_dir,
        inventory,
        alignment,
        evidence,
        safety,
        workload,
        vertical_specific,
    )
    write_eda_readme(output_dir)

    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "total_prompt_count": inventory["total_prompt_count"],
        "total_gold_count": inventory["total_gold_count"],
        "total_kb_count": inventory["total_kb_count"],
        "vertical_count": inventory["vertical_count"],
        "critical_issue_count": inventory["critical_issue_count"],
        "warning_count": inventory["warning_count"],
        "research_ai_full_retrieval_corpus_available": vertical_specific["research_ai"][
            "full_retrieval_corpus_exists"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--research-ai-retrieval-corpus", default=str(DEFAULT_RESEARCH_AI_CORPUS))
    parser.add_argument(
        "--research-ai-retrieval-manifest", default=str(DEFAULT_RESEARCH_AI_MANIFEST)
    )
    parser.add_argument(
        "--research-ai-retrieval-quality-report",
        default=str(DEFAULT_RESEARCH_AI_QUALITY),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.write_report:
        parser.error("Pass --write-report to generate Phase 2A promoted dataset EDA.")
    try:
        summary = write_reports(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
