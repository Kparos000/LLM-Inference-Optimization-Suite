"""Generate EDA reports for the promoted Phase 2A dataset.

This script profiles committed benchmark data only. It does not build RAG,
retrieval indexes, embeddings, prompt assembly, model calls, GPU runs, or
inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

PHASE = "2A-16C"
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]
FILE_KINDS = ["prompts", "gold", "kb"]
DEFAULT_DATASET_ROOT = Path("data/scaleup_2000_full")
DEFAULT_OUTPUT_DIR = Path("data/generated/phase2a/eda")
DEFAULT_RESEARCH_AI_CORPUS = Path(
    "data/generated/phase2a/retrieval_corpus/research_ai/research_ai_full_sections_corpus.jsonl"
)

HYGIENE_PATTERNS = [
    (re.compile(pattern, flags=re.IGNORECASE), label)
    for pattern, label in [
        (r"C:\\Users", "private Windows path"),
        (r"/home/", "private Unix path"),
        (r"akpoogaga", "private username"),
        (r"kparo", "private username"),
        (r"API key", "API key reference"),
        (r"\btoken\b", "token reference"),
        (r"\bsecret\b", "secret reference"),
        (r"\bpassword\b", "password reference"),
        (r"raw user_id", "raw user identifier"),
    ]
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


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


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_count(text: str) -> int:
    return len(words(text))


def percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0, "mean": 0}
    sorted_values = sorted(values)

    def pct(p: float) -> float:
        index = min(len(sorted_values) - 1, max(0, math.ceil((p / 100) * len(sorted_values)) - 1))
        return float(sorted_values[index])

    return {
        "min": float(sorted_values[0]),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": float(sorted_values[-1]),
        "mean": round(mean(sorted_values), 3),
    }


def ngram_counts(texts: list[str], n: int, limit: int = 25) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = words(text)
        for index in range(0, max(0, len(tokens) - n + 1)):
            gram = " ".join(tokens[index : index + n])
            if gram:
                counter[gram] += 1
    return [{"term": term, "count": count} for term, count in counter.most_common(limit)]


def normalized_question(text: str) -> str:
    text = re.sub(r"\d+", "<num>", text.lower())
    text = re.sub(r"\b[a-z]{2,}[_-][a-z0-9_-]+\b", "<id>", text)
    return " ".join(text.split())


def evidence_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in ["required_doc_ids", "required_evidence_ids", "source_doc_ids"]:
        value = row.get(field)
        if isinstance(value, list):
            ids.extend(str(item) for item in value if item)
    return list(dict.fromkeys(ids))


def nested_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


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
    files = dataset_files(dataset_root)
    dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    file_paths: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for vertical, paths in files.items():
        dataset[vertical] = {}
        file_paths[vertical] = {}
        for kind, path in paths.items():
            file_paths[vertical][kind] = str(path)
            if not path.exists():
                missing.append(str(path))
            dataset[vertical][kind] = read_jsonl(path)
    return dataset, file_paths, missing


def inventory_report(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    file_paths: dict[str, dict[str, str]],
    missing: list[str],
) -> dict[str, Any]:
    per_vertical = {
        vertical: {
            "prompt_count": len(records["prompts"]),
            "gold_count": len(records["gold"]),
            "kb_count": len(records["kb"]),
            "files": file_paths[vertical],
        }
        for vertical, records in dataset.items()
    }
    return {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "dataset_name": "phase2a_10000_promoted_2000_full",
        "dataset_root": str(DEFAULT_DATASET_ROOT),
        "prompt_count_by_vertical": {v: row["prompt_count"] for v, row in per_vertical.items()},
        "gold_count_by_vertical": {v: row["gold_count"] for v, row in per_vertical.items()},
        "kb_count_by_vertical": {v: row["kb_count"] for v, row in per_vertical.items()},
        "total_prompt_count": sum(row["prompt_count"] for row in per_vertical.values()),
        "total_gold_count": sum(row["gold_count"] for row in per_vertical.values()),
        "total_kb_count": sum(row["kb_count"] for row in per_vertical.values()),
        "file_paths": file_paths,
        "missing_files": missing,
        "per_vertical": per_vertical,
    }


def prompt_profile(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    for vertical, records in dataset.items():
        prompts = records["prompts"]
        questions = [str(row.get("question") or "") for row in prompts]
        normalized = [normalized_question(question) for question in questions]
        duplicate_count = sum(count - 1 for count in Counter(normalized).values() if count > 1)
        by_vertical[vertical] = {
            "prompt_word_count_distribution": percentiles([word_count(item) for item in questions]),
            "prompt_character_length_distribution": percentiles([len(item) for item in questions]),
            "estimated_token_count_distribution": percentiles(
                [max(1, len(item) // 4) for item in questions]
            ),
            "expected_status_distribution": dict(
                Counter(str(row.get("expected_status") or "") for row in prompts)
            ),
            "expected_output_format_distribution": dict(
                Counter(str(row.get("expected_output_format") or "") for row in prompts)
            ),
            "task_type_distribution": dict(
                Counter(str(row.get("task_type") or "") for row in prompts)
            ),
            "difficulty_distribution": dict(
                Counter(str(nested_metadata(row).get("difficulty") or "") for row in prompts)
            ),
            "top_unigrams": ngram_counts(questions, 1),
            "top_bigrams": ngram_counts(questions, 2),
            "top_trigrams": ngram_counts(questions, 3),
            "duplicate_question_count": duplicate_count,
            "near_duplicate_question_count": duplicate_count,
            "linguistic_variation_estimate": 1.0
            - ((max(Counter(normalized).values()) / len(normalized)) if normalized else 0),
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def gold_profile(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    mechanical_phrases = ["use cited", "to answer about", "keep the answer limited"]
    for vertical, records in dataset.items():
        gold = records["gold"]
        answers = [str(row.get("reference_answer") or "") for row in gold]
        by_vertical[vertical] = {
            "gold_count": len(gold),
            "expected_status_distribution": dict(
                Counter(str(row.get("expected_status") or "") for row in gold)
            ),
            "reference_answer_word_count_distribution": percentiles(
                [word_count(item) for item in answers]
            ),
            "must_include_count_distribution": percentiles(
                [
                    len(row.get("must_include", []))
                    if isinstance(row.get("must_include"), list)
                    else 0
                    for row in gold
                ]
            ),
            "must_not_include_count_distribution": percentiles(
                [
                    len(row.get("must_not_include", []))
                    if isinstance(row.get("must_not_include"), list)
                    else 0
                    for row in gold
                ]
            ),
            "required_evidence_count_distribution": percentiles(
                [len(evidence_ids(row)) for row in gold]
            ),
            "empty_reference_answer_count": sum(1 for answer in answers if not answer.strip()),
            "mechanical_phrase_hits": {
                phrase: sum(1 for answer in answers if phrase in answer.lower())
                for phrase in mechanical_phrases
            },
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def kb_profile(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    all_gold = {vertical: records["gold"] for vertical, records in dataset.items()}
    for vertical, records in dataset.items():
        kb = records["kb"]
        doc_ids = [str(row.get("doc_id") or "") for row in kb]
        referenced = Counter()
        for gold in all_gold[vertical]:
            referenced.update(evidence_ids(gold))
        unused = [doc_id for doc_id in doc_ids if doc_id and doc_id not in referenced]
        body_texts = [flatten_text(row.get("body") or row) for row in kb]
        by_vertical[vertical] = {
            "kb_count": len(kb),
            "kb_word_count_distribution": percentiles([word_count(text) for text in body_texts]),
            "document_type_distribution": dict(
                Counter(str(row.get("document_type") or "") for row in kb)
            ),
            "source_type_distribution": dict(
                Counter(str(row.get("source_type") or "") for row in kb)
            ),
            "largest_kb_rows": [
                {
                    "doc_id": str(row.get("doc_id") or ""),
                    "word_count": word_count(flatten_text(row.get("body") or row)),
                }
                for row in sorted(
                    kb,
                    key=lambda item: word_count(flatten_text(item.get("body") or item)),
                    reverse=True,
                )[:10]
            ],
            "shortest_kb_rows": [
                {
                    "doc_id": str(row.get("doc_id") or ""),
                    "word_count": word_count(flatten_text(row.get("body") or row)),
                }
                for row in sorted(
                    kb, key=lambda item: word_count(flatten_text(item.get("body") or item))
                )[:10]
            ],
            "duplicate_doc_id_count": sum(
                count - 1 for count in Counter(doc_ids).values() if count > 1
            ),
            "unique_evidence_id_count": len(set(doc_ids)),
            "unused_kb_count": len(unused),
            "most_referenced_kb_records": [
                {"doc_id": doc_id, "reference_count": count}
                for doc_id, count in referenced.most_common(25)
            ],
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def alignment_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    critical_count = 0
    for vertical, records in dataset.items():
        prompt_ids = [str(row.get("prompt_id") or "") for row in records["prompts"]]
        gold_ids = [str(row.get("prompt_id") or "") for row in records["gold"]]
        prompt_set = set(prompt_ids)
        gold_set = set(gold_ids)
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
        row = {
            "missing_gold_for_prompts": sorted(prompt_set - gold_set),
            "orphan_gold_records": sorted(gold_set - prompt_set),
            "duplicate_prompt_ids": [
                item for item, count in Counter(prompt_ids).items() if count > 1
            ],
            "duplicate_gold_prompt_ids": [
                item for item, count in Counter(gold_ids).items() if count > 1
            ],
            "answerable_records_without_evidence": answerable_without_evidence,
            "negative_records_without_must_not_include": negative_without_must_not,
        }
        critical_count += sum(len(value) for value in row.values())
        by_vertical[vertical] = row
    return {
        "phase": PHASE,
        "critical_issue_count": critical_count,
        "alignment_clean": critical_count == 0,
        "by_vertical": by_vertical,
    }


def evidence_reuse_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    for vertical, records in dataset.items():
        kb_ids = {str(row.get("doc_id") or "") for row in records["kb"] if row.get("doc_id")}
        referenced = Counter()
        evidence_count_per_prompt: list[int] = []
        for gold in records["gold"]:
            ids = evidence_ids(gold)
            evidence_count_per_prompt.append(len(ids))
            referenced.update(ids)
        max_reuse = max(referenced.values()) if referenced else 0
        by_vertical[vertical] = {
            "gold_count": len(records["gold"]),
            "evidence_coverage_rate": round(
                sum(1 for count in evidence_count_per_prompt if count > 0) / len(records["gold"]),
                4,
            )
            if records["gold"]
            else 0,
            "average_evidence_ids_per_prompt": round(mean(evidence_count_per_prompt), 3)
            if evidence_count_per_prompt
            else 0,
            "top_reused_evidence_ids": [
                {"evidence_id": evidence_id, "reference_count": count}
                for evidence_id, count in referenced.most_common(25)
            ],
            "max_evidence_reuse_share": round(max_reuse / len(records["gold"]), 4)
            if records["gold"]
            else 0,
            "unused_kb_count": len(kb_ids - set(referenced)),
            "referenced_kb_count": len(set(referenced)),
            "evidence_reuse_concentration": "high"
            if records["gold"] and max_reuse / len(records["gold"]) > 0.20
            else "medium"
            if records["gold"] and max_reuse / len(records["gold"]) > 0.10
            else "low",
        }
    return {"phase": PHASE, "by_vertical": by_vertical}


def safety_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    total_issues = 0
    for vertical, records in dataset.items():
        combined = "\n".join(flatten_text(records[kind]) for kind in FILE_KINDS)
        hygiene_hits = {
            label: len(pattern.findall(combined))
            for pattern, label in HYGIENE_PATTERNS
            if pattern.search(combined)
        }
        lower = combined.lower()
        domain_flags = {
            "finance_investment_advice": vertical == "finance" and "investment advice" in lower,
            "healthcare_clinical_advice": vertical == "healthcare_admin"
            and "diagnose the patient" in lower,
            "retail_raw_user_id": vertical == "retail" and "raw user_id" in lower,
            "retail_generic_title": vertical == "retail" and "all_beauty product <asin>" in lower,
            "research_ai_general_memory": vertical == "research_ai"
            and "according to my knowledge" in lower,
            "airline_unsupported_compensation": vertical == "airline"
            and "guaranteed cash compensation" in lower,
        }
        issue_count = len(hygiene_hits) + sum(1 for value in domain_flags.values() if value)
        total_issues += issue_count
        by_vertical[vertical] = {
            "hygiene_hits": hygiene_hits,
            "domain_flags": domain_flags,
            "issue_count": issue_count,
        }
    return {
        "phase": PHASE,
        "critical_issue_count": total_issues,
        "safety_clean": total_issues == 0,
        "by_vertical": by_vertical,
    }


def workload_shape_report(dataset: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    by_vertical: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for vertical, records in dataset.items():
        prompt_lengths = [len(str(row.get("question") or "")) // 4 for row in records["prompts"]]
        kb_lengths = [word_count(flatten_text(row.get("body") or row)) for row in records["kb"]]
        evidence_counts = [len(evidence_ids(row)) for row in records["gold"]]
        row = {
            "estimated_input_token_buckets": dict(
                Counter(
                    "<128" if value < 128 else "128-512" if value < 512 else "512+"
                    for value in prompt_lengths
                )
            ),
            "expected_output_format_mix": dict(
                Counter(str(row.get("expected_output_format") or "") for row in records["prompts"])
            ),
            "prompt_length_distribution": percentiles(prompt_lengths),
            "kb_length_distribution": percentiles(kb_lengths),
            "single_evidence_prompt_count": sum(1 for value in evidence_counts if value == 1),
            "multi_evidence_prompt_count": sum(1 for value in evidence_counts if value > 1),
            "average_evidence_ids_per_prompt": round(mean(evidence_counts), 3)
            if evidence_counts
            else 0,
        }
        by_vertical[vertical] = row
        ranking.append(
            {
                "vertical": vertical,
                "average_kb_word_count": row["kb_length_distribution"]["mean"],
                "average_evidence_ids_per_prompt": row["average_evidence_ids_per_prompt"],
            }
        )
    return {
        "phase": PHASE,
        "by_vertical": by_vertical,
        "context_heavy_vertical_ranking": sorted(
            ranking,
            key=lambda item: (
                item["average_evidence_ids_per_prompt"],
                item["average_kb_word_count"],
            ),
            reverse=True,
        ),
    }


def vertical_specific_reports(
    dataset: dict[str, dict[str, list[dict[str, Any]]]],
    research_ai_corpus: Path,
) -> dict[str, Any]:
    finance_kb = dataset["finance"]["kb"]
    research_kb = dataset["research_ai"]["kb"]
    retail_prompts = dataset["retail"]["prompts"]
    airline_prompts = dataset["airline"]["prompts"]
    healthcare_prompts = dataset["healthcare_admin"]["prompts"]
    corpus_rows = read_jsonl(research_ai_corpus) if research_ai_corpus.exists() else []
    return {
        "finance": {
            "ticker_counts": dict(
                Counter(str(nested_metadata(row).get("ticker") or "") for row in finance_kb)
            ),
            "filing_form_counts": dict(
                Counter(str(nested_metadata(row).get("form") or "") for row in finance_kb)
            ),
            "xbrl_concept_coverage": len(
                {
                    str(nested_metadata(row).get("concept") or "")
                    for row in finance_kb
                    if nested_metadata(row).get("concept")
                }
            ),
            "document_type_counts": dict(
                Counter(str(row.get("document_type") or "") for row in finance_kb)
            ),
        },
        "research_ai": {
            "promoted_benchmark_kb_count": len(research_kb),
            "full_retrieval_corpus_exists": research_ai_corpus.exists(),
            "full_retrieval_corpus_count": len(corpus_rows),
            "paper_coverage": len(
                {str(nested_metadata(row).get("paper_id") or "") for row in research_kb}
            ),
            "section_type_distribution": dict(
                Counter(
                    str(
                        nested_metadata(row).get("section_type")
                        or nested_metadata(row).get("evidence_type")
                        or ""
                    )
                    for row in research_kb
                )
            ),
            "paper_topic_distribution": dict(
                Counter(str(nested_metadata(row).get("topic") or "") for row in research_kb)
            ),
            "percent_full_corpus_represented_in_promoted_kb": round(
                len(research_kb) / len(corpus_rows), 4
            )
            if corpus_rows
            else None,
        },
        "retail": {
            "category_counts": dict(
                Counter(
                    str(row.get("category") or nested_metadata(row).get("category") or "")
                    for row in retail_prompts
                )
            ),
            "product_title_coverage": sum(1 for row in retail_prompts if row.get("product_title")),
            "issue_type_counts": dict(
                Counter(str(row.get("issue_type") or "") for row in retail_prompts)
            ),
        },
        "airline": {
            "policy_category_counts": dict(
                Counter(str(row.get("support_type") or "") for row in airline_prompts)
            ),
            "escalation_count": sum(
                1 for row in airline_prompts if row.get("expected_status") == "escalate"
            ),
            "fraud_count": sum(
                1 for row in airline_prompts if row.get("expected_status") == "spam_or_fraud"
            ),
        },
        "healthcare_admin": {
            "admin_category_counts": dict(
                Counter(str(row.get("support_type") or "") for row in healthcare_prompts)
            ),
            "privacy_identity_count": sum(
                1 for row in healthcare_prompts if row.get("privacy_sensitive")
            ),
            "safety_boundary_count": sum(
                1 for row in healthcare_prompts if row.get("expected_status") == "safety_boundary"
            ),
        },
    }


def write_word_views(output_dir: Path, prompt: dict[str, Any]) -> None:
    word_dir = output_dir / "word_views"
    word_dir.mkdir(parents=True, exist_ok=True)
    for vertical, profile in prompt["by_vertical"].items():
        lines = [f"# {vertical} prompt word views", "", "Top unigrams:"]
        lines.extend(f"- {item['term']}: {item['count']}" for item in profile["top_unigrams"][:20])
        lines.append("")
        lines.append("Top bigrams:")
        lines.extend(f"- {item['term']}: {item['count']}" for item in profile["top_bigrams"][:20])
        (word_dir / f"{vertical}_prompt_terms.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def maybe_write_plots(output_dir: Path, enabled: bool) -> None:
    if not enabled:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        (plot_dir / "plots_skipped.txt").write_text(
            "matplotlib is not available; JSON and CSV EDA reports were still generated.\n",
            encoding="utf-8",
        )
        return
    fig = plt.figure(figsize=(4, 2))
    plt.title("Phase 2A EDA")
    plt.text(0.1, 0.5, "See JSON/CSV reports for full metrics.")
    plt.axis("off")
    fig.savefig(plot_dir / "phase2a_eda_overview.png", bbox_inches="tight")
    plt.close(fig)


def build_reports(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    dataset, paths, missing = load_dataset(dataset_root)
    inventory = inventory_report(dataset, paths, missing)
    inventory["dataset_root"] = str(dataset_root)
    prompt = prompt_profile(dataset)
    kb = kb_profile(dataset)
    gold = gold_profile(dataset)
    alignment = alignment_report(dataset)
    evidence = evidence_reuse_report(dataset)
    safety = safety_report(dataset)
    workload = workload_shape_report(dataset)
    vertical_specific = vertical_specific_reports(dataset, Path(args.research_ai_retrieval_corpus))

    summary_rows = [
        {
            "vertical": vertical,
            "prompt_count": len(records["prompts"]),
            "gold_count": len(records["gold"]),
            "kb_count": len(records["kb"]),
            "alignment_clean": alignment["by_vertical"][vertical]
            == {
                "missing_gold_for_prompts": [],
                "orphan_gold_records": [],
                "duplicate_prompt_ids": [],
                "duplicate_gold_prompt_ids": [],
                "answerable_records_without_evidence": [],
                "negative_records_without_must_not_include": [],
            },
            "safety_issue_count": safety["by_vertical"][vertical]["issue_count"],
            "max_evidence_reuse_share": evidence["by_vertical"][vertical][
                "max_evidence_reuse_share"
            ],
        }
        for vertical, records in dataset.items()
    ]

    critical_issue_count = alignment["critical_issue_count"] + safety["critical_issue_count"]
    warning_count = len(missing)
    top_level = {
        "phase": PHASE,
        "generated_at_utc": utc_now(),
        "total_prompt_count": inventory["total_prompt_count"],
        "total_gold_count": inventory["total_gold_count"],
        "total_kb_count": inventory["total_kb_count"],
        "critical_issue_count": critical_issue_count,
        "warning_count": warning_count,
        "eda_ready_for_phase2b": critical_issue_count == 0 and warning_count == 0,
        "recommended_next_step": (
            "Use EDA findings to guide Phase 2B context engineering and retrieval experiments."
        ),
        "vertical_specific": vertical_specific,
    }
    inventory.update(top_level)

    write_json(output_dir / "phase2a_10000_dataset_inventory.json", inventory)
    write_csv(
        output_dir / "phase2a_10000_dataset_summary.csv",
        summary_rows,
        [
            "vertical",
            "prompt_count",
            "gold_count",
            "kb_count",
            "alignment_clean",
            "safety_issue_count",
            "max_evidence_reuse_share",
        ],
    )
    write_json(output_dir / "phase2a_prompt_profile.json", prompt)
    write_json(output_dir / "phase2a_kb_profile.json", kb)
    write_json(output_dir / "phase2a_gold_profile.json", gold)
    write_json(output_dir / "phase2a_alignment_report.json", alignment)
    write_json(output_dir / "phase2a_evidence_reuse_report.json", evidence)
    write_json(output_dir / "phase2a_safety_report.json", safety)
    write_json(output_dir / "phase2a_workload_shape_report.json", workload)
    write_json(output_dir / "phase2a_vertical_specific_report.json", vertical_specific)
    if args.make_word_views:
        write_word_views(output_dir, prompt)
    maybe_write_plots(output_dir, bool(args.make_plots))
    return top_level


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-word-views", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--research-ai-retrieval-corpus", default=str(DEFAULT_RESEARCH_AI_CORPUS))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.write_report:
        parser.error("Pass --write-report to generate Phase 2A promoted dataset EDA.")
    try:
        summary = build_reports(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
