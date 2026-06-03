"""Gold-evidence and corpus alignment audit for retrieval quality gates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from inference_bench.context_schema import ContextRecord
from inference_bench.retrieval import (
    FINANCE_METRIC_TERMS,
    context_match_ids,
    normalize_identifier,
    tokenize,
)

FINANCE_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")


def prompt_text(prompt: dict[str, Any]) -> str:
    """Return visible prompt text plus public prompt metadata."""

    parts = [
        str(prompt.get("question") or ""),
        str(prompt.get("issue") or ""),
        str(prompt.get("company") or ""),
        str(prompt.get("ticker") or ""),
        str(prompt.get("filing_form") or ""),
    ]
    return " ".join(part for part in parts if part)


def finance_prompt_has_entity(prompt: dict[str, Any]) -> bool:
    """Return whether a finance prompt names a visible company or ticker."""

    text = prompt_text(prompt)
    if FINANCE_TICKER_RE.search(text):
        return True
    return any(
        term in text.lower()
        for term in (
            "apple",
            "microsoft",
            "nvidia",
            "tesla",
            "amazon",
            "alphabet",
            "google",
            "meta",
            "amd",
        )
    )


def finance_prompt_has_metric(prompt: dict[str, Any]) -> bool:
    """Return whether a finance prompt exposes a metric term."""

    tokens = set(tokenize(prompt_text(prompt)))
    return bool(tokens & FINANCE_METRIC_TERMS)


def finance_prompt_has_period(prompt: dict[str, Any]) -> bool:
    """Return whether a finance prompt exposes a year, quarter, or period phrase."""

    text = prompt_text(prompt).lower()
    return bool(re.search(r"\b20\d{2}\b", text)) or any(
        term in text for term in ("q1", "q2", "q3", "q4", "quarter", "annual", "fiscal")
    )


def corpus_ids_by_vertical(
    corpora_by_vertical: dict[str, list[ContextRecord]],
) -> dict[str, set[str]]:
    """Return all context match IDs by vertical."""

    return {
        vertical: {match_id for record in records for match_id in context_match_ids(record)}
        for vertical, records in corpora_by_vertical.items()
    }


def vertical_prompt_lookup(
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return prompt records by vertical and prompt ID."""

    return {
        vertical: {str(row.get("prompt_id")): row for row in rows}
        for vertical, rows in prompts_by_vertical.items()
    }


def gold_ids_from_gold_record(gold_record: dict[str, Any]) -> list[str]:
    """Return unique gold evidence IDs from one gold row."""

    values: list[str] = []
    for field_name in ("required_doc_ids", "required_evidence_ids", "required_chunk_ids"):
        value = gold_record.get(field_name)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return list(dict.fromkeys(values))


def build_gold_evidence_audit_report(
    *,
    prompts_by_vertical: dict[str, list[dict[str, Any]]],
    gold_by_vertical: dict[str, dict[str, dict[str, Any]]],
    corpora_by_vertical: dict[str, list[ContextRecord]],
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the gold/corpus alignment audit report."""

    known_ids = corpus_ids_by_vertical(corpora_by_vertical)
    match_ids_by_context_id = {
        vertical: {record.context_id: context_match_ids(record) for record in records}
        for vertical, records in corpora_by_vertical.items()
    }
    prompts_lookup = vertical_prompt_lookup(prompts_by_vertical)
    eval_rows_by_vertical: dict[str, list[dict[str, Any]]] = {}
    for row in evaluation_rows:
        if (
            row.get("split") == "final_10000"
            and row.get("memory_mode") == "mm2_hybrid_top5"
            and row.get("ablation_mode") in {"prompt_text_only", "prompt_plus_metadata"}
        ):
            eval_rows_by_vertical.setdefault(str(row.get("vertical")), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    by_vertical: dict[str, Any] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for vertical, gold_records in sorted(gold_by_vertical.items()):
        prompt_rows = prompts_lookup.get(vertical, {})
        missing_gold_ids_count = 0
        gold_not_in_corpus_count = 0
        prompt_missing_entity_count = 0
        prompt_missing_metric_count = 0
        prompt_missing_period_count = 0
        suspected_gold_misalignment_count = 0
        chunk_token_counts = [
            record.token_estimate for record in corpora_by_vertical.get(vertical, [])
        ]
        vertical_examples: list[dict[str, Any]] = []

        for prompt_id, gold_record in gold_records.items():
            gold_ids = gold_ids_from_gold_record(gold_record)
            prompt = prompt_rows.get(prompt_id, {})
            if not gold_ids:
                missing_gold_ids_count += 1
            missing_ids = [
                evidence_id
                for evidence_id in gold_ids
                if evidence_id not in known_ids.get(vertical, set())
            ]
            gold_not_in_corpus_count += len(missing_ids)
            if missing_ids and len(vertical_examples) < 10:
                vertical_examples.append(
                    {
                        "prompt_id": prompt_id,
                        "missing_gold_ids": missing_ids,
                        "audit_reason": "gold_id_not_found_in_context_corpus",
                    }
                )
            if vertical == "finance":
                if not finance_prompt_has_entity(prompt):
                    prompt_missing_entity_count += 1
                if not finance_prompt_has_metric(prompt):
                    prompt_missing_metric_count += 1
                if not finance_prompt_has_period(prompt):
                    prompt_missing_period_count += 1
                if missing_ids and not finance_prompt_has_period(prompt):
                    suspected_gold_misalignment_count += 1

        row_counter: Counter[str] = Counter()
        for row in eval_rows_by_vertical.get(vertical, []):
            gold_ids = [str(item) for item in row.get("gold_evidence_ids", [])]
            candidate_ids = [
                str(item)
                for item in row.get("candidate_context_ids", [])
                if item in match_ids_by_context_id.get(vertical, {})
            ]
            top5_includes_gold = candidate_window_includes_gold(
                candidate_ids=candidate_ids,
                gold_ids=gold_ids,
                match_ids_by_context_id=match_ids_by_context_id.get(vertical, {}),
                top_k=5,
            )
            top50_includes_gold = candidate_window_includes_gold(
                candidate_ids=candidate_ids,
                gold_ids=gold_ids,
                match_ids_by_context_id=match_ids_by_context_id.get(vertical, {}),
                top_k=50,
            )
            top100_includes_gold = candidate_window_includes_gold(
                candidate_ids=candidate_ids,
                gold_ids=gold_ids,
                match_ids_by_context_id=match_ids_by_context_id.get(vertical, {}),
                top_k=100,
            )
            if top50_includes_gold and not top5_includes_gold:
                row_counter["gold_in_top50_but_not_top5_count"] += 1
            if not top50_includes_gold:
                row_counter["gold_absent_from_top50_count"] += 1
            if not top100_includes_gold:
                row_counter["gold_absent_from_top100_count"] += 1

        avg_chunk_tokens = (
            round(sum(chunk_token_counts) / len(chunk_token_counts), 6)
            if chunk_token_counts
            else 0.0
        )
        payload = {
            "vertical": vertical,
            "gold_record_count": len(gold_records),
            "context_record_count": len(corpora_by_vertical.get(vertical, [])),
            "missing_gold_ids_count": missing_gold_ids_count,
            "gold_not_in_corpus_count": gold_not_in_corpus_count,
            "prompt_missing_entity_count": prompt_missing_entity_count,
            "prompt_missing_metric_count": prompt_missing_metric_count,
            "prompt_missing_period_count": prompt_missing_period_count,
            "gold_in_top50_but_not_top5_count": row_counter["gold_in_top50_but_not_top5_count"],
            "gold_absent_from_top50_count": row_counter["gold_absent_from_top50_count"],
            "gold_absent_from_top100_count": row_counter["gold_absent_from_top100_count"],
            "suspected_gold_misalignment_count": suspected_gold_misalignment_count,
            "avg_context_token_estimate": avg_chunk_tokens,
        }
        by_vertical[vertical] = payload
        examples[vertical] = vertical_examples
        summary_rows.append(payload)

    report = {
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "audit_scope": "gold_context_alignment_and_candidate_failure_modes",
        "by_vertical": by_vertical,
        "examples": examples,
        "diagnostic_interpretation": (
            "Large gold_in_top50_but_not_top5 counts indicate a final selector/reranking "
            "problem. Large gold_absent_from_top100 counts indicate candidate retrieval, "
            "chunking, prompt ambiguity, or gold/corpus alignment problems."
        ),
    }
    return report, summary_rows


def candidate_window_includes_gold(
    *,
    candidate_ids: list[str],
    gold_ids: list[str],
    match_ids_by_context_id: dict[str, set[str]],
    top_k: int,
) -> bool:
    """Return whether a candidate window contains any gold evidence ID."""

    if not gold_ids:
        return False
    gold_variants = {
        variant
        for evidence_id in gold_ids
        if evidence_id
        for variant in (evidence_id, normalize_identifier(evidence_id))
    }
    for context_id in candidate_ids[:top_k]:
        if match_ids_by_context_id.get(context_id, set()) & gold_variants:
            return True
    return False


GOLD_EVIDENCE_AUDIT_SUMMARY_FIELDS = [
    "vertical",
    "gold_record_count",
    "context_record_count",
    "missing_gold_ids_count",
    "gold_not_in_corpus_count",
    "prompt_missing_entity_count",
    "prompt_missing_metric_count",
    "prompt_missing_period_count",
    "gold_in_top50_but_not_top5_count",
    "gold_absent_from_top50_count",
    "gold_absent_from_top100_count",
    "suspected_gold_misalignment_count",
    "avg_context_token_estimate",
]
