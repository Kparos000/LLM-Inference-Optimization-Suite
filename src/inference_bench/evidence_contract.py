"""Evidence contract utilities for retrieval-to-generation handoff.

The contract describes selected evidence passed to future generation runs. It
does not include gold labels and does not call models or external services.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from inference_bench.context_schema import VALID_VERTICALS
from inference_bench.retrieval import RetrievalResult

FORBIDDEN_EVIDENCE_CONTRACT_KEYS = {
    "gold",
    "gold_evidence_ids",
    "required_doc_ids",
    "required_evidence_ids",
    "required_chunk_ids",
}


@dataclass(frozen=True)
class EvidenceContractItem:
    """One selected evidence item for future generation."""

    evidence_id: str
    source_id: str
    parent_id: str
    vertical: str
    title: str
    section_type: str
    company: str
    ticker: str
    metric: str
    concept: str
    period: str
    retrieval_score: float
    rerank_score: float
    selection_reason: str
    evidence_text: str
    citation_label: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            msg = "evidence_id must be non-empty"
            raise ValueError(msg)
        if self.vertical not in VALID_VERTICALS:
            msg = f"vertical must be one of: {', '.join(sorted(VALID_VERTICALS))}"
            raise ValueError(msg)
        if not self.evidence_text.strip():
            msg = "evidence_text must be non-empty"
            raise ValueError(msg)
        if any(key in asdict(self) for key in FORBIDDEN_EVIDENCE_CONTRACT_KEYS):
            msg = "evidence contract must not include gold labels"
            raise ValueError(msg)


def metadata_value(metadata: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty scalar metadata value for the supplied keys."""

    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def evidence_contract_from_result(
    result: RetrievalResult,
    *,
    selection_reason: str,
) -> EvidenceContractItem:
    """Build an evidence contract item from a retrieval result."""

    record = result.context_record
    metadata = record.metadata
    evidence_id = record.context_id or record.chunk_id or record.source_id
    section_type = metadata_value(metadata, "section_type", "section_title", "document_type")
    company = metadata_value(metadata, "company_name", "company")
    ticker = metadata_value(metadata, "ticker")
    concept = metadata_value(metadata, "concept", "concepts", "label")
    metric = metadata_value(metadata, "metric", "metric_name", "concept", "label")
    period = metadata_value(
        metadata,
        "period",
        "report_date",
        "fiscal_year",
        "fiscal_periods_present",
        "latest_end",
    )
    citation_label = f"{record.vertical}:{evidence_id}"
    return EvidenceContractItem(
        evidence_id=evidence_id,
        source_id=record.source_id,
        parent_id=record.parent_id,
        vertical=record.vertical,
        title=record.title,
        section_type=section_type,
        company=company,
        ticker=ticker,
        metric=metric,
        concept=concept,
        period=period,
        retrieval_score=round(float(result.score), 6),
        rerank_score=round(float(result.component_scores.get("rerank_score", result.score)), 6),
        selection_reason=selection_reason,
        evidence_text=record.text,
        citation_label=citation_label,
    )


def evidence_contracts_from_results(
    results: list[RetrievalResult],
    *,
    selection_reasons_by_context_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return validated evidence contract dictionaries for selected results."""

    reasons = selection_reasons_by_context_id or {}
    contracts = [
        evidence_contract_from_result(
            result,
            selection_reason=reasons.get(
                result.context_record.context_id,
                "hybrid_score_rank",
            ),
        )
        for result in results
    ]
    return [asdict(item) for item in contracts]


def validate_evidence_contract(payload: dict[str, Any]) -> None:
    """Validate one evidence contract payload."""

    forbidden = set(payload) & FORBIDDEN_EVIDENCE_CONTRACT_KEYS
    if forbidden:
        msg = f"evidence contract contains forbidden gold-label fields: {sorted(forbidden)}"
        raise ValueError(msg)
    EvidenceContractItem(**payload)


def build_evidence_selection_report(
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build evidence selection diagnostics from workload evaluation rows."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in evaluation_rows:
        key = (
            str(row["split"]),
            str(row["ablation_mode"]),
            str(row["memory_mode"]),
            str(row["vertical"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    by_split: dict[str, Any] = {}
    for (split, ablation_mode, memory_mode, vertical), rows in sorted(grouped.items()):
        selected_counts = [int(row.get("context_rows_selected") or 0) for row in rows]
        exact_top5_count = sum(1 for count in selected_counts if count == 5)
        strategy_values = sorted(
            {str(row.get("evidence_selector_strategy") or "unavailable") for row in rows}
        )
        valid_count = sum(1 for row in rows if bool(row.get("evidence_contract_valid")))
        duplicate_avoidance_count = sum(
            1 for row in rows if bool(row.get("duplicate_avoidance_applied"))
        )
        reason_counter: dict[str, int] = {}
        for row in rows:
            for reason in row.get("selection_reasons", []):
                reason_counter[str(reason)] = reason_counter.get(str(reason), 0) + 1
        payload = {
            "record_count": len(rows),
            "evidence_selector_strategy": ",".join(strategy_values),
            "exact_top5_count": exact_top5_count,
            "avg_selected_evidence_count": round(mean(selected_counts), 6)
            if selected_counts
            else 0.0,
            "duplicate_avoidance_applied_count": duplicate_avoidance_count,
            "evidence_contract_valid_count": valid_count,
            "selection_reason_sample": sorted(reason_counter, key=reason_counter.get, reverse=True)[
                :5
            ],
        }
        by_split.setdefault(split, {}).setdefault(ablation_mode, {}).setdefault(memory_mode, {})[
            vertical
        ] = payload
        summary_rows.append(
            {
                "split": split,
                "ablation_mode": ablation_mode,
                "memory_mode": memory_mode,
                "vertical": vertical,
                "record_count": payload["record_count"],
                "evidence_selector_strategy": payload["evidence_selector_strategy"],
                "exact_top5_count": payload["exact_top5_count"],
                "avg_selected_evidence_count": payload["avg_selected_evidence_count"],
                "duplicate_avoidance_applied_count": payload["duplicate_avoidance_applied_count"],
                "evidence_contract_valid_count": payload["evidence_contract_valid_count"],
                "selection_reason_sample": ",".join(payload["selection_reason_sample"]),
            }
        )

    report = {
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "contract_excludes_gold_labels": True,
        "selector_strategies": [
            "calibrated_top5",
            "finance_calibrated_top5",
            "finance_diverse_metric_period_top5",
            "oracle_diagnostic_only",
        ],
        "oracle_strategy_used_for_final_selection": False,
        "by_split": by_split,
    }
    return report, summary_rows


EVIDENCE_SELECTION_SUMMARY_FIELDS = [
    "split",
    "ablation_mode",
    "memory_mode",
    "vertical",
    "record_count",
    "evidence_selector_strategy",
    "exact_top5_count",
    "avg_selected_evidence_count",
    "duplicate_avoidance_applied_count",
    "evidence_contract_valid_count",
    "selection_reason_sample",
]
