"""Promote repaired retrieval validation artifacts as the canonical baseline.

This module only records and validates existing retrieval artifacts. It does
not rebuild corpora, run retrieval, run inference, use GPU, or call APIs.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from inference_bench.slo import SLO_VERTICALS

DEFAULT_CONTEXT_ROOT = Path("data/generated/context_engineering")
ACTIVE_DATASET_VARIANT = "repaired_generated"
ACTIVE_STAGE_SIZE = 2000
ACTIVE_ABLATION_MODE = "prompt_plus_metadata"
ACTIVE_DENSE_BACKEND = "qdrant_vector"
ACTIVE_VECTOR_STORE = "qdrant_local"
PROMOTION_REASON = (
    "All five repaired 2,000-record vertical retrieval validations pass "
    "candidate@20, candidate@50, final recall@5, and MRR SLOs with Qdrant-backed "
    "prompt_plus_metadata retrieval, leakage protections, and compression checks."
)


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a sorted JSON object."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read CSV rows from disk."""

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _is_active_repaired_row(row: dict[str, Any]) -> bool:
    """Return whether a repaired validation row belongs to the promoted baseline."""

    return (
        row.get("dataset_variant") == ACTIVE_DATASET_VARIANT
        and str(row.get("stage_size")) == str(ACTIVE_STAGE_SIZE)
        and row.get("ablation_mode") == ACTIVE_ABLATION_MODE
        and row.get("measurement") == "retrieval_dataset_alignment"
        and row.get("dense_backend") == ACTIVE_DENSE_BACKEND
        and row.get("vector_store") == ACTIVE_VECTOR_STORE
        and row.get("vertical") in SLO_VERTICALS
    )


def promoted_validation_rows(summary_path: str | Path) -> list[dict[str, Any]]:
    """Load active repaired 2,000-record validation rows."""

    rows: list[dict[str, Any]] = [
        dict(row) for row in read_csv_rows(summary_path) if _is_active_repaired_row(row)
    ]
    rows_by_vertical = {str(row["vertical"]): row for row in rows}
    missing = [vertical for vertical in SLO_VERTICALS if vertical not in rows_by_vertical]
    if missing:
        msg = "Promoted retrieval validation summary missing verticals: " + ", ".join(missing)
        raise ValueError(msg)
    return [rows_by_vertical[vertical] for vertical in SLO_VERTICALS]


def metrics_by_vertical(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build compact metric payloads by vertical."""

    metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        vertical = str(row["vertical"])
        metrics[vertical] = {
            "candidate_recall_at_20": float(row["candidate_recall_at_20"]),
            "candidate_recall_at_50": float(row["candidate_recall_at_50"]),
            "final_recall_at_5": float(row["final_recall_at_5"]),
            "mrr": float(row["mrr"]),
            "slo_status": str(row["slo_status"]),
            "record_count": int(row["record_count"]),
            "query_rewrite_count": int(row["query_rewrite_count"]),
            "dense_backend": str(row["dense_backend"]),
            "vector_store": str(row["vector_store"]),
        }
    return metrics


def all_promoted_rows_pass(rows: list[dict[str, Any]]) -> bool:
    """Return whether every active promoted row passed retrieval SLOs."""

    return len(rows) == len(SLO_VERTICALS) and all(
        row.get("slo_status") == "PASSED" for row in rows
    )


def _load_optional_json(path: Path) -> dict[str, Any]:
    """Load a JSON object if it exists, otherwise return an empty object."""

    return read_json(path) if path.exists() else {}


def build_retrieval_promotion_registry(
    *,
    context_root: str | Path = DEFAULT_CONTEXT_ROOT,
    promotion_timestamp_utc: str | None = None,
    promotion_reason: str = PROMOTION_REASON,
) -> dict[str, Any]:
    """Build the retrieval promotion registry payload."""

    root = Path(context_root)
    validation_summary_path = root / "repaired_retrieval_validation_summary.csv"
    promotion_plan_path = root / "repaired_retrieval_promotion_plan.json"
    rows = promoted_validation_rows(validation_summary_path)
    plan = _load_optional_json(promotion_plan_path)
    rows_pass = all_promoted_rows_pass(rows)
    plan_recommends = plan.get("promotion_recommended") is True
    promotion_status = "PROMOTED" if rows_pass and plan_recommends else "NOT_PROMOTED"
    timestamp = promotion_timestamp_utc or utc_now()
    return {
        "artifact_type": "retrieval_promotion_registry",
        "generated_at_utc": timestamp,
        "promotion_date_utc": timestamp,
        "promotion_reason": promotion_reason,
        "promotion_status": promotion_status,
        "promotion_recommended": bool(plan_recommends),
        "all_repaired_2000_slos_pass": rows_pass,
        "active_dataset_variant": ACTIVE_DATASET_VARIANT,
        "active_stage_size": ACTIVE_STAGE_SIZE,
        "active_ablation_mode": ACTIVE_ABLATION_MODE,
        "active_dense_backend": ACTIVE_DENSE_BACKEND,
        "active_vector_store": ACTIVE_VECTOR_STORE,
        "retrieval_metrics_by_vertical": metrics_by_vertical(rows),
        "source_artifacts": {
            "repaired_validation_summary": str(validation_summary_path),
            "repaired_validation_report": str(root / "repaired_retrieval_validation_report.json"),
            "repaired_promotion_plan": str(promotion_plan_path),
            "qdrant_index_summary": str(root / "qdrant_index_summary.csv"),
            "compression_diagnostic_summary": str(root / "compression_diagnostic_summary.csv"),
            "retrieval_quality_gate_report": str(root / "retrieval_quality_gate_report.json"),
        },
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
    }


def build_retrieval_source_of_truth_manifest(
    *,
    context_root: str | Path = DEFAULT_CONTEXT_ROOT,
    registry: dict[str, Any] | None = None,
    promotion_timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build the canonical retrieval source-of-truth manifest."""

    root = Path(context_root)
    active_registry = registry or build_retrieval_promotion_registry(
        context_root=root,
        promotion_timestamp_utc=promotion_timestamp_utc,
    )
    timestamp = promotion_timestamp_utc or str(active_registry["promotion_date_utc"])
    retrieval_slo_status = (
        "PASS"
        if active_registry["promotion_status"] == "PROMOTED"
        and active_registry["all_repaired_2000_slos_pass"] is True
        else "BLOCKED"
    )
    return {
        "artifact_type": "retrieval_source_of_truth_manifest",
        "generated_at_utc": timestamp,
        "promotion_timestamp_utc": timestamp,
        "active_retrieval_dataset": str(root / "repaired_retrieval_dataset"),
        "active_retrieval_validation_report": str(
            root / "repaired_retrieval_validation_report.json"
        ),
        "active_retrieval_validation_summary": str(
            root / "repaired_retrieval_validation_summary.csv"
        ),
        "active_qdrant_validation_report": str(root / "qdrant_index_report.json"),
        "active_qdrant_validation_summary": str(root / "qdrant_index_summary.csv"),
        "retrieval_promotion_status": active_registry["promotion_status"],
        "retrieval_slo_status": retrieval_slo_status,
        "retrieval_ready_for_phase4": retrieval_slo_status == "PASS",
        "source_hint_mode_is_not_canonical": True,
        "artifacts": {
            "promotion_registry": str(root / "retrieval_promotion_registry.json"),
            "active_retrieval_dataset": str(root / "repaired_retrieval_dataset"),
            "active_retrieval_validation_report": str(
                root / "repaired_retrieval_validation_report.json"
            ),
            "active_retrieval_validation_summary": str(
                root / "repaired_retrieval_validation_summary.csv"
            ),
            "active_qdrant_validation_report": str(root / "qdrant_index_report.json"),
            "active_qdrant_validation_summary": str(root / "qdrant_index_summary.csv"),
            "compression_diagnostic_summary": str(root / "compression_diagnostic_summary.csv"),
            "retrieval_quality_gate_report": str(root / "retrieval_quality_gate_report.json"),
        },
        "metrics_by_vertical": active_registry["retrieval_metrics_by_vertical"],
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_api_calls_triggered": True,
    }


def write_retrieval_promotion_artifacts(
    *,
    context_root: str | Path = DEFAULT_CONTEXT_ROOT,
    promotion_timestamp_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write the promotion registry and source-of-truth manifest."""

    root = Path(context_root)
    timestamp = promotion_timestamp_utc or utc_now()
    registry = build_retrieval_promotion_registry(
        context_root=root,
        promotion_timestamp_utc=timestamp,
    )
    manifest = build_retrieval_source_of_truth_manifest(
        context_root=root,
        registry=registry,
        promotion_timestamp_utc=timestamp,
    )
    write_json(root / "retrieval_promotion_registry.json", registry)
    write_json(root / "retrieval_source_of_truth_manifest.json", manifest)
    return registry, manifest
