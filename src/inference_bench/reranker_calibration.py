"""Split-safe reranker calibration scaffolding for Phase 3 retrieval.

This module uses gold labels only for offline calibration/reporting. Runtime
retrieval still receives only ablation-allowed query features.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

FORBIDDEN_STRICT_FEATURES = {
    "gold_evidence_id",
    "gold_evidence_ids",
    "source_id",
    "parent_id",
    "document_id",
    "filing_id",
    "accession_number",
    "required_doc_ids",
    "required_evidence_ids",
}

BASE_RUNTIME_FEATURES = [
    "qdrant_score",
    "bm25_score",
    "hybrid_score",
    "query_document_lexical_overlap",
    "title_overlap",
    "company_match",
    "ticker_match",
    "metric_synonym_match",
    "xbrl_concept_match",
    "period_match",
    "section_match",
    "vertical_match",
    "document_text_similarity",
    "section_type_feature",
]

SOURCE_HINT_FEATURES = [
    "source_hint_match",
    "parent_hint_match",
    "document_hint_match",
]


@dataclass(frozen=True)
class CalibratedRerankerWeights:
    """Serializable deterministic reranker weights."""

    backend: str
    weights: dict[str, float]
    intercept: float = 0.0
    version: str = "phase3_block12_v1"

    def save(self, path: str | Path) -> Path:
        """Persist calibrated weights as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> CalibratedRerankerWeights:
        """Load calibrated weights from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


DEFAULT_CALIBRATED_WEIGHTS = CalibratedRerankerWeights(
    backend="calibrated_linear",
    weights={
        "qdrant_score": 0.45,
        "bm25_score": 0.55,
        "query_document_lexical_overlap": 0.30,
        "title_overlap": 0.30,
        "company_match": 0.45,
        "ticker_match": 0.45,
        "metric_synonym_match": 0.55,
        "xbrl_concept_match": 0.55,
        "period_match": 0.35,
        "section_match": 0.35,
        "vertical_match": 0.10,
        "document_text_similarity": 0.20,
        "section_type_feature": 0.25,
    },
)


def split_for_prompt_id(prompt_id: str) -> str:
    """Return a deterministic train/dev/test split for one prompt ID."""

    digest = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def allowed_features_for_ablation(ablation_mode: str) -> list[str]:
    """Return allowed runtime features for one ablation mode."""

    if ablation_mode == "prompt_plus_source_hints":
        return [*BASE_RUNTIME_FEATURES, *SOURCE_HINT_FEATURES]
    return list(BASE_RUNTIME_FEATURES)


def assert_no_forbidden_features(
    feature_names: list[str],
    *,
    ablation_mode: str,
) -> None:
    """Validate that strict ablations do not include source/gold features."""

    forbidden = set()
    if ablation_mode in {"prompt_text_only", "prompt_plus_metadata"}:
        forbidden = FORBIDDEN_STRICT_FEATURES | set(SOURCE_HINT_FEATURES)
    else:
        forbidden = {"gold_evidence_id", "gold_evidence_ids", "required_evidence_ids"}
    used_forbidden = sorted(set(feature_names) & forbidden)
    if used_forbidden:
        msg = f"Forbidden reranker features for {ablation_mode}: {used_forbidden}"
        raise ValueError(msg)


def choose_reranker_backend(preferred_backend: str = "calibrated_linear") -> str:
    """Return the configured reranker backend.

    Cross-encoder support is intentionally config-gated and disabled by default.
    """

    if preferred_backend in {"heuristic", "calibrated_linear"}:
        return preferred_backend
    if preferred_backend == "cross_encoder_optional":
        return "cross_encoder_optional_disabled"
    msg = f"Unknown reranker backend '{preferred_backend}'"
    raise ValueError(msg)


def build_reranker_calibration_report(
    evaluation_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build calibration diagnostics from generated retrieval rows."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in evaluation_rows:
        key = (
            str(row.get("ablation_mode") or "prompt_plus_source_hints"),
            str(row.get("memory_mode") or ""),
            str(row.get("vertical") or ""),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
    for row in evaluation_rows:
        split_counts[split_for_prompt_id(str(row.get("prompt_id") or ""))] += 1

    for (ablation_mode, memory_mode, vertical), rows in sorted(grouped.items()):
        feature_names = allowed_features_for_ablation(ablation_mode)
        assert_no_forbidden_features(feature_names, ablation_mode=ablation_mode)
        dev_rows = [
            row for row in rows if split_for_prompt_id(str(row.get("prompt_id") or "")) == "dev"
        ]
        test_rows = [
            row for row in rows if split_for_prompt_id(str(row.get("prompt_id") or "")) == "test"
        ]
        summary_rows.append(
            {
                "ablation_mode": ablation_mode,
                "memory_mode": memory_mode,
                "vertical": vertical,
                "reranker_backend": "calibrated_linear",
                "feature_count": len(feature_names),
                "forbidden_feature_count": 0,
                "train_records": sum(
                    1
                    for row in rows
                    if split_for_prompt_id(str(row.get("prompt_id") or "")) == "train"
                ),
                "dev_records": len(dev_rows),
                "test_records": len(test_rows),
                "dev_recall_at_5": round(
                    mean(float(row.get("recall_at_5") or 0.0) for row in dev_rows),
                    6,
                )
                if dev_rows
                else 0.0,
                "test_recall_at_5": round(
                    mean(float(row.get("recall_at_5") or 0.0) for row in test_rows),
                    6,
                )
                if test_rows
                else 0.0,
                "candidate_recall_at_100": round(
                    mean(float(row.get("candidate_recall_at_100") or 0.0) for row in rows),
                    6,
                )
                if rows
                else 0.0,
                "calibrated_weights_version": DEFAULT_CALIBRATED_WEIGHTS.version,
            }
        )

    report = {
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "calibration_scope": "offline_gold_labels_for_weight_calibration_only",
        "runtime_gold_features_used": False,
        "direct_source_id_features_used_in_strict_modes": False,
        "reranker_backend": "calibrated_linear",
        "cross_encoder_backend": "cross_encoder_optional_disabled",
        "split_counts": split_counts,
        "allowed_features_by_ablation": {
            ablation: allowed_features_for_ablation(ablation)
            for ablation in (
                "prompt_text_only",
                "prompt_plus_metadata",
                "prompt_plus_source_hints",
            )
        },
        "default_weights": asdict(DEFAULT_CALIBRATED_WEIGHTS),
        "summary": summary_rows,
    }
    return report, summary_rows


def write_reranker_calibration_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write reranker calibration rows."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RERANKER_CALIBRATION_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


RERANKER_CALIBRATION_SUMMARY_FIELDS = [
    "ablation_mode",
    "memory_mode",
    "vertical",
    "reranker_backend",
    "feature_count",
    "forbidden_feature_count",
    "train_records",
    "dev_records",
    "test_records",
    "dev_recall_at_5",
    "test_recall_at_5",
    "candidate_recall_at_100",
    "calibrated_weights_version",
]
