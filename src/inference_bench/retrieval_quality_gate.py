"""Hard retrieval quality gate for Phase 3 RAG readiness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

QUALITY_TARGETS = [
    {
        "target_name": "overall_prompt_plus_metadata_hybrid_recall_at_5",
        "split": "final_10000",
        "ablation_mode": "prompt_plus_metadata",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "overall",
        "metric": "recall_at_5",
        "threshold": 0.80,
        "recommended_action": (
            "Improve calibrated final top-5 selection and finance metadata features."
        ),
    },
    {
        "target_name": "finance_prompt_plus_metadata_hybrid_recall_at_5",
        "split": "final_10000",
        "ablation_mode": "prompt_plus_metadata",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "finance",
        "metric": "recall_at_5",
        "threshold": 0.80,
        "recommended_action": (
            "Repair finance reranking, period extraction, and chunk disambiguation."
        ),
    },
    {
        "target_name": "overall_prompt_text_only_hybrid_recall_at_5",
        "split": "final_10000",
        "ablation_mode": "prompt_text_only",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "overall",
        "metric": "recall_at_5",
        "threshold": 0.70,
        "recommended_action": "Improve prompt-visible entity, metric, and section inference.",
    },
    {
        "target_name": "finance_prompt_text_only_hybrid_recall_at_5",
        "split": "final_10000",
        "ablation_mode": "prompt_text_only",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "finance",
        "metric": "recall_at_5",
        "threshold": 0.65,
        "recommended_action": (
            "Audit whether finance prompts contain enough visible company/metric/period signal."
        ),
    },
    {
        "target_name": "source_hint_assisted_hybrid_recall_at_5",
        "split": "final_10000",
        "ablation_mode": "prompt_plus_source_hints",
        "memory_mode": "mm2_hybrid_top5",
        "vertical": "overall",
        "metric": "recall_at_5",
        "threshold": 0.95,
        "recommended_action": (
            "Preserve assisted upper-bound retrieval while keeping it labeled as hint-assisted."
        ),
    },
]

COMPRESSION_TARGETS = [
    {
        "target_name": "mm3_compression_token_reduction_pct",
        "split": "final_10000",
        "metric": "token_reduction_pct",
        "threshold": 0.20,
        "comparison": ">=",
        "recommended_action": "Increase deterministic compression while preserving evidence.",
    },
    {
        "target_name": "mm3_compression_recall_loss",
        "split": "final_10000",
        "metric": "recall_loss",
        "threshold": 0.05,
        "comparison": "<=",
        "recommended_action": "Reduce compression aggressiveness if gold evidence is dropped.",
    },
]


def weighted_metric(rows: list[dict[str, Any]], metric: str) -> float:
    """Return a record-count weighted metric."""

    total_records = sum(int(row.get("record_count") or 0) for row in rows)
    if total_records <= 0:
        return 0.0
    return round(
        sum(float(row.get(metric) or 0.0) * int(row.get("record_count") or 0) for row in rows)
        / total_records,
        6,
    )


def matching_retrieval_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    ablation_mode: str,
    memory_mode: str,
    vertical: str,
) -> list[dict[str, Any]]:
    """Return rows matching a target selector."""

    return [
        row
        for row in rows
        if str(row.get("split")) == split
        and str(row.get("ablation_mode")) == ablation_mode
        and str(row.get("memory_mode")) == memory_mode
        and (vertical == "overall" or str(row.get("vertical")) == vertical)
    ]


def evaluate_target(
    target: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one retrieval target."""

    matched_rows = matching_retrieval_rows(
        rows,
        split=str(target["split"]),
        ablation_mode=str(target["ablation_mode"]),
        memory_mode=str(target["memory_mode"]),
        vertical=str(target["vertical"]),
    )
    actual = weighted_metric(matched_rows, str(target["metric"]))
    threshold = float(target["threshold"])
    passed = actual >= threshold
    return {
        "target_name": target["target_name"],
        "split": target["split"],
        "ablation_mode": target["ablation_mode"],
        "memory_mode": target["memory_mode"],
        "vertical": target["vertical"],
        "metric": target["metric"],
        "threshold": threshold,
        "actual": actual,
        "margin": round(actual - threshold, 6),
        "status": "PASSED" if passed else "FAILED",
        "recommended_next_repair_action": target["recommended_action"],
    }


def evaluate_compression_target(
    target: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one compression target."""

    matched_rows = [row for row in rows if str(row.get("split")) == str(target["split"])]
    actual = weighted_metric(matched_rows, str(target["metric"]))
    threshold = float(target["threshold"])
    comparison = str(target["comparison"])
    passed = actual >= threshold if comparison == ">=" else actual <= threshold
    margin = actual - threshold if comparison == ">=" else threshold - actual
    return {
        "target_name": target["target_name"],
        "split": target["split"],
        "ablation_mode": "all",
        "memory_mode": "mm3_compressed_hybrid_top5",
        "vertical": "overall",
        "metric": target["metric"],
        "threshold": threshold,
        "actual": actual,
        "margin": round(margin, 6),
        "status": "PASSED" if passed else "FAILED",
        "recommended_next_repair_action": target["recommended_action"],
    }


def build_retrieval_quality_gate_report(
    retrieval_summary_rows: list[dict[str, Any]],
    compression_summary_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the hard retrieval quality gate report."""

    rows = [evaluate_target(target, retrieval_summary_rows) for target in QUALITY_TARGETS]
    rows.extend(
        evaluate_compression_target(target, compression_summary_rows)
        for target in COMPRESSION_TARGETS
    )
    failed = [row for row in rows if row["status"] != "PASSED"]
    status = "PASSED" if not failed else "BLOCKED"
    report = {
        "quality_gate_status": status,
        "passed": status == "PASSED",
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "strict_no_hint_rules_weakened": False,
        "targets": rows,
        "failed_targets": failed,
        "recommended_next_repair_action": failed[0]["recommended_next_repair_action"]
        if failed
        else "Proceed to Phase 4 retrieval-aware inference smoke tests.",
    }
    return report, rows


def load_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV rows for standalone gate checks."""

    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_quality_gate_outputs(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    report: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Write quality gate JSON and CSV outputs."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary_output = Path(summary_path)
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_GATE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


QUALITY_GATE_SUMMARY_FIELDS = [
    "target_name",
    "split",
    "ablation_mode",
    "memory_mode",
    "vertical",
    "metric",
    "threshold",
    "actual",
    "margin",
    "status",
    "recommended_next_repair_action",
]
