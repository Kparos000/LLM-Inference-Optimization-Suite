import json
from pathlib import Path

from inference_bench.retrieval_promotion import (
    ACTIVE_ABLATION_MODE,
    ACTIVE_DENSE_BACKEND,
    ACTIVE_STAGE_SIZE,
    ACTIVE_VECTOR_STORE,
    build_retrieval_promotion_registry,
    build_retrieval_source_of_truth_manifest,
    write_retrieval_promotion_artifacts,
)
from inference_bench.slo import SLO_VERTICALS, build_slo_readiness_report, load_slo_config


def write_promoted_fixture(context_root: Path) -> None:
    context_root.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_variant",
        "vertical",
        "stage_size",
        "ablation_mode",
        "measurement",
        "dense_backend",
        "vector_store",
        "candidate_recall_at_20",
        "candidate_recall_at_50",
        "final_recall_at_5",
        "mrr",
        "slo_status",
        "primary_blocker",
        "recommended_next_action",
        "record_count",
        "query_rewrite_count",
    ]
    rows = []
    for vertical in SLO_VERTICALS:
        rows.append(
            {
                "dataset_variant": "repaired_generated",
                "vertical": vertical,
                "stage_size": str(ACTIVE_STAGE_SIZE),
                "ablation_mode": ACTIVE_ABLATION_MODE,
                "measurement": "retrieval_dataset_alignment",
                "dense_backend": ACTIVE_DENSE_BACKEND,
                "vector_store": ACTIVE_VECTOR_STORE,
                "candidate_recall_at_20": "0.96",
                "candidate_recall_at_50": "0.97",
                "final_recall_at_5": "0.94",
                "mrr": "0.95",
                "slo_status": "PASSED",
                "primary_blocker": "none",
                "recommended_next_action": "Proceed to Phase 4.",
                "record_count": "2000",
                "query_rewrite_count": "2000",
            }
        )
    summary_lines = [",".join(fieldnames)]
    summary_lines.extend(",".join(row[field] for field in fieldnames) for row in rows)
    (context_root / "repaired_retrieval_validation_summary.csv").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    (context_root / "repaired_retrieval_validation_report.json").write_text(
        json.dumps({"summary_rows": rows}),
        encoding="utf-8",
    )
    (context_root / "repaired_retrieval_promotion_plan.json").write_text(
        json.dumps(
            {
                "promotion_recommended": True,
                "all_repaired_2000_slos_pass": True,
                "remaining_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    for artifact_name in (
        "qdrant_index_report.json",
        "qdrant_index_summary.csv",
        "compression_diagnostic_summary.csv",
        "retrieval_quality_gate_report.json",
    ):
        (context_root / artifact_name).write_text("{}\n", encoding="utf-8")


def test_promotion_registry_generation(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    write_promoted_fixture(context_root)

    registry = build_retrieval_promotion_registry(
        context_root=context_root,
        promotion_timestamp_utc="2026-06-04T00:00:00+00:00",
    )

    assert registry["artifact_type"] == "retrieval_promotion_registry"
    assert registry["promotion_status"] == "PROMOTED"
    assert registry["all_repaired_2000_slos_pass"] is True
    assert set(registry["retrieval_metrics_by_vertical"]) == set(SLO_VERTICALS)


def test_source_of_truth_manifest_generation(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    write_promoted_fixture(context_root)
    registry = build_retrieval_promotion_registry(
        context_root=context_root,
        promotion_timestamp_utc="2026-06-04T00:00:00+00:00",
    )

    manifest = build_retrieval_source_of_truth_manifest(
        context_root=context_root,
        registry=registry,
        promotion_timestamp_utc="2026-06-04T00:00:00+00:00",
    )

    assert manifest["artifact_type"] == "retrieval_source_of_truth_manifest"
    assert manifest["retrieval_promotion_status"] == "PROMOTED"
    assert manifest["retrieval_slo_status"] == "PASS"
    assert manifest["retrieval_ready_for_phase4"] is True
    assert manifest["artifacts"]["active_retrieval_validation_summary"].endswith(
        "repaired_retrieval_validation_summary.csv"
    )


def test_readiness_uses_promoted_retrieval_reports(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    write_promoted_fixture(context_root)
    _registry, _manifest = write_retrieval_promotion_artifacts(
        context_root=context_root,
        promotion_timestamp_utc="2026-06-04T00:00:00+00:00",
    )

    report, rows = build_slo_readiness_report(
        slo_config=load_slo_config("configs/slo_targets.yaml"),
        retrieval_report_path=context_root / "retrieval_source_of_truth_manifest.json",
    )

    retrieval_rows = [row for row in rows if row["metric_family"] == "retrieval_slo"]
    assert report["retrieval_slo_blocked_count"] == 0
    assert report["inference_scaling_blocked_by_retrieval_slos"] is False
    assert report["summary"]["overall_status"] == "READY_WITH_GAPS"
    assert {row["status"] for row in retrieval_rows} == {"PASS"}
    assert report["not_available_metric_count"] > 0


def test_backward_compatible_legacy_retrieval_report_still_blocks(tmp_path: Path) -> None:
    legacy_report = tmp_path / "retrieval_evaluation_report.json"
    legacy_report.write_text(
        json.dumps(
            {
                "by_split": {
                    "final_10000": {
                        "prompt_plus_metadata": {
                            "mm2_hybrid_top5": {
                                vertical: {
                                    "candidate_recall_at_20": 0.1,
                                    "candidate_recall_at_50": 0.1,
                                    "recall_at_5": 0.1,
                                    "mrr": 0.1,
                                }
                                for vertical in SLO_VERTICALS
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report, rows = build_slo_readiness_report(
        slo_config=load_slo_config("configs/slo_targets.yaml"),
        retrieval_report_path=legacy_report,
    )

    assert report["retrieval_slo_blocked_count"] == len(SLO_VERTICALS) * 4
    assert any(
        row["metric_family"] == "retrieval_slo" and row["status"] == "BLOCKED" for row in rows
    )
