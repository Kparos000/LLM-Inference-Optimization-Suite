from inference_bench.memory_workloads import build_evaluation_report
from inference_bench.retrieval_quality_gate import build_retrieval_quality_gate_report


def retrieval_row(
    *,
    ablation_mode: str,
    vertical: str,
    recall_at_5: float,
    record_count: int = 100,
) -> dict[str, object]:
    return {
        "split": "final_10000",
        "ablation_mode": ablation_mode,
        "memory_mode": "mm2_hybrid_top5",
        "vertical": vertical,
        "record_count": record_count,
        "recall_at_5": recall_at_5,
    }


def compression_row(
    token_reduction_pct: float = 0.25, recall_loss: float = 0.0
) -> dict[str, object]:
    return {
        "split": "final_10000",
        "ablation_mode": "prompt_plus_metadata",
        "vertical": "finance",
        "record_count": 100,
        "token_reduction_pct": token_reduction_pct,
        "recall_loss": recall_loss,
    }


def passing_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for vertical in ["airline", "healthcare_admin", "retail", "finance", "research_ai"]:
        rows.append(
            retrieval_row(ablation_mode="prompt_plus_metadata", vertical=vertical, recall_at_5=0.84)
        )
        rows.append(
            retrieval_row(ablation_mode="prompt_text_only", vertical=vertical, recall_at_5=0.72)
        )
        rows.append(
            retrieval_row(
                ablation_mode="prompt_plus_source_hints", vertical=vertical, recall_at_5=0.97
            )
        )
    return rows


def test_quality_gate_fails_when_thresholds_are_not_met() -> None:
    rows = passing_rows()
    rows.append(
        retrieval_row(
            ablation_mode="prompt_plus_metadata",
            vertical="finance",
            recall_at_5=0.20,
        )
    )

    report, summary_rows = build_retrieval_quality_gate_report(rows, [compression_row()])

    assert report["quality_gate_status"] == "BLOCKED"
    assert any(row["status"] == "FAILED" for row in summary_rows)
    assert report["failed_targets"]


def test_quality_gate_passes_on_synthetic_passing_fixture() -> None:
    report, summary_rows = build_retrieval_quality_gate_report(
        passing_rows(),
        [compression_row()],
    )

    assert report["quality_gate_status"] == "PASSED"
    assert all(row["status"] == "PASSED" for row in summary_rows)
    assert report["no_model_inference_triggered"] is True


def test_recall_10_20_50_100_diagnostics_are_present() -> None:
    _report, summary_rows = build_evaluation_report(
        [
            {
                "split": "final_10000",
                "ablation_mode": "prompt_plus_metadata",
                "memory_mode": "mm2_hybrid_top5",
                "vertical": "finance",
                "recall_at_5": 0.5,
                "mrr": 0.5,
                "retrieval_latency_ms": 1.0,
                "context_token_count": 100,
                "context_rows_selected": 5,
                "distinct_context_ids": ["a", "b"],
                "retrieval_backend_label": "qdrant_vector",
                "dense_backend": "qdrant_vector",
                "vector_store": "qdrant_local",
                "source_hints_used": False,
                "query_enrichment_used": True,
                "reranking_used": True,
                "reranker_enabled": True,
                "leakage_guard_applied": True,
                "blocked_direct_hint_count": 0,
                "gold_evidence_included": True,
                "missing_gold_evidence_count": 0,
                "token_reduction": 0,
                "candidate_recall_at_10": 0.6,
                "candidate_recall_at_20": 0.7,
                "candidate_recall_at_50": 0.8,
                "candidate_recall_at_100": 0.9,
                "candidate_recall_at_200": 1.0,
                "candidate_mrr_at_100": 0.9,
            }
        ]
    )

    row = summary_rows[0]
    assert row["candidate_recall_at_10"] == 0.6
    assert row["candidate_recall_at_20"] == 0.7
    assert row["candidate_recall_at_50"] == 0.8
    assert row["candidate_recall_at_100"] == 0.9
