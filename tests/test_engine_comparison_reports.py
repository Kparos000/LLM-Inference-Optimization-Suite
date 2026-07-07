from __future__ import annotations

import csv
from pathlib import Path


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_engine_comparison_report_keeps_vllm_and_sglang_explicit() -> None:
    rows = _rows("results/processed/controlled_final_simulation_engine_comparison.csv")

    assert {row["engine"] for row in rows if row["model_alias"] == "model3_7b"} == {
        "vllm",
        "sglang",
    }
    assert all(row["status"] == "COMPLETED" for row in rows)
    assert all(row["requests_attempted"] == "400" for row in rows)
    assert all(row["requests_failed"] == "0" for row in rows)


def test_memory_and_concurrency_comparison_reports_cover_full_matrix() -> None:
    memory_rows = _rows("results/processed/controlled_final_simulation_memory_mode_comparison.csv")
    concurrency_rows = _rows(
        "results/processed/controlled_final_simulation_concurrency_comparison.csv"
    )

    assert len(memory_rows) == 25
    assert len(concurrency_rows) == 25
    assert {row["memory_mode"] for row in memory_rows} == {
        "mm0_no_context",
        "mm1_dense_top5",
        "mm2_hybrid_top5",
        "mm3_compressed_hybrid_top5",
        "mm4_bounded_agentic",
    }
    assert {row["concurrency"] for row in concurrency_rows} == {"4", "16", "32"}


def test_api_track_comparison_contains_no_gpu_telemetry_columns() -> None:
    rows = _rows("results/processed/controlled_final_simulation_api_track_comparison.csv")

    assert len(rows) == 5
    assert {row["model_alias"] for row in rows} == {"model6_gated"}
    assert "gpu_utilization" not in rows[0]
    assert "gpu_hourly_cost" not in rows[0]


def test_api_vs_self_hosted_and_model_comparison_reports_cover_final_matrix() -> None:
    api_vs_self_hosted = _rows(
        "results/processed/controlled_final_simulation_api_vs_self_hosted_comparison.csv"
    )
    model_rows = _rows("results/processed/controlled_final_simulation_model_comparison.csv")

    assert len(api_vs_self_hosted) == 25
    assert len(model_rows) == 25
    assert {row["model_alias"] for row in model_rows} == {"model3_7b", "model6_gated"}
