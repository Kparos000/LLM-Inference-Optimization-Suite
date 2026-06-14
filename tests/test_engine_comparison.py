from __future__ import annotations

from inference_bench.engine_comparison import (
    build_comparison_report,
    build_engine_row,
    build_pairwise_delta,
)


def _result_rows(count: int = 50) -> list[dict[str, object]]:
    return [{"prompt_id": f"prompt-{index}", "success": True} for index in range(count)]


def _evaluation() -> dict[str, float]:
    return {
        "json_valid_rate": 1.0,
        "generation_contract_valid_rate": 0.8,
        "evidence_match_rate": 0.6,
        "grounded_rate": 0.6,
        "safety_violation_rate": 0.0,
    }


def test_engine_comparison_keeps_missing_metrics_explicit() -> None:
    row = build_engine_row(
        backend="api",
        comparison_scope="contextual",
        result_rows=_result_rows(5),
        evaluation_summary=_evaluation(),
    )

    assert row["mean_ttft_ms"] is None
    assert "mean_ttft_ms" in row["missing_metrics"]
    assert "max_gpu_memory_used_mb" in row["missing_metrics"]


def test_pairwise_delta_does_not_estimate_missing_values() -> None:
    baseline = build_engine_row(
        backend="vllm",
        comparison_scope="matched",
        result_rows=_result_rows(),
        evaluation_summary=_evaluation(),
        latency_summary={"mean_ttft_ms": 100.0},
    )
    candidate = build_engine_row(
        backend="sglang",
        comparison_scope="matched",
        result_rows=_result_rows(),
        evaluation_summary=_evaluation(),
    )

    delta = build_pairwise_delta(baseline, candidate)

    assert delta["mean_ttft_ms"] is None
    assert delta["grounded_rate"] == 0.0


def test_matched_prompt_ids_make_gpu_engines_comparable() -> None:
    result_rows = _result_rows()
    vllm = build_engine_row(
        backend="vllm",
        comparison_scope="matched",
        result_rows=result_rows,
        evaluation_summary=_evaluation(),
    )
    sglang = build_engine_row(
        backend="sglang",
        comparison_scope="matched",
        result_rows=result_rows,
        evaluation_summary=_evaluation(),
    )

    report = build_comparison_report(
        rows=[vllm, sglang],
        prompt_ids_by_backend={
            "vllm": {str(row["prompt_id"]) for row in result_rows},
            "sglang": {str(row["prompt_id"]) for row in result_rows},
        },
    )

    assert report["comparison_status"] == "COMPARABLE"
    assert report["prompt_id_sets_match"] is True


def test_prompt_id_mismatch_blocks_full_comparison() -> None:
    vllm = build_engine_row(
        backend="vllm",
        comparison_scope="matched",
        result_rows=_result_rows(),
        evaluation_summary=_evaluation(),
    )
    sglang = build_engine_row(
        backend="sglang",
        comparison_scope="matched",
        result_rows=_result_rows(),
        evaluation_summary=_evaluation(),
    )

    report = build_comparison_report(
        rows=[vllm, sglang],
        prompt_ids_by_backend={
            "vllm": {"a"},
            "sglang": {"b"},
        },
    )

    assert report["comparison_status"] == "NOT_FULLY_COMPARABLE"
    assert report["prompt_id_sets_match"] is False
