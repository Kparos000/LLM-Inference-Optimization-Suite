import json
from pathlib import Path

import pytest

from inference_bench.api_local_comparison import (
    build_metric_comparison,
    build_readiness_gate,
    compare_workloads,
    write_comparison_artifacts,
)


def _row(prompt_id: str, prompt: str, *, latency: float) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "vertical": "finance",
        "memory_mode": "mm2_hybrid_top5",
        "ablation_mode": "prompt_plus_metadata",
        "prompt": prompt,
        "citation_id_aliases": '{"E1":["doc-1"]}',
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_ms": latency,
        "success": True,
    }


def _eval_report(*, grounded: float) -> dict[str, object]:
    return {
        "summary": {
            "json_valid_rate": 1.0,
            "generation_contract_valid_rate": 1.0,
            "evidence_id_presence_rate": 1.0,
            "evidence_match_rate": grounded,
            "grounded_rate": grounded,
            "safety_violation_rate": 0.0,
        }
    }


def test_workload_comparison_requires_matching_prompt_ids() -> None:
    with pytest.raises(ValueError, match="prompt IDs do not match"):
        compare_workloads(
            [_row(f"local-{index}", "prompt", latency=10.0) for index in range(5)],
            [_row(f"api-{index}", "prompt", latency=1.0) for index in range(5)],
        )


def test_workload_comparison_reports_renderer_drift() -> None:
    local = [_row(f"p{index}", f"local {index}", latency=10.0) for index in range(5)]
    api = [_row(f"p{index}", f"api {index}", latency=1.0) for index in range(5)]

    comparison = compare_workloads(local, api)

    assert comparison["prompt_id_set_matches"] is True
    assert comparison["memory_modes_match"] is True
    assert comparison["exact_prompt_rendering_matches"] is False
    assert comparison["comparison_scope"].endswith("renderer_drift")


def test_metric_comparison_keeps_local_cost_unavailable() -> None:
    local = [_row(f"p{index}", "same", latency=10.0) for index in range(5)]
    api = [_row(f"p{index}", "same", latency=1.0) for index in range(5)]
    comparison, rows = build_metric_comparison(
        local_eval_report=_eval_report(grounded=0.2),
        api_eval_report=_eval_report(grounded=0.6),
        local_rows=local,
        api_rows=api,
        api_cost_report={
            "cost_per_request_usd": 0.001,
            "cost_per_successful_answer_usd": 0.001,
            "cost_per_grounded_answer_usd": 0.002,
        },
    )

    assert comparison["grounded_rate"]["api_minus_local"] == pytest.approx(0.4)
    assert comparison["cost_per_request_usd"]["local_qwen_0_5b"] is None
    assert comparison["cost_per_request_usd"]["api_llama_3_1_8b"] == 0.001
    cost_row = next(row for row in rows if row["name"] == "cost_per_request_usd")
    assert "not measured" in cost_row["details"]


def test_comparison_artifacts_write_json_and_csv(tmp_path: Path) -> None:
    report_path, summary_path = write_comparison_artifacts(
        report_path=tmp_path / "report.json",
        summary_path=tmp_path / "summary.csv",
        report={"decision": "READY_FOR_SMALL_GPU_SMOKE"},
        summary_rows=[
            {
                "row_type": "readiness_check",
                "name": "telemetry_exists",
                "status": "PASS",
                "local_qwen_0_5b": "",
                "api_llama_3_1_8b": "",
                "api_minus_local": "",
                "artifact": "src/inference_bench/telemetry.py",
                "details": "available",
            }
        ],
    )

    assert json.loads(report_path.read_text(encoding="utf-8"))["decision"] == (
        "READY_FOR_SMALL_GPU_SMOKE"
    )
    assert "telemetry_exists" in summary_path.read_text(encoding="utf-8")


def test_readiness_gate_passes_with_required_artifacts(tmp_path: Path) -> None:
    retrieval_path = tmp_path / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "retrieval_slo_status": "PASS",
                "retrieval_promotion_status": "PROMOTED",
                "retrieval_ready_for_phase4": True,
            }
        ),
        encoding="utf-8",
    )
    gpu_costs_path = tmp_path / "gpu_costs.yaml"
    gpu_costs_path.write_text(
        """
runpod_default:
  provider: runpod
  gpu_type: null
  hourly_price_usd: null
  region: null
  instance_id_optional: null
  measured_start_time: null
  measured_end_time: null
  cost_formula: elapsed_hours * hourly_price_usd
""".strip()
        + "\n",
        encoding="utf-8",
    )
    local_rows = [_row(f"p{index}", "same", latency=10.0) for index in range(5)]

    decision, checks, blockers = build_readiness_gate(
        repo_root=tmp_path,
        local_eval_report={"summary": {"generation_contract_valid_rate": 0.8, "joined_count": 5}},
        api_eval_report={"summary": {"generation_contract_valid_rate": 1.0}},
        local_rows=local_rows,
        api_cost_report={
            "execution_complete": True,
            "request_count": 5,
            "success_count": 5,
            "pricing_source_url": "https://example.test/pricing",
            "total_cost_usd": 0.001,
            "cost_per_request_usd": 0.0002,
            "cost_per_successful_answer_usd": 0.0002,
            "cost_per_grounded_answer_usd": 0.0003,
        },
        retrieval_manifest_path=retrieval_path.name,
        gpu_costs_path=gpu_costs_path.name,
    )

    assert decision == "READY_FOR_SMALL_GPU_SMOKE"
    assert blockers == []
    assert all(check.status == "PASS" for check in checks)
