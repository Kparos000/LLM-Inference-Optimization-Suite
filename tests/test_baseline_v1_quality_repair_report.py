import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase4/build_baseline_v1_quality_repair_report.py"
SPEC = importlib.util.spec_from_file_location("baseline_v1_quality_repair_report", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _runtime_report() -> dict[str, Any]:
    return {
        "runtime_seconds": 100.0,
        "latency_summary": {
            "mean_ttft_ms": 100.0,
            "p50_ttft_ms": 100.0,
            "p95_ttft_ms": 200.0,
            "p99_ttft_ms": 300.0,
            "mean_tpot_ms": 10.0,
            "p50_tpot_ms": 10.0,
            "p95_tpot_ms": 20.0,
            "p99_tpot_ms": 30.0,
            "p50_e2e_latency_ms": 1000.0,
            "p95_e2e_latency_ms": 2000.0,
            "p99_e2e_latency_ms": 3000.0,
            "mean_total_tokens_per_second": 100.0,
        },
        "cost_report": {
            "gpu_cost_usd": 1.0,
            "api_cost_usd": 0.1,
            "self_hosted_request_count": 1000,
            "api_request_count": 1000,
            "total_cost_usd": 1.1,
        },
        "gpu_summary": {
            "max_memory_used_mb": 40960.0,
            "max_temperature_c": 60.0,
            "mean_power_draw_w": 250.0,
            "mean_utilization_gpu_percent": 60.0,
            "memory_total_mb": {"max": 81920.0},
            "power_draw_w": {"max": 400.0},
        },
    }


def _eval_report() -> dict[str, Any]:
    return {
        "summary": {
            "json_valid_rate": 1.0,
            "generation_contract_valid_rate": 0.8,
            "evidence_match_rate": 0.7,
            "grounded_rate": 0.6,
            "safety_violation_count": 1,
        },
        "total_requests_completed": 2000,
        "config_summaries": [
            {
                "backend_type": "self_hosted_gpu",
                "total_input_tokens": 1000,
                "total_output_tokens": 100,
            },
            {
                "backend_type": "api_provider",
                "total_input_tokens": 500,
                "total_output_tokens": 50,
            },
        ],
    }


def test_slo_scorecard_does_not_invent_json_target() -> None:
    rows = module.build_slo_scorecard(runtime_report=_runtime_report(), eval_report=_eval_report())
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["json_validity_pct"]["status"] == "NOT_EVALUATED_NO_CONFIGURED_TARGET"
    assert by_metric["json_validity_pct"]["slo_target"] is None
    assert by_metric["contract_validity_pct"]["status"] == "FAIL"
    assert by_metric["gpu_utilization_mean_pct"]["status"] == "PASS"


def test_repaired_metric_rows_use_repository_slos() -> None:
    rows = module.build_repaired_metric_rows(
        {
            "before_summary": {
                "json_valid_rate": 1.0,
                "generation_contract_valid_rate": 0.84,
                "evidence_match_rate": 0.79,
                "grounded_rate": 0.76,
                "safety_violation_count": 14,
            },
            "after_summary": {
                "json_valid_rate": 1.0,
                "generation_contract_valid_rate": 1.0,
                "evidence_match_rate": 1.0,
                "grounded_rate": 1.0,
                "safety_violation_count": 1,
            },
        }
    )
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["contract_validity_pct"]["status_after"] == "PASS"
    assert by_metric["evidence_match_pct"]["status_after"] == "PASS"
    assert by_metric["groundedness_pct"]["status_after"] == "PASS"
    assert by_metric["safety_findings"]["status_after"] == "FAIL"


def test_report_preserves_baseline_archive_and_uses_targeted_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "experiments/baseline_v1"
    processed = baseline / "processed"
    processed.mkdir(parents=True)
    (processed / "final_10000_baseline_v1_runtime_report.json").write_text(
        json.dumps(_runtime_report()),
        encoding="utf-8",
    )
    (processed / "final_10000_baseline_v1_eval_report.json").write_text(
        json.dumps(_eval_report()),
        encoding="utf-8",
    )
    (processed / "final_10000_baseline_v1_slo_report.json").write_text(
        json.dumps(
            {
                "benchmark_execution_verdict": "COMPLETED",
                "runtime_slo_verdict": "PASS",
                "cost_slo_verdict": "PASS",
                "quality_slo_verdict": "FAIL",
                "safety_slo_verdict": "FAIL",
                "overall_deployability_verdict": "NOT_DEPLOYABLE_SLO_FAILURES",
            }
        ),
        encoding="utf-8",
    )
    repair = tmp_path / "repair.json"
    repair.write_text(
        json.dumps(
            {
                "repairs_applied": ["final_answer_contract_normalization"],
                "total_requests": 1600,
                "candidate_config_ids": ["cfg"],
                "success_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    before_after = tmp_path / "before_after.json"
    before_after.write_text(
        json.dumps(
            {
                "before_summary": {
                    "json_valid_rate": 1.0,
                    "generation_contract_valid_rate": 0.84,
                    "evidence_match_rate": 0.79,
                    "grounded_rate": 0.76,
                    "safety_violation_count": 14,
                },
                "after_summary": {
                    "json_valid_rate": 1.0,
                    "generation_contract_valid_rate": 1.0,
                    "evidence_match_rate": 1.0,
                    "grounded_rate": 1.0,
                    "safety_violation_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(
        json.dumps({"final_main_10000_experiment_allowed": True}),
        encoding="utf-8",
    )
    diagnosis = ROOT / "results/processed/phase2_optimization_diagnosis_report.json"
    args = argparse.Namespace(
        output_root=tmp_path / "experiments/baseline/baseline_v1_quality_repair_v1",
        baseline_archive=baseline,
        targeted_repair_report_path=repair,
        before_after_path=before_after,
        readiness_path=readiness,
        slo_targets_path=ROOT / "configs/slo_targets.yaml",
        slo_profiles_path=ROOT / "configs/slo_profiles.yaml",
    )
    assert diagnosis.exists()
    report = module.build_report(args)
    assert report["baseline_archive"].endswith("experiments/baseline_v1")
    assert report["baseline_verdicts"]["quality_slo_verdict"] == "FAIL"
    assert report["validation_scope"]["full_10000_rerun_performed"] is False
    assert report["main_inference_v1_allowed"] is True
