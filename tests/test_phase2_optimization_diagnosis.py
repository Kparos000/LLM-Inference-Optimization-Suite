from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def _load_module() -> ModuleType:
    path = Path("scripts/phase4/diagnose_phase2_optimization.py")
    spec = importlib.util.spec_from_file_location("phase2_optimization_diagnosis", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnosis = _load_module()


def _report() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            Path("results/processed/phase2_optimization_diagnosis_report.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _candidates() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            Path("results/processed/phase2_selected_optimization_candidates.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def _rerun_plan() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            Path("results/processed/phase2_before_after_rerun_plan.json").read_text(
                encoding="utf-8"
            )
        ),
    )


def test_phase2_diagnosis_reports_config_level_failures() -> None:
    report = _report()

    assert report["status"] == "PHASE2_OPTIMIZATION_DIAGNOSIS_COMPLETE"
    assert report["no_inference_rerun"] is True
    assert len(report["diagnosis_rows"]) == 25
    assert report["baseline_verdicts"]["runtime_slo_verdict"] == "PASS"
    assert report["baseline_verdicts"]["quality_slo_verdict"] == "FAIL"
    assert report["baseline_verdicts"]["safety_slo_verdict"] == "FAIL"
    assert report["worst_configs"]
    assert report["best_configs"]


def test_mm0_is_treated_as_ablation_not_evidence_target() -> None:
    report = _report()
    mm0_rows = [row for row in report["diagnosis_rows"] if row["memory_mode"] == "mm0_no_context"]

    assert len(mm0_rows) == 5
    assert all(row["ablation_only"] is True for row in mm0_rows)
    assert all(row["worth_optimizing"] is False for row in mm0_rows)
    assert all("mm0_expected_no_context_failure" in row["bottleneck_classes"] for row in mm0_rows)


def test_safety_failures_are_prioritized() -> None:
    report = _report()

    assert report["bottleneck_counts"]["safety_wording_failure"] > 0
    assert report["bottleneck_counts"]["mm4_agentic_safety_trace_issue"] > 0
    assert report["optimization_counts"]["safety_wording_cleanup"] > 0
    assert report["optimization_counts"]["mm4_final_answer_guard"] > 0


def test_candidates_are_deterministic_and_exclude_mm0() -> None:
    payload = _candidates()
    selected = payload["selected_candidates"]
    ids = [row["config_id"] for row in selected]

    assert len(selected) == 8
    assert ids == sorted(ids, key=ids.index)
    assert all(row["memory_mode"] != "mm0_no_context" for row in selected)
    assert any(row["memory_mode"] == "mm4_bounded_agentic" for row in selected)
    assert any(row["backend_type"] == "api_provider" for row in selected)
    assert any(row["memory_mode"] == "mm2_hybrid_top5" for row in selected)
    assert any(row["memory_mode"] == "mm3_compressed_hybrid_top5" for row in selected)


def test_phase2_before_after_plan_does_not_rerun_inference() -> None:
    plan = json.loads(
        Path("results/processed/phase2_before_after_rerun_plan.json").read_text(encoding="utf-8")
    )

    assert plan["status"] == "PHASE2_BEFORE_AFTER_RERUN_PLAN_READY"
    assert plan["do_not_rerun_full_10000_yet"] is True
    assert plan["do_not_modify_gold_or_evaluators"] is True


def test_bottleneck_and_recommendation_rules_are_deterministic() -> None:
    row = {
        "memory_mode": "mm4_bounded_agentic",
        "failed_metric_family": "contract_validity;evidence_match;groundedness;safety",
        "concurrency": "32",
        "backend_type": "self_hosted_gpu",
        "engine": "sglang",
        "evidence_match_rate": "0.70",
        "grounded_rate": "0.69",
    }

    bottlenecks = diagnosis._bottlenecks(row)
    recommendations = diagnosis._recommendations(row, bottlenecks)

    assert "generation_contract_failure" in bottlenecks
    assert "evidence_selection_failure" in bottlenecks
    assert "mm4_agentic_safety_trace_issue" in bottlenecks
    assert "concurrency_degradation" in bottlenecks
    assert "final_answer_contract_normalization" in recommendations
    assert "citation_whitelist" in recommendations
    assert "mm4_final_answer_guard" in recommendations
    assert "lower_concurrency" in recommendations
