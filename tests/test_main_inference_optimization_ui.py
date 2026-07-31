from __future__ import annotations

import json
from pathlib import Path

from inference_bench.main_inference_optimization_ui import (
    build_ui_diagnosis,
    write_ui_artifacts,
)


def test_main_inference_ui_diagnosis_uses_saved_artifacts_only() -> None:
    payload = build_ui_diagnosis()
    diagnosis = payload["diagnosis"]

    assert diagnosis["run_id"] == "main_inference_v1"
    assert diagnosis["inference_executed"] is False
    assert diagnosis["llm_used"] is False
    assert diagnosis["run_context"]["deployability_verdict"] == "NOT_DEPLOYABLE_SLO_FAILURES"
    assert {item["metric_id"] for item in diagnosis["failed_slos"]} == {
        "contract_validity",
        "format_validity",
        "evidence_match",
        "groundedness",
        "safety_findings",
    }


def test_ui_options_are_filtered_by_failed_slo_bottleneck() -> None:
    payload = build_ui_diagnosis()
    diagnosis = payload["diagnosis"]
    options = payload["optimization_options"]["options_by_failed_slo"]
    evidence_slo = next(
        item for item in diagnosis["failed_slos"] if item["metric_id"] == "evidence_match"
    )
    evidence_option_ids = {item["optimization_id"] for item in options[evidence_slo["slo_id"]]}

    assert evidence_option_ids
    assert "increase_concurrency" not in evidence_option_ids
    assert "enable_prefix_cache" not in evidence_option_ids
    assert "prompt_contract_repair" in evidence_option_ids


def test_negative_rules_reject_invalid_stronger_model_option() -> None:
    payload = build_ui_diagnosis()
    diagnosis = payload["diagnosis"]
    rejected = payload["optimization_options"]["rejected_optimizations_by_failed_slo"]
    grounded_slo = next(
        item for item in diagnosis["failed_slos"] if item["metric_id"] == "groundedness"
    )
    grounded_rejections = rejected[grounded_slo["slo_id"]]
    stronger = next(
        item for item in grounded_rejections if item["optimization_id"] == "use_stronger_model"
    )

    assert stronger["negative_rule_triggered"] == "stronger_model_escalation"
    assert any(check["triggered"] for check in stronger["negative_rule_checks"])


def test_apply_plan_is_plan_only_and_does_not_create_optimized_result() -> None:
    payload = build_ui_diagnosis()
    apply_plan = payload["apply_plan"]

    assert apply_plan["inference_executed"] is False
    assert apply_plan["optimized_result_created"] is False
    assert apply_plan["apply_all_plan"]["does_not_execute"] is True
    assert all(item["execution_mode"] == "plan_only_no_inference" for item in apply_plan["plans"])


def test_two_track_taxonomy_separates_repairs_from_core_optimizations() -> None:
    payload = build_ui_diagnosis()
    repairs = payload["deployability_repairs"]
    core_catalog = payload["core_optimization_catalog"]
    repair_ids = {item["repair_id"] for item in repairs["repairs"]}
    core_ids = {item["optimization_id"] for item in core_catalog["optimizations"]}

    assert repairs["track"] == "deployability_repairs"
    assert core_catalog["track"] == "core_inference_optimizations"
    assert repair_ids.isdisjoint(core_ids)
    assert "prompt_contract_repair" in repair_ids
    assert "enable_prefix_cache" in core_ids
    assert "concurrency_sweep" in core_ids


def test_repair_gate_blocks_core_optimization_until_measured_validation() -> None:
    payload = build_ui_diagnosis()
    repair_gate = payload["repair_gate"]
    experiment_stage = payload["experiment_stage"]

    assert repair_gate["gate_status"] == "NOT_MEASURED"
    assert repair_gate["core_optimization_eligible"] is False
    assert experiment_stage["current_stage"] == "DEPLOYABILITY_REPAIR_PLANNED"
    assert experiment_stage["gates"]["core_optimization_eligible"] is False


def test_core_optimization_applicability_applies_negative_rules() -> None:
    payload = build_ui_diagnosis()
    states = {
        item["optimization_id"]: item
        for item in payload["core_optimization_applicability"]["states"]
    }

    assert states["use_quantized_model"]["state"] == "blocked_by_negative_rule"
    assert states["use_quantized_model"]["negative_rule_triggered"] == "quantization"
    assert states["concurrency_sweep"]["negative_rule_triggered"] == "concurrency_increase"
    assert states["enable_prefix_cache"]["negative_rule_triggered"] == "prefix_caching"
    assert states["use_pagedattention_capable_engine"]["state"] == "already_measured_in_baseline"


def test_write_ui_artifacts_writes_expected_json_files(tmp_path: Path) -> None:
    paths = write_ui_artifacts(output_root=tmp_path)

    assert set(paths) == {
        "diagnosis",
        "optimization_options",
        "apply_plan",
        "story",
        "deployability_repairs",
        "repair_gate",
        "core_optimization_catalog",
        "core_optimization_applicability",
        "experiment_stage",
        "optimization_story",
    }
    for path in paths.values():
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == "main_inference_v1"
