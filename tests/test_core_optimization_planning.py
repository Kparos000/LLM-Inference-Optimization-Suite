from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from inference_bench.core_optimization_planning import (
    ALLOWED_ACTIVATION_STATES,
    build_core_optimization_taxonomy,
)

PROCESSED = Path("experiments/main/main_inference_v1/processed")


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_core_taxonomy_separates_repairs_from_core_optimizations() -> None:
    taxonomy = build_core_optimization_taxonomy()
    layers = taxonomy["layers"]
    core = layers["B_engineer_applied_core_optimizations"]
    core_ids = {str(item["optimization_id"]) for item in core}
    removed = set(taxonomy["removed_from_core_list"])

    assert taxonomy["status"] == "planning_audit_complete"
    assert len(core) == 15
    assert removed.isdisjoint(core_ids)
    assert "prompt_contract_repair" in removed
    assert "prompt_prefix_layout_optimization" in core_ids
    assert "distributed_capacity_serving_architecture" in core_ids


def test_generated_taxonomy_config_is_authoritative_and_readable() -> None:
    payload = cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/core_optimization_taxonomy.yaml").read_text()),
    )
    layers = payload["layers"]

    assert payload["status"] == "planning_audit_complete"
    assert len(layers["A_engine_baseline_capabilities"]) >= 18
    assert len(layers["B_engineer_applied_core_optimizations"]) == 15
    assert len(layers["C_applicable_experiment_candidates"]) == 15


def test_engine_baseline_capability_report_does_not_overclaim_activation() -> None:
    report = _json(PROCESSED / "main_inference_v1_engine_baseline_capability_report.json")
    capabilities = report["capabilities"]
    states = {str(item["baseline_activation_state"]) for item in capabilities}
    confirmed = [
        item for item in capabilities if item["baseline_activation_state"] == "confirmed_active"
    ]

    assert report["status"] == "ENGINE_BASELINE_CAPABILITY_AUDITED"
    assert report["main_inference_v1_manifest_optimization_flags"] == []
    assert states <= ALLOWED_ACTIVATION_STATES
    assert "unknown" in states
    assert confirmed == []
    assert report["baseline_capabilities_are_new_optimizations"] is False


def test_workload_opportunity_audit_keeps_cache_claims_unmeasured() -> None:
    report = _json(PROCESSED / "core_optimization_workload_opportunity_report.json")

    assert report["status"] == "WORKLOAD_OPPORTUNITY_AUDITED"
    assert report["semantic_similarity_counted"] is False
    assert report["measured_cache_hits_available"] is False
    assert report["prefix_cache_hit_rate"] is None
    assert report["workload_scan"]["rows_scanned"] == 10000
    assert report["workload_scan"]["prefix_reuse"]["prefix_family_count"] > 0


def test_applicability_matrix_filters_current_project_candidates() -> None:
    payload = _json(PROCESSED / "core_optimization_applicability_matrix.json")
    states = {str(item["optimization_id"]): item for item in payload["states"]}

    assert payload["status"] == "CORE_OPTIMIZATION_APPLICABILITY_READY"
    assert states["prompt_prefix_layout_optimization"]["applicability_state"] == (
        "selected_for_one_factor_test"
    )
    assert states["scheduler_batch_tuning"]["applicability_state"] == (
        "selected_for_one_factor_test"
    )
    assert states["prefix_cache_verification_tuning"]["instrument_first"] is True
    assert states["quantization"]["applicability_state"] == "blocked_by_negative_rule"
    assert states["multi_gpu_parallelism"]["applicability_state"] == (
        "not_applicable_current_project"
    )


def test_one_factor_plan_changes_exactly_one_variable_per_experiment() -> None:
    plan = _json(PROCESSED / "core_optimization_one_factor_experiment_plan.json")

    assert plan["status"] == "ONE_FACTOR_EXPERIMENT_PROGRAM_PLANNED"
    assert plan["does_not_execute_inference"] is True
    for experiment in plan["experiments"]:
        assert experiment["changed_variable"]
        assert experiment["held_constant_variables"]
        assert experiment["does_not_execute_automatically"] is True
        assert experiment["changed_variable"] not in experiment["held_constant_variables"]


def test_scenario_registry_marks_planned_results_as_unmeasured() -> None:
    registry = cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/core_optimization_scenario_registry.yaml").read_text()),
    )
    scenarios = {item["scenario_id"]: item for item in registry["scenarios"]}

    assert registry["status"] == "SCENARIO_REGISTRY_PLANNED_WITH_OBSERVABILITY"
    assert scenarios["main_inference_v1"]["result_type"] == "measured"
    assert scenarios["deployability_repair_validation_v1"]["result_type"] == "measured"
    assert scenarios["coreopt_scheduler_batch_vllm_v1"]["result_type"] == "planned"
    assert scenarios["coreopt_scheduler_batch_vllm_v1"]["instrumentation_readiness"] == (
        "requires_runner_instrumentation"
    )
    assert scenarios["optimized_inference_v1"]["result_type"] == "missing_not_created"
    assert scenarios["optimized_inference_v1"]["artifact_paths"] == []


def test_ui_contract_preserves_measured_planned_and_missing_labels() -> None:
    contract = _json(PROCESSED / "core_optimization_ui_contract.json")
    labels = contract["ui_state_labels"]

    assert contract["status"] == "UI_CONTRACT_DESIGNED_NO_MEASURED_OPTIMIZATION_RESULTS"
    assert contract["does_not_create_optimized_inference_v1"] is True
    assert contract["frontend_must_not_display_as_measured"] == [
        "core optimization candidates",
        "modeled expected improvements",
        "future TensorRT-LLM or multi-GPU concepts",
    ]
    assert "measured_baseline_capability" in labels
    assert "planned_candidate" in labels
    assert "missing_result" in labels
