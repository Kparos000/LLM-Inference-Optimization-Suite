from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from inference_bench.core_optimization_observability import (
    EVENT_SCHEMA_JSON_PATH,
    MAIN_PROCESSED,
    OBSERVABILITY_EVENT_ADAPTER,
    READINESS_STATES,
    REPO_ROOT,
    SCENARIO_PLAN_PATHS,
    all_adapters,
    analyze_prefix_opportunity,
    build_event_schema,
    build_observability_registry,
    build_updated_scenario_registry,
    compare_prefix_layouts,
)
from inference_bench.core_optimization_planning import REPAIR_IDS
from inference_bench.product_platform import ProductPlatformData


def _json(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def test_observability_registry_covers_core_optimizations_and_excludes_repairs() -> None:
    registry = build_observability_registry()
    entries = cast(list[dict[str, Any]], registry["optimizations"])
    ids = {str(entry["optimization_id"]) for entry in entries}

    assert len(entries) == 15
    assert ids.isdisjoint(REPAIR_IDS)
    assert "prompt_prefix_layout_optimization" in ids
    assert "distributed_capacity_serving_architecture" in ids
    assert set(registry["readiness_states"]) == READINESS_STATES
    for entry in entries:
        assert entry["primary_metrics"]
        assert entry["required_instrumentation"]
        assert entry["instrumentation_readiness_state"] in READINESS_STATES
        assert entry["optimization_domain"] in registry["optimization_domains"]


def test_metric_semantics_are_distinct_and_do_not_overclaim_missing_counters() -> None:
    registry = build_observability_registry()
    semantics = cast(dict[str, str], registry["measurement_semantics"])
    by_id = {
        str(entry["optimization_id"]): entry
        for entry in cast(list[dict[str, Any]], registry["optimizations"])
    }

    assert semantics["measured"] != semantics["derived"]
    assert semantics["derived"] != semantics["estimated"]
    assert semantics["estimated"] != semantics["unavailable"]
    assert by_id["prefix_cache_verification_tuning"]["instrumentation_readiness_state"] == (
        "requires_engine_metrics"
    )
    assert (
        "cache hit/miss counters"
        in by_id["prefix_cache_verification_tuning"]["missing_instrumentation"]
    )
    assert by_id["speculative_decoding"]["instrumentation_readiness_state"] == (
        "unsupported_current_runtime"
    )
    assert "drafted_tokens" in by_id["speculative_decoding"]["required_instrumentation"]


def test_prefix_analysis_is_static_estimated_and_exact_prefix_only() -> None:
    first = analyze_prefix_opportunity()
    second = analyze_prefix_opportunity()

    assert first["inference_executed"] is False
    assert first["optimization_applied"] is False
    assert first["semantic_similarity_counted"] is False
    assert first["historical_cache_reuse_claimed"] is False
    assert first["summary"]["rows_scanned"] == 10000
    assert first["summary"]["prefix_family_count"] > 0
    assert first["summary"] == second["summary"]
    assert first["source_hashes"] == second["source_hashes"]


def test_prefix_comparison_uses_exact_leading_tokens_not_semantic_similarity() -> None:
    result = compare_prefix_layouts(
        ["alpha beta customer question", "alpha beta policy question"],
        ["alpha beta gamma customer question", "alpha beta gamma policy question"],
    )

    assert result["semantic_similarity_used"] is False
    assert (
        result["candidate_longest_exact_common_prefix"]
        > result["baseline_longest_exact_common_prefix"]
    )


def test_event_schema_validates_discriminated_events_and_rejects_invalid_payload() -> None:
    schema = build_event_schema()
    valid_event = schema["example_event"]

    parsed = OBSERVABILITY_EVENT_ADAPTER.validate_python(valid_event)
    assert parsed.schema_version == "core_optimization_observability.v1"
    assert parsed.event_type == "optimization_decision"

    invalid_event = {
        **valid_event,
        "event_type": "prefix_cache_hit",
        "payload": {
            "payload_type": "prefix_cache_hit",
            "cache_hit_count": 1,
            "cache_miss_count": 0,
            "invalid_field": "must reject",
        },
    }
    with pytest.raises(ValidationError):
        OBSERVABILITY_EVENT_ADAPTER.validate_python(invalid_event)


def test_saved_event_schema_artifact_is_planning_only() -> None:
    payload = _json(REPO_ROOT / EVENT_SCHEMA_JSON_PATH)

    assert payload["status"] == "EVENT_SCHEMA_READY_PLANNING_ONLY"
    assert "prefix_cache_hit" in payload["event_types"]
    assert "prompt_layout_rendered" in payload["event_types"]
    assert "prefix_family_assigned" in payload["event_types"]
    assert "static_metric_computed" in payload["event_types"]
    assert payload["example_event"]["measurement_type"] == "estimated"


def test_adapters_parse_existing_artifacts_without_live_engine_or_mutation() -> None:
    reports = [adapter.inspect().to_dict() for adapter in all_adapters()]
    by_name = {str(report["adapter_id"]): report for report in reports}

    assert by_name["main_inference_gpu_telemetry"]["live_engine_required"] is False
    assert by_name["main_inference_gpu_telemetry"]["mutates_artifacts"] is False
    assert all(
        field["support_state"] == "missing"
        for field in by_name["engine_metrics_unavailable"]["field_statuses"]
    )
    missing_fields = {
        str(field["field_name"])
        for field in by_name["engine_metrics_unavailable"]["field_statuses"]
        if field["support_state"] == "missing"
    }
    assert "cache_hit_count" in missing_fields
    assert "active_batch_size" in missing_fields


def test_scenario_registry_keeps_unrun_one_factor_scenarios_planned() -> None:
    registry = build_updated_scenario_registry()
    scenarios = {
        str(item["scenario_id"]): item for item in cast(list[dict[str, Any]], registry["scenarios"])
    }

    assert registry["status"] == "SCENARIO_REGISTRY_PLANNED_WITH_OBSERVABILITY"
    assert registry["champion_selected"] is False
    for scenario_id in SCENARIO_PLAN_PATHS:
        if scenario_id == "coreopt_prefix_layout_static_v1":
            assert scenarios[scenario_id]["result_type"] == "measured_static_analysis"
            assert scenarios[scenario_id]["decision"] == "MISSING_CONFIGURATION"
        else:
            assert scenarios[scenario_id]["result_type"] == "planned"
        assert scenarios[scenario_id]["instrumentation_readiness"] in READINESS_STATES
    assert scenarios["optimized_inference_v1"]["result_type"] == "missing_not_created"
    assert scenarios["optimized_inference_v1"]["artifact_paths"] == []


def test_generated_yaml_registry_matches_scenario_and_has_no_measured_optimization() -> None:
    observability = cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/core_optimization_observability.yaml").read_text()),
    )
    scenario = cast(
        dict[str, Any],
        yaml.safe_load(Path("configs/core_optimization_scenario_registry.yaml").read_text()),
    )

    assert observability["does_not_execute_inference"] is True
    assert observability["does_not_create_optimized_inference_v1"] is True
    assert len(observability["optimizations"]) == 15
    scenarios = {
        str(item["scenario_id"]): item for item in cast(list[dict[str, Any]], scenario["scenarios"])
    }
    assert scenarios["coreopt_prefix_layout_static_v1"]["result_type"] == (
        "measured_static_analysis"
    )
    assert scenarios["coreopt_prefix_layout_static_v1"]["inference_executed"] is False
    assert scenario["observability_framework"]["optimized_inference_v1_created"] is False


def test_product_platform_exposes_observability_cards_without_fake_zeroes() -> None:
    data = ProductPlatformData()
    cards = data.core_observability_cards()
    missing = data.core_observability_missing_instrumentation()
    prefix = data.prefix_opportunity_analysis()

    assert cards["result_type"] == "planned"
    assert len(cards["cards"]) == 15
    assert cards["prefix_summary"]["rows_scanned"] == 10000
    assert prefix["summary"]["rows_scanned"] == 10000
    assert missing["missing_is_not_zero"] is True
    assert all(row["measurement_type"] == "unavailable" for row in missing["rows"])


def test_generated_observability_artifacts_exist() -> None:
    required = [
        "core_optimization_observability_registry.json",
        "core_optimization_observability_readiness.json",
        "core_optimization_observability_readiness.csv",
        "main_inference_v1_observability_inventory.json",
        "main_inference_v1_observability_inventory.csv",
        "core_optimization_event_schema.json",
        "core_optimization_ui_observability_contract.json",
        "coreopt_prefix_layout_static_v1_instrumentation_plan.json",
        "coreopt_scheduler_batch_vllm_v1_instrumentation_plan.json",
        "coreopt_prefix_cache_vllm_v1_instrumentation_plan.json",
        "coreopt_chunked_prefill_sglang_v1_instrumentation_plan.json",
    ]

    for filename in required:
        assert (REPO_ROOT / MAIN_PROCESSED / filename).exists()
