from __future__ import annotations

import json
from typing import Any

from inference_bench.b7_controlled_baseline import (
    B7_EXPECTED_PROMPT_COUNT,
    B7_EXPECTED_PROMPTS_PER_VERTICAL,
    B7_MODEL_ALIAS,
    B7_MODEL_ID,
    classify_b7_quality_gate,
    preflight_b7_runner_rows,
)
from inference_bench.context_corpora import VERTICALS


def _runtime_selection() -> dict[str, object]:
    return {
        "model_alias": B7_MODEL_ALIAS,
        "model_id": B7_MODEL_ID,
        "runtime": "vllm",
        "engine": "vllm",
        "backend_type": "self_hosted_gpu",
        "hardware_type": "remote_rtx3070",
        "live_run_allowed": True,
    }


def _runner_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        for index in range(B7_EXPECTED_PROMPTS_PER_VERTICAL):
            prompt_id = f"{vertical}_{index:04d}"
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": "SYSTEM:\nUse [EVIDENCE 1].",
                    "expected_output": "generation_contract_json",
                    "metadata": {
                        "vertical": vertical,
                        "canonical_ids_exposed_to_model": "false",
                        "gold_evidence_ids": json.dumps([f"{vertical}-gold-{index}"]),
                        "citation_id_aliases": json.dumps({"E1": [f"{vertical}-gold-{index}"]}),
                        "b5_required_labels": "E1",
                    },
                }
            )
    assert len(rows) == B7_EXPECTED_PROMPT_COUNT
    return rows


def test_b7_preflight_passes_balanced_leakage_free_runner_input() -> None:
    report = preflight_b7_runner_rows(
        _runner_rows(),
        model_alias=B7_MODEL_ALIAS,
        model_id=B7_MODEL_ID,
        runtime_selection=_runtime_selection(),
        artifact_sync_dry_run_passed=True,
        checkpoint_resume_enabled=True,
        manifest_enabled=True,
    )

    assert report["status"] == "PREFLIGHT_PASSED_B7_CONTROLLED_1000_BASELINE"
    assert report["passed"] is True
    assert report["prompts_per_vertical"] == {
        vertical: B7_EXPECTED_PROMPTS_PER_VERTICAL for vertical in VERTICALS
    }


def test_b7_preflight_blocks_missing_required_evidence_and_leakage() -> None:
    rows = _runner_rows()
    rows[0]["metadata"] = {
        **dict(rows[0]["metadata"]),
        "b5_required_labels": "",
        "citation_id_aliases": "{}",
        "canonical_ids_exposed_to_model": "true",
    }

    report = preflight_b7_runner_rows(
        rows,
        model_alias=B7_MODEL_ALIAS,
        model_id=B7_MODEL_ID,
        runtime_selection=_runtime_selection(),
        artifact_sync_dry_run_passed=True,
        checkpoint_resume_enabled=True,
        manifest_enabled=True,
    )

    assert report["status"] == "PREFLIGHT_BLOCKED_B7_CONTROLLED_1000_BASELINE"
    assert "required_evidence_present_in_e1_e5" in report["failed_checks"]
    assert "no_canonical_id_leakage_flagged" in report["failed_checks"]


def test_b7_quality_gate_passes_only_with_quality_sync_and_telemetry() -> None:
    summary = {
        "json_valid_rate": 0.99,
        "generation_contract_valid_rate": 0.98,
        "evidence_match_rate": 0.94,
        "grounded_rate": 0.93,
        "safety_violation_count": 0,
        "truncation_rate": 0.01,
    }
    per_vertical = [
        {"vertical": vertical, "evidence_match_rate": 0.90, "grounded_rate": 0.90}
        for vertical in VERTICALS
    ]

    gate = classify_b7_quality_gate(
        summary=summary,
        per_vertical_quality=per_vertical,
        completed_count=1000,
        artifact_sync_verified=True,
        telemetry_sample_count=5,
    )

    assert gate["status"] == "B7_CONTROLLED_1000_BASELINE_READY"
    assert gate["next_api_load_probe_allowed"] is True


def test_b7_quality_gate_blocks_low_vertical_floor() -> None:
    summary = {
        "json_valid_rate": 0.99,
        "generation_contract_valid_rate": 0.98,
        "evidence_match_rate": 0.94,
        "grounded_rate": 0.93,
        "safety_violation_count": 0,
        "truncation_rate": 0.01,
    }
    per_vertical = [
        {
            "vertical": vertical,
            "evidence_match_rate": 0.84 if vertical == "research_ai" else 0.90,
            "grounded_rate": 0.90,
        }
        for vertical in VERTICALS
    ]

    gate = classify_b7_quality_gate(
        summary=summary,
        per_vertical_quality=per_vertical,
        completed_count=1000,
        artifact_sync_verified=True,
        telemetry_sample_count=5,
    )

    assert gate["status"] == "B7_CONTROLLED_1000_BASELINE_BLOCKED"
    assert "vertical_evidence_match_rate_min" in gate["failed_metrics"]
