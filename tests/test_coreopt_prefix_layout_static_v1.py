from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inference_bench import coreopt_prefix_layout_static as static
from inference_bench.product_platform import ProductPlatformData
from inference_bench.workload_adapter import workload_record_to_runner_item


def _patched_tokenizer() -> tuple[dict[str, Any], static.TokenizeFn]:
    return (
        {
            "tokenizer_available": False,
            "tokenizer_source": "unit_test_tokenizer",
            "fallback_used": True,
        },
        lambda text: text.replace("\n", " ").split(),
    )


def test_baseline_layout_preserves_authoritative_runner_rendering() -> None:
    record = next(
        iter(
            static.load_workload_records(
                static.workload_paths(("mm2_hybrid_top5",))[0],
                limit=1,
            )
        )
    )

    rendered = static.render_baseline_prompt(record)

    assert rendered.prompt == workload_record_to_runner_item(record).prompt
    assert tuple(section.section_id for section in rendered.sections) == (
        "system",
        "memory_mode",
        "retrieved_evidence",
        "user_question",
        "output_contract",
    )


def test_candidate_layout_reorders_only_existing_section_content() -> None:
    record = next(
        iter(
            static.load_workload_records(
                static.workload_paths(("mm2_hybrid_top5",))[0],
                limit=1,
            )
        )
    )

    baseline = static.render_baseline_prompt(record)
    candidate = static.render_prefix_optimized_prompt(record)
    baseline_hashes = {
        section.section_id: static._sha256_text(section.text) for section in baseline.sections
    }
    candidate_hashes = {
        section.section_id: static._sha256_text(section.text) for section in candidate.sections
    }

    assert baseline_hashes == candidate_hashes
    assert candidate.prompt.index("OUTPUT CONTRACT:") < candidate.prompt.index(
        "RETRIEVED EVIDENCE:"
    )
    assert candidate.prompt.index("RETRIEVED EVIDENCE:") < candidate.prompt.index("USER QUESTION:")


def test_static_analysis_reports_prefix_opportunity_without_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(static, "_load_tokenizer", _patched_tokenizer)

    analysis = static.analyze_prefix_layout_static(
        workload_file_paths=static.workload_paths(("mm1_dense_top5",)),
        limit_per_memory_mode=20,
    )
    summary = analysis["prefix_summary"]

    assert summary["workload_rows_scanned"] == 20
    assert summary["inference_executed"] is False
    assert summary["cache_hits_measured"] is False
    assert summary["latency_claimed"] is False
    assert summary["deltas"]["candidate_minus_baseline_total_input_tokens"] == 0
    assert summary["deltas"]["candidate_minus_baseline_mean_common_prefix_tokens"] > 0
    assert analysis["equivalence_report"]["status"] == "PASS"
    assert analysis["decision"]["decision"] == "MISSING_CONFIGURATION"


def test_static_artifact_writer_produces_complete_tmp_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(static, "_load_tokenizer", _patched_tokenizer)

    result = static.write_coreopt_prefix_layout_static_artifacts(
        output_root=tmp_path,
        workload_file_paths=static.workload_paths(("mm0_no_context",)),
        limit_per_memory_mode=5,
        update_registry=False,
    )

    for key in (
        "manifest",
        "baseline_layout",
        "candidate_layout",
        "prefix_summary",
        "equivalence_report",
        "decision",
        "checksums",
    ):
        assert Path(result["artifact_paths"][key]).exists()

    summary = json.loads(Path(result["artifact_paths"]["prefix_summary"]).read_text())
    assert summary["workload_rows_scanned"] == 5
    assert summary["result_type"] == static.RESULT_TYPE


def test_generated_static_artifacts_are_full_scale_and_plan_safe() -> None:
    root = static.EXPERIMENT_ROOT
    summary = json.loads((root / "coreopt_prefix_layout_static_v1_prefix_summary.json").read_text())
    decision = json.loads((root / "coreopt_prefix_layout_static_v1_decision.json").read_text())
    equivalence = json.loads(
        (root / "coreopt_prefix_layout_static_v1_equivalence_report.json").read_text()
    )

    assert summary["workload_rows_scanned"] == 40000
    assert summary["result_type"] == "measured_static_analysis"
    assert summary["inference_executed"] is False
    assert summary["latency_claimed"] is False
    assert summary["deltas"]["candidate_minus_baseline_mean_common_prefix_tokens"] > 0
    assert decision["decision"] == "MISSING_CONFIGURATION"
    assert equivalence["status"] == "PASS"
    assert equivalence["requires_inference_validation"] is True


def test_product_platform_exposes_static_prefix_layout_experiment() -> None:
    data = ProductPlatformData()
    payload = data.coreopt_prefix_layout_static_experiment()

    assert payload["summary"]["scenario_id"] == "coreopt_prefix_layout_static_v1"
    assert payload["summary"]["workload_rows_scanned"] == 40000
    assert payload["decision"]["decision"] == "MISSING_CONFIGURATION"
    assert payload["equivalence"]["status"] == "PASS"
    assert payload["layouts"]["baseline"]["raw_prompt_text_included"] is False
    assert payload["metrics"]["prefix_families"]
