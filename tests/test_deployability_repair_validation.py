from __future__ import annotations

import json
import subprocess
from pathlib import Path

from inference_bench.deployability_repair_validation import (
    CASE_SPECS,
    CORE_OPTIMIZATION_FLAGS,
    REPAIR_FLAGS,
    detect_execution_device,
    execute_repair_validation,
    run_deployability_repair_validation,
    select_targeted_sample,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _evidence_ids_for_case(case_index: int, count: int) -> list[str]:
    return [f"DOC-{case_index}-{index}" for index in range(1, count + 1)]


def _build_dataset(root: Path) -> None:
    by_vertical: dict[str, dict[str, list[dict[str, object]]]] = {}
    for index, spec in enumerate(CASE_SPECS, start=1):
        evidence_count = max(spec.min_evidence_count, 1)
        if spec.max_evidence_count is not None:
            evidence_count = min(evidence_count, spec.max_evidence_count)
        if spec.min_evidence_count == 0 and spec.expected_status == "out_of_scope":
            evidence_count = 0
        evidence_ids = _evidence_ids_for_case(index, evidence_count)
        prompt_id = f"{spec.vertical}_{index:04d}"
        vertical_rows = by_vertical.setdefault(
            spec.vertical,
            {"prompts": [], "gold": [], "kb": []},
        )
        vertical_rows["prompts"].append(
            {
                "prompt_id": prompt_id,
                "vertical": spec.vertical,
                "question": f"Question for {spec.case_id}",
                "task_type": "answer_grounded",
                "expected_status": spec.expected_status,
                "expected_action": spec.expected_status,
                "expected_output_format": "generation_contract_json",
                "required_doc_ids": evidence_ids,
                "required_chunk_ids": evidence_ids,
            }
        )
        vertical_rows["gold"].append(
            {
                "prompt_id": prompt_id,
                "vertical": spec.vertical,
                "expected_status": spec.expected_status,
                "expected_action": spec.expected_status,
                "expected_output_format": "generation_contract_json",
                "required_doc_ids": evidence_ids,
                "required_chunk_ids": evidence_ids,
                "metadata": {
                    "expected_action": spec.expected_status,
                    "required_evidence_ids": evidence_ids,
                },
                "must_not_include": ["diagnosis", "treatment advice"],
            }
        )
        for evidence_id in evidence_ids:
            vertical_rows["kb"].append(
                {
                    "doc_id": evidence_id,
                    "chunk_id": evidence_id,
                    "source_id": f"{spec.vertical}_source",
                    "source_type": "synthetic_test_kb",
                    "title": f"Evidence {evidence_id}",
                    "body": f"Evidence {evidence_id} supports {spec.case_id}.",
                    "metadata": {"case_id": spec.case_id},
                }
            )

    for vertical, rows in by_vertical.items():
        vertical_root = root / vertical
        _write_jsonl(vertical_root / f"{vertical}_prompts_2000.jsonl", rows["prompts"])
        _write_jsonl(vertical_root / f"{vertical}_gold_2000.jsonl", rows["gold"])
        _write_jsonl(vertical_root / f"{vertical}_kb_2000.jsonl", rows["kb"])


def test_targeted_sample_covers_repairs_verticals_and_uses_no_leaked_gold(tmp_path: Path) -> None:
    dataset_root = tmp_path / "scaleup"
    _build_dataset(dataset_root)

    sample_rows, gold_by_prompt_id = select_targeted_sample(dataset_root=dataset_root)

    assert len(sample_rows) == len(CASE_SPECS)
    assert {row["vertical"] for row in sample_rows} == {
        "airline",
        "healthcare_admin",
        "retail",
        "finance",
        "research_ai",
    }
    assert {row["repair_family"] for row in sample_rows} == {
        "bounded_citation",
        "escalation",
        "evidence_formatting",
        "mm4_bounded",
        "prompt_contract",
        "safety_wording",
    }
    assert len({row["prompt_id"] for row in sample_rows}) == len(sample_rows)
    assert set(gold_by_prompt_id) == {str(row["prompt_id"]) for row in sample_rows}
    assert all("citation_aliases:" not in str(row["rendered_prompt"]) for row in sample_rows)
    assert all(
        row["core_optimization_flags"] == list(CORE_OPTIMIZATION_FLAGS) for row in sample_rows
    )


def test_repair_validation_uses_existing_evaluator_and_passes_sample(tmp_path: Path) -> None:
    dataset_root = tmp_path / "scaleup"
    _build_dataset(dataset_root)
    sample_rows, gold_by_prompt_id = select_targeted_sample(dataset_root=dataset_root)

    results, traces, evaluations = execute_repair_validation(sample_rows, gold_by_prompt_id)

    assert len(results) == len(CASE_SPECS)
    assert len(traces) == len(CASE_SPECS)
    assert len(evaluations) == len(CASE_SPECS)
    assert all(result["inference_executed"] is False for result in results)
    assert all(result["repair_logic_executed"] is True for result in results)
    assert all(result["repair_successful"] is True for result in results)
    assert all(evaluation["generation_contract_valid"] is True for evaluation in evaluations)
    assert all(evaluation["safety_violation"] is False for evaluation in evaluations)
    assert any(trace["repair_family"] == "mm4_bounded" for trace in traces)
    assert all(set(trace["enabled_repair_ids"]).issubset(set(REPAIR_FLAGS)) for trace in traces)


def test_device_fallback_avoids_a100_and_allows_cpu() -> None:
    def no_gpu_runner(_: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout="")

    cpu_report = detect_execution_device(command_runner=no_gpu_runner)
    assert cpu_report["selected_device"] == "cpu"
    assert cpu_report["a100_selected"] is False
    assert cpu_report["inference_executed"] is False

    def rtx_runner(_: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090\n",
        )

    rtx_report = detect_execution_device(command_runner=rtx_runner)
    assert rtx_report["selected_device"] == "local_rtx"
    assert rtx_report["a100_selected"] is False


def test_validation_run_writes_artifacts_and_does_not_create_optimized_result(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "scaleup"
    artifact_root = tmp_path / "experiments" / "repairs" / "deployability_repair_validation_v1"
    backup_root = tmp_path / "backups"
    main_root = tmp_path / "experiments" / "main" / "main_inference_v1"
    main_root.mkdir(parents=True)
    _build_dataset(dataset_root)

    def no_gpu_runner(_: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["nvidia-smi"], returncode=0, stdout="")

    summary = run_deployability_repair_validation(
        dataset_root=dataset_root,
        artifact_root=artifact_root,
        main_experiment_root=main_root,
        backup_root=backup_root,
        command_runner=no_gpu_runner,
    )

    assert summary["status"] == "SAMPLE_VALIDATED"
    assert summary["sample_count"] == len(CASE_SPECS)
    assert summary["core_optimization_eligible"] is True
    assert summary["full_scale_validated"] is False
    assert summary["backup_verification"]["passed"] is True
    assert (artifact_root / "raw/deployability_repair_validation_v1_manifest.json").exists()
    assert (
        artifact_root / "processed/deployability_repair_validation_v1_validation_gate_report.json"
    ).exists()
    assert not (tmp_path / "experiments" / "optimized" / "optimized_inference_v1").exists()
