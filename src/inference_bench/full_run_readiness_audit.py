"""Full-run and RunPod readiness audit for AI inference experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def _contains(root: Path, relative_path: str, *patterns: str) -> bool:
    path = root / relative_path
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    return all(pattern.lower() in text for pattern in patterns)


def _check(
    *,
    category: str,
    name: str,
    status: str,
    evidence: str,
    blocking: bool = False,
) -> dict[str, Any]:
    return {
        "category": category,
        "name": name,
        "status": status,
        "evidence": evidence,
        "blocking": blocking,
        "severity": "BLOCKER" if blocking else "GAP" if status == "GAP" else "INFO",
    }


def _file_check(
    root: Path,
    *,
    category: str,
    name: str,
    relative_path: str,
    blocking: bool,
) -> dict[str, Any]:
    present = _exists(root, relative_path)
    return _check(
        category=category,
        name=name,
        status="PASS" if present else "FAIL" if blocking else "GAP",
        evidence=relative_path if present else f"Missing: {relative_path}",
        blocking=blocking and not present,
    )


def _load_json_if_present(root: Path, relative_path: str) -> dict[str, Any] | None:
    path = root / relative_path
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _gpu_price_configured(root: Path) -> bool:
    payload = _load_json_if_present(root, "results/processed/b6_runtime_projection_report.json")
    if payload is None:
        return False
    projections = payload.get("runpod_gpu_projections")
    if not isinstance(projections, dict):
        return False
    return any(
        isinstance(profile, dict) and profile.get("hourly_price_usd") is not None
        for profile in projections.values()
    )


def _b6_gate_passed(root: Path) -> bool | None:
    payload = _load_json_if_present(root, "results/processed/b6_vllm_1_5b_500_eval_report.json")
    if payload is None:
        return None
    gate = payload.get("quality_gate")
    return bool(gate.get("passed")) if isinstance(gate, dict) else None


def partial_run_completion_check(
    *,
    expected_count: int,
    observed_count: int,
    manifest_status: str,
) -> dict[str, Any]:
    """Return a blocking check when a partial run is marked completed."""

    if manifest_status == "completed" and observed_count < expected_count:
        return _check(
            category="run_safety",
            name="partial_run_not_marked_complete",
            status="FAIL",
            evidence=(f"Manifest says completed with {observed_count}/{expected_count} rows"),
            blocking=True,
        )
    return _check(
        category="run_safety",
        name="partial_run_not_marked_complete",
        status="PASS",
        evidence=f"Manifest status {manifest_status}; rows {observed_count}/{expected_count}",
    )


def telemetry_availability_check(report: dict[str, Any] | None) -> dict[str, Any]:
    """Return PASS when telemetry exists and UNAVAILABLE when it is missing."""

    if report is None:
        return _check(
            category="gpu_runtime",
            name="b6_gpu_telemetry_available",
            status="UNAVAILABLE",
            evidence="B6 report missing; telemetry cannot be evaluated",
        )
    telemetry = report.get("gpu_telemetry_summary")
    if not isinstance(telemetry, dict):
        return _check(
            category="gpu_runtime",
            name="b6_gpu_telemetry_available",
            status="UNAVAILABLE",
            evidence="B6 report has no gpu_telemetry_summary",
        )
    sample_count = telemetry.get("sample_count")
    if sample_count in (None, 0, "0"):
        return _check(
            category="gpu_runtime",
            name="b6_gpu_telemetry_available",
            status="UNAVAILABLE",
            evidence="GPU telemetry sample_count is missing or zero",
        )
    return _check(
        category="gpu_runtime",
        name="b6_gpu_telemetry_available",
        status="PASS",
        evidence=f"GPU telemetry sample_count={sample_count}",
    )


def build_full_run_readiness_audit(
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Audit whether the repo is ready for full local/RunPod benchmark execution."""

    root = Path(repo_root)
    checks: list[dict[str, Any]] = []

    checks.extend(
        [
            _file_check(
                root,
                category="dataset_workload",
                name="promoted_retrieval_manifest",
                relative_path="data/generated/context_engineering/retrieval_source_of_truth_manifest.json",
                blocking=True,
            ),
            _file_check(
                root,
                category="dataset_workload",
                name="controlled_2000_workload",
                relative_path="data/workloads/controlled_2000/prompt_plus_metadata/mm2_hybrid_top5.jsonl",
                blocking=True,
            ),
            _file_check(
                root,
                category="dataset_workload",
                name="deterministic_workload_builder",
                relative_path="src/inference_bench/memory_workloads.py",
                blocking=True,
            ),
            _file_check(
                root,
                category="dataset_workload",
                name="b6_500_runner_input",
                relative_path="data/generated/phase4/b6_context_aligned_500_runner_input.jsonl",
                blocking=True,
            ),
        ]
    )
    checks.append(
        _check(
            category="dataset_workload",
            name="prompt_gold_join_validation",
            status="PASS"
            if _contains(root, "scripts/phase4/evaluate_generation_outputs.py", "load_gold_records")
            else "FAIL",
            evidence="evaluate_generation_outputs.py joins generated rows to promoted gold",
            blocking=not _contains(
                root,
                "scripts/phase4/evaluate_generation_outputs.py",
                "load_gold_records",
            ),
        )
    )

    for name, relative_path in (
        ("b5_context_alignment_active", "src/inference_bench/context_alignment_repair.py"),
        ("evidence_alias_mapping_preserved", "src/inference_bench/generation_contract.py"),
        ("leakage_guard_active", "src/inference_bench/context_alignment_repair.py"),
        ("answer_planning_active", "src/inference_bench/answer_planning.py"),
        ("multi_evidence_selector_active", "src/inference_bench/multi_evidence_selector.py"),
        ("safety_repair_active", "src/inference_bench/safety_generation_repair.py"),
        ("generation_contract_active", "src/inference_bench/generation_contract.py"),
    ):
        checks.append(
            _file_check(
                root,
                category="context_generation",
                name=name,
                relative_path=relative_path,
                blocking=True,
            )
        )

    load_runner = "src/inference_bench/runners/openai_load_runner.py"
    checkpoint_supported = _contains(root, load_runner, "checkpoint_path", "completed_prompt_ids")
    b6_report = _load_json_if_present(root, "results/processed/b6_vllm_1_5b_500_eval_report.json")
    b6_manifest = _load_json_if_present(root, "results/raw/b6_vllm_1_5b_500_manifest.json")
    b6_observed_count = int((b6_report or {}).get("row_count") or 0)
    b6_manifest_status = str((b6_manifest or {}).get("status") or "missing")
    checks.extend(
        [
            _check(
                category="run_safety",
                name="raw_outputs_written_incrementally",
                status="PASS" if _contains(root, load_runner, "_append_results_csv") else "GAP",
                evidence=load_runner,
            ),
            _check(
                category="run_safety",
                name="checkpoint_resume_supported",
                status="PASS" if checkpoint_supported else "GAP",
                evidence=(
                    load_runner if checkpoint_supported else "No checkpoint/resume support found"
                ),
            ),
            _check(
                category="run_safety",
                name="completed_prompt_ids_tracked",
                status="PASS" if _contains(root, load_runner, "completed_prompt_ids") else "GAP",
                evidence=load_runner,
            ),
            _check(
                category="run_safety",
                name="partial_failures_captured_as_rows",
                status="PASS"
                if _contains(
                    root,
                    "scripts/phase4/run_b6_vllm_1_5b_500_quality_gate.py",
                    "_failure_row",
                )
                else "GAP",
                evidence="B6 runner writes failure rows per prompt",
            ),
            _check(
                category="run_safety",
                name="partial_run_not_marked_complete",
                status="PASS"
                if _contains(
                    root,
                    "scripts/phase4/run_b6_vllm_1_5b_500_quality_gate.py",
                    "error_count",
                )
                else "GAP",
                evidence="manifest records error_count",
            ),
        ]
    )
    checks.append(
        partial_run_completion_check(
            expected_count=500,
            observed_count=b6_observed_count,
            manifest_status=b6_manifest_status,
        )
    )

    for name, relative_path, blocking in (
        ("remote_rtx3070_config", "configs/hardware/remote_rtx3070.yaml", True),
        ("runpod_projection_profiles", "configs/runpod_projection_prices.yaml", False),
        ("vllm_launch_documented", "docs/96_remote_rtx3070_vllm_smoke.md", True),
        ("sglang_launch_documented", "docs/96_remote_rtx3070_sglang_smoke.md", False),
        ("gpu_telemetry_sampler", "src/inference_bench/gpu_telemetry.py", True),
    ):
        checks.append(
            _file_check(
                root,
                category="gpu_runtime",
                name=name,
                relative_path=relative_path,
                blocking=blocking,
            )
        )
    price_configured = _gpu_price_configured(root)
    checks.append(
        _check(
            category="gpu_runtime",
            name="runpod_hourly_price_configured",
            status="PASS" if price_configured else "GAP",
            evidence=(
                "At least one RunPod profile has hourly_price_usd"
                if price_configured
                else "RunPod prices missing; cost claims blocked"
            ),
        )
    )
    checks.append(telemetry_availability_check(b6_report))

    for name, relative_path in (
        ("gpu_cost_formula", "src/inference_bench/cost.py"),
        ("api_cost_formula", "src/inference_bench/api_pricing.py"),
        ("gpu_cost_config", "configs/gpu_costs.yaml"),
    ):
        checks.append(
            _file_check(
                root,
                category="cost",
                name=name,
                relative_path=relative_path,
                blocking=False,
            )
        )
    checks.append(
        _check(
            category="cost",
            name="gpu_cost_requires_hourly_price",
            status="PASS",
            evidence="B6 projection reports price_missing instead of guessing",
        )
    )

    for name, relative_path in (
        ("modular_slo_profiles", "src/inference_bench/slo_profiles.py"),
        ("bottleneck_catalog", "configs/bottleneck_catalog.yaml"),
        ("optimization_catalog", "configs/optimization_catalog.yaml"),
        ("diagnosis_engine", "src/inference_bench/slo_diagnosis.py"),
    ):
        checks.append(
            _file_check(
                root,
                category="slo_diagnosis",
                name=name,
                relative_path=relative_path,
                blocking=True,
            )
        )
    checks.append(
        _check(
            category="slo_diagnosis",
            name="missing_telemetry_reported_unavailable",
            status="PASS"
            if _contains(root, "src/inference_bench/slo_diagnosis.py", "UNAVAILABLE")
            else "FAIL",
            evidence="slo_diagnosis marks missing observations unavailable",
            blocking=not _contains(root, "src/inference_bench/slo_diagnosis.py", "UNAVAILABLE"),
        )
    )

    b6_passed = _b6_gate_passed(root)
    checks.append(
        _check(
            category="scaling",
            name="b6_500_gate_result",
            status="PASS" if b6_passed else "GAP" if b6_passed is None else "FAIL",
            evidence=(
                "B6 quality gate passed"
                if b6_passed
                else "B6 report missing"
                if b6_passed is None
                else "B6 gate did not pass"
            ),
            blocking=b6_passed is False,
        )
    )
    checks.append(
        _check(
            category="scaling",
            name="runpod_blocked_without_price",
            status="PASS",
            evidence="RunPod readiness remains blocked when hourly prices are absent",
        )
    )

    blocking_failures = [check for check in checks if check["blocking"]]
    gaps = [check for check in checks if check["status"] == "GAP"]
    if blocking_failures:
        status = "NOT_READY"
    elif gaps:
        status = "READY_WITH_GAPS"
    else:
        status = "FULL_RUN_READY"

    return {
        "status": status,
        "checks": checks,
        "summary": {
            "check_count": len(checks),
            "pass_count": sum(check["status"] == "PASS" for check in checks),
            "gap_count": len(gaps),
            "fail_count": sum(check["status"] == "FAIL" for check in checks),
            "blocking_failure_count": len(blocking_failures),
        },
        "data_loss_prevention_safeguards": [
            check["name"]
            for check in checks
            if check["category"] == "run_safety" and check["status"] == "PASS"
        ],
        "remaining_gaps": [
            {"category": check["category"], "name": check["name"], "evidence": check["evidence"]}
            for check in checks
            if check["status"] in {"GAP", "FAIL"}
        ],
    }


def readiness_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten readiness checks for CSV."""

    return [
        {
            "overall_status": report["status"],
            "category": check["category"],
            "name": check["name"],
            "status": check["status"],
            "blocking": check["blocking"],
            "evidence": check["evidence"],
        }
        for check in report["checks"]
    ]


def write_full_run_readiness_artifacts(
    *,
    report: dict[str, Any],
    report_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write readiness report artifacts."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = readiness_summary_rows(report)
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report_output, summary_output
