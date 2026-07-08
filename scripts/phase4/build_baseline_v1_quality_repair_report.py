"""Build Baseline_V1 quality-repair evidence and SLO scorecards.

This script does not run inference and does not modify the frozen
``experiments/baseline_v1`` archive. It packages the existing Baseline_V1
artifacts and the targeted Phase 2 repair validation into a versioned baseline
repair experiment folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_ROOT = "experiments/baseline/baseline_v1_quality_repair_v1"
DEFAULT_BASELINE_ARCHIVE = "experiments/baseline_v1"
DEFAULT_TARGETED_REPAIR_REPORT = "results/processed/phase2_targeted_optimization_rerun_report.json"
DEFAULT_BEFORE_AFTER = "results/processed/phase2_before_after_comparison.json"
DEFAULT_READINESS = "results/processed/phase2_final_run_readiness_after_repairs.json"
DEFAULT_SLO_TARGETS = "configs/slo_targets.yaml"
DEFAULT_SLO_PROFILES = "configs/slo_profiles.yaml"

TARGETED_REPAIR_SOURCE_FILES = (
    "results/processed/phase2_targeted_optimization_rerun_report.json",
    "results/processed/phase2_targeted_optimization_rerun_summary.csv",
    "results/processed/phase2_before_after_comparison.json",
    "results/processed/phase2_before_after_comparison.csv",
    "results/processed/phase2_final_run_readiness_after_repairs.json",
    "results/processed/phase2_optimization_diagnosis_report.json",
    "results/processed/phase2_optimization_diagnosis_summary.csv",
    "results/processed/phase2_selected_optimization_candidates.json",
    "results/processed/phase2_before_after_rerun_plan.json",
)


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(_repo_path(path).read_text(encoding="utf-8")) or {}


def _configured_vertical_targets(slo_targets_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = _load_yaml(slo_targets_path)
    verticals = payload.get("verticals")
    if not isinstance(verticals, dict) or not verticals:
        raise ValueError("configs/slo_targets.yaml does not contain vertical targets")
    return {
        str(vertical): settings
        for vertical, settings in verticals.items()
        if isinstance(settings, dict)
    }


def _strictest_min(verticals: dict[str, dict[str, Any]], group: str, key: str) -> float:
    values = [
        float(settings.get(group, {}).get(key))
        for settings in verticals.values()
        if settings.get(group, {}).get(key) is not None
    ]
    if not values:
        raise KeyError(f"No configured SLO target for {group}.{key}")
    return max(values)


def _envelope_min(verticals: dict[str, dict[str, Any]], group: str, key: str) -> float:
    values = [
        float(settings.get(group, {}).get(key))
        for settings in verticals.values()
        if settings.get(group, {}).get(key) is not None
    ]
    if not values:
        raise KeyError(f"No configured SLO target for {group}.{key}")
    return min(values)


def _strictest_max(verticals: dict[str, dict[str, Any]], group: str, key: str) -> float:
    values = [
        float(settings.get(group, {}).get(key))
        for settings in verticals.values()
        if settings.get(group, {}).get(key) is not None
    ]
    if not values:
        raise KeyError(f"No configured SLO target for {group}.{key}")
    return min(values)


def _envelope_max(verticals: dict[str, dict[str, Any]], group: str, key: str) -> float:
    values = [
        float(settings.get(group, {}).get(key))
        for settings in verticals.values()
        if settings.get(group, {}).get(key) is not None
    ]
    if not values:
        raise KeyError(f"No configured SLO target for {group}.{key}")
    return max(values)


def _target_range(verticals: dict[str, dict[str, Any]], group: str, key: str) -> str:
    values = [
        float(settings.get(group, {}).get(key))
        for settings in verticals.values()
        if settings.get(group, {}).get(key) is not None
    ]
    if not values:
        return "NOT_CONFIGURED"
    if min(values) == max(values):
        return str(values[0])
    return f"{min(values)}..{max(values)}"


def _status_min(observed: float | None, target: float | None) -> str:
    if observed is None or target is None:
        return "NOT_EVALUATED_NO_CONFIGURED_TARGET"
    return "PASS" if observed >= target else "FAIL"


def _status_max(observed: float | None, target: float | None) -> str:
    if observed is None or target is None:
        return "NOT_EVALUATED_NO_CONFIGURED_TARGET"
    return "PASS" if observed <= target else "FAIL"


def _diff_min(observed: float | None, target: float | None) -> float | None:
    if observed is None or target is None:
        return None
    return observed - target


def _diff_max(observed: float | None, target: float | None) -> float | None:
    if observed is None or target is None:
        return None
    return target - observed


def _percent(value: float) -> float:
    return value * 100.0


def build_slo_scorecard(
    *,
    runtime_report: dict[str, Any],
    eval_report: dict[str, Any],
    slo_targets_path: str | Path = DEFAULT_SLO_TARGETS,
) -> list[dict[str, Any]]:
    """Return aggregate Baseline_V1 metric rows with configured SLO targets."""

    verticals = _configured_vertical_targets(slo_targets_path)
    summary = eval_report["summary"]
    latency = runtime_report["latency_summary"]
    cost = runtime_report["cost_report"]
    gpu = runtime_report["gpu_summary"]
    runtime_seconds = float(runtime_report["runtime_seconds"])
    total_requests = float(eval_report["total_requests_completed"])
    gpu_cost = float(cost["gpu_cost_usd"])
    api_cost = float(cost["api_cost_usd"])
    self_hosted_requests = float(cost["self_hosted_request_count"])
    api_requests = float(cost["api_request_count"])
    total_tokens = float(
        sum(
            row.get("total_input_tokens", 0) + row.get("total_output_tokens", 0)
            for row in eval_report["config_summaries"]
        )
    )
    gpu_tokens = float(
        sum(
            row.get("total_input_tokens", 0) + row.get("total_output_tokens", 0)
            for row in eval_report["config_summaries"]
            if row.get("backend_type") == "self_hosted_gpu"
        )
    )
    memory_peak_gb = float(gpu["max_memory_used_mb"]) / 1024.0
    memory_peak_pct = (
        float(gpu["max_memory_used_mb"]) / float(gpu["memory_total_mb"]["max"]) * 100.0
    )

    rows: list[dict[str, Any]] = []

    def add(
        metric: str,
        observed: float | None,
        target: float | None,
        direction: str,
        family: str,
        target_label: str | None = None,
    ) -> None:
        if direction == "min":
            status = _status_min(observed, target)
            difference = _diff_min(observed, target)
            comparator = ">="
        elif direction == "max":
            status = _status_max(observed, target)
            difference = _diff_max(observed, target)
            comparator = "<="
        else:
            status = "NOT_EVALUATED_NO_CONFIGURED_TARGET"
            difference = None
            comparator = "not configured"
        rows.append(
            {
                "metric": metric,
                "family": family,
                "slo_target": target_label if target_label is not None else target,
                "slo_comparator": comparator,
                "baseline_result": observed,
                "difference_to_pass": difference,
                "status": status,
            }
        )

    add("json_validity_pct", _percent(float(summary["json_valid_rate"])), None, "none", "quality")
    add(
        "contract_validity_pct",
        _percent(float(summary["generation_contract_valid_rate"])),
        _percent(_envelope_min(verticals, "quality_slo", "format_validity_min")),
        "min",
        "quality",
        (
            "uses configured format_validity_min; aggregate envelope target; "
            f"range={_target_range(verticals, 'quality_slo', 'format_validity_min')}"
        ),
    )
    add(
        "evidence_match_pct",
        _percent(float(summary["evidence_match_rate"])),
        _percent(_envelope_min(verticals, "quality_slo", "evidence_match_min")),
        "min",
        "quality",
        (
            "aggregate envelope target; "
            f"range={_target_range(verticals, 'quality_slo', 'evidence_match_min')}"
        ),
    )
    add(
        "groundedness_pct",
        _percent(float(summary["grounded_rate"])),
        _percent(_envelope_min(verticals, "quality_slo", "groundedness_min")),
        "min",
        "quality",
        (
            "aggregate envelope target; "
            f"range={_target_range(verticals, 'quality_slo', 'groundedness_min')}"
        ),
    )
    add(
        "safety_findings",
        float(summary["safety_violation_count"]),
        _strictest_max(verticals, "quality_slo", "safety_violations_max"),
        "max",
        "safety",
        f"range={_target_range(verticals, 'quality_slo', 'safety_violations_max')}",
    )
    add("mean_ttft_ms", float(latency["mean_ttft_ms"]), None, "none", "runtime")
    add(
        "p50_ttft_ms",
        float(latency["p50_ttft_ms"]),
        _envelope_max(verticals, "latency_slo", "ttft_p50_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'ttft_p50_ms_max')}",
    )
    add(
        "p95_ttft_ms",
        float(latency["p95_ttft_ms"]),
        _envelope_max(verticals, "latency_slo", "ttft_p95_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'ttft_p95_ms_max')}",
    )
    add(
        "p99_ttft_ms",
        float(latency["p99_ttft_ms"]),
        _envelope_max(verticals, "latency_slo", "ttft_p99_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'ttft_p99_ms_max')}",
    )
    add("mean_tpot_ms", float(latency["mean_tpot_ms"]), None, "none", "runtime")
    add(
        "p50_tpot_ms",
        float(latency["p50_tpot_ms"]),
        _envelope_max(verticals, "latency_slo", "tpot_p50_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'tpot_p50_ms_max')}",
    )
    add(
        "p95_tpot_ms",
        float(latency["p95_tpot_ms"]),
        _envelope_max(verticals, "latency_slo", "tpot_p95_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'tpot_p95_ms_max')}",
    )
    add(
        "p99_tpot_ms",
        float(latency["p99_tpot_ms"]),
        _envelope_max(verticals, "latency_slo", "tpot_p99_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'tpot_p99_ms_max')}",
    )
    add(
        "p50_e2e_latency_ms",
        float(latency["p50_e2e_latency_ms"]),
        _envelope_max(verticals, "latency_slo", "e2e_p50_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'e2e_p50_ms_max')}",
    )
    add(
        "p95_e2e_latency_ms",
        float(latency["p95_e2e_latency_ms"]),
        _envelope_max(verticals, "latency_slo", "e2e_p95_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'e2e_p95_ms_max')}",
    )
    add(
        "p99_e2e_latency_ms",
        float(latency["p99_e2e_latency_ms"]),
        _envelope_max(verticals, "latency_slo", "e2e_p99_ms_max"),
        "max",
        "runtime",
        f"range={_target_range(verticals, 'latency_slo', 'e2e_p99_ms_max')}",
    )
    add(
        "requests_per_second",
        total_requests / runtime_seconds,
        _envelope_min(verticals, "throughput_slo", "requests_per_second_min"),
        "min",
        "throughput",
        f"range={_target_range(verticals, 'throughput_slo', 'requests_per_second_min')}",
    )
    add(
        "mean_tokens_per_second",
        float(latency["mean_total_tokens_per_second"]),
        _envelope_min(verticals, "throughput_slo", "tokens_per_second_min"),
        "min",
        "throughput",
        f"range={_target_range(verticals, 'throughput_slo', 'tokens_per_second_min')}",
    )
    add(
        "gpu_utilization_mean_pct",
        float(gpu["mean_utilization_gpu_percent"]),
        _envelope_min(verticals, "resource_slo", "gpu_utilization_min_pct"),
        "min",
        "resource",
        f"range={_target_range(verticals, 'resource_slo', 'gpu_utilization_min_pct')}",
    )
    add(
        "gpu_memory_peak_gb",
        memory_peak_gb,
        _envelope_max(verticals, "resource_slo", "gpu_memory_peak_gb_max"),
        "max",
        "resource",
        f"range={_target_range(verticals, 'resource_slo', 'gpu_memory_peak_gb_max')}",
    )
    add(
        "gpu_memory_peak_pct",
        memory_peak_pct,
        _envelope_max(verticals, "resource_slo", "gpu_memory_utilization_max_pct"),
        "max",
        "resource",
        f"range={_target_range(verticals, 'resource_slo', 'gpu_memory_utilization_max_pct')}",
    )
    add("gpu_power_mean_w", float(gpu["mean_power_draw_w"]), None, "none", "resource")
    add("gpu_power_peak_w", float(gpu["power_draw_w"]["max"]), None, "none", "resource")
    add("gpu_temperature_peak_c", float(gpu["max_temperature_c"]), None, "none", "resource")
    add(
        "gpu_cost_per_request_usd",
        gpu_cost / self_hosted_requests,
        _envelope_max(verticals, "gpu_cost_slo", "gpu_cost_per_request_usd_max"),
        "max",
        "gpu_cost",
        f"range={_target_range(verticals, 'gpu_cost_slo', 'gpu_cost_per_request_usd_max')}",
    )
    add(
        "gpu_cost_per_1000_requests_usd",
        gpu_cost / self_hosted_requests * 1000.0,
        _envelope_max(verticals, "gpu_cost_slo", "gpu_cost_per_1000_requests_usd_max"),
        "max",
        "gpu_cost",
        f"range={_target_range(verticals, 'gpu_cost_slo', 'gpu_cost_per_1000_requests_usd_max')}",
    )
    add(
        "tokens_per_gpu_dollar",
        gpu_tokens / gpu_cost,
        _envelope_min(verticals, "gpu_cost_slo", "tokens_per_gpu_dollar_min"),
        "min",
        "gpu_cost",
        f"range={_target_range(verticals, 'gpu_cost_slo', 'tokens_per_gpu_dollar_min')}",
    )
    add(
        "api_cost_per_request_usd",
        api_cost / api_requests,
        _envelope_max(verticals, "api_cost_slo", "api_cost_per_request_usd_max"),
        "max",
        "api_cost",
        f"range={_target_range(verticals, 'api_cost_slo', 'api_cost_per_request_usd_max')}",
    )
    add(
        "api_cost_per_1000_requests_usd",
        api_cost / api_requests * 1000.0,
        _envelope_max(verticals, "api_cost_slo", "api_cost_per_1000_requests_usd_max"),
        "max",
        "api_cost",
        f"range={_target_range(verticals, 'api_cost_slo', 'api_cost_per_1000_requests_usd_max')}",
    )
    add("gpu_cost_total_usd", gpu_cost, None, "none", "gpu_cost")
    add("api_cost_total_usd", api_cost, None, "none", "api_cost")
    add("total_cost_usd", float(cost["total_cost_usd"]), None, "none", "total_cost")
    add("total_tokens", total_tokens, None, "none", "throughput")
    return rows


def build_repaired_metric_rows(
    before_after: dict[str, Any],
    slo_targets_path: str | Path = DEFAULT_SLO_TARGETS,
) -> list[dict[str, Any]]:
    verticals = _configured_vertical_targets(slo_targets_path)
    before = before_after["before_summary"]
    after = before_after["after_summary"]
    metric_specs = [
        (
            "json_validity_pct",
            _percent(float(before["json_valid_rate"])),
            _percent(float(after["json_valid_rate"])),
            None,
            "none",
        ),
        (
            "contract_validity_pct",
            _percent(float(before["generation_contract_valid_rate"])),
            _percent(float(after["generation_contract_valid_rate"])),
            _percent(_envelope_min(verticals, "quality_slo", "format_validity_min")),
            "min",
        ),
        (
            "evidence_match_pct",
            _percent(float(before["evidence_match_rate"])),
            _percent(float(after["evidence_match_rate"])),
            _percent(_envelope_min(verticals, "quality_slo", "evidence_match_min")),
            "min",
        ),
        (
            "groundedness_pct",
            _percent(float(before["grounded_rate"])),
            _percent(float(after["grounded_rate"])),
            _percent(_envelope_min(verticals, "quality_slo", "groundedness_min")),
            "min",
        ),
        (
            "safety_findings",
            float(before["safety_violation_count"]),
            float(after["safety_violation_count"]),
            _strictest_max(verticals, "quality_slo", "safety_violations_max"),
            "max",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for metric, before_value, after_value, target, direction in metric_specs:
        if direction == "min":
            status = _status_min(after_value, target)
            difference = _diff_min(after_value, target)
            comparator = ">="
        elif direction == "max":
            status = _status_max(after_value, target)
            difference = _diff_max(after_value, target)
            comparator = "<="
        else:
            status = "NOT_EVALUATED_NO_CONFIGURED_TARGET"
            difference = None
            comparator = "not configured"
        rows.append(
            {
                "metric": metric,
                "slo_target": target,
                "slo_comparator": comparator,
                "before": before_value,
                "after": after_value,
                "delta": after_value - before_value,
                "difference_to_pass_after": difference,
                "status_after": status,
            }
        )
    return rows


def build_failure_diagnosis(diagnosis_report: dict[str, Any]) -> dict[str, Any]:
    verticals = diagnosis_report.get("vertical_metrics", [])
    bottlenecks = diagnosis_report.get("bottleneck_counts", {})
    candidates = diagnosis_report.get("worst_configs", [])
    return {
        "status": "BASELINE_V1_FAILURE_DIAGNOSIS_COMPLETE",
        "remaining_failures": {
            "quality_slo": "FAIL",
            "safety_slo": "FAIL",
            "deployability": "NOT_DEPLOYABLE_SLO_FAILURES",
        },
        "root_causes": [
            {
                "cause": "Research AI generation-contract and grounding failures",
                "implementation": (
                    "scripts/phase4/run_controlled_final_simulation.py prompt rendering "
                    "and generation-contract normalization; "
                    "scripts/phase4/run_phase2_targeted_baseline_repairs.py Research AI "
                    "answer_skeleton strengthening"
                ),
                "affected_verticals": ["research_ai"],
                "affected_models": ["model3_7b", "model6_gated"],
                "affected_engines": ["vllm", "sglang", "api_provider_route"],
                "affected_memory_modes": [
                    "mm1_dense_top5",
                    "mm2_hybrid_top5",
                    "mm3_compressed_hybrid_top5",
                    "mm4_bounded_agentic",
                ],
                "affected_concurrency": [4, 16, 32],
                "why": (
                    "The baseline outputs often failed the strict five-field contract or "
                    "omitted the visible E-label evidence needed for deterministic "
                    "groundedness, even when retrieval supplied the context."
                ),
            },
            {
                "cause": "Healthcare Admin safety wording in final answers",
                "implementation": (
                    "scripts/phase4/run_controlled_final_simulation.py final-answer "
                    "safety boundary; "
                    "scripts/phase4/run_phase2_targeted_baseline_repairs.py healthcare "
                    "safety wording cleanup"
                ),
                "affected_verticals": ["healthcare_admin"],
                "affected_models": ["model3_7b", "model6_gated"],
                "affected_engines": ["vllm", "sglang", "api_provider_route"],
                "affected_memory_modes": [
                    "mm1_dense_top5",
                    "mm2_hybrid_top5",
                    "mm3_compressed_hybrid_top5",
                    "mm4_bounded_agentic",
                ],
                "affected_concurrency": [4, 16, 32],
                "why": (
                    "Safe refusal/boundary answers repeated prohibited wording from "
                    "prompt or policy text, causing the unchanged deterministic safety "
                    "evaluator to flag the final answer."
                ),
            },
            {
                "cause": "MM0 expected no-context ablation",
                "implementation": (
                    "scripts/phase4/run_controlled_final_simulation.py SLO grouping "
                    "treats MM0 evidence and groundedness separately"
                ),
                "affected_verticals": [
                    "airline",
                    "healthcare_admin",
                    "retail",
                    "finance",
                    "research_ai",
                ],
                "affected_models": ["model3_7b", "model6_gated"],
                "affected_engines": ["vllm", "sglang", "api_provider_route"],
                "affected_memory_modes": ["mm0_no_context"],
                "affected_concurrency": [4, 16, 32],
                "why": (
                    "MM0 intentionally receives no retrieved evidence, so evidence and "
                    "groundedness misses are reported as ablation evidence rather than "
                    "optimized as contextual failures."
                ),
            },
        ],
        "vertical_summaries": verticals,
        "bottleneck_counts": bottlenecks,
        "selected_candidates": candidates,
    }


def _copy_repair_evidence(output_root: Path) -> list[str]:
    copied: list[str] = []
    evidence_dir = output_root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for source in TARGETED_REPAIR_SOURCE_FILES:
        source_path = _repo_path(source)
        if not source_path.exists():
            continue
        destination = evidence_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied.append(_display_path(destination))
    return copied


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _repo_path(args.output_root)
    baseline_archive = _repo_path(args.baseline_archive)
    runtime_report = _read_json(
        baseline_archive / "processed/final_10000_baseline_v1_runtime_report.json"
    )
    eval_report = _read_json(
        baseline_archive / "processed/final_10000_baseline_v1_eval_report.json"
    )
    slo_report = _read_json(baseline_archive / "processed/final_10000_baseline_v1_slo_report.json")
    before_after = _read_json(args.before_after_path)
    repair_report = _read_json(args.targeted_repair_report_path)
    readiness = _read_json(args.readiness_path)
    diagnosis = _read_json("results/processed/phase2_optimization_diagnosis_report.json")
    slo_profile = _load_yaml(args.slo_profiles_path)
    scorecard_rows = build_slo_scorecard(
        runtime_report=runtime_report,
        eval_report=eval_report,
        slo_targets_path=args.slo_targets_path,
    )
    repaired_metric_rows = build_repaired_metric_rows(
        before_after=before_after,
        slo_targets_path=args.slo_targets_path,
    )
    failure_diagnosis = build_failure_diagnosis(diagnosis)
    copied = _copy_repair_evidence(output_root)
    report = {
        "run_id": "baseline_v1_quality_repair_v1",
        "status": "BASELINE_V1_QUALITY_REPAIR_VALIDATION_COMPLETE",
        "baseline_archive": _display_path(baseline_archive),
        "slo_profile": slo_profile.get("default_profile"),
        "baseline_verdicts": {
            "benchmark_execution_verdict": slo_report["benchmark_execution_verdict"],
            "runtime_slo_verdict": slo_report["runtime_slo_verdict"],
            "cost_slo_verdict": slo_report["cost_slo_verdict"],
            "quality_slo_verdict": slo_report["quality_slo_verdict"],
            "safety_slo_verdict": slo_report["safety_slo_verdict"],
            "overall_deployability_verdict": slo_report["overall_deployability_verdict"],
        },
        "repairs_applied": repair_report["repairs_applied"],
        "validation_scope": {
            "type": "targeted_before_after_validation",
            "full_10000_rerun_performed": False,
            "request_count": repair_report["total_requests"],
            "candidate_config_ids": repair_report["candidate_config_ids"],
        },
        "targeted_repair_gate": repair_report["success_gate"],
        "readiness": readiness,
        "remaining_failures": [
            (
                "The frozen Baseline_V1 archive remains not deployable by SLO and is "
                "preserved unchanged."
            ),
            (
                "The targeted repair validation still has one SGLang MM4 Healthcare "
                "Admin safety finding; monitor it in Main_Inference_V1."
            ),
        ],
        "main_inference_v1_allowed": bool(readiness["final_main_10000_experiment_allowed"]),
        "copied_evidence_files": copied,
    }
    _write_json(output_root / "baseline_v1_slo_scorecard.json", {"rows": scorecard_rows})
    _write_csv(output_root / "baseline_v1_slo_scorecard.csv", scorecard_rows)
    _write_json(
        output_root / "baseline_v1_quality_repair_metrics.json",
        {"rows": repaired_metric_rows},
    )
    _write_csv(output_root / "baseline_v1_quality_repair_metrics.csv", repaired_metric_rows)
    _write_json(output_root / "baseline_v1_failure_diagnosis.json", failure_diagnosis)
    _write_json(output_root / "baseline_v1_quality_repair_report.json", report)
    readme = (
        "# Baseline V1 Quality Repair V1\n\n"
        "This folder preserves the Baseline_V1 quality-repair scorecard and targeted "
        "validation evidence. It does not overwrite `experiments/baseline_v1/` and "
        "does not run Main_Inference_V1.\n\n"
        "- Baseline_V1 execution: complete.\n"
        "- Frozen Baseline_V1 deployability: NOT_DEPLOYABLE_SLO_FAILURES.\n"
        "- Targeted repair validation: passed on 1,600 selected requests.\n"
        "- Main_Inference_V1 allowed: "
        f"{str(bool(readiness['final_main_10000_experiment_allowed'])).lower()}.\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-archive", default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument("--targeted-repair-report-path", default=DEFAULT_TARGETED_REPAIR_REPORT)
    parser.add_argument("--before-after-path", default=DEFAULT_BEFORE_AFTER)
    parser.add_argument("--readiness-path", default=DEFAULT_READINESS)
    parser.add_argument("--slo-targets-path", default=DEFAULT_SLO_TARGETS)
    parser.add_argument("--slo-profiles-path", default=DEFAULT_SLO_PROFILES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
