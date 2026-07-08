"""Diagnose controlled-final baseline SLO failures and plan Phase 2 optimization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
PHASE4 = Path(__file__).resolve().parent
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import (  # noqa: E402
    load_gold_records,
    load_result_rows,
    result_row_to_generated_answer,
)

from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402

RUN_ID = "controlled_final_simulation"
DEFAULT_RAW_RESULTS = "results/raw/controlled_final_simulation_results.jsonl"
DEFAULT_SLO_REPORT = "results/processed/controlled_final_simulation_slo_report_fixed.json"
DEFAULT_SLO_SUMMARY = "results/processed/controlled_final_simulation_slo_summary_fixed.csv"
DEFAULT_DATASET_ROOT = "data/scaleup_2000_full"
DEFAULT_REPORT = "results/processed/phase2_optimization_diagnosis_report.json"
DEFAULT_SUMMARY = "results/processed/phase2_optimization_diagnosis_summary.csv"
DEFAULT_CANDIDATES = "results/processed/phase2_selected_optimization_candidates.json"
DEFAULT_RERUN_PLAN = "results/processed/phase2_before_after_rerun_plan.json"

QUALITY_FIELDS = (
    "generation_contract_valid",
    "format_valid",
    "evidence_match",
    "groundedness",
)
CONTEXTUAL_MODES = {
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
    "mm4_bounded_agentic",
}
HIGH_VALUE_MODES = {"mm2_hybrid_top5", "mm3_compressed_hybrid_top5"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results-path", default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--slo-report-path", default=DEFAULT_SLO_REPORT)
    parser.add_argument("--slo-summary-path", default=DEFAULT_SLO_SUMMARY)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    parser.add_argument("--candidates-path", default=DEFAULT_CANDIDATES)
    parser.add_argument("--rerun-plan-path", default=DEFAULT_RERUN_PLAN)
    return parser


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


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


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with _repo_path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def _group_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "") for field in fields)


def _summarize_group(
    *,
    group_name: str,
    rows: list[dict[str, Any]],
    evals: list[dict[str, Any]],
) -> dict[str, Any]:
    success_rows = [row for row in rows if bool(row.get("success"))]
    latencies = [float(row.get("end_to_end_latency_ms") or 0.0) for row in success_rows]
    tps = [float(row.get("throughput_tokens_per_second") or 0.0) for row in success_rows]
    safety_count = sum(bool(row.get("safety_violation")) for row in evals)
    contract = _rate(evals, "generation_contract_valid")
    evidence = _rate(evals, "evidence_match")
    grounded = _rate(evals, "groundedness")
    quality_score = (contract + evidence + grounded) / 3.0
    return {
        "group": group_name,
        "requests": len(rows),
        "successes": len(success_rows),
        "failures": len(rows) - len(success_rows),
        "json_valid_rate": _rate(evals, "json_validity"),
        "contract_valid_rate": contract,
        "format_valid_rate": _rate(evals, "format_valid"),
        "evidence_match_rate": evidence,
        "grounded_rate": grounded,
        "safety_violation_count": safety_count,
        "quality_score": quality_score,
        "mean_e2e_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "mean_tokens_per_second": sum(tps) / len(tps) if tps else 0.0,
    }


def _bottlenecks(row: dict[str, Any]) -> list[str]:
    memory_mode = str(row.get("memory_mode") or "")
    failed = set(str(row.get("failed_metric_family") or "").split(";")) - {""}
    bottlenecks: list[str] = []
    if "contract_validity" in failed or "format_validity" in failed:
        bottlenecks.append("generation_contract_failure")
        bottlenecks.append("prompt_context_formatting_issue")
    if memory_mode == "mm0_no_context":
        bottlenecks.append("mm0_expected_no_context_failure")
    else:
        if "evidence_match" in failed:
            bottlenecks.append("evidence_selection_failure")
        if "groundedness" in failed:
            bottlenecks.append("groundedness_failure")
    if "safety" in failed:
        if memory_mode == "mm4_bounded_agentic":
            bottlenecks.append("mm4_agentic_safety_trace_issue")
        else:
            bottlenecks.append("safety_wording_failure")
    if str(row.get("concurrency")) == "32":
        bottlenecks.append("concurrency_degradation")
    if str(row.get("backend_type")) == "api_provider":
        bottlenecks.append("api_specific_issue")
    if str(row.get("engine")) == "sglang":
        bottlenecks.append("engine_specific_issue")
    if memory_mode in CONTEXTUAL_MODES and (
        float(row.get("evidence_match_rate") or 0.0) < 0.80
        or float(row.get("grounded_rate") or 0.0) < 0.80
    ):
        bottlenecks.append("model_capacity_or_instruction_following_issue")
    return sorted(set(bottlenecks))


def _recommendations(row: dict[str, Any], bottlenecks: list[str]) -> list[str]:
    recommendations: list[str] = []
    if "generation_contract_failure" in bottlenecks:
        recommendations.extend(["final_answer_contract_normalization", "max_token_adjustment"])
    if "evidence_selection_failure" in bottlenecks:
        recommendations.extend(["evidence_selector_repair", "citation_whitelist"])
    if "groundedness_failure" in bottlenecks:
        recommendations.extend(["context_compression", "evidence_selector_repair"])
    if "safety_wording_failure" in bottlenecks:
        recommendations.append("safety_wording_cleanup")
    if "mm4_agentic_safety_trace_issue" in bottlenecks:
        recommendations.append("mm4_final_answer_guard")
    if "concurrency_degradation" in bottlenecks:
        recommendations.extend(["lower_concurrency", "prefix_cache"])
    if "engine_specific_issue" in bottlenecks and str(row.get("engine")) == "sglang":
        recommendations.append("engine_switch")
    if "api_specific_issue" in bottlenecks:
        recommendations.append("api_prompt_contract_normalization")
    if "model_capacity_or_instruction_following_issue" in bottlenecks:
        recommendations.append("stronger_model_escalation")
    if str(row.get("memory_mode")) == "mm0_no_context":
        recommendations = [
            item
            for item in recommendations
            if item in {"final_answer_contract_normalization", "max_token_adjustment"}
        ]
    return sorted(dict.fromkeys(recommendations))


def _candidate_priority(row: dict[str, Any]) -> tuple[int, float, str]:
    memory_mode = str(row.get("memory_mode") or "")
    is_mm0 = memory_mode == "mm0_no_context"
    safety = int(float(row.get("safety_violation_count") or 0))
    quality_distance = (
        max(0.0, 0.95 - float(row.get("generation_contract_valid_rate") or 0.0))
        + (0.0 if is_mm0 else max(0.0, 0.95 - float(row.get("evidence_match_rate") or 0.0)))
        + (0.0 if is_mm0 else max(0.0, 0.95 - float(row.get("grounded_rate") or 0.0)))
    )
    mode_bonus = 0 if memory_mode in HIGH_VALUE_MODES else 1
    safety_bonus = 0 if safety > 0 else 1
    return (is_mm0 + mode_bonus + safety_bonus, quality_distance, str(row.get("config_id")))


def _select_candidates(diagnosis_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deployable_rows = [
        row for row in diagnosis_rows if row["slo_scope"] == "deployability_contextual"
    ]
    selected: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        item = dict(row)
        item["selection_reason"] = reason
        item["planned_prompt_count"] = 200
        item["run_policy"] = "targeted_before_after_only_after_plan_approval"
        selected[str(row["config_id"])] = item

    for row in sorted(deployable_rows, key=_candidate_priority):
        memory = str(row["memory_mode"])
        if (
            memory in HIGH_VALUE_MODES
            and len([item for item in selected.values() if item["memory_mode"] in HIGH_VALUE_MODES])
            < 4
        ):
            add(row, "high_value_contextual_mode_preserves_engine_concurrency_comparison")
    for row in sorted(deployable_rows, key=_candidate_priority):
        if (
            row["memory_mode"] == "mm4_bounded_agentic"
            and len(
                [item for item in selected.values() if item["memory_mode"] == "mm4_bounded_agentic"]
            )
            < 2
        ):
            add(row, "mm4_agentic_safety_guard_candidate")
    for row in sorted(deployable_rows, key=_candidate_priority):
        if (
            row["backend_type"] == "api_provider"
            and len([item for item in selected.values() if item["backend_type"] == "api_provider"])
            < 2
        ):
            add(row, "api_model6_comparison_candidate")
    for row in sorted(deployable_rows, key=_candidate_priority):
        if len(selected) >= 8:
            break
        add(row, "closest_remaining_deployability_gap")
    return list(selected.values())


def build_phase2_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    raw_rows = load_result_rows(_repo_path(args.raw_results_path))
    generated = [result_row_to_generated_answer(row) for row in raw_rows]
    eval_rows = evaluate_generated_answers(
        generated, load_gold_records(_repo_path(args.dataset_root))
    )
    slo_report = _load_json(args.slo_report_path)
    slo_rows = _read_csv(args.slo_summary_path)
    eval_by_request = {
        f"{row['config_id']}::{row['prompt_id']}": evaluation
        for row, evaluation in zip(raw_rows, eval_rows, strict=False)
    }

    diagnosis_rows: list[dict[str, Any]] = []
    for row in slo_rows:
        bottlenecks = _bottlenecks(row)
        recs = _recommendations(row, bottlenecks)
        diagnosis_rows.append(
            {
                **row,
                "bottleneck_classes": ";".join(bottlenecks),
                "recommended_optimizations": ";".join(recs),
                "runtime_pass_quality_fail": (
                    row.get("slo_latency") == "PASS"
                    and row.get("slo_throughput") == "PASS"
                    and bool(row.get("failed_metric_family"))
                ),
                "ablation_only": row.get("slo_scope") == "no_context_ablation",
                "worth_optimizing": row.get("slo_scope") == "deployability_contextual",
            }
        )

    by_config = defaultdict(lambda: ([], []))
    by_vertical = defaultdict(lambda: ([], []))
    for row in raw_rows:
        key = f"{row['config_id']}::{row['prompt_id']}"
        evaluation = eval_by_request.get(key)
        if evaluation is None:
            continue
        by_config[str(row["config_id"])][0].append(row)
        by_config[str(row["config_id"])][1].append(evaluation)
        by_vertical[str(row.get("vertical") or "")][0].append(row)
        by_vertical[str(row.get("vertical") or "")][1].append(evaluation)

    config_metrics = [
        _summarize_group(group_name=config_id, rows=rows, evals=evals)
        for config_id, (rows, evals) in by_config.items()
    ]
    vertical_metrics = [
        _summarize_group(group_name=vertical, rows=rows, evals=evals)
        for vertical, (rows, evals) in by_vertical.items()
    ]
    worst_configs = sorted(
        [row for row in diagnosis_rows if row["slo_scope"] == "deployability_contextual"],
        key=lambda row: (
            -int(row["failed_slos"]),
            float(row["evidence_match_rate"]),
            float(row["grounded_rate"]),
            str(row["config_id"]),
        ),
    )[:8]
    best_configs = sorted(
        [row for row in diagnosis_rows if row["slo_scope"] == "deployability_contextual"],
        key=lambda row: (
            -float(row["evidence_match_rate"]),
            -float(row["grounded_rate"]),
            int(row["failed_slos"]),
            str(row["config_id"]),
        ),
    )[:8]
    selected = _select_candidates(diagnosis_rows)
    bottleneck_counts = defaultdict(int)
    optimization_counts = defaultdict(int)
    for row in diagnosis_rows:
        for item in str(row["bottleneck_classes"]).split(";"):
            if item:
                bottleneck_counts[item] += 1
        for item in str(row["recommended_optimizations"]).split(";"):
            if item:
                optimization_counts[item] += 1
    report = {
        "run_id": RUN_ID,
        "status": "PHASE2_OPTIMIZATION_DIAGNOSIS_COMPLETE",
        "no_inference_rerun": True,
        "source_slo_report": args.slo_report_path,
        "source_slo_summary": args.slo_summary_path,
        "baseline_verdicts": {
            key: slo_report.get(key)
            for key in (
                "runtime_slo_verdict",
                "quality_slo_verdict",
                "safety_slo_verdict",
                "cost_slo_verdict",
                "overall_deployability_verdict",
                "benchmark_execution_verdict",
            )
        },
        "aggregate_slo_results": slo_report.get("aggregate_slo_results", []),
        "diagnosis_rows": diagnosis_rows,
        "config_metrics": sorted(config_metrics, key=lambda row: row["group"]),
        "vertical_metrics": sorted(vertical_metrics, key=lambda row: row["group"]),
        "worst_configs": worst_configs,
        "best_configs": best_configs,
        "bottleneck_counts": dict(sorted(bottleneck_counts.items())),
        "optimization_counts": dict(sorted(optimization_counts.items())),
        "selected_candidate_count": len(selected),
        "optimization_rerun_can_begin": True,
        "optimization_rerun_requires_approval": True,
    }
    candidates = {
        "run_id": RUN_ID,
        "status": "PHASE2_OPTIMIZATION_CANDIDATES_SELECTED",
        "selection_policy": [
            "exclude MM0 from evidence/groundedness optimization because it is no-context ablation",
            "prioritize MM2/MM3 contextual modes",
            "preserve vLLM/SGLang and concurrency comparisons",
            "include MM4 safety guard candidate",
            "include API model6 comparison candidate",
        ],
        "selected_candidates": selected,
    }
    rerun_plan = {
        "run_id": RUN_ID,
        "status": "PHASE2_BEFORE_AFTER_RERUN_PLAN_READY",
        "baseline_artifacts": {
            "raw_results": args.raw_results_path,
            "fixed_slo_report": args.slo_report_path,
            "fixed_slo_summary": args.slo_summary_path,
        },
        "do_not_rerun_full_10000_yet": True,
        "do_not_modify_gold_or_evaluators": True,
        "candidate_config_ids": [row["config_id"] for row in selected],
        "planned_optimizations": sorted(optimization_counts),
        "success_gate": {
            "runtime_slo_verdict": "PASS",
            "quality_slo_verdict": "PASS",
            "safety_slo_verdict": "PASS",
            "overall_deployability_verdict": "DEPLOYABLE_BASELINE",
        },
    }
    return {"report": report, "candidates": candidates, "rerun_plan": rerun_plan}


def main() -> int:
    args = build_parser().parse_args()
    payload = build_phase2_diagnosis(args)
    _write_json(args.report_path, payload["report"])
    _write_csv(args.summary_path, payload["report"]["diagnosis_rows"])
    _write_json(args.candidates_path, payload["candidates"])
    _write_json(args.rerun_plan_path, payload["rerun_plan"])
    print(json.dumps(payload["report"], ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
