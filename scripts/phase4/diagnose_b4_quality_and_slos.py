"""Generate B4 quality-audit and SLO-diagnosis reports from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.generation_quality_audit import (  # noqa: E402
    build_generation_quality_audit,
    write_generation_quality_audit_artifacts,
)
from inference_bench.slo_diagnosis import diagnose_slos  # noqa: E402
from inference_bench.slo_profiles import resolve_slo_profile  # noqa: E402

VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")

DEFAULT_EVAL_REPORT = "results/processed/b4_vllm_1_5b_context_aligned_eval_report.json"
DEFAULT_RESULTS = "results/raw/b4_vllm_1_5b_context_aligned_results.jsonl"
DEFAULT_RUNNER_INPUT = "data/generated/phase4/b4_context_aligned_runner_input.jsonl"
DEFAULT_LATENCY = "results/processed/b4_vllm_1_5b_context_aligned_latency_summary.csv"
DEFAULT_MANIFEST = "results/raw/b4_vllm_1_5b_context_aligned_manifest.json"
DEFAULT_GPU = "results/processed/b4_vllm_1_5b_context_aligned_gpu_telemetry_summary.json"
DEFAULT_CONTEXT_REPORT = "results/processed/b4_context_alignment_report.json"
DEFAULT_QUALITY_AUDIT = "results/processed/b4_generation_quality_audit_report.json"
DEFAULT_QUALITY_SUMMARY = "results/processed/b4_generation_quality_audit_summary.csv"
DEFAULT_FINANCE_EXAMPLES = "results/processed/b4_finance_failure_examples.jsonl"
DEFAULT_FAILURE_EXAMPLES = "results/processed/b4_quality_failure_examples.jsonl"
DEFAULT_SLO_REPORT = "results/processed/b4_slo_diagnosis_report.json"
DEFAULT_SLO_SUMMARY = "results/processed/b4_slo_diagnosis_summary.csv"
DEFAULT_RECOMMENDATIONS = "results/processed/b4_optimization_recommendation_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the offline B4 diagnosis CLI."""

    parser = argparse.ArgumentParser(
        description="Diagnose B4 context-aligned quality and SLO failures."
    )
    parser.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--latency-summary", default=DEFAULT_LATENCY)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu-summary", default=DEFAULT_GPU)
    parser.add_argument("--context-report", default=DEFAULT_CONTEXT_REPORT)
    parser.add_argument("--quality-audit-report", default=DEFAULT_QUALITY_AUDIT)
    parser.add_argument("--quality-audit-summary", default=DEFAULT_QUALITY_SUMMARY)
    parser.add_argument("--finance-examples", default=DEFAULT_FINANCE_EXAMPLES)
    parser.add_argument("--failure-examples", default=DEFAULT_FAILURE_EXAMPLES)
    parser.add_argument("--slo-report", default=DEFAULT_SLO_REPORT)
    parser.add_argument("--slo-summary", default=DEFAULT_SLO_SUMMARY)
    parser.add_argument("--recommendations-summary", default=DEFAULT_RECOMMENDATIONS)
    return parser


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads((ROOT / Path(path)).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / Path(path)).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], payload)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / Path(path)).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row: {path}")
            rows.append(cast(dict[str, Any], payload))
    return rows


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with (ROOT / Path(path)).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _quality_metrics(
    evaluation_rows: list[dict[str, Any]],
    *,
    prompt_ids: set[str],
) -> dict[str, Any]:
    rows = [row for row in evaluation_rows if str(row.get("prompt_id")) in prompt_ids]
    return {
        "grounded_rate": _rate(rows, "groundedness"),
        "evidence_match_rate": _rate(rows, "evidence_match"),
        "task_success_rate": _rate(rows, "status_matches"),
        "generation_contract_valid_rate": _rate(rows, "generation_contract_valid"),
        "safety_violation_count": sum(bool(row.get("safety_violation")) for row in rows),
    }


def _latency_by_vertical(path: str | Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for row in _read_csv(path):
        metrics = {
            key: float(value)
            for key, value in row.items()
            if key != "vertical" and value not in (None, "")
        }
        for source, target in (
            ("mean_itl_p50_ms", "itl_p50_ms"),
            ("mean_itl_p95_ms", "itl_p95_ms"),
            ("mean_itl_p99_ms", "itl_p99_ms"),
        ):
            if source in metrics:
                metrics[target] = metrics[source]
        rows[row["vertical"]] = metrics
    return rows


def _retrieval_by_vertical() -> dict[str, dict[str, float]]:
    manifest = _read_json(
        "data/generated/context_engineering/retrieval_source_of_truth_manifest.json"
    )
    raw_metrics = cast(dict[str, Any], manifest["metrics_by_vertical"])
    return {
        vertical: {
            "candidate_recall_at_20_min": float(metrics["candidate_recall_at_20"]),
            "candidate_recall_at_50_min": float(metrics["candidate_recall_at_50"]),
            "final_recall_at_5_min": float(metrics["final_recall_at_5"]),
            "mrr_min": float(metrics["mrr"]),
        }
        for vertical, metrics in raw_metrics.items()
    }


def _throughput_metrics(
    manifest: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> dict[str, float]:
    start = datetime.fromisoformat(str(manifest["start_time"]))
    end = datetime.fromisoformat(str(manifest["end_time"]))
    elapsed = (end - start).total_seconds()
    successes = sum(bool(row.get("success")) for row in result_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in result_rows)
    return {
        "requests_per_second_min": len(result_rows) / elapsed,
        "successful_requests_per_second_min": successes / elapsed,
        "aggregate_tokens_per_second": output_tokens / elapsed,
    }


def _telemetry_metrics(path: str | Path) -> dict[str, float]:
    telemetry = _read_json(path)
    utilization = cast(dict[str, Any], telemetry.get("utilization_gpu_percent", {}))
    memory_used = cast(dict[str, Any], telemetry.get("memory_used_mb", {}))
    memory_total = cast(dict[str, Any], telemetry.get("memory_total_mb", {}))
    metrics: dict[str, float] = {}
    if utilization.get("mean") not in (None, ""):
        metrics["mean_gpu_utilization_percent"] = float(utilization["mean"])
    if memory_used.get("max") not in (None, ""):
        metrics["max_gpu_memory_used_mb"] = float(memory_used["max"])
    if memory_total.get("max") not in (None, ""):
        metrics["gpu_memory_total_mb"] = float(memory_total["max"])
    return metrics


def _model_metadata(model_alias: str) -> dict[str, Any]:
    models = _read_yaml("configs/models.yaml")
    aliases = cast(dict[str, str], models["model_aliases"])
    return cast(dict[str, Any], models[aliases[model_alias]])


def _summary_row(diagnosis: dict[str, Any]) -> dict[str, Any]:
    primary = diagnosis.get("primary_recommendation")
    return {
        "block": "B4",
        "vertical": diagnosis["context"]["vertical"],
        "engine": diagnosis["context"]["engine"],
        "memory_mode": diagnosis["context"]["memory_mode"],
        "selected_slo_count": len(diagnosis["selected_slos"]),
        "passed_slo_count": len(diagnosis["passed_slos"]),
        "failed_slo_count": len(diagnosis["failed_slos"]),
        "not_applicable_slo_count": len(diagnosis["not_applicable_slos"]),
        "unavailable_metric_count": len(diagnosis["unavailable_metrics"]),
        "bottleneck_ids": ";".join(item["id"] for item in diagnosis["bottlenecks"]),
        "primary_optimization": primary["optimization_id"] if isinstance(primary, dict) else "",
        "llm_used": False,
    }


def _recommendation_rows(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recommendation in diagnosis["recommended_optimizations"]:
        rows.append(
            {
                "block": "B4",
                "vertical": diagnosis["context"]["vertical"],
                "engine": diagnosis["context"]["engine"],
                "memory_mode": diagnosis["context"]["memory_mode"],
                "optimization_id": recommendation["optimization_id"],
                "recommendation_tier": recommendation["recommendation_tier"],
                "rank_score": recommendation["rank_score"],
                "reason": recommendation["reason"],
                "matched_bottlenecks": ";".join(recommendation["matched_bottlenecks"]),
                "implementation_status": recommendation["implementation_status"],
                "application_method": recommendation["application_method"],
                "llm_used": False,
            }
        )
    return rows


def build_b4_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    """Build quality audit and SLO diagnosis without new inference."""

    eval_report = _read_json(args.eval_report)
    evaluation_rows = cast(list[dict[str, Any]], eval_report["evaluation_rows"])
    result_rows = _read_jsonl(args.results_path)
    runner_inputs = _read_jsonl(args.runner_input_path)
    quality_audit = build_generation_quality_audit(
        evaluation_rows=evaluation_rows,
        result_rows=result_rows,
        runner_inputs=runner_inputs,
        block="B4",
        status="AUDIT_COMPLETE_CONTEXT_PRESENT_QUALITY_BLOCKED",
        source_block="B4",
        model_inference_triggered=False,
        finance_interpretation=(
            "After B4 context alignment, Finance failures are no longer caused by "
            "missing E1-E5 gold evidence or missing Finance metadata. The remaining "
            "Finance failures are model citation-selection and instruction-following "
            "errors, with one truncation and no Finance safety/advice/projection wording."
        ),
    )
    write_generation_quality_audit_artifacts(
        report=quality_audit,
        report_path=ROOT / args.quality_audit_report,
        summary_path=ROOT / args.quality_audit_summary,
        finance_examples_path=ROOT / args.finance_examples,
        failure_examples_path=ROOT / args.failure_examples,
    )

    profile = resolve_slo_profile()
    retrieval = _retrieval_by_vertical()
    latency = _latency_by_vertical(args.latency_summary)
    manifest = _read_json(args.manifest)
    shared = {
        **_throughput_metrics(manifest, result_rows),
        **_telemetry_metrics(args.gpu_summary),
    }
    raw_by_prompt = {str(row["prompt_id"]): row for row in result_rows}
    hardware = _read_yaml("configs/hardware/remote_rtx3070.yaml")
    experiment_config = {
        "block": "B4",
        "engine": "vllm",
        "model_alias": "model2_1_5b",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "memory_mode": "mm2_hybrid_top5",
        "ablation_mode": "prompt_plus_metadata",
        "concurrency": 1,
        "max_records": 100,
        "max_new_tokens": 160,
        "runner_input_path": args.runner_input_path,
    }
    diagnoses: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        prompt_ids = {
            prompt_id for prompt_id, row in raw_by_prompt.items() if row.get("vertical") == vertical
        }
        metrics = {
            **retrieval[vertical],
            **_quality_metrics(evaluation_rows, prompt_ids=prompt_ids),
            **latency[vertical],
            **shared,
        }
        diagnosis = diagnose_slos(
            run_metrics=metrics,
            profile=profile,
            experiment_config=experiment_config,
            model_metadata=_model_metadata("model2_1_5b"),
            hardware_profile=hardware,
            engine="vllm",
            memory_mode="mm2_hybrid_top5",
            vertical=vertical,
            telemetry_available=True,
            backend_type="self_hosted",
        )
        diagnosis["block"] = "B4"
        diagnosis["source_artifacts"] = {
            "eval_report": args.eval_report,
            "results_path": args.results_path,
            "runner_input_path": args.runner_input_path,
            "latency_summary": args.latency_summary,
            "gpu_summary": args.gpu_summary,
            "context_report": args.context_report,
        }
        diagnoses.append(diagnosis)
        summary_rows.append(_summary_row(diagnosis))
        recommendation_rows.extend(_recommendation_rows(diagnosis))

    primary_counts = Counter(
        str(row["primary_optimization"]) for row in summary_rows if row["primary_optimization"]
    )
    gold_absent_remaining = int(quality_audit["overall"]["rows_with_any_gold_evidence_absent"])
    report = {
        "block": "B4",
        "profile": profile.name,
        "priority_mode": profile.priority_mode,
        "diagnosis_scope": "B4_context_aligned_vllm_1_5b_artifacts_no_new_inference",
        "quality_gate": eval_report.get("quality_gate"),
        "quality_audit_paths": {
            "report": args.quality_audit_report,
            "summary": args.quality_audit_summary,
            "finance_examples": args.finance_examples,
            "failure_examples": args.failure_examples,
        },
        "gold_absent_after_alignment": gold_absent_remaining,
        "recommendation_guardrail": {
            "gold_absent_remaining": gold_absent_remaining,
            "stronger_model_only_recommendation_allowed": gold_absent_remaining == 0,
            "interpretation": (
                "B4 has no remaining gold-absent failed rows. If a future rerun has "
                "gold-absent rows, context alignment/retrieval must be diagnosed before "
                "treating a stronger model as the sole repair."
            ),
        },
        "diagnoses": diagnoses,
        "aggregate": {
            "diagnosis_count": len(diagnoses),
            "failed_slo_count": sum(len(item["failed_slos"]) for item in diagnoses),
            "unavailable_metric_count": sum(len(item["unavailable_metrics"]) for item in diagnoses),
            "not_applicable_slo_count": sum(len(item["not_applicable_slos"]) for item in diagnoses),
            "primary_recommendation_counts": dict(sorted(primary_counts.items())),
        },
        "llm_used": False,
    }
    _write_json(args.slo_report, report)
    _write_csv(args.slo_summary, summary_rows)
    _write_csv(args.recommendations_summary, recommendation_rows)
    return report


def main() -> None:
    """Run the offline B4 quality/SLO diagnosis."""

    args = build_parser().parse_args()
    report = build_b4_diagnosis(args)
    print(
        json.dumps(
            {
                "status": "B4_DIAGNOSIS_COMPLETE",
                "gold_absent_after_alignment": report["gold_absent_after_alignment"],
                "failed_slo_count": report["aggregate"]["failed_slo_count"],
                "quality_gate_status": report["quality_gate"]["status"],
                "outputs": [
                    args.quality_audit_report,
                    args.quality_audit_summary,
                    args.finance_examples,
                    args.failure_examples,
                    args.slo_report,
                    args.slo_summary,
                    args.recommendations_summary,
                ],
                "llm_used": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
