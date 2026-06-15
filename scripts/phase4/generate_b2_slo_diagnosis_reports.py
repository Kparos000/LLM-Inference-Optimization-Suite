"""Generate deterministic B2 diagnosis reports from existing measured artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from inference_bench.slo_diagnosis import diagnose_slos
from inference_bench.slo_profiles import resolve_slo_profile

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "results" / "processed"
RAW = ROOT / "results" / "raw"
VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")

RUNS = (
    {
        "block": "A1",
        "experiment_config": "configs/experiments/a1_remote_rtx3070_vllm_smoke.yaml",
        "manifest": "results/raw/a1_remote_rtx3070_vllm_smoke_manifest.json",
        "raw_results": "results/raw/a1_remote_rtx3070_vllm_smoke_results.jsonl",
        "eval_report": "results/processed/a1_remote_rtx3070_vllm_eval_report.json",
        "latency_summary": "results/processed/a1_remote_rtx3070_vllm_latency_summary.csv",
        "telemetry_summary": ("results/processed/a1_remote_rtx3070_gpu_telemetry_summary.json"),
    },
    {
        "block": "A2_A3",
        "experiment_config": "configs/experiments/a2_remote_rtx3070_sglang_smoke.yaml",
        "manifest": "results/raw/a2_remote_rtx3070_sglang_smoke_manifest.json",
        "raw_results": "results/raw/a2_remote_rtx3070_sglang_smoke_results.jsonl",
        "eval_report": "results/processed/a2_remote_rtx3070_sglang_eval_report.json",
        "latency_summary": "results/processed/a2_remote_rtx3070_sglang_latency_summary.csv",
        "telemetry_summary": (
            "results/processed/a2_remote_rtx3070_sglang_gpu_telemetry_summary.json"
        ),
    },
    {
        "block": "A5_A6",
        "experiment_config": "configs/experiments/a5_mm4_bounded_agentic_smoke.yaml",
        "manifest": "results/raw/a6_mm4_agentic_smoke_manifest.json",
        "raw_results": "results/raw/a6_mm4_agentic_smoke_results.jsonl",
        "eval_report": "results/processed/a6_mm4_agentic_eval_report.json",
        "latency_summary": "results/processed/a6_mm4_agentic_latency_summary.csv",
        "trace_summary": "results/processed/a6_mm4_agentic_trace_summary.csv",
        "telemetry_summary": None,
    },
)


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def _read_yaml(path: str) -> dict[str, Any]:
    payload = yaml.safe_load((ROOT / path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = f"Expected YAML mapping in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], payload)


def _read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(cast(dict[str, Any], payload))
    return rows


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(str(row[field])) for row in rows if row.get(field) not in (None, "")]
    return sum(values) / len(values) if values else None


def _rate(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [bool(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _quality_metrics(
    evaluation_rows: list[dict[str, Any]],
    *,
    vertical_prompt_ids: set[str],
) -> dict[str, Any]:
    rows = [row for row in evaluation_rows if str(row.get("prompt_id")) in vertical_prompt_ids]
    return {
        "grounded_rate": _rate(rows, "groundedness"),
        "evidence_match_rate": _rate(rows, "evidence_match"),
        "task_success_rate": _rate(rows, "status_matches"),
        "generation_contract_valid_rate": _rate(rows, "generation_contract_valid"),
        "safety_violation_count": sum(bool(row.get("safety_violation")) for row in rows),
    }


def _latency_by_vertical(path: str) -> dict[str, dict[str, float]]:
    return {
        row["vertical"]: {
            key: float(value)
            for key, value in row.items()
            if key != "vertical" and value not in (None, "")
        }
        for row in _read_csv(path)
    }


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
    manifest: dict[str, Any], raw_rows: list[dict[str, Any]]
) -> dict[str, float]:
    start = datetime.fromisoformat(str(manifest["start_time"]))
    end = datetime.fromisoformat(str(manifest["end_time"]))
    elapsed = (end - start).total_seconds()
    success_count = sum(bool(row.get("success")) for row in raw_rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in raw_rows)
    return {
        "requests_per_second_min": len(raw_rows) / elapsed,
        "successful_requests_per_second_min": success_count / elapsed,
        "aggregate_tokens_per_second": output_tokens / elapsed,
    }


def _telemetry_metrics(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    telemetry = _read_json(path)
    utilization = cast(dict[str, Any], telemetry.get("utilization_gpu_percent", {}))
    memory_used = cast(dict[str, Any], telemetry.get("memory_used_mb", {}))
    memory_total = cast(dict[str, Any], telemetry.get("memory_total_mb", {}))
    return {
        "mean_gpu_utilization_percent": float(utilization["mean"]),
        "max_gpu_memory_used_mb": float(memory_used["max"]),
        "gpu_memory_total_mb": float(memory_total["max"]),
    }


def _agentic_metrics(run: dict[str, Any], raw_rows: list[dict[str, Any]]) -> dict[str, float]:
    trace_path = run.get("trace_summary")
    if not isinstance(trace_path, str):
        return {}
    row = _read_csv(trace_path)[0]
    complete_count = sum(bool(item.get("node_latencies")) for item in raw_rows)
    return {
        "mean_retrieval_rounds": float(row["mean_retrieval_rounds"]),
        "mean_generation_attempts": cast(float, _mean(raw_rows, "generation_attempts")),
        "mean_repair_attempts": cast(float, _mean(raw_rows, "repair_attempts")),
        "mean_tool_call_count": float(row["mean_tool_call_count"]),
        "trace_completeness": complete_count / len(raw_rows),
    }


def _model_metadata(model_alias: str) -> dict[str, Any]:
    models = _read_yaml("configs/models.yaml")
    aliases = cast(dict[str, str], models["model_aliases"])
    return cast(dict[str, Any], models[aliases[model_alias]])


def _summary_row(block: str, diagnosis: dict[str, Any]) -> dict[str, Any]:
    primary = diagnosis.get("primary_recommendation")
    return {
        "block": block,
        "vertical": diagnosis["context"]["vertical"],
        "engine": diagnosis["context"]["engine"],
        "memory_mode": diagnosis["context"]["memory_mode"],
        "selected_slo_count": len(diagnosis["selected_slos"]),
        "passed_slo_count": len(diagnosis["passed_slos"]),
        "failed_slo_count": len(diagnosis["failed_slos"]),
        "not_applicable_slo_count": len(diagnosis["not_applicable_slos"]),
        "unavailable_metric_count": len(diagnosis["unavailable_metrics"]),
        "bottleneck_ids": ";".join(item["id"] for item in diagnosis["bottlenecks"]),
        "primary_optimization": (primary["optimization_id"] if isinstance(primary, dict) else ""),
        "llm_used": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate the four ignored local B2 reports."""

    profile = resolve_slo_profile()
    retrieval = _retrieval_by_vertical()
    hardware_profile = _read_yaml("configs/hardware/remote_rtx3070.yaml")
    diagnoses: list[dict[str, Any]] = []
    diagnosis_summary: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []
    for run in RUNS:
        experiment = _read_yaml(str(run["experiment_config"]))
        manifest = _read_json(str(run["manifest"]))
        raw_rows = _read_jsonl(str(run["raw_results"]))
        raw_by_prompt = {str(row["prompt_id"]): row for row in raw_rows}
        evaluation = _read_json(str(run["eval_report"]))
        evaluation_rows = cast(list[dict[str, Any]], evaluation["evaluation_rows"])
        latency = _latency_by_vertical(str(run["latency_summary"]))
        shared = {
            **_throughput_metrics(manifest, raw_rows),
            **_telemetry_metrics(run.get("telemetry_summary")),
            **_agentic_metrics(run, raw_rows),
        }
        for vertical in VERTICALS:
            prompt_ids = {
                prompt_id
                for prompt_id, row in raw_by_prompt.items()
                if row.get("vertical") == vertical
            }
            metrics = {
                **retrieval[vertical],
                **_quality_metrics(evaluation_rows, vertical_prompt_ids=prompt_ids),
                **latency[vertical],
                **shared,
            }
            diagnosis = diagnose_slos(
                run_metrics=metrics,
                profile=profile,
                experiment_config=experiment,
                model_metadata=_model_metadata(str(experiment["model_alias"])),
                hardware_profile=hardware_profile,
                engine=str(experiment["engine"]),
                memory_mode=str(experiment["memory_mode"]),
                vertical=vertical,
                telemetry_available=run.get("telemetry_summary") is not None,
            )
            diagnosis["block"] = run["block"]
            diagnosis["source_artifacts"] = {
                key: value
                for key, value in run.items()
                if key not in {"block"} and value is not None
            }
            diagnoses.append(diagnosis)
            diagnosis_summary.append(_summary_row(str(run["block"]), diagnosis))
            for recommendation in diagnosis["recommended_optimizations"]:
                recommendation_rows.append(
                    {
                        "block": run["block"],
                        "vertical": vertical,
                        "engine": experiment["engine"],
                        "memory_mode": experiment["memory_mode"],
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

    primary_counts = Counter(
        str(row["primary_optimization"]) for row in diagnosis_summary if row["primary_optimization"]
    )
    report = {
        "phase": "B2",
        "profile": profile.name,
        "priority_mode": profile.priority_mode,
        "diagnosis_scope": "existing_A1_A2_A6_artifacts_no_new_inference",
        "diagnoses": diagnoses,
        "aggregate": {
            "diagnosis_count": len(diagnoses),
            "failed_slo_count": sum(len(item["failed_slos"]) for item in diagnoses),
            "unavailable_metric_count": sum(len(item["unavailable_metrics"]) for item in diagnoses),
            "not_applicable_slo_count": sum(len(item["not_applicable_slos"]) for item in diagnoses),
        },
        "llm_used": False,
    }
    recommendation_report = {
        "phase": "B2",
        "decision_source": "deterministic_rules_and_yaml_catalogs",
        "primary_recommendation_counts": dict(sorted(primary_counts.items())),
        "recommendation_rows": recommendation_rows,
        "already_active_vllm_capability": (
            "PagedAttention is represented as engine_builtin and is not proposed "
            "as a new toggle for vLLM."
        ),
        "future_explainer_model_candidate": "model6_gated",
        "future_explainer_is_source_of_truth": False,
        "llm_used": False,
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "b2_slo_diagnosis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(PROCESSED / "b2_slo_diagnosis_summary.csv", diagnosis_summary)
    (PROCESSED / "b2_optimization_recommendation_report.json").write_text(
        json.dumps(recommendation_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        PROCESSED / "b2_optimization_recommendation_summary.csv",
        recommendation_rows,
    )


if __name__ == "__main__":
    main()
