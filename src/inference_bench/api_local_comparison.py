"""Offline API-versus-local comparison and small-GPU readiness gate."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from inference_bench.phase4_readiness import (
    REQUIRED_TELEMETRY_FIELDS,
    load_gpu_cost_configs,
)
from inference_bench.telemetry import TELEMETRY_FIELDS

COMPARISON_METRICS = (
    "json_valid_rate",
    "generation_contract_valid_rate",
    "evidence_id_presence_rate",
    "evidence_match_rate",
    "grounded_rate",
    "safety_violation_rate",
    "input_tokens",
    "output_tokens",
    "mean_latency_ms",
    "median_latency_ms",
    "cost_per_request_usd",
    "cost_per_successful_answer_usd",
    "cost_per_grounded_answer_usd",
)


@dataclass(frozen=True)
class GateCheck:
    """One required small-GPU readiness check."""

    criterion: str
    status: str
    artifact: str
    details: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable JSON/CSV mapping."""

        return {
            "criterion": self.criterion,
            "status": self.status,
            "artifact": self.artifact,
            "details": self.details,
        }


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Expected a JSON object in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL object rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                msg = f"Expected JSON object at {path}:{line_number}"
                raise ValueError(msg)
            rows.append(cast(dict[str, Any], loaded))
    return rows


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _runtime_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    latencies = sorted(_numeric_values(rows, "latency_ms"))
    return {
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }


def _quality_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        msg = "Evaluation report is missing its summary object"
        raise ValueError(msg)
    metrics: dict[str, float | None] = {}
    for key in (
        "json_valid_rate",
        "generation_contract_valid_rate",
        "evidence_id_presence_rate",
        "evidence_match_rate",
        "grounded_rate",
        "safety_violation_rate",
    ):
        value = summary.get(key)
        metrics[key] = (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )
    return metrics


def _sha256(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def compare_workloads(
    local_rows: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify record alignment and report prompt-rendering parity."""

    if len(local_rows) != 5 or len(api_rows) != 5:
        msg = "Block 28 requires exactly five local and five API rows"
        raise ValueError(msg)
    local_by_id = {str(row.get("prompt_id") or ""): row for row in local_rows}
    api_by_id = {str(row.get("prompt_id") or ""): row for row in api_rows}
    if set(local_by_id) != set(api_by_id):
        msg = "Local and API result prompt IDs do not match"
        raise ValueError(msg)
    comparisons: list[dict[str, Any]] = []
    for prompt_id in sorted(local_by_id):
        local = local_by_id[prompt_id]
        api = api_by_id[prompt_id]
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "vertical": local.get("vertical"),
                "vertical_matches": local.get("vertical") == api.get("vertical"),
                "memory_mode_matches": local.get("memory_mode") == api.get("memory_mode"),
                "local_ablation_mode": local.get("ablation_mode"),
                "api_ablation_mode": api.get("ablation_mode"),
                "prompt_rendering_matches": local.get("prompt") == api.get("prompt"),
                "citation_aliases_match": (
                    local.get("citation_id_aliases") == api.get("citation_id_aliases")
                ),
                "local_prompt_sha256": _sha256(local.get("prompt")),
                "api_prompt_sha256": _sha256(api.get("prompt")),
            }
        )
    return {
        "prompt_id_set_matches": True,
        "verticals_match": all(row["vertical_matches"] for row in comparisons),
        "memory_modes_match": all(row["memory_mode_matches"] for row in comparisons),
        "exact_prompt_rendering_matches": all(
            row["prompt_rendering_matches"] for row in comparisons
        ),
        "citation_aliases_match": all(row["citation_aliases_match"] for row in comparisons),
        "comparison_scope": (
            "prompt_id_aligned_same_generation_contract_with_renderer_drift"
            if not all(row["prompt_rendering_matches"] for row in comparisons)
            else "exact_same_rendered_workload"
        ),
        "records": comparisons,
    }


def build_metric_comparison(
    *,
    local_eval_report: dict[str, Any],
    api_eval_report: dict[str, Any],
    local_rows: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
    api_cost_report: dict[str, Any],
) -> tuple[dict[str, dict[str, float | int | None]], list[dict[str, Any]]]:
    """Build metric comparison values and CSV rows."""

    local_metrics: dict[str, float | int | None] = {}
    local_metrics.update(_quality_metrics(local_eval_report))
    local_metrics.update(_runtime_metrics(local_rows))
    local_metrics.update(
        {
            "cost_per_request_usd": None,
            "cost_per_successful_answer_usd": None,
            "cost_per_grounded_answer_usd": None,
        }
    )
    api_metrics: dict[str, float | int | None] = {}
    api_metrics.update(_quality_metrics(api_eval_report))
    api_metrics.update(_runtime_metrics(api_rows))
    for key in (
        "cost_per_request_usd",
        "cost_per_successful_answer_usd",
        "cost_per_grounded_answer_usd",
    ):
        value = api_cost_report.get(key)
        api_metrics[key] = (
            float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None
        )
    comparison: dict[str, dict[str, float | int | None]] = {}
    rows: list[dict[str, Any]] = []
    for metric in COMPARISON_METRICS:
        local_value = local_metrics.get(metric)
        api_value = api_metrics.get(metric)
        delta = (
            float(api_value) - float(local_value)
            if api_value is not None and local_value is not None
            else None
        )
        comparison[metric] = {
            "local_qwen_0_5b": local_value,
            "api_llama_3_1_8b": api_value,
            "api_minus_local": delta,
        }
        rows.append(
            {
                "row_type": "metric",
                "name": metric,
                "status": "",
                "local_qwen_0_5b": local_value,
                "api_llama_3_1_8b": api_value,
                "api_minus_local": delta,
                "artifact": "",
                "details": (
                    "Local infrastructure cost was not measured."
                    if metric.startswith("cost_")
                    else ""
                ),
            }
        )
    return comparison, rows


def _retrieval_slo_check(manifest: dict[str, Any], artifact: str) -> GateCheck:
    passed = (
        manifest.get("retrieval_slo_status") == "PASS"
        and manifest.get("retrieval_promotion_status") == "PROMOTED"
        and manifest.get("retrieval_ready_for_phase4") is True
    )
    return GateCheck(
        criterion="retrieval_slo_pass",
        status="PASS" if passed else "FAIL",
        artifact=artifact,
        details="Promoted retrieval source of truth passes all vertical SLOs.",
    )


def build_readiness_gate(
    *,
    repo_root: str | Path,
    local_eval_report: dict[str, Any],
    api_eval_report: dict[str, Any],
    local_rows: list[dict[str, Any]],
    api_cost_report: dict[str, Any],
    retrieval_manifest_path: str | Path,
    gpu_costs_path: str | Path,
) -> tuple[str, list[GateCheck], list[str]]:
    """Evaluate the explicit Block 28 small-GPU readiness criteria."""

    root = Path(repo_root)
    retrieval_manifest = read_json(root / retrieval_manifest_path)
    checks = [_retrieval_slo_check(retrieval_manifest, str(retrieval_manifest_path))]
    local_summary = local_eval_report.get("summary", {})
    api_summary = api_eval_report.get("summary", {})
    contract_works = (
        isinstance(local_summary, dict)
        and isinstance(api_summary, dict)
        and float(local_summary.get("generation_contract_valid_rate") or 0) > 0
        and float(api_summary.get("generation_contract_valid_rate") or 0) > 0
    )
    checks.append(
        GateCheck(
            criterion="generation_contract_works",
            status="PASS" if contract_works else "FAIL",
            artifact="src/inference_bench/generation_contract.py",
            details="Both measured runs produced evaluator-recognized contract-valid output.",
        )
    )
    api_success = (
        api_cost_report.get("execution_complete") is True
        and api_cost_report.get("request_count") == 5
        and api_cost_report.get("success_count") == 5
    )
    checks.append(
        GateCheck(
            criterion="api_smoke_successful",
            status="PASS" if api_success else "FAIL",
            artifact="results/processed/phase4_api_priced_cost_report.json",
            details="Five API-priced requests completed successfully.",
        )
    )
    cost_works = all(
        api_cost_report.get(key) is not None
        for key in (
            "pricing_source_url",
            "total_cost_usd",
            "cost_per_request_usd",
            "cost_per_successful_answer_usd",
            "cost_per_grounded_answer_usd",
        )
    )
    checks.append(
        GateCheck(
            criterion="cost_accounting_works",
            status="PASS" if cost_works else "FAIL",
            artifact="results/processed/phase4_api_priced_cost_report.json",
            details="Provider pricing and per-request/success/grounded costs are measured.",
        )
    )
    local_success = (
        len(local_rows) == 5
        and all(bool(row.get("success")) for row in local_rows)
        and isinstance(local_summary, dict)
        and local_summary.get("joined_count") == 5
    )
    checks.append(
        GateCheck(
            criterion="local_hf_smoke_successful",
            status="PASS" if local_success else "FAIL",
            artifact="results/processed/phase4_generation_contract_hardened_eval_report.json",
            details="Five local HF outputs completed and joined to promoted gold records.",
        )
    )
    telemetry_missing = sorted(REQUIRED_TELEMETRY_FIELDS.difference(TELEMETRY_FIELDS))
    checks.append(
        GateCheck(
            criterion="telemetry_exists",
            status="PASS" if not telemetry_missing else "FAIL",
            artifact="src/inference_bench/telemetry.py",
            details=(
                "Request telemetry and nullable GPU fields are defined."
                if not telemetry_missing
                else f"Missing telemetry fields: {', '.join(telemetry_missing)}"
            ),
        )
    )
    gpu_configs = load_gpu_cost_configs(root / gpu_costs_path)
    gpu_config_exists = "runpod_default" in gpu_configs
    checks.append(
        GateCheck(
            criterion="gpu_cost_config_exists",
            status="PASS" if gpu_config_exists else "FAIL",
            artifact=str(gpu_costs_path),
            details=(
                "RunPod provider and elapsed-hours cost formula are configured; live values "
                "must be filled after provisioning."
            ),
        )
    )
    blockers = [f"{check.criterion}: {check.details}" for check in checks if check.status != "PASS"]
    decision = "READY_FOR_SMALL_GPU_SMOKE" if not blockers else "NOT_READY"
    return decision, checks, blockers


def exact_next_gpu_plan() -> list[dict[str, str]]:
    """Return the reviewed command plan without executing GPU work."""

    return [
        {
            "step": "1_record_cost_inputs",
            "command": (
                "Copy configs/gpu_costs.yaml to a run-specific file and fill gpu_type, "
                "hourly_price_usd, region, instance_id_optional, and measured_start_time."
            ),
        },
        {
            "step": "2_start_vllm",
            "command": (
                "vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 "
                "--port 8000 --dtype auto --api-key EMPTY"
            ),
        },
        {
            "step": "3_check_server",
            "command": "curl http://localhost:8000/v1/models",
        },
        {
            "step": "4_run_five_prompt_smoke",
            "command": (
                "python scripts/phase4/run_openai_compatible_smoke.py "
                "--input-path data/generated/phase4/api_priced_contract_runner_input.jsonl "
                "--output-path results/raw/phase4_vllm_gpu_smoke_results.jsonl "
                "--model-alias model1_0_5b "
                "--model-name Qwen/Qwen2.5-0.5B-Instruct "
                "--base-url http://localhost:8000/v1 --api-key EMPTY "
                "--limit 5 --max-new-tokens 256"
            ),
        },
        {
            "step": "5_evaluate",
            "command": (
                "python scripts/phase4/evaluate_generation_outputs.py "
                "--results-path results/raw/phase4_vllm_gpu_smoke_results.jsonl "
                "--dataset-root data/scaleup_2000_full "
                "--output-root results/processed "
                "--report-name phase4_vllm_gpu_smoke_eval_report.json "
                "--summary-name phase4_vllm_gpu_smoke_eval_summary.csv"
            ),
        },
        {
            "step": "6_capture_gpu_telemetry_and_cost",
            "command": (
                "Capture TTFT, TPOT, throughput, GPU utilization, GPU memory, power, "
                "temperature, measured_end_time, and elapsed RunPod cost before scaling."
            ),
        },
    ]


def build_comparison_report(
    *,
    repo_root: str | Path,
    local_results_path: str | Path,
    local_eval_path: str | Path,
    api_results_path: str | Path,
    api_eval_path: str | Path,
    api_cost_path: str | Path,
    retrieval_manifest_path: str | Path,
    gpu_costs_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the complete comparison and decision report."""

    root = Path(repo_root)
    local_rows = read_jsonl(root / local_results_path)
    api_rows = read_jsonl(root / api_results_path)
    local_eval = read_json(root / local_eval_path)
    api_eval = read_json(root / api_eval_path)
    api_cost = read_json(root / api_cost_path)
    workload_comparison = compare_workloads(local_rows, api_rows)
    metric_comparison, summary_rows = build_metric_comparison(
        local_eval_report=local_eval,
        api_eval_report=api_eval,
        local_rows=local_rows,
        api_rows=api_rows,
        api_cost_report=api_cost,
    )
    decision, checks, blockers = build_readiness_gate(
        repo_root=root,
        local_eval_report=local_eval,
        api_eval_report=api_eval,
        local_rows=local_rows,
        api_cost_report=api_cost,
        retrieval_manifest_path=retrieval_manifest_path,
        gpu_costs_path=gpu_costs_path,
    )
    for check in checks:
        summary_rows.append(
            {
                "row_type": "readiness_check",
                "name": check.criterion,
                "status": check.status,
                "local_qwen_0_5b": "",
                "api_llama_3_1_8b": "",
                "api_minus_local": "",
                "artifact": check.artifact,
                "details": check.details,
            }
        )
    report = {
        "decision": decision,
        "remaining_blockers": blockers,
        "comparison_limitations": [
            (
                "Prompt IDs, verticals, memory mode, and evaluator contract align, but the "
                "Block 24 and Block 27 rendered prompts and citation alias maps are not "
                "byte-identical. Treat latency and token deltas as directional."
            )
        ],
        "workload_comparison": workload_comparison,
        "metrics": metric_comparison,
        "readiness_checks": [check.to_dict() for check in checks],
        "exact_next_gpu_smoke_command_plan": exact_next_gpu_plan(),
        "operational_prerequisites_before_execution": [
            "Provision a GPU with sufficient VRAM for model1_0_5b.",
            "Record the actual GPU type, region, and hourly price.",
            "Start vLLM and verify the OpenAI-compatible /v1/models endpoint.",
            "Enable request-level streaming and GPU telemetry capture.",
        ],
        "local_cost_status": (
            "NOT_AVAILABLE: local CPU infrastructure and energy cost were not measured."
        ),
        "api_cost_status": "MEASURED",
        "no_gpu_work_triggered": True,
        "no_vllm_triggered": True,
        "no_sglang_triggered": True,
        "no_additional_paid_api_calls_triggered": True,
        "input_artifacts": {
            "local_results": str(local_results_path),
            "local_evaluation": str(local_eval_path),
            "api_results": str(api_results_path),
            "api_evaluation": str(api_eval_path),
            "api_cost": str(api_cost_path),
            "retrieval_manifest": str(retrieval_manifest_path),
            "gpu_cost_config": str(gpu_costs_path),
        },
    }
    return report, summary_rows


def write_comparison_artifacts(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    report: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write Block 28 JSON and CSV outputs."""

    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_output = Path(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "name",
        "status",
        "local_qwen_0_5b",
        "api_llama_3_1_8b",
        "api_minus_local",
        "artifact",
        "details",
    ]
    with summary_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return report_output, summary_output
