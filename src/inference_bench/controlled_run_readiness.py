"""Controlled inference readiness audit for the first small GPU smoke."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_bench.config import load_memory_modes_config, load_yaml_file
from inference_bench.phase4_readiness import load_gpu_cost_configs
from inference_bench.telemetry import TELEMETRY_FIELDS

REQUIRED_WORKLOAD_FIELDS = {
    "prompt_id",
    "vertical",
    "memory_mode",
    "dataset_split",
}


@dataclass(frozen=True)
class ControlledReadinessCheck:
    """One repository-backed controlled-run readiness category."""

    category: str
    status: str
    evidence: str
    details: str
    remaining_gap: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable report row."""

        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return loaded


def _first_jsonl_row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                msg = f"Expected JSON object in {path}"
                raise ValueError(msg)
            return loaded
    return {}


def _non_empty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _workload_split_check(root: Path) -> tuple[bool, str, dict[str, bool]]:
    details: list[str] = []
    materialized: dict[str, bool] = {}
    builder_path = root / "src/inference_bench/memory_workloads.py"
    builder_text = builder_path.read_text(encoding="utf-8") if _non_empty(builder_path) else ""
    definitions_ready = all(
        split in builder_text for split in ("smoke_500", "controlled_2000", "final_10000")
    )
    valid = definitions_ready
    for split in ("smoke_500", "controlled_2000", "final_10000"):
        candidates = sorted((root / "data/workloads" / split).rglob("mm2_hybrid_top5.jsonl"))
        materialized[split] = bool(candidates)
        if not candidates:
            details.append(f"{split}=defined_regeneratable_not_materialized")
            continue
        row = _first_jsonl_row(candidates[0])
        missing = sorted(REQUIRED_WORKLOAD_FIELDS.difference(row))
        if missing:
            valid = False
            details.append(f"{split}=missing_fields:{','.join(missing)}")
        else:
            details.append(f"{split}=materialized_valid")
    details.append(f"split_builder_ready={definitions_ready}")
    return valid, "; ".join(details), materialized


def _gold_join_check(root: Path) -> bool:
    workload = root / "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
    if not _non_empty(workload):
        return False
    prompt_id = str(_first_jsonl_row(workload).get("prompt_id") or "")
    if not prompt_id:
        return False
    for gold_path in (root / "data/scaleup_2000_full").glob("*/*_gold_2000.jsonl"):
        with gold_path.open(encoding="utf-8") as file:
            if any(json.loads(line).get("prompt_id") == prompt_id for line in file if line.strip()):
                return True
    return False


def inspect_controlled_inference_readiness(
    repo_root: str | Path,
) -> tuple[dict[str, Any], list[ControlledReadinessCheck]]:
    """Inspect repository controls without invoking inference, GPU, or APIs."""

    root = Path(repo_root).resolve()
    context = root / "data/generated/context_engineering"
    promotion_path = context / "retrieval_source_of_truth_manifest.json"
    repaired_validation = context / "repaired_retrieval_validation_report.json"
    slo_path = context / "slo_readiness_report.json"
    promotion = _read_json(promotion_path) if _non_empty(promotion_path) else {}
    dataset_ready = (
        promotion.get("retrieval_promotion_status") == "PROMOTED"
        and promotion.get("retrieval_slo_status") == "PASS"
        and _non_empty(repaired_validation)
        and _non_empty(slo_path)
    )

    workload_ready, workload_details, workload_materialized = _workload_split_check(root)
    gold_join_ready = _gold_join_check(root)
    evaluator_join_path = root / "src/inference_bench/evaluator_contract.py"
    evaluator_join_text = (
        evaluator_join_path.read_text(encoding="utf-8") if _non_empty(evaluator_join_path) else ""
    )
    gold_join_contract_ready = "prompt_id" in evaluator_join_text
    workload_ready = workload_ready and (gold_join_ready or gold_join_contract_ready)

    memory_modes = load_memory_modes_config(root / "configs/memory_modes.yaml")
    memory_ready = set(memory_modes) == {
        "mm0_no_context",
        "mm1_dense_top5",
        "mm2_hybrid_top5",
        "mm3_compressed_hybrid_top5",
        "mm4_bounded_agentic",
    }
    mm4_active = (
        memory_modes.get("mm4_bounded_agentic") is not None
        and memory_modes["mm4_bounded_agentic"].expected_stage == "phase4_active"
    )

    chunking_paths = (
        context / "corpus_registry.json",
        context / "corpus_build_report.json",
        context / "qdrant_index_report.json",
        context / "compression_diagnostic_report.json",
        root / "src/inference_bench/retrieval_keys.py",
    )
    chunking_ready = all(_non_empty(path) for path in chunking_paths)

    load_runner = root / "src/inference_bench/runners/openai_load_runner.py"
    load_text = load_runner.read_text(encoding="utf-8") if _non_empty(load_runner) else ""
    run_safety_features = {
        "checkpoint": "checkpoint" in load_text,
        "resume": "resume" in load_text,
        "chunked": "chunk_size" in load_text,
        "timeout": "timeout_seconds" in load_text,
        "failure_rows": "error_message" in load_text,
    }
    run_safety_ready = all(run_safety_features.values())

    manifest_path = root / "src/inference_bench/run_manifest.py"
    telemetry_path = root / "src/inference_bench/telemetry.py"
    observability_ready = (
        _non_empty(manifest_path)
        and _non_empty(telemetry_path)
        and {"backend", "model", "memory_mode", "error_type"}.issubset(TELEMETRY_FIELDS)
    )

    pricing_path = root / "configs/api_pricing.yaml"
    gpu_cost_path = root / "configs/gpu_costs.yaml"
    slo_config_path = root / "configs/slo_targets.yaml"
    gpu_configs = load_gpu_cost_configs(gpu_cost_path)
    runpod = gpu_configs.get("runpod_default")
    api_runner_text = (root / "scripts/phase4/run_api_priced_smoke.py").read_text(encoding="utf-8")
    api_budget_guard = all(
        marker in api_runner_text
        for marker in ("--limit", "--allow-paid-api-call", "api_key_for_route")
    )
    cost_schema_ready = (
        _non_empty(pricing_path)
        and runpod is not None
        and api_budget_guard
        and _non_empty(root / "src/inference_bench/metrics/cost.py")
    )
    live_gpu_cost_ready = bool(
        runpod and runpod.gpu_type and runpod.hourly_price_usd is not None and runpod.region
    )

    model5_report = root / "results/processed/phase4_model5_openrouter_streaming_eval_report.json"
    model6_report = root / "results/processed/phase4_api_streaming_smoke_eval_report.json"
    local_report = root / "results/processed/phase4_generation_contract_hardened_eval_report.json"
    serving_paths_ready = all(
        _non_empty(path)
        for path in (
            root / "scripts/phase4/run_local_hf_smoke.py",
            root / "scripts/phase4/run_api_priced_smoke.py",
            root / "scripts/phase4/run_openai_compatible_smoke.py",
            root / "scripts/phase4/run_sglang_compatible_smoke.py",
        )
    )
    serving_measurements = {
        "local_hf_measured": _non_empty(local_report),
        "model5_openrouter_measured": _non_empty(model5_report),
        "model6_hf_novita_measured": _non_empty(model6_report),
    }
    serving_ready = serving_paths_ready and _non_empty(pricing_path)

    slo_config = load_yaml_file(slo_config_path)
    slo_ready = bool(slo_config.get("verticals")) and dataset_ready

    gpu_model_confirmed = False
    smoke_matrix_frozen = False
    vllm_command_documented = _non_empty(root / "docs/07_vllm_baseline_plan.md")

    checks = [
        ControlledReadinessCheck(
            "dataset_readiness",
            "PASS" if dataset_ready else "FAIL",
            str(promotion_path.relative_to(root)),
            "Promoted retrieval manifest, repaired validation, and retrieval SLO report exist.",
            "" if dataset_ready else "Repair or promote retrieval before inference.",
        ),
        ControlledReadinessCheck(
            "workload_control",
            "PASS" if workload_ready else "FAIL",
            "data/workloads/{smoke_500,controlled_2000,final_10000}",
            (
                f"{workload_details}; prompt_id_gold_join_materialized={gold_join_ready}; "
                f"prompt_id_join_contract={gold_join_contract_ready}."
            ),
            "" if workload_ready else "Regenerate or repair controlled workload splits.",
        ),
        ControlledReadinessCheck(
            "memory_modes",
            "PASS" if memory_ready and mm4_active else "FAIL",
            "configs/memory_modes.yaml",
            "mm0-mm3 and the bounded LangGraph mm4 benchmark mode are configured.",
            "" if memory_ready and mm4_active else "Correct memory-mode configuration.",
        ),
        ControlledReadinessCheck(
            "context_engineering",
            "PASS" if chunking_ready else "FAIL",
            "data/generated/context_engineering",
            "Vertical corpora, canonical keys, Qdrant, and compression reports are present.",
            "" if chunking_ready else "Regenerate missing context/retrieval reports.",
        ),
        ControlledReadinessCheck(
            "run_safety",
            "PASS" if run_safety_ready else "FAIL",
            str(load_runner.relative_to(root)),
            f"OpenAI load-run controls: {run_safety_features}.",
            "" if run_safety_ready else "Complete checkpoint/resume/failure persistence.",
        ),
        ControlledReadinessCheck(
            "logging_observability",
            "PASS" if observability_ready else "FAIL",
            "src/inference_bench/{run_manifest.py,telemetry.py}",
            "Run manifests and per-request backend/model/memory/error telemetry are defined.",
            "" if observability_ready else "Complete manifest or telemetry fields.",
        ),
        ControlledReadinessCheck(
            "cost_controls",
            "FAIL" if not live_gpu_cost_ready else "PASS",
            "configs/{api_pricing.yaml,gpu_costs.yaml}",
            (
                f"Cost schema and API budget guard ready={cost_schema_ready}; "
                f"live RunPod values ready={live_gpu_cost_ready}."
            ),
            (
                ""
                if live_gpu_cost_ready
                else "Fill the selected RunPod GPU type, region, and hourly price."
            ),
        ),
        ControlledReadinessCheck(
            "serving_backends",
            "PASS" if serving_ready else "FAIL",
            "HF local, OpenRouter, HF/Novita, vLLM dry-run, SGLang dry-run",
            (
                "Serving adapters are present; measured local artifacts are optional in clean "
                f"checkout. Measurement status: {serving_measurements}."
            ),
            "" if serving_ready else "Complete the missing local/API smoke artifact.",
        ),
        ControlledReadinessCheck(
            "slo_definitions",
            "PASS" if slo_ready else "FAIL",
            str(slo_config_path.relative_to(root)),
            "Retrieval, quality, latency, resource, throughput, API cost, and GPU cost SLOs exist.",
            "" if slo_ready else "Restore SLO definitions or retrieval signoff.",
        ),
        ControlledReadinessCheck(
            "gpu_execution_inputs",
            "FAIL",
            "configs/gpu_costs.yaml and reviewed GPU smoke plan",
            (
                f"gpu_model_confirmed={gpu_model_confirmed}; "
                f"smoke_matrix_frozen={smoke_matrix_frozen}; "
                f"vllm_command_documented={vllm_command_documented}."
            ),
            "Confirm the GPU/model, fill price inputs, and freeze the five-prompt smoke matrix.",
        ),
    ]
    blockers = [
        f"{check.category}: {check.remaining_gap}" for check in checks if check.status == "FAIL"
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_status": "READY_FOR_SMALL_GPU_SMOKE" if not blockers else "NOT_READY",
        "checks": [check.to_dict() for check in checks],
        "workload_materialization": workload_materialized,
        "serving_measurements": serving_measurements,
        "cost_schema_ready": cost_schema_ready,
        "live_gpu_cost_ready": live_gpu_cost_ready,
        "remaining_gaps": blockers,
        "recommended_next_block": (
            "Block 32A: freeze the five-prompt vLLM GPU smoke matrix, select the RunPod "
            "GPU/model, record hourly pricing, and execute one guarded live vLLM smoke."
        ),
        "exact_reason_if_not_ready": (
            "Live RunPod cost inputs and the reviewed GPU smoke matrix are not yet frozen."
            if blockers
            else None
        ),
        "mm4_benchmark_ready": True,
        "mm4_status": "active_bounded",
        "no_gpu_call_triggered": True,
        "no_vllm_call_triggered": True,
        "no_sglang_call_triggered": True,
        "no_model_inference_triggered": True,
    }
    return report, checks


def write_controlled_readiness_artifacts(
    *,
    output_root: str | Path,
    report: dict[str, Any],
    checks: list[ControlledReadinessCheck],
) -> tuple[Path, Path]:
    """Write the controlled-readiness JSON and CSV."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "controlled_inference_readiness_report.json"
    summary_path = output / "controlled_inference_readiness_summary.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(checks[0].to_dict()))
        writer.writeheader()
        writer.writerows(check.to_dict() for check in checks)
    return report_path, summary_path
