"""Run the controlled final-experiment simulation safety gate and reports."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.api_load_probe import load_probe_environment  # noqa: E402
from inference_bench.artifact_sync import (  # noqa: E402
    ArtifactSyncConfig,
    build_artifact_specs,
    sync_artifacts,
    verify_backup,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.post_run_automation import (  # noqa: E402
    PostRunAutomationInputs,
    build_post_run_automation_report,
    write_post_run_automation_artifacts,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    hash_existing_paths,
    utc_now,
    write_run_manifest,
)
from inference_bench.runtime_registry import select_runtime_for_model  # noqa: E402

RUN_ID = "controlled_final_simulation"
SELF_HOSTED_MODEL_ALIAS = "model3_7b"
SELF_HOSTED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
API_MODEL_ALIAS = "model6_gated"
API_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
GPU_ID = "a100_sxm_80gb"
GPU_NAME = "NVIDIA A100-SXM4-80GB"
TRAFFIC_PROFILE = "online_low_latency"
MEMORY_MODES = (
    "mm0_no_context",
    "mm1_dense_top5",
    "mm2_hybrid_top5",
    "mm3_compressed_hybrid_top5",
    "mm4_bounded_agentic",
)
SELF_HOSTED_ENGINES = ("vllm", "sglang")
SELF_HOSTED_CONCURRENCY = (16, 32)
API_CONCURRENCY = (4, 8)
DEFAULT_PROMPTS_PER_VERTICAL = 100
DEFAULT_MATRIX_PATH = (
    "data/generated/phase4/controlled_final_simulation_100_per_vertical_matrix.jsonl"
)
DEFAULT_RAW_RESULTS = "results/raw/controlled_final_simulation_results.jsonl"
DEFAULT_MANIFEST = "results/raw/controlled_final_simulation_manifest.json"
DEFAULT_GPU_TELEMETRY = "results/raw/controlled_final_simulation_gpu_telemetry.jsonl"
DEFAULT_EVAL_REPORT = "results/processed/controlled_final_simulation_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/controlled_final_simulation_eval_summary.csv"
DEFAULT_ENGINE_COMPARISON = "results/processed/controlled_final_simulation_engine_comparison.csv"
DEFAULT_MEMORY_COMPARISON = (
    "results/processed/controlled_final_simulation_memory_mode_comparison.csv"
)
DEFAULT_CONCURRENCY_COMPARISON = (
    "results/processed/controlled_final_simulation_concurrency_comparison.csv"
)
DEFAULT_API_COMPARISON = "results/processed/controlled_final_simulation_api_track_comparison.csv"
DEFAULT_SLO_REPORT = "results/processed/controlled_final_simulation_slo_report.json"
DEFAULT_SLO_SUMMARY = "results/processed/controlled_final_simulation_slo_summary.csv"
DEFAULT_COST_REPORT = "results/processed/controlled_final_simulation_cost_report.json"
DEFAULT_ARTIFACT_SYNC_REPORT = (
    "results/processed/controlled_final_simulation_artifact_sync_report.json"
)
DEFAULT_POST_RUN_AUTOMATION_REPORT = (
    "results/processed/controlled_final_simulation_post_run_automation_report.json"
)
DEFAULT_PLOTTING_DATASET = "results/processed/controlled_final_simulation_plotting_dataset.csv"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_SGLANG_BASE_URL = "http://localhost:30000/v1"
SGLANG_STARTUP_COMMAND = (
    "python -m sglang.launch_server "
    "--model-path Qwen/Qwen2.5-7B-Instruct "
    "--served-model-name Qwen/Qwen2.5-7B-Instruct "
    "--host 0.0.0.0 "
    "--port 30000 "
    "--mem-fraction-static 0.90 "
    "--context-length 4096 "
    "--max-running-requests 32 "
    "--chunked-prefill-size 8192"
)


@dataclass(frozen=True)
class ConfigSpec:
    """One controlled final-simulation config."""

    config_id: str
    model_alias: str
    model_id: str
    backend_type: str
    engine: str
    runtime: str
    memory_mode: str
    concurrency: int


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with _repo_path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def build_parser() -> argparse.ArgumentParser:
    """Build the controlled final-simulation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--matrix-path", default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--raw-results-path", default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu-telemetry-path", default=DEFAULT_GPU_TELEMETRY)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--engine-comparison-path", default=DEFAULT_ENGINE_COMPARISON)
    parser.add_argument("--memory-comparison-path", default=DEFAULT_MEMORY_COMPARISON)
    parser.add_argument("--concurrency-comparison-path", default=DEFAULT_CONCURRENCY_COMPARISON)
    parser.add_argument("--api-comparison-path", default=DEFAULT_API_COMPARISON)
    parser.add_argument("--slo-report-path", default=DEFAULT_SLO_REPORT)
    parser.add_argument("--slo-summary-path", default=DEFAULT_SLO_SUMMARY)
    parser.add_argument("--cost-report-path", default=DEFAULT_COST_REPORT)
    parser.add_argument("--artifact-sync-report-path", default=DEFAULT_ARTIFACT_SYNC_REPORT)
    parser.add_argument(
        "--post-run-automation-report-path",
        default=DEFAULT_POST_RUN_AUTOMATION_REPORT,
    )
    parser.add_argument("--plotting-dataset-path", default=DEFAULT_PLOTTING_DATASET)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sglang-base-url", default=DEFAULT_SGLANG_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--prompt-count-per-vertical", type=int, default=100)
    parser.add_argument("--traffic-profile", default=TRAFFIC_PROFILE)
    parser.add_argument("--gpu-id", default=GPU_ID)
    parser.add_argument("--hourly-price", type=float, default=1.49)
    parser.add_argument("--backup-root", default="backups")
    parser.add_argument("--run-full", action="store_true")
    return parser


def build_config_specs() -> list[ConfigSpec]:
    """Return the frozen self-hosted and API config matrix."""

    specs: list[ConfigSpec] = []
    for engine in SELF_HOSTED_ENGINES:
        for memory_mode in MEMORY_MODES:
            for concurrency in SELF_HOSTED_CONCURRENCY:
                specs.append(
                    ConfigSpec(
                        config_id=(
                            f"self_hosted_{SELF_HOSTED_MODEL_ALIAS}_{engine}_"
                            f"{memory_mode}_c{concurrency}"
                        ),
                        model_alias=SELF_HOSTED_MODEL_ALIAS,
                        model_id=SELF_HOSTED_MODEL_ID,
                        backend_type="self_hosted_gpu",
                        engine=engine,
                        runtime=engine,
                        memory_mode=memory_mode,
                        concurrency=concurrency,
                    )
                )
    for memory_mode in MEMORY_MODES:
        for concurrency in API_CONCURRENCY:
            specs.append(
                ConfigSpec(
                    config_id=f"api_{API_MODEL_ALIAS}_api_provider_route_{memory_mode}_c{concurrency}",
                    model_alias=API_MODEL_ALIAS,
                    model_id=API_MODEL_ID,
                    backend_type="api_provider",
                    engine="api_provider_route",
                    runtime="api_provider_route",
                    memory_mode=memory_mode,
                    concurrency=concurrency,
                )
            )
    return specs


def _prompt_text(row: dict[str, Any]) -> str:
    for key in ("question", "issue", "prompt", "user_prompt"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(row, ensure_ascii=True, sort_keys=True)


def _evidence_ids(prompt: dict[str, Any], gold: dict[str, Any] | None) -> list[str]:
    for source in (prompt, gold or {}):
        for key in ("required_evidence_ids", "required_doc_ids", "required_chunk_ids"):
            value = source.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
    citations = (gold or {}).get("required_citations")
    if isinstance(citations, list):
        ids = []
        for citation in citations:
            if isinstance(citation, dict):
                ids.append(str(citation.get("doc_id") or citation.get("chunk_id") or ""))
        return [item for item in ids if item]
    return []


def _load_prompt_rows(dataset_root: str | Path, prompts_per_vertical: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = _repo_path(dataset_root)
    for vertical in VERTICALS:
        prompt_path = root / vertical / f"{vertical}_prompts_2000.jsonl"
        gold_path = root / vertical / f"{vertical}_gold_2000.jsonl"
        prompts = _read_jsonl(prompt_path)
        gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in _read_jsonl(gold_path)}
        for prompt in prompts[:prompts_per_vertical]:
            prompt_id = str(prompt.get("prompt_id") or "")
            gold = gold_by_prompt.get(prompt_id)
            rows.append(
                {
                    "vertical": vertical,
                    "prompt_id": prompt_id,
                    "prompt": _prompt_text(prompt),
                    "input_context": "",
                    "expected_evidence_ids": _evidence_ids(prompt, gold),
                    "expected_status": prompt.get("expected_status")
                    or (gold or {}).get("expected_status"),
                    "traffic_profile": TRAFFIC_PROFILE,
                }
            )
    return rows


def build_matrix_rows(
    *, dataset_root: str | Path, prompts_per_vertical: int
) -> list[dict[str, Any]]:
    """Build the 15,000-request controlled simulation matrix."""

    prompt_rows = _load_prompt_rows(dataset_root, prompts_per_vertical)
    rows: list[dict[str, Any]] = []
    for spec in build_config_specs():
        for prompt in prompt_rows:
            rows.append(
                {
                    **prompt,
                    "config_id": spec.config_id,
                    "model_alias": spec.model_alias,
                    "model_id": spec.model_id,
                    "backend_type": spec.backend_type,
                    "engine": spec.engine,
                    "runtime": spec.runtime,
                    "memory_mode": spec.memory_mode,
                    "concurrency": spec.concurrency,
                }
            )
    return rows


def summarize_matrix(rows: list[dict[str, Any]], prompts_per_vertical: int) -> dict[str, Any]:
    """Validate matrix cardinality and dimensions."""

    config_ids = {str(row["config_id"]) for row in rows}
    vertical_counts = Counter(str(row["vertical"]) for row in rows)
    expected_config_count = 30
    expected_prompt_rows = prompts_per_vertical * len(VERTICALS)
    expected_total = expected_config_count * expected_prompt_rows
    return {
        "row_count": len(rows),
        "expected_row_count": expected_total,
        "config_count": len(config_ids),
        "expected_config_count": expected_config_count,
        "prompt_count_per_config": expected_prompt_rows,
        "prompts_per_vertical_per_config": prompts_per_vertical,
        "self_hosted_request_count": sum(
            1 for row in rows if row["backend_type"] == "self_hosted_gpu"
        ),
        "api_request_count": sum(1 for row in rows if row["backend_type"] == "api_provider"),
        "vertical_counts": dict(sorted(vertical_counts.items())),
        "passed": len(rows) == expected_total and len(config_ids) == expected_config_count,
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _server_models(base_url: str, api_key: str, timeout_seconds: float = 5.0) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return sorted(
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def check_runtime_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Check runtime, server, API, and mm4 gates before any full run."""

    project_config = load_project_config()
    checks: dict[str, Any] = {
        "vllm_model3_7b": {
            "status": "NOT_CHECKED",
            "model_alias": SELF_HOSTED_MODEL_ALIAS,
            "model_id": SELF_HOSTED_MODEL_ID,
            "health_check_url": _models_url(args.base_url),
        },
        "sglang_model3_7b": {
            "status": "NOT_CHECKED",
            "model_alias": SELF_HOSTED_MODEL_ALIAS,
            "model_id": SELF_HOSTED_MODEL_ID,
            "startup_command": SGLANG_STARTUP_COMMAND,
            "health_check_url": _models_url(args.sglang_base_url),
        },
        "api_model6_gated": {
            "status": "NOT_CHECKED",
            "model_alias": API_MODEL_ALIAS,
            "model_id": API_MODEL_ID,
        },
        "mm4_bounded_agentic": {"status": "NOT_CHECKED"},
    }
    try:
        selection = select_runtime_for_model(
            model_alias=SELF_HOSTED_MODEL_ALIAS,
            runtime="vllm",
            hardware_type=args.gpu_id,
            backend_route="openai_compatible_vllm",
            live_run=True,
        )
        models = _server_models(args.base_url, args.api_key)
        checks["vllm_model3_7b"].update(
            {
                "runtime_selection": selection.to_dict(),
                "server_models": models,
                "status": (
                    "SMOKE_READY" if SELF_HOSTED_MODEL_ID in models else "SERVER_MODEL_NOT_READY"
                ),
                "reason": (
                    "model available on local vLLM server"
                    if SELF_HOSTED_MODEL_ID in models
                    else f"{SELF_HOSTED_MODEL_ID} not available at {args.base_url}"
                ),
            }
        )
    except (RuntimeError, ValueError, OSError, urllib.error.URLError) as exc:
        checks["vllm_model3_7b"].update({"status": "BLOCKED", "reason": str(exc)})

    try:
        selection = select_runtime_for_model(
            model_alias=SELF_HOSTED_MODEL_ALIAS,
            runtime="sglang",
            hardware_type=args.gpu_id,
            backend_route="sglang_openai_compatible",
            live_run=True,
        )
        package_available = _module_available("sglang")
        models = _server_models(args.sglang_base_url, args.api_key)
        model_available = SELF_HOSTED_MODEL_ID in models
        smoke_ready = package_available and model_available
        checks["sglang_model3_7b"].update(
            {
                "runtime_selection": selection.to_dict(),
                "runtime_registry_allows_sglang": True,
                "server_models": models,
                "python_package_available": package_available,
                "health_check": {
                    "endpoint": _models_url(args.sglang_base_url),
                    "passed": bool(models),
                    "model_available": model_available,
                },
                "status": "SMOKE_READY" if smoke_ready else "BLOCKED",
                "reason": (
                    "model available on SGLang server"
                    if smoke_ready
                    else (
                        "SGLang package is importable, but /v1/models did not list "
                        f"{SELF_HOSTED_MODEL_ID}. Start SGLang with: "
                        f"{SGLANG_STARTUP_COMMAND}"
                    )
                ),
            }
        )
    except ValueError as exc:
        checks["sglang_model3_7b"].update(
            {
                "status": "BLOCKED",
                "python_package_available": _module_available("sglang"),
                "runtime_registry_allows_sglang": False,
                "health_check": {
                    "endpoint": _models_url(args.sglang_base_url),
                    "passed": False,
                    "model_available": False,
                },
                "reason": str(exc),
            }
        )
    except (RuntimeError, OSError, urllib.error.URLError) as exc:
        checks["sglang_model3_7b"].update(
            {
                "status": "BLOCKED",
                "python_package_available": _module_available("sglang"),
                "runtime_registry_allows_sglang": True,
                "health_check": {
                    "endpoint": _models_url(args.sglang_base_url),
                    "passed": False,
                    "model_available": False,
                },
                "reason": (
                    f"SGLang /v1/models health check failed at "
                    f"{_models_url(args.sglang_base_url)}: {exc}. "
                    f"Start SGLang with: {SGLANG_STARTUP_COMMAND}"
                ),
            }
        )

    model6 = project_config.resolve_model_config(API_MODEL_ALIAS)
    environment = load_probe_environment(env_path=ROOT / args.env_file)
    hf_token = environment.get("HF_TOKEN", "")
    provider_keys = {
        "OPENROUTER_API_KEY": bool(environment.get("OPENROUTER_API_KEY")),
        "NOVITA_API_KEY": bool(environment.get("NOVITA_API_KEY")),
    }
    api_ready = bool(hf_token) and any(provider_keys.values())
    checks["api_model6_gated"].update(
        {
            "requires_hf_token": model6.requires_hf_token,
            "requires_license_acceptance": model6.requires_license_acceptance,
            "hf_token_present": bool(hf_token),
            "provider_keys_present": provider_keys,
            "status": "SMOKE_READY" if api_ready else "BLOCKED",
            "reason": (
                "HF token and provider key present"
                if api_ready
                else "HF_TOKEN and a provider API key are required for model6_gated"
            ),
        }
    )

    mm4_available = (
        _module_available("langgraph")
        and (ROOT / "scripts/phase4/run_mm4_agentic_smoke.py").exists()
        and (ROOT / "src/inference_bench/agents/langgraph_mm4.py").exists()
    )
    checks["mm4_bounded_agentic"].update(
        {
            "langgraph_available": _module_available("langgraph"),
            "script_available": (ROOT / "scripts/phase4/run_mm4_agentic_smoke.py").exists(),
            "status": "SMOKE_READY" if mm4_available else "MM4_NOT_RUNNABLE",
            "reason": (
                "bounded LangGraph mm4 runner is importable"
                if mm4_available
                else "bounded LangGraph mm4 runner dependencies are incomplete"
            ),
        }
    )
    full_ready = all(
        checks[name]["status"] == "SMOKE_READY"
        for name in (
            "vllm_model3_7b",
            "sglang_model3_7b",
            "api_model6_gated",
            "mm4_bounded_agentic",
        )
    )
    return {"checks": checks, "full_simulation_allowed": full_ready}


def build_smoke_report(gates: dict[str, Any]) -> dict[str, Any]:
    """Build a smoke report without faking unavailable tracks."""

    checks = cast(dict[str, dict[str, Any]], gates["checks"])
    tracks = {}
    for track, check_name in {
        "vllm_model3_7b": "vllm_model3_7b",
        "sglang_model3_7b": "sglang_model3_7b",
        "api_model6_gated": "api_model6_gated",
    }.items():
        check = checks[check_name]
        gate_ready = check.get("status") == "SMOKE_READY"
        tracks[track] = {
            "attempted": False,
            "status": "SMOKE_READY" if gate_ready else "SMOKE_BLOCKED",
            "reason": check.get("reason"),
            "gate_status": check.get("status"),
        }
    all_ready = bool(gates["full_simulation_allowed"])
    return {
        "status": "SMOKE_READY" if all_ready else "SMOKE_BLOCKED",
        "reason": (
            "All required smoke gates are ready; full simulation still requires explicit "
            "--run-full."
            if all_ready
            else "One or more required smoke gates are not ready."
        ),
        "tracks": tracks,
        "full_simulation_allowed": all_ready,
        "serving_commands": {
            "sglang_model3_7b": SGLANG_STARTUP_COMMAND,
        },
    }


def _artifact_specs(args: argparse.Namespace) -> list[Any]:
    return build_artifact_specs(
        raw_jsonl=args.raw_results_path,
        manifest=args.manifest_path,
        telemetry=args.gpu_telemetry_path,
        processed_reports=[
            args.eval_report_path,
            args.eval_summary_path,
            args.engine_comparison_path,
            args.memory_comparison_path,
            args.concurrency_comparison_path,
            args.api_comparison_path,
            args.slo_report_path,
            args.slo_summary_path,
            args.cost_report_path,
            args.artifact_sync_report_path,
            args.post_run_automation_report_path,
            args.plotting_dataset_path,
        ],
    )


def _write_manifest(
    args: argparse.Namespace, *, status: str, matrix_summary: dict[str, Any]
) -> None:
    now = utc_now()
    manifest = RunManifest(
        run_id=RUN_ID,
        timestamp_utc=now,
        backend="mixed",
        model_alias="model3_7b+model6_gated",
        model_id=f"{SELF_HOSTED_MODEL_ID}+{API_MODEL_ID}",
        memory_mode="mm0-mm4",
        split="controlled_final_simulation_100_per_vertical",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=args.matrix_path,
        output_path=args.raw_results_path,
        max_records=int(matrix_summary["row_count"]),
        git_commit=current_git_commit(ROOT),
        command=" ".join(sys.argv),
        status=status,
        start_time=now,
        end_time=now,
        error_count=0,
        telemetry_path=args.gpu_telemetry_path,
        config_id=RUN_ID,
        vertical="all",
        runtime="mixed",
        engine="vllm+sglang+api_provider_route",
        backend_type="mixed",
        hardware=args.gpu_id,
        provider="self_hosted+api_provider",
        concurrency=max((*SELF_HOSTED_CONCURRENCY, *API_CONCURRENCY)),
        traffic_profile=args.traffic_profile,
        prompt_count=int(matrix_summary["row_count"]),
        dataset_workload_hash=hash_existing_paths([_repo_path(args.matrix_path)]),
        config_hash=hash_existing_paths(
            ["configs/runtime_engines.yaml", "configs/slo_targets.yaml"]
        ),
        started_at=now,
        updated_at=now,
        completed_at=now,
        completed_count=0,
        failed_count=0,
        expected_count=int(matrix_summary["row_count"]),
        artifact_paths={
            "matrix": args.matrix_path,
            "raw_results": args.raw_results_path,
            "manifest": args.manifest_path,
            "gpu_telemetry": args.gpu_telemetry_path,
            "eval_report": args.eval_report_path,
            "slo_report": args.slo_report_path,
            "post_run_automation_report": args.post_run_automation_report_path,
            "plotting_dataset": args.plotting_dataset_path,
        },
        baseline_or_optimized="baseline",
        optimization_flags=(),
        dataset_version="controlled_2000",
    )
    write_run_manifest(manifest, _repo_path(args.manifest_path))


def _blocked_rows(configs: list[ConfigSpec], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "config_id": config.config_id,
            "model_alias": config.model_alias,
            "engine": config.engine,
            "memory_mode": config.memory_mode,
            "concurrency": config.concurrency,
            "status": "NOT_RUN",
            "reason": reason,
            "requests_attempted": 0,
            "requests_completed": 0,
        }
        for config in configs
    ]


def write_blocked_reports(
    args: argparse.Namespace,
    *,
    matrix_summary: dict[str, Any],
    gates: dict[str, Any],
    smoke_report: dict[str, Any],
) -> dict[str, Any]:
    """Write all required reports for a safety-gated blocked run."""

    reason = "Safety gates blocked full controlled simulation before provider/model requests."
    configs = build_config_specs()
    blocked_rows = _blocked_rows(configs, reason)
    _write_jsonl(args.raw_results_path, [])
    _repo_path(args.gpu_telemetry_path).parent.mkdir(parents=True, exist_ok=True)
    _repo_path(args.gpu_telemetry_path).write_text("", encoding="utf-8")
    _write_manifest(args, status="failed", matrix_summary=matrix_summary)
    eval_report = {
        "run_id": RUN_ID,
        "status": "CONTROLLED_FINAL_SIMULATION_BLOCKED_BY_SAFETY_GATES",
        "matrix_summary": matrix_summary,
        "smoke_report": smoke_report,
        "gate_report": gates,
        "serving_commands": {
            "sglang_model3_7b": SGLANG_STARTUP_COMMAND,
        },
        "configs_completed": 0,
        "configs_failed": len(configs),
        "total_requests_attempted": 0,
        "total_requests_planned": matrix_summary["row_count"],
        "vllm_ran": False,
        "sglang_ran": False,
        "api_route_ran": False,
        "mm4_ran": False,
        "final_10000_prompt_experiment_allowed": False,
    }
    _write_json(args.eval_report_path, eval_report)
    _write_csv(args.eval_summary_path, blocked_rows)
    _write_csv(args.engine_comparison_path, blocked_rows)
    _write_csv(args.memory_comparison_path, blocked_rows)
    _write_csv(args.concurrency_comparison_path, blocked_rows)
    _write_csv(
        args.api_comparison_path,
        [row for row in blocked_rows if row["model_alias"] == API_MODEL_ALIAS],
    )
    slo_rows = [
        {
            **row,
            "passed_slos": 0,
            "failed_slos": 0,
            "failed_metric_family": "not_evaluated",
            "bottleneck_category": "safety_gate",
            "recommended_optimization_candidates": "",
        }
        for row in blocked_rows
    ]
    slo_report = {
        "run_id": RUN_ID,
        "status": "SLO_COMPARISON_NOT_RUN_SAFETY_GATED",
        "deployability_verdict": "NOT_DEPLOYABLE_SIMULATION_BLOCKED",
        "benchmark_execution_verdict": "NOT_READY",
        "optimization_needed_verdict": "NOT_EVALUATED",
        "config_slo_results": slo_rows,
        "gate_report": gates,
    }
    _write_json(args.slo_report_path, slo_report)
    _write_csv(args.slo_summary_path, slo_rows)
    cost_report = {
        "run_id": RUN_ID,
        "status": "COST_NOT_MEASURED_SAFETY_GATED",
        "self_hosted_gpu_hourly_price_usd": args.hourly_price,
        "api_cost_usd": 0.0,
        "gpu_cost_usd": 0.0,
        "cost_notes": [
            "No provider API call was made.",
            "No GPU inference request was made by this runner.",
        ],
    }
    _write_json(args.cost_report_path, cost_report)
    automation_report = build_post_run_automation_report(
        PostRunAutomationInputs(
            run_id=RUN_ID,
            manifest_path=args.manifest_path,
            eval_summary_path=args.eval_summary_path,
            latency_summary_path=None,
            telemetry_summary_path=None,
            cost_report_path=args.cost_report_path,
            comparison_paths=(
                args.engine_comparison_path,
                args.memory_comparison_path,
                args.concurrency_comparison_path,
                args.api_comparison_path,
                args.slo_summary_path,
            ),
        )
    )
    write_post_run_automation_artifacts(
        report=automation_report,
        report_path=args.post_run_automation_report_path,
        plotting_dataset_path=args.plotting_dataset_path,
    )
    config = ArtifactSyncConfig(run_id=RUN_ID, backup_root=args.backup_root)
    sync = sync_artifacts(
        specs=_artifact_specs(args), config=config, event="blocked_preflight", repo_root=ROOT
    )
    verification = verify_backup(specs=_artifact_specs(args), config=config, repo_root=ROOT)
    artifact_report = {
        "run_id": RUN_ID,
        "artifact_sync_enabled": True,
        "sync": sync,
        "verification": verification,
        "success": bool(verification.get("passed")),
    }
    _write_json(args.artifact_sync_report_path, artifact_report)
    return eval_report


def run_controlled_final_simulation(args: argparse.Namespace) -> dict[str, Any]:
    """Build matrix, run safety gates, and write reports."""

    if args.prompt_count_per_vertical != DEFAULT_PROMPTS_PER_VERTICAL:
        msg = "controlled final simulation is locked to 100 prompts per vertical"
        raise ValueError(msg)
    if args.gpu_id != GPU_ID:
        msg = "controlled final simulation is locked to A100 SXM 80GB"
        raise ValueError(msg)
    started = time.perf_counter()
    matrix_rows = build_matrix_rows(
        dataset_root=args.dataset_root,
        prompts_per_vertical=args.prompt_count_per_vertical,
    )
    _write_jsonl(args.matrix_path, matrix_rows)
    matrix_summary = summarize_matrix(matrix_rows, args.prompt_count_per_vertical)
    gates = check_runtime_gate(args)
    smoke_report = build_smoke_report(gates)
    if not matrix_summary["passed"]:
        gates["full_simulation_allowed"] = False
        smoke_report["status"] = "SMOKE_BLOCKED"
        smoke_report["reason"] = "Matrix cardinality validation failed."
    if args.run_full and gates["full_simulation_allowed"]:
        msg = (
            "Full 15,000-request execution is intentionally not implemented in this "
            "safety-gate runner until all smoke tracks are available."
        )
        raise RuntimeError(msg)
    report = write_blocked_reports(
        args,
        matrix_summary=matrix_summary,
        gates=gates,
        smoke_report=smoke_report,
    )
    report["wall_seconds"] = time.perf_counter() - started
    return report


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    try:
        report = run_controlled_final_simulation(args)
    except Exception as exc:  # noqa: BLE001
        print(f"controlled final simulation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
