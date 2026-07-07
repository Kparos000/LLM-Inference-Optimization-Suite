"""Run live A100 SXM Qwen2.5-3B vLLM calibration."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import (  # noqa: E402
    build_summary_rows,
    load_gold_records,
    result_row_to_generated_answer,
)
from run_b6r6_research_ai_quality_recovery import (  # noqa: E402
    _failure_row,
    _run_default_item,
    _run_research_ai_item,
    _workload_item,
)
from run_openai_compatible_smoke import DEFAULT_API_KEY, check_server_readiness  # noqa: E402
from run_remote_vllm_smoke import latency_summary_rows, sanitized_command, write_json  # noqa: E402

from inference_bench.artifact_sync import (  # noqa: E402
    ArtifactSyncConfig,
    build_artifact_specs,
    sync_artifacts,
    verify_backup,
)
from inference_bench.b1_quality import build_per_vertical_quality  # noqa: E402
from inference_bench.b6r6_research_ai_recovery import (  # noqa: E402
    STRATEGY_D_ANSWER_SKELETON,
)
from inference_bench.b7_controlled_baseline import (  # noqa: E402
    B7_REQUEST_ARRIVAL_MODE,
    B7_RESEARCH_AI_STRATEGY,
    build_b7_load_and_cache_report,
    build_b7_runtime_projection,
    classify_b7_quality_gate,
    required_labels_for_runner_row,
)
from inference_bench.checkpoint_resume import (  # noqa: E402
    build_resume_plan,
    checkpoint_from_rows,
    read_jsonl_rows,
    write_checkpoint,
    write_jsonl_rows,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_alignment_repair import (  # noqa: E402
    build_b6_context_aligned_runner_input,
)
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.gpu_telemetry import (  # noqa: E402
    GpuTelemetrySample,
    collect_gpu_sample,
    summarize_gpu_telemetry,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    hash_existing_paths,
    utc_now,
    write_run_manifest,
)
from inference_bench.runtime_registry import select_runtime_for_model  # noqa: E402
from inference_bench.schema import WorkloadItem  # noqa: E402

DEFAULT_SOURCE_WORKLOAD = (
    "data/workloads/controlled_2000/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
)
DEFAULT_SOURCE_OF_TRUTH_MANIFEST = (
    "data/generated/context_engineering/retrieval_source_of_truth_manifest.json"
)
DEFAULT_DATASET_ROOT = "data/scaleup_2000_full"
DEFAULT_CONTEXT_ROOT = "data/generated/context_engineering"
DEFAULT_RUN_ID_PREFIX = "a100_sxm_model2_3b_mm2_c1"
DEFAULT_PROMPT_COUNT = 200
DEFAULT_RUN_ID = f"{DEFAULT_RUN_ID_PREFIX}_{DEFAULT_PROMPT_COUNT}"
DEFAULT_RUNNER_INPUT = f"data/generated/phase4/{DEFAULT_RUN_ID}_runner_input.jsonl"
DEFAULT_RAW_OUTPUT = f"results/raw/{DEFAULT_RUN_ID}_results.jsonl"
DEFAULT_MANIFEST = f"results/raw/{DEFAULT_RUN_ID}_manifest.json"
DEFAULT_GPU_TELEMETRY = f"results/raw/{DEFAULT_RUN_ID}_gpu_telemetry.jsonl"
DEFAULT_EVAL_REPORT = f"results/processed/{DEFAULT_RUN_ID}_eval_report.json"
DEFAULT_EVAL_SUMMARY = f"results/processed/{DEFAULT_RUN_ID}_eval_summary.csv"
DEFAULT_RUNTIME_PROJECTION = f"results/processed/{DEFAULT_RUN_ID}_runtime_projection.json"
DEFAULT_ARTIFACT_SYNC_REPORT = f"results/processed/{DEFAULT_RUN_ID}_artifact_sync_report.json"
DEFAULT_READINESS_REPORT = f"results/processed/{DEFAULT_RUN_ID}_readiness_report.json"
DEFAULT_CHECKPOINT = f"results/raw/{DEFAULT_RUN_ID}_checkpoint.json"
DEFAULT_FAILED_ROWS = f"results/raw/{DEFAULT_RUN_ID}_failed_rows.jsonl"
DEFAULT_CONTEXT_PREFLIGHT_REPORT = (
    f"results/processed/{DEFAULT_RUN_ID}_context_preflight_report.json"
)
DEFAULT_CONTEXT_PREFLIGHT_SUMMARY = (
    f"results/processed/{DEFAULT_RUN_ID}_context_preflight_summary.csv"
)
DEFAULT_CONTEXT_PREFLIGHT_EXAMPLES = (
    f"results/processed/{DEFAULT_RUN_ID}_context_preflight_examples.jsonl"
)
A100_MODEL_ALIAS = "model2_3b"
A100_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
A100_MEMORY_MODE = "mm2_hybrid_top5"
A100_ENGINE = "vllm"
A100_RUNTIME = "vllm"
A100_BACKEND_TYPE = "self_hosted_gpu"
A100_PROVIDER = "runpod"
A100_HARDWARE = "a100_sxm_80gb"
A100_TRAFFIC_PROFILE = "online_low_latency"
A100_HOURLY_PRICE = 1.49
A100_TEMPERATURE = 0.0
A100_ALLOWED_PROMPT_COUNTS = {100, 200}
A100_VERTICALS = ("airline", "healthcare_admin", "retail", "finance", "research_ai")
A100_VLLM_COMMAND = (
    "python -m vllm.entrypoints.openai.api_server "
    "--model Qwen/Qwen2.5-3B-Instruct "
    "--host 0.0.0.0 --port 8000 "
    "--gpu-memory-utilization 0.90 "
    "--max-model-len 4096 "
    "--max-num-seqs 32 "
    "--max-num-batched-tokens 8192"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the A100 SXM calibration CLI parser."""

    parser = argparse.ArgumentParser(description="Run live A100 SXM vLLM calibration.")
    parser.add_argument("--source-workload", default=DEFAULT_SOURCE_WORKLOAD)
    parser.add_argument("--source-of-truth-manifest", default=DEFAULT_SOURCE_OF_TRUTH_MANIFEST)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--context-root", default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument(
        "--prompt-count", type=int, choices=sorted(A100_ALLOWED_PROMPT_COUNTS), default=200
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--model-alias", default=A100_MODEL_ALIAS)
    parser.add_argument("--model-id", default=A100_MODEL_ID)
    parser.add_argument("--memory-mode", default=A100_MEMORY_MODE)
    parser.add_argument("--engine", default=A100_ENGINE)
    parser.add_argument("--gpu-id", default=A100_HARDWARE)
    parser.add_argument("--hourly-price", type=float, default=A100_HOURLY_PRICE)
    parser.add_argument("--traffic-profile", default=A100_TRAFFIC_PROFILE)
    parser.add_argument("--temperature", type=float, default=A100_TEMPERATURE)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--output-path", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu-telemetry-path", default=DEFAULT_GPU_TELEMETRY)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--runtime-projection-path", default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--artifact-sync-report-path", default=DEFAULT_ARTIFACT_SYNC_REPORT)
    parser.add_argument("--readiness-report-path", default=DEFAULT_READINESS_REPORT)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--failed-rows-path", default=DEFAULT_FAILED_ROWS)
    parser.add_argument("--context-preflight-report", default=DEFAULT_CONTEXT_PREFLIGHT_REPORT)
    parser.add_argument("--context-preflight-summary", default=DEFAULT_CONTEXT_PREFLIGHT_SUMMARY)
    parser.add_argument("--context-preflight-examples", default=DEFAULT_CONTEXT_PREFLIGHT_EXAMPLES)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--backup-root", default="backups")
    parser.add_argument("--sync-every-n-requests", type=int, default=50)
    parser.add_argument("--telemetry-ssh-host", default="")
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--artifact-sync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-telemetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-input-rebuild",
        action="store_true",
        help="Use an existing A100 runner input if present.",
    )
    return parser


def _run_id(args: argparse.Namespace) -> str:
    suffix = "mm2" if args.memory_mode == A100_MEMORY_MODE else args.memory_mode
    return f"a100_sxm_{args.model_alias}_{suffix}_c{args.concurrency}_{args.prompt_count}"


def _default_path(args: argparse.Namespace, template_path: str) -> str:
    default = DEFAULT_RUN_ID
    run_id = _run_id(args)
    return template_path.replace(default, run_id)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate safety-critical options and update default paths for 100-prompt runs."""

    if args.prompt_count not in A100_ALLOWED_PROMPT_COUNTS:
        msg = "--prompt-count must be 100 or 200"
        raise ValueError(msg)
    if args.concurrency != 1:
        msg = "A100 SXM calibration is locked to --concurrency 1 for this run"
        raise ValueError(msg)
    if args.model_alias != A100_MODEL_ALIAS or args.model_id != A100_MODEL_ID:
        msg = "A100 SXM calibration is locked to model2_3b / Qwen/Qwen2.5-3B-Instruct"
        raise ValueError(msg)
    if args.memory_mode != A100_MEMORY_MODE:
        msg = "A100 SXM calibration is locked to --memory-mode mm2_hybrid_top5"
        raise ValueError(msg)
    if args.engine.lower() != A100_ENGINE:
        msg = "A100 SXM calibration is locked to --engine vllm"
        raise ValueError(msg)
    args.engine = A100_ENGINE
    if args.gpu_id != A100_HARDWARE:
        msg = "A100 SXM calibration is locked to --gpu-id a100_sxm_80gb"
        raise ValueError(msg)
    if args.traffic_profile != A100_TRAFFIC_PROFILE:
        msg = "A100 SXM calibration is locked to --traffic-profile online_low_latency"
        raise ValueError(msg)
    if args.temperature != 0:
        msg = "A100 SXM calibration is locked to --temperature 0"
        raise ValueError(msg)
    if not (args.artifact_sync and args.checkpoint_resume and args.manifest and args.gpu_telemetry):
        msg = "artifact sync, checkpoint/resume, manifest, and GPU telemetry must remain enabled"
        raise ValueError(msg)
    default_paths = {
        "runner_input_path": DEFAULT_RUNNER_INPUT,
        "output_path": DEFAULT_RAW_OUTPUT,
        "manifest_path": DEFAULT_MANIFEST,
        "gpu_telemetry_path": DEFAULT_GPU_TELEMETRY,
        "eval_report_path": DEFAULT_EVAL_REPORT,
        "eval_summary_path": DEFAULT_EVAL_SUMMARY,
        "runtime_projection_path": DEFAULT_RUNTIME_PROJECTION,
        "artifact_sync_report_path": DEFAULT_ARTIFACT_SYNC_REPORT,
        "readiness_report_path": DEFAULT_READINESS_REPORT,
        "checkpoint_path": DEFAULT_CHECKPOINT,
        "failed_rows_path": DEFAULT_FAILED_ROWS,
        "context_preflight_report": DEFAULT_CONTEXT_PREFLIGHT_REPORT,
        "context_preflight_summary": DEFAULT_CONTEXT_PREFLIGHT_SUMMARY,
        "context_preflight_examples": DEFAULT_CONTEXT_PREFLIGHT_EXAMPLES,
    }
    for attr, default_value in default_paths.items():
        if getattr(args, attr) == default_value:
            setattr(args, attr, _default_path(args, default_value))
    return args


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_rows(_repo_path(path))


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    write_jsonl_rows(_repo_path(path), rows, append=append)


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        msg = "at least one row is required"
        raise ValueError(msg)
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(_repo_path(path), payload)


def _update_a100_runner_input_metadata(
    path: str | Path,
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    updated: list[dict[str, Any]] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "a100_run_id": _run_id(args),
                "a100_config_id": _run_id(args),
                "a100_model_alias": args.model_alias,
                "a100_model_id": args.model_id,
                "a100_gpu_id": args.gpu_id,
                "a100_hourly_price": str(args.hourly_price),
                "a100_traffic_profile": args.traffic_profile,
                "b7_request_arrival_mode": B7_REQUEST_ARRIVAL_MODE,
                "a100_concurrency": str(args.concurrency),
            }
        )
        row["metadata"] = metadata
        row["workload_name"] = f"a100_sxm_{args.prompt_count}_mm2_hybrid_top5_b6r6_repairs"
        updated.append(row)
    _write_jsonl(path, updated, append=False)
    return updated


def build_or_load_runner_input(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the frozen A100 input unless an existing one is explicitly reused."""

    output_path = _repo_path(args.runner_input_path)
    if not args.skip_input_rebuild or not output_path.exists():
        build_b6_context_aligned_runner_input(
            source_workload_path=_repo_path(args.source_workload),
            source_of_truth_manifest_path=_repo_path(args.source_of_truth_manifest),
            dataset_root=_repo_path(args.dataset_root),
            context_root=_repo_path(args.context_root),
            output_path=output_path,
            report_path=_repo_path(args.context_preflight_report),
            summary_path=_repo_path(args.context_preflight_summary),
            examples_path=_repo_path(args.context_preflight_examples),
            prompts_per_vertical=args.prompt_count // len(A100_VERTICALS),
        )
    rows = _update_a100_runner_input_metadata(args.runner_input_path, args=args)
    context_report = json.loads(
        _repo_path(args.context_preflight_report).read_text(encoding="utf-8")
    )
    context_report["block"] = "A100_SXM_CALIBRATION"
    context_report["status"] = (
        "PREFLIGHT_PASSED_A100_SXM_CONTEXT_ALIGNMENT"
        if bool(context_report.get("inference_allowed"))
        else "PREFLIGHT_BLOCKED_A100_SXM_CONTEXT_ALIGNMENT"
    )
    context_report["runner_input_path"] = args.runner_input_path
    context_report["row_count"] = len(rows)
    _write_json(args.context_preflight_report, context_report)
    return rows, cast(dict[str, Any], context_report)


def _build_manifest(
    *,
    args: argparse.Namespace,
    status: str,
    started_at: str,
    updated_at: str,
    completed_at: str | None,
    completed_count: int,
    failed_count: int,
    expected_count: int,
    command: str,
) -> RunManifest:
    artifact_paths = {
        "runner_input": args.runner_input_path,
        "raw_results": args.output_path,
        "manifest": args.manifest_path,
        "gpu_telemetry": args.gpu_telemetry_path,
        "eval_report": args.eval_report_path,
        "eval_summary": args.eval_summary_path,
        "runtime_projection": args.runtime_projection_path,
        "artifact_sync_report": args.artifact_sync_report_path,
        "readiness_report": args.readiness_report_path,
        "checkpoint": args.checkpoint_path,
        "failed_rows": args.failed_rows_path,
    }
    return RunManifest(
        run_id=_run_id(args),
        timestamp_utc=updated_at,
        backend=A100_RUNTIME,
        model_alias=args.model_alias,
        model_id=args.model_id,
        memory_mode=args.memory_mode,
        split=f"a100_sxm_calibration_{args.prompt_count}",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=args.runner_input_path,
        output_path=args.output_path,
        max_records=expected_count,
        git_commit=current_git_commit(ROOT),
        command=command,
        status=status,
        start_time=started_at,
        end_time=completed_at,
        error_count=failed_count,
        telemetry_path=args.gpu_telemetry_path,
        config_id=_run_id(args),
        vertical="all",
        runtime=A100_RUNTIME,
        engine=A100_ENGINE,
        backend_type=A100_BACKEND_TYPE,
        hardware=args.gpu_id,
        provider=A100_PROVIDER,
        concurrency=args.concurrency,
        traffic_profile=args.traffic_profile,
        prompt_count=expected_count,
        dataset_workload_hash=hash_existing_paths([_repo_path(args.runner_input_path)]),
        config_hash=hash_existing_paths(
            [
                "configs/models.yaml",
                "configs/runtime_engines.yaml",
                "configs/load_profiles.yaml",
            ]
        ),
        started_at=started_at,
        updated_at=updated_at,
        completed_at=completed_at,
        completed_count=completed_count,
        failed_count=failed_count,
        expected_count=expected_count,
        artifact_paths=artifact_paths,
    )


def _write_manifest(
    *,
    args: argparse.Namespace,
    status: str,
    started_at: str,
    completed_at: str | None,
    rows: list[dict[str, Any]],
    expected_count: int,
    command: str,
) -> None:
    completed_count = sum(bool(row.get("success")) for row in rows)
    failed_count = len(rows) - completed_count
    manifest = _build_manifest(
        args=args,
        status=status,
        started_at=started_at,
        updated_at=utc_now(),
        completed_at=completed_at,
        completed_count=completed_count,
        failed_count=failed_count,
        expected_count=expected_count,
        command=command,
    )
    write_run_manifest(manifest, _repo_path(args.manifest_path))


def _artifact_specs(args: argparse.Namespace) -> list[Any]:
    return build_artifact_specs(
        raw_jsonl=args.output_path,
        manifest=args.manifest_path,
        telemetry=args.gpu_telemetry_path,
        processed_reports=[
            args.eval_report_path,
            args.eval_summary_path,
            args.runtime_projection_path,
            args.readiness_report_path,
            args.artifact_sync_report_path,
        ],
    )


def _artifact_sync_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    dry_root = ROOT / ".tmp" / "a100_sxm_artifact_sync_dry_run"
    try:
        dry_root.mkdir(parents=True, exist_ok=True)
        raw = dry_root / "raw.jsonl"
        manifest = dry_root / "manifest.json"
        telemetry = dry_root / "telemetry.jsonl"
        raw.write_text('{"prompt_id":"dry_run_probe","success":true}\n', encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "run_id": "a100-sxm-artifact-sync-dry-run",
                    "status": "completed",
                    "expected_count": 1,
                    "completed_count": 1,
                    "failed_count": 0,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        telemetry.write_text('{"sample":"dry_run"}\n', encoding="utf-8")
        config = ArtifactSyncConfig(
            run_id="a100-sxm-artifact-sync-dry-run",
            backup_root=args.backup_root,
            incremental_every_n_requests=args.sync_every_n_requests,
        )
        specs = build_artifact_specs(raw_jsonl=raw, manifest=manifest, telemetry=telemetry)
        sync = sync_artifacts(specs=specs, config=config, event="dry_run", repo_root=ROOT)
        verification = verify_backup(specs=specs, config=config, repo_root=ROOT)
        return {
            "sync": sync,
            "verification": verification,
            "success": bool(sync.get("success")) and bool(verification.get("passed")),
        }
    finally:
        shutil.rmtree(dry_root, ignore_errors=True)
        parent = dry_root.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def _sync_current_artifacts(
    *,
    args: argparse.Namespace,
    event: str,
) -> dict[str, Any]:
    config = ArtifactSyncConfig(
        run_id=_run_id(args),
        backup_root=args.backup_root,
        incremental_every_n_requests=args.sync_every_n_requests,
    )
    return sync_artifacts(specs=_artifact_specs(args), config=config, event=event, repo_root=ROOT)


def _verify_current_backup(args: argparse.Namespace) -> dict[str, Any]:
    config = ArtifactSyncConfig(
        run_id=_run_id(args),
        backup_root=args.backup_root,
        incremental_every_n_requests=args.sync_every_n_requests,
    )
    return verify_backup(specs=_artifact_specs(args), config=config, repo_root=ROOT)


def _telemetry_loop(
    *,
    path: Path,
    ssh_host: str | None,
    interval_seconds: float,
    stop_event: threading.Event,
    errors: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while not stop_event.is_set():
        try:
            samples = collect_gpu_sample(ssh_host=ssh_host)
            with path.open("a", encoding="utf-8", newline="\n") as file:
                for sample in samples:
                    file.write(
                        json.dumps(sample.to_dict(), ensure_ascii=True, sort_keys=True) + "\n"
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")
        stop_event.wait(interval_seconds)


def _read_gpu_samples(path: str | Path) -> list[GpuTelemetrySample]:
    samples: list[GpuTelemetrySample] = []
    for row in _read_jsonl(path):
        if "gpu_name" not in row:
            continue
        samples.append(
            GpuTelemetrySample(
                timestamp=str(row["timestamp"]),
                gpu_name=str(row["gpu_name"]),
                utilization_gpu_percent=float(row["utilization_gpu_percent"]),
                memory_used_mb=float(row["memory_used_mb"]),
                memory_total_mb=float(row["memory_total_mb"]),
                power_draw_w=float(row["power_draw_w"]),
                temperature_c=float(row["temperature_c"]),
                process_info=str(row.get("process_info") or ""),
            )
        )
    return samples


def preflight_a100_runner_rows(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    model_id: str,
    runtime_selection: dict[str, Any],
    artifact_sync_dry_run_passed: bool,
) -> dict[str, Any]:
    """Validate the A100 SXM runner input and controls before inference."""

    prompt_ids = [str(row.get("prompt_id") or "") for row in rows]
    duplicate_prompt_ids = sorted(
        prompt_id for prompt_id, count in Counter(prompt_ids).items() if count > 1
    )
    vertical_counts = Counter(
        str((row.get("metadata") or {}).get("vertical") or row.get("vertical") or "")
        for row in rows
    )
    expected_per_vertical = args.prompt_count // len(A100_VERTICALS)
    missing_required = [
        {
            "prompt_id": row.get("prompt_id"),
            "vertical": (row.get("metadata") or {}).get("vertical"),
            "reason": "no_required_evidence_label_in_e1_e5",
        }
        for row in rows
        if not required_labels_for_runner_row(row)
    ]
    checks = {
        "row_count": len(rows) == args.prompt_count,
        "prompt_count_allowed": args.prompt_count in A100_ALLOWED_PROMPT_COUNTS,
        "per_vertical_split": all(
            vertical_counts.get(vertical, 0) == expected_per_vertical for vertical in A100_VERTICALS
        ),
        "unique_prompt_ids": not duplicate_prompt_ids and all(prompt_ids),
        "required_evidence_present_in_e1_e5": not missing_required,
        "model_alias_resolves": args.model_alias == A100_MODEL_ALIAS and model_id == A100_MODEL_ID,
        "runtime_registry_allows_vllm_a100_sxm": (
            runtime_selection.get("model_alias") == A100_MODEL_ALIAS
            and runtime_selection.get("model_id") == A100_MODEL_ID
            and runtime_selection.get("runtime") == A100_RUNTIME
            and runtime_selection.get("engine") == A100_ENGINE
            and runtime_selection.get("backend_type") == A100_BACKEND_TYPE
            and runtime_selection.get("hardware_type") == A100_HARDWARE
            and bool(runtime_selection.get("live_run_allowed"))
        ),
        "artifact_sync_dry_run_passed": artifact_sync_dry_run_passed,
        "checkpoint_resume_enabled": bool(args.checkpoint_resume),
        "manifest_enabled": bool(args.manifest),
        "gpu_telemetry_enabled": bool(args.gpu_telemetry),
        "concurrency_locked_to_one": args.concurrency == 1,
        "engine_locked_to_vllm": args.engine == A100_ENGINE,
        "memory_mode_locked_to_mm2": args.memory_mode == A100_MEMORY_MODE,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "block": "A100_SXM_CALIBRATION",
        "status": (
            "PREFLIGHT_PASSED_A100_SXM_CALIBRATION"
            if not failed_checks
            else "PREFLIGHT_BLOCKED_A100_SXM_CALIBRATION"
        ),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "row_count": len(rows),
        "expected_count": args.prompt_count,
        "prompts_per_vertical": dict(vertical_counts),
        "expected_prompts_per_vertical": expected_per_vertical,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "missing_required_evidence_rows": missing_required,
        "model_alias": args.model_alias,
        "model_id": model_id,
        "runtime_selection": runtime_selection,
        "artifact_sync_dry_run_passed": artifact_sync_dry_run_passed,
        "checkpoint_resume_enabled": bool(args.checkpoint_resume),
        "manifest_enabled": bool(args.manifest),
        "gpu_telemetry_enabled": bool(args.gpu_telemetry),
        "model_inference_triggered": False,
    }


def _mark_a100_row(
    row: dict[str, Any],
    item: WorkloadItem,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = dict(item.metadata)
    row.update(
        {
            "run_id": _run_id(args),
            "config_id": _run_id(args),
            "model_alias": args.model_alias,
            "model_id": args.model_id,
            "model_name": args.model_id,
            "backend": A100_RUNTIME,
            "runtime": A100_RUNTIME,
            "engine": A100_ENGINE,
            "backend_type": A100_BACKEND_TYPE,
            "hardware": args.gpu_id,
            "provider": A100_PROVIDER,
            "concurrency": args.concurrency,
            "traffic_profile": args.traffic_profile,
            "request_arrival_mode": B7_REQUEST_ARRIVAL_MODE,
            "selected_context_ids": metadata.get("selected_context_ids"),
            "b5_required_labels": metadata.get("b5_required_labels"),
            "a100_finance_repair_active": metadata.get("vertical") == "finance",
            "a100_research_ai_repair_active": metadata.get("vertical") == "research_ai",
            "a100_research_ai_strategy": (
                B7_RESEARCH_AI_STRATEGY if metadata.get("vertical") == "research_ai" else ""
            ),
            "artifact_sync_enabled": True,
            "checkpoint_resume_enabled": True,
            "manifest_enabled": True,
            "gpu_telemetry_enabled": True,
        }
    )
    return row


def _run_one_item(
    *,
    args: argparse.Namespace,
    item: WorkloadItem,
    gold_by_prompt: dict[str, dict[str, Any]],
    base_url: str,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if item.metadata.get("vertical") == "research_ai":
            row = _run_research_ai_item(
                item=item,
                strategy_id=STRATEGY_D_ANSWER_SKELETON,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                run_id=_run_id(args),
                optimization="a100_sxm_b6r6_research_ai_answer_skeleton_calibration",
                gold_by_prompt=gold_by_prompt,
            )
        else:
            row = _run_default_item(
                item=item,
                strategy_id=STRATEGY_D_ANSWER_SKELETON,
                gold_by_prompt=gold_by_prompt,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                run_id=_run_id(args),
                optimization="a100_sxm_b6r6_finance_research_repairs_calibration",
            )
    except Exception as exc:  # noqa: BLE001
        row = _failure_row(
            item=item,
            prompt=item.prompt,
            exc=exc,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            strategy_id=STRATEGY_D_ANSWER_SKELETON,
            run_id=_run_id(args),
            optimization="a100_sxm_b6r6_finance_research_repairs_calibration",
        )
    return _mark_a100_row(row, item, args=args)


def _evaluate_rows(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_path: str,
    wall_seconds: float,
    gpu_summary: dict[str, object],
    telemetry_errors: list[str],
    artifact_verification: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    generated = [result_row_to_generated_answer(row) for row in rows]
    evaluation_rows = evaluate_generated_answers(
        generated,
        load_gold_records("data/scaleup_2000_full"),
    )
    summary = build_summary_rows(
        results_path=output_path,
        result_rows=rows,
        evaluation_rows=evaluation_rows,
    )[0]
    latency_rows = latency_summary_rows(rows)
    latency = latency_rows[0]
    per_vertical = build_per_vertical_quality(evaluation_rows, rows, verticals=VERTICALS)
    load_and_cache = build_b7_load_and_cache_report(
        rows=rows,
        traffic_profile=args.traffic_profile,
        concurrency=args.concurrency,
        request_arrival_mode=B7_REQUEST_ARRIVAL_MODE,
    )
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    request_success_count = sum(bool(row.get("success")) for row in rows)
    request_failure_count = len(rows) - request_success_count
    request_error_types = Counter(
        str(row.get("error_message") or "unknown").split(":", 1)[0]
        for row in rows
        if not bool(row.get("success"))
    )
    request_throughput = len(rows) / wall_seconds if wall_seconds > 0 else None
    token_throughput = total_tokens / wall_seconds if wall_seconds > 0 else None
    output_token_throughput = output_tokens / wall_seconds if wall_seconds > 0 else None
    estimated_cost = (wall_seconds / 3600.0) * args.hourly_price
    summary.update(
        {
            "wall_seconds": wall_seconds,
            "completed_prompts": len(rows),
            "request_success_count": request_success_count,
            "request_failure_count": request_failure_count,
            "request_success_rate": request_success_count / len(rows) if rows else 0.0,
            "request_error_types": dict(sorted(request_error_types.items())),
            "mean_ttft_ms": latency.get("mean_ttft_ms"),
            "p50_ttft_ms": latency.get("p50_ttft_ms"),
            "p95_ttft_ms": latency.get("p95_ttft_ms"),
            "p99_ttft_ms": latency.get("p99_ttft_ms"),
            "mean_tpot_ms": latency.get("mean_tpot_ms"),
            "p50_tpot_ms": latency.get("p50_tpot_ms"),
            "p95_tpot_ms": latency.get("p95_tpot_ms"),
            "p99_tpot_ms": latency.get("p99_tpot_ms"),
            "mean_e2e_latency_ms": latency.get("mean_e2e_latency_ms"),
            "p50_e2e_latency_ms": latency.get("p50_e2e_latency_ms"),
            "p95_e2e_latency_ms": latency.get("p95_e2e_latency_ms"),
            "p99_e2e_latency_ms": latency.get("p99_e2e_latency_ms"),
            "mean_total_tokens_per_second": latency.get("mean_total_tokens_per_second"),
            "requests_per_second": request_throughput,
            "aggregate_tokens_per_second": token_throughput,
            "aggregate_output_tokens_per_second": output_token_throughput,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "gpu_telemetry_sample_count": gpu_summary.get("sample_count"),
            "gpu_telemetry_error_count": len(telemetry_errors),
            "artifact_sync_verified": artifact_verification.get("passed"),
            "hourly_price_usd": args.hourly_price,
            "estimated_cost_usd": estimated_cost,
        }
    )
    report = {
        "block": "A100_SXM_CALIBRATION",
        "experiment": _run_id(args),
        "model_alias": args.model_alias,
        "model_id": args.model_id,
        "runtime": A100_RUNTIME,
        "engine": A100_ENGINE,
        "hardware": args.gpu_id,
        "provider": A100_PROVIDER,
        "memory_mode": args.memory_mode,
        "traffic_profile": args.traffic_profile,
        "request_arrival_mode": B7_REQUEST_ARRIVAL_MODE,
        "concurrency": args.concurrency,
        "prompt_count": args.prompt_count,
        "temperature": args.temperature,
        "hourly_price_usd": args.hourly_price,
        "estimated_cost_usd": estimated_cost,
        "summary": summary,
        "per_vertical_quality": per_vertical,
        "latency_summary": latency_rows,
        "load_profile_report": load_and_cache["load_profile"],
        "cache_readiness": load_and_cache["cache_readiness"],
        "gpu_telemetry_summary": gpu_summary,
        "gpu_telemetry_errors": telemetry_errors,
        "artifact_backup_verification": artifact_verification,
        "evaluation_rows": evaluation_rows,
        "evaluator_modified": False,
        "gold_data_modified": False,
        "promoted_retrieval_modified": False,
        "runpod_readiness_claimed": False,
        "a100_1000_prompt_baseline_allowed": False,
    }
    return report, per_vertical, summary


def _project_wall_seconds(rows: list[dict[str, Any]], measured_wall_seconds: float) -> float:
    row_wall_seconds = sum(float(row.get("end_to_end_latency_ms") or 0.0) for row in rows) / 1000.0
    return max(measured_wall_seconds, row_wall_seconds)


def _server_required_message(base_url: str, reason: str) -> str:
    return (
        f"A100 SXM calibration requires a local vLLM OpenAI-compatible server at {base_url}. "
        f"{reason}\nStart it with:\n{A100_VLLM_COMMAND}"
    )


def _baseline_allowed(gate: dict[str, Any]) -> bool:
    return bool(gate.get("passed"))


def _a100_gate(gate: dict[str, Any]) -> dict[str, Any]:
    """Translate shared B7 quality-gate labels into A100 calibration labels."""

    translated = dict(gate)
    status_map = {
        "B7_CONTROLLED_1000_BASELINE_READY": "A100_SXM_200_PROMPT_CALIBRATION_READY",
        "B7_CONTROLLED_1000_RUN_SAFETY_BLOCKED": "A100_SXM_200_PROMPT_RUN_SAFETY_BLOCKED",
        "B7_CONTROLLED_1000_BASELINE_BLOCKED": "A100_SXM_200_PROMPT_CALIBRATION_BLOCKED",
    }
    translated["status"] = status_map.get(str(gate.get("status")), str(gate.get("status")))
    return translated


def run_a100_calibration(args: argparse.Namespace) -> dict[str, Any]:
    """Run A100 SXM preflight, optional inference, evaluation, and reporting."""

    command = sanitized_command(sys.argv)
    started_at = utc_now()
    runner_rows, context_report = build_or_load_runner_input(args)
    project_config = load_project_config()
    model = project_config.resolve_model_config(args.model_alias)
    runtime_selection = select_runtime_for_model(
        model_alias=args.model_alias,
        runtime=A100_RUNTIME,
        hardware_type=args.gpu_id,
        backend_route="openai_compatible_vllm",
        live_run=True,
    )
    dry_sync = _artifact_sync_dry_run(args)
    preflight = preflight_a100_runner_rows(
        runner_rows,
        model_id=model.model_id,
        args=args,
        runtime_selection=cast(dict[str, Any], runtime_selection.to_dict()),
        artifact_sync_dry_run_passed=bool(dry_sync["success"]),
    )
    if not bool(context_report.get("inference_allowed")):
        preflight["passed"] = False
        preflight["failed_checks"] = sorted(
            set(cast(list[str], preflight["failed_checks"]) + ["context_alignment"])
        )
        preflight["status"] = "PREFLIGHT_BLOCKED_A100_SXM_CALIBRATION"
    _write_manifest(
        args=args,
        status="initialized",
        started_at=started_at,
        completed_at=None,
        rows=[],
        expected_count=len(runner_rows),
        command=command,
    )
    initial_sync = _sync_current_artifacts(args=args, event="run_start")
    if args.dry_run or not bool(preflight["passed"]):
        readiness = {
            "block": "A100_SXM_CALIBRATION",
            "status": preflight["status"],
            "preflight": preflight,
            "context_alignment_preflight": context_report,
            "artifact_sync_dry_run": dry_sync,
            "initial_artifact_sync": initial_sync,
            "inference_triggered": False,
            "a100_1000_prompt_baseline_allowed": False,
        }
        _write_json(args.readiness_report_path, readiness)
        return readiness

    try:
        server_readiness = check_server_readiness(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_id,
            timeout_seconds=args.timeout_seconds,
        )
    except RuntimeError as exc:
        raise RuntimeError(_server_required_message(args.base_url, str(exc))) from exc
    if not server_readiness.reachable or server_readiness.model_available is False:
        msg = _server_required_message(args.base_url, server_readiness.message)
        raise RuntimeError(msg)

    raw_path = _repo_path(args.output_path)
    checkpoint_path = _repo_path(args.checkpoint_path)
    existing_rows = _read_jsonl(raw_path)
    resumed_from_checkpoint = checkpoint_path.exists() or bool(existing_rows)
    if not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("", encoding="utf-8")
    if not _repo_path(args.gpu_telemetry_path).exists() or not resumed_from_checkpoint:
        _repo_path(args.gpu_telemetry_path).parent.mkdir(parents=True, exist_ok=True)
        _repo_path(args.gpu_telemetry_path).write_text("", encoding="utf-8")

    resume_plan = build_resume_plan(
        run_id=_run_id(args),
        prompt_rows=runner_rows,
        checkpoint_path=checkpoint_path,
        partial_raw_jsonl_path=raw_path,
    )
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    items_by_prompt = {str(row["prompt_id"]): _workload_item(row) for row in runner_rows}
    rows = existing_rows
    completed_ids = {str(row.get("prompt_id")) for row in rows}
    stop_event = threading.Event()
    telemetry_errors: list[str] = []
    telemetry_thread = threading.Thread(
        target=_telemetry_loop,
        kwargs={
            "path": _repo_path(args.gpu_telemetry_path),
            "ssh_host": args.telemetry_ssh_host or None,
            "interval_seconds": args.telemetry_interval_seconds,
            "stop_event": stop_event,
            "errors": telemetry_errors,
        },
        daemon=True,
    )
    _write_manifest(
        args=args,
        status="running",
        started_at=started_at,
        completed_at=None,
        rows=rows,
        expected_count=len(runner_rows),
        command=command,
    )
    telemetry_thread.start()
    run_started = time.perf_counter()
    sync_events = [initial_sync]
    try:
        for prompt_id in resume_plan.pending_prompt_ids:
            item = items_by_prompt[prompt_id]
            row = _run_one_item(
                args=args,
                item=item,
                gold_by_prompt=gold_by_prompt,
                base_url=args.base_url,
                api_key=args.api_key,
                timeout_seconds=args.timeout_seconds,
            )
            if prompt_id in completed_ids:
                continue
            rows.append(row)
            completed_ids.add(prompt_id)
            _write_jsonl(raw_path, [row], append=True)
            if row.get("success") is False:
                _write_jsonl(args.failed_rows_path, [row], append=True)
            checkpoint = checkpoint_from_rows(
                run_id=_run_id(args),
                expected_count=len(runner_rows),
                result_rows=rows,
                raw_output_path=args.output_path,
                failed_output_path=args.failed_rows_path,
            )
            write_checkpoint(checkpoint, checkpoint_path)
            if len(rows) % args.sync_every_n_requests == 0:
                _write_manifest(
                    args=args,
                    status="running",
                    started_at=started_at,
                    completed_at=None,
                    rows=rows,
                    expected_count=len(runner_rows),
                    command=command,
                )
                sync_events.append(
                    _sync_current_artifacts(args=args, event=f"incremental_{len(rows)}")
                )
                print(f"A100 SXM progress: {len(rows)}/{len(runner_rows)} prompts")
    except Exception:
        _write_manifest(
            args=args,
            status="partial",
            started_at=started_at,
            completed_at=None,
            rows=rows,
            expected_count=len(runner_rows),
            command=command,
        )
        sync_events.append(_sync_current_artifacts(args=args, event="failure_partial"))
        raise
    finally:
        stop_event.set()
        telemetry_thread.join(timeout=max(args.telemetry_interval_seconds + 5.0, 6.0))

    measured_wall_seconds = time.perf_counter() - run_started
    completed_at = utc_now()
    status = "completed" if len(rows) >= len(runner_rows) else "partial"
    _write_manifest(
        args=args,
        status=status,
        started_at=started_at,
        completed_at=completed_at if status == "completed" else None,
        rows=rows,
        expected_count=len(runner_rows),
        command=command,
    )
    gpu_samples = _read_gpu_samples(args.gpu_telemetry_path)
    gpu_summary = summarize_gpu_telemetry(
        gpu_samples,
        interval_seconds=args.telemetry_interval_seconds,
        requested_duration_seconds=measured_wall_seconds,
    )
    preliminary_verification = _verify_current_backup(args)
    report, per_vertical, summary = _evaluate_rows(
        args=args,
        rows=rows,
        output_path=args.output_path,
        wall_seconds=_project_wall_seconds(rows, measured_wall_seconds),
        gpu_summary=gpu_summary,
        telemetry_errors=telemetry_errors,
        artifact_verification=cast(dict[str, Any], preliminary_verification),
    )
    projection = build_b7_runtime_projection(
        measured_prompt_count=len(rows),
        measured_wall_seconds=_project_wall_seconds(rows, measured_wall_seconds),
        mean_latency_ms=float(summary.get("mean_e2e_latency_ms") or 0.0),
        p50_latency_ms=float(summary.get("p50_e2e_latency_ms") or 0.0),
        p95_latency_ms=float(summary.get("p95_e2e_latency_ms") or 0.0),
    )
    _write_json(args.runtime_projection_path, projection)
    projected_hours = _project_wall_seconds(rows, measured_wall_seconds) / 3600.0
    cost_estimate = {
        "hourly_price_usd": args.hourly_price,
        "measured_wall_seconds": _project_wall_seconds(rows, measured_wall_seconds),
        "measured_hours": projected_hours,
        "estimated_cost_usd": projected_hours * args.hourly_price,
    }
    artifact_verification = _verify_current_backup(args)
    gate = _a100_gate(
        classify_b7_quality_gate(
            summary=summary,
            per_vertical_quality=per_vertical,
            completed_count=len(rows),
            expected_count=len(runner_rows),
            artifact_sync_verified=bool(artifact_verification.get("passed")),
            telemetry_sample_count=int(gpu_summary.get("sample_count") or 0),
        )
    )
    report["quality_gate"] = gate
    report["status"] = gate["status"]
    report["runtime_projection"] = projection
    report["cost_estimate"] = cost_estimate
    report["a100_1000_prompt_baseline_allowed"] = _baseline_allowed(gate)
    _write_json(args.eval_report_path, report)
    _write_csv(args.eval_summary_path, [{"vertical": "all", **summary}, *per_vertical])
    readiness = {
        "block": "A100_SXM_CALIBRATION",
        "status": gate["status"],
        "preflight_status": preflight["status"],
        "preflight": preflight,
        "context_alignment_preflight": context_report,
        "server_readiness": server_readiness.to_dict(),
        "resume_plan": resume_plan.to_dict(),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "completed_prompts": len(rows),
        "expected_prompts": len(runner_rows),
        "quality_gate": gate,
        "artifact_backup_verification": artifact_verification,
        "gpu_telemetry_summary": gpu_summary,
        "runtime_projection": projection,
        "cost_estimate": cost_estimate,
        "benchmark_execution_readiness": gate["benchmark_execution_readiness"],
        "next_api_load_probe_allowed": gate["next_api_load_probe_allowed"],
        "runpod_readiness_claimed": False,
        "a100_1000_prompt_baseline_allowed": _baseline_allowed(gate),
        "a100_1000_prompt_baseline_rule": (
            "Allowed only if the 200-prompt calibration completes all prompts, passes "
            "quality gates, verifies artifact sync, and captures GPU telemetry."
        ),
    }
    _write_json(args.readiness_report_path, readiness)
    final_sync = _sync_current_artifacts(args=args, event="run_end")
    sync_events.append(final_sync)
    final_verification = _verify_current_backup(args)
    artifact_sync_report = {
        "block": "A100_SXM_CALIBRATION",
        "run_id": _run_id(args),
        "backup_root": args.backup_root,
        "artifact_sync_enabled": True,
        "dry_run": dry_sync,
        "sync_events": sync_events,
        "final_verification": final_verification,
        "success": bool(final_verification.get("passed")),
    }
    _write_json(args.artifact_sync_report_path, artifact_sync_report)
    _sync_current_artifacts(args=args, event="artifact_sync_report_written")
    readiness["artifact_backup_verification"] = final_verification
    readiness["artifact_sync_report"] = artifact_sync_report
    readiness["quality_gate"] = _a100_gate(
        classify_b7_quality_gate(
            summary=summary,
            per_vertical_quality=per_vertical,
            completed_count=len(rows),
            expected_count=len(runner_rows),
            artifact_sync_verified=bool(final_verification.get("passed")),
            telemetry_sample_count=int(gpu_summary.get("sample_count") or 0),
        )
    )
    readiness["status"] = cast(dict[str, Any], readiness["quality_gate"])["status"]
    readiness["a100_1000_prompt_baseline_allowed"] = _baseline_allowed(
        cast(dict[str, Any], readiness["quality_gate"])
    )
    readiness["benchmark_execution_readiness"] = cast(
        dict[str, Any],
        readiness["quality_gate"],
    )["benchmark_execution_readiness"]
    readiness["next_api_load_probe_allowed"] = cast(
        dict[str, Any],
        readiness["quality_gate"],
    )["next_api_load_probe_allowed"]
    report["quality_gate"] = readiness["quality_gate"]
    report["status"] = readiness["status"]
    report["a100_1000_prompt_baseline_allowed"] = readiness["a100_1000_prompt_baseline_allowed"]
    _write_json(args.eval_report_path, report)
    _write_json(args.readiness_report_path, readiness)
    return readiness


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    try:
        result = run_a100_calibration(normalize_args(args))
    except Exception as exc:  # noqa: BLE001
        print(f"A100 SXM calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
