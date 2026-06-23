"""Run B7R1 vLLM CUDA stability repair for the frozen B7 1,000-row input."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import load_gold_records  # noqa: E402
from run_b6r6_research_ai_quality_recovery import (  # noqa: E402
    _failure_row,
    _run_default_item,
    _run_research_ai_item,
    _workload_item,
)
from run_b7_controlled_1000_baseline import (  # noqa: E402
    DEFAULT_RUNNER_INPUT,
    _artifact_sync_dry_run,
    _evaluate_rows,
    _project_wall_seconds,
    _read_gpu_samples,
    _read_jsonl,
    _telemetry_loop,
    _write_csv,
    _write_json,
    _write_jsonl,
)
from run_openai_compatible_smoke import DEFAULT_API_KEY, check_server_readiness  # noqa: E402
from run_remote_vllm_smoke import sanitized_command  # noqa: E402

from inference_bench.artifact_sync import (  # noqa: E402
    ArtifactSyncConfig,
    build_artifact_specs,
    sync_artifacts,
    verify_backup,
)
from inference_bench.b6r6_research_ai_recovery import (  # noqa: E402
    STRATEGY_D_ANSWER_SKELETON,
)
from inference_bench.b7_controlled_baseline import (  # noqa: E402
    B7_BACKEND_TYPE,
    B7_CONCURRENCY,
    B7_HARDWARE,
    B7_MEMORY_MODE,
    B7_MODEL_ALIAS,
    B7_MODEL_ID,
    B7_PROVIDER,
    B7_REQUEST_ARRIVAL_MODE,
    B7_RESEARCH_AI_STRATEGY,
    B7_RUNTIME,
    B7_TRAFFIC_PROFILE,
    build_b7_load_and_cache_report,
    build_b7_runtime_projection,
    classify_b7_quality_gate,
    preflight_b7_runner_rows,
)
from inference_bench.checkpoint_resume import (  # noqa: E402
    build_resume_plan,
    checkpoint_from_rows,
    write_checkpoint,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.gpu_telemetry import summarize_gpu_telemetry  # noqa: E402
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    hash_existing_paths,
    utc_now,
    write_run_manifest,
)
from inference_bench.runtime_registry import select_runtime_for_model  # noqa: E402
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.serving_profiles import (  # noqa: E402
    ServingProfile,
    select_serving_profile,
)
from inference_bench.vllm_stability_audit import (  # noqa: E402
    build_vllm_stability_audit,
    classify_b7r1_stability_gate,
    is_backend_connection_failure,
    is_fatal_engine_error,
)

B7R1_RUN_ID = "b7r1-model2-3b-1000-vllm-safe"
B7R1_CONFIG_ID = "b7r1_model2_3b_1000_vllm_safe_profile"
B7R1_SERVING_PROFILE = "remote_rtx3070_qwen3b_safe_v1"
B7R1_OPTIMIZATION = "b7r1_vllm_cuda_stability_repair"

DEFAULT_RAW_OUTPUT = "results/raw/b7r1_model2_3b_1000_results.jsonl"
DEFAULT_MANIFEST = "results/raw/b7r1_model2_3b_1000_manifest.json"
DEFAULT_GPU_TELEMETRY = "results/raw/b7r1_model2_3b_1000_gpu_telemetry.jsonl"
DEFAULT_EVAL_REPORT = "results/processed/b7r1_model2_3b_1000_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/b7r1_model2_3b_1000_eval_summary.csv"
DEFAULT_RUNTIME_PROJECTION = "results/processed/b7r1_runtime_projection.json"
DEFAULT_ARTIFACT_SYNC_REPORT = "results/processed/b7r1_artifact_sync_report.json"
DEFAULT_COMPARISON = "results/processed/b7_vs_b7r1_comparison.json"
DEFAULT_PREFLIGHT = "results/processed/b7r1_preflight_report.json"
DEFAULT_READINESS_REPORT = "results/processed/b7r1_readiness_report.json"
DEFAULT_CHECKPOINT = "results/raw/b7r1_model2_3b_1000_checkpoint.json"
DEFAULT_FAILED_ROWS = "results/raw/b7r1_model2_3b_1000_failed_rows.jsonl"
B7_EVAL_REPORT = "results/processed/b7_model2_3b_1000_eval_report.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the B7R1 CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--output-path", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu-telemetry-path", default=DEFAULT_GPU_TELEMETRY)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--runtime-projection-path", default=DEFAULT_RUNTIME_PROJECTION)
    parser.add_argument("--artifact-sync-report-path", default=DEFAULT_ARTIFACT_SYNC_REPORT)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON)
    parser.add_argument("--preflight-report-path", default=DEFAULT_PREFLIGHT)
    parser.add_argument("--readiness-report-path", default=DEFAULT_READINESS_REPORT)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--failed-rows-path", default=DEFAULT_FAILED_ROWS)
    parser.add_argument("--serving-profile", default=B7R1_SERVING_PROFILE)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--backup-root", default="backups")
    parser.add_argument("--sync-every-n-requests", type=int, default=25)
    parser.add_argument("--telemetry-ssh-host", default="zeever-gpu")
    parser.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _artifact_specs(args: argparse.Namespace) -> list[Any]:
    return build_artifact_specs(
        raw_jsonl=args.output_path,
        manifest=args.manifest_path,
        telemetry=args.gpu_telemetry_path,
        processed_reports=[
            args.eval_report_path,
            args.eval_summary_path,
            args.runtime_projection_path,
            args.artifact_sync_report_path,
            args.comparison_path,
            args.preflight_report_path,
            args.readiness_report_path,
        ],
    )


def _sync_current_artifacts(*, args: argparse.Namespace, event: str) -> dict[str, Any]:
    config = ArtifactSyncConfig(
        run_id=B7R1_RUN_ID,
        backup_root=args.backup_root,
        incremental_every_n_requests=args.sync_every_n_requests,
    )
    return cast(
        dict[str, Any],
        sync_artifacts(specs=_artifact_specs(args), config=config, event=event, repo_root=ROOT),
    )


def _verify_current_backup(args: argparse.Namespace) -> dict[str, Any]:
    config = ArtifactSyncConfig(
        run_id=B7R1_RUN_ID,
        backup_root=args.backup_root,
        incremental_every_n_requests=args.sync_every_n_requests,
    )
    return cast(
        dict[str, Any],
        verify_backup(specs=_artifact_specs(args), config=config, repo_root=ROOT),
    )


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
        "comparison": args.comparison_path,
        "preflight_report": args.preflight_report_path,
        "readiness_report": args.readiness_report_path,
        "checkpoint": args.checkpoint_path,
        "failed_rows": args.failed_rows_path,
    }
    return RunManifest(
        run_id=B7R1_RUN_ID,
        timestamp_utc=updated_at,
        backend=B7_RUNTIME,
        model_alias=B7_MODEL_ALIAS,
        model_id=B7_MODEL_ID,
        memory_mode=B7_MEMORY_MODE,
        split="controlled_1000",
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
        config_id=B7R1_CONFIG_ID,
        vertical="all",
        runtime=B7_RUNTIME,
        engine=B7_RUNTIME,
        backend_type=B7_BACKEND_TYPE,
        hardware=B7_HARDWARE,
        provider=B7_PROVIDER,
        concurrency=B7_CONCURRENCY,
        traffic_profile=B7_TRAFFIC_PROFILE,
        prompt_count=expected_count,
        dataset_workload_hash=hash_existing_paths([_repo_path(args.runner_input_path)]),
        config_hash=hash_existing_paths(
            [
                "configs/models.yaml",
                "configs/runtime_engines.yaml",
                "configs/load_profiles.yaml",
                "configs/serving_profiles.yaml",
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
    successful = sum(bool(row.get("success")) for row in rows)
    failed = len(rows) - successful
    manifest = _build_manifest(
        args=args,
        status=status,
        started_at=started_at,
        updated_at=utc_now(),
        completed_at=completed_at,
        completed_count=successful,
        failed_count=failed,
        expected_count=expected_count,
        command=command,
    )
    write_run_manifest(manifest, _repo_path(args.manifest_path))


def _prompt_token_risk(rows: list[dict[str, Any]], profile: ServingProfile) -> dict[str, Any]:
    prompt_lengths = [len(str(row.get("prompt") or "").split()) for row in rows]
    max_prompt_words = max(prompt_lengths, default=0)
    above_safe_window = sum(length > profile.max_model_len for length in prompt_lengths)
    return {
        "row_count": len(rows),
        "max_prompt_word_estimate": max_prompt_words,
        "profile_max_model_len": profile.max_model_len,
        "prompt_word_estimate_above_profile_window": above_safe_window,
        "load_and_cache_report_available_after_inference": True,
    }


def _mark_b7r1_row(row: dict[str, Any], item: WorkloadItem) -> dict[str, Any]:
    metadata = dict(item.metadata)
    row.update(
        {
            "run_id": B7R1_RUN_ID,
            "config_id": B7R1_CONFIG_ID,
            "model_alias": B7_MODEL_ALIAS,
            "model_id": B7_MODEL_ID,
            "model_name": B7_MODEL_ID,
            "backend": B7_RUNTIME,
            "runtime": B7_RUNTIME,
            "engine": B7_RUNTIME,
            "backend_type": B7_BACKEND_TYPE,
            "hardware": B7_HARDWARE,
            "provider": B7_PROVIDER,
            "concurrency": B7_CONCURRENCY,
            "traffic_profile": B7_TRAFFIC_PROFILE,
            "request_arrival_mode": B7_REQUEST_ARRIVAL_MODE,
            "selected_context_ids": metadata.get("selected_context_ids"),
            "b5_required_labels": metadata.get("b5_required_labels"),
            "b7r1_serving_profile": B7R1_SERVING_PROFILE,
            "b7_finance_repair_active": metadata.get("vertical") == "finance",
            "b7_research_ai_repair_active": metadata.get("vertical") == "research_ai",
            "b7_research_ai_strategy": (
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
                run_id=B7R1_RUN_ID,
                optimization=B7R1_OPTIMIZATION,
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
                run_id=B7R1_RUN_ID,
                optimization=B7R1_OPTIMIZATION,
            )
    except Exception as exc:  # noqa: BLE001
        row = _failure_row(
            item=item,
            prompt=item.prompt,
            exc=exc,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            strategy_id=STRATEGY_D_ANSWER_SKELETON,
            run_id=B7R1_RUN_ID,
            optimization=B7R1_OPTIMIZATION,
        )
    return _mark_b7r1_row(row, item)


def _build_preflight(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    profile: ServingProfile,
) -> dict[str, Any]:
    project_config = load_project_config()
    model = project_config.resolve_model_config(B7_MODEL_ALIAS)
    runtime_selection = select_runtime_for_model(
        model_alias=B7_MODEL_ALIAS,
        runtime=B7_RUNTIME,
        hardware_type=B7_HARDWARE,
        backend_route="openai_compatible_vllm",
        live_run=True,
    )
    dry_sync = _artifact_sync_dry_run(args)
    preflight = preflight_b7_runner_rows(
        rows,
        model_alias=B7_MODEL_ALIAS,
        model_id=model.model_id,
        runtime_selection=runtime_selection.to_dict(),
        artifact_sync_dry_run_passed=bool(dry_sync["success"]),
        checkpoint_resume_enabled=True,
        manifest_enabled=True,
    )
    profile_check = (
        profile.profile_id == args.serving_profile
        and profile.model_alias == B7_MODEL_ALIAS
        and profile.model_id == B7_MODEL_ID
        and profile.engine == B7_RUNTIME
        and profile.hardware == B7_HARDWARE
        and profile.status == "ready"
        and profile.live_run_allowed
        and profile.gpu_memory_utilization <= 0.82
        and profile.max_num_seqs == 1
        and profile.max_model_len <= 3584
    )
    preflight["block"] = "B7R1"
    preflight["status"] = (
        "PREFLIGHT_PASSED_B7R1_VLLM_STABILITY_REPAIR"
        if bool(preflight["passed"]) and profile_check
        else "PREFLIGHT_BLOCKED_B7R1_VLLM_STABILITY_REPAIR"
    )
    preflight["passed"] = bool(preflight["passed"]) and profile_check
    preflight["serving_profile_check_passed"] = profile_check
    preflight["serving_profile"] = profile.to_dict()
    preflight["serving_profile_vllm_args"] = profile.vllm_server_args()
    preflight["artifact_sync_dry_run"] = dry_sync
    preflight["prompt_token_risk_report"] = _prompt_token_risk(rows, profile)
    preflight["b6r6_repairs_active"] = True
    preflight["same_b7_input_reused"] = True
    preflight["model_inference_triggered"] = False
    if not profile_check:
        failed = list(preflight.get("failed_checks") or [])
        failed.append("safe_serving_profile")
        preflight["failed_checks"] = sorted(set(failed))
    return preflight


def _build_comparison(
    b7_report: dict[str, Any] | None,
    b7r1_report: dict[str, Any],
) -> dict[str, Any]:
    b7_summary = (b7_report or {}).get("summary") if b7_report else {}
    b7r1_summary = b7r1_report.get("summary") if isinstance(b7r1_report, dict) else {}
    if not isinstance(b7_summary, dict):
        b7_summary = {}
    if not isinstance(b7r1_summary, dict):
        b7r1_summary = {}
    return {
        "block": "B7R1",
        "comparison": "b7_vs_b7r1",
        "b7_status": (b7_report or {}).get("status") if b7_report else None,
        "b7r1_status": b7r1_report.get("status"),
        "request_success_count_delta": int(b7r1_summary.get("request_success_count") or 0)
        - int(b7_summary.get("request_success_count") or 0),
        "request_failure_count_delta": int(b7r1_summary.get("request_failure_count") or 0)
        - int(b7_summary.get("request_failure_count") or 0),
        "b7_summary": b7_summary,
        "b7r1_summary": b7r1_summary,
    }


def run_b7r1(args: argparse.Namespace) -> dict[str, Any]:
    """Run B7R1 preflight and optional inference."""

    command = sanitized_command(sys.argv)
    started_at = utc_now()
    runner_input = _repo_path(args.runner_input_path)
    if not runner_input.exists():
        msg = f"Frozen B7 runner input is required: {runner_input}"
        raise FileNotFoundError(msg)
    runner_rows = _read_jsonl(args.runner_input_path)
    profile = select_serving_profile(args.serving_profile, live_run=True)
    preflight = _build_preflight(args=args, rows=runner_rows, profile=profile)
    _write_json(args.preflight_report_path, preflight)
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
    if args.dry_run or args.preflight_only or not bool(preflight["passed"]):
        readiness = {
            "block": "B7R1",
            "status": preflight["status"],
            "preflight": preflight,
            "initial_artifact_sync": initial_sync,
            "inference_triggered": False,
        }
        _write_json(args.readiness_report_path, readiness)
        return readiness

    server_readiness = check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=B7_MODEL_ID,
        timeout_seconds=args.timeout_seconds,
    )
    if not server_readiness.reachable or server_readiness.model_available is False:
        readiness = {
            "block": "B7R1",
            "status": "B7R1_STABILITY_BLOCKED",
            "preflight": preflight,
            "server_readiness": server_readiness.to_dict(),
            "inference_triggered": False,
        }
        _write_json(args.readiness_report_path, readiness)
        return readiness

    raw_path = _repo_path(args.output_path)
    checkpoint_path = _repo_path(args.checkpoint_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raw_path.write_text("", encoding="utf-8")
    telemetry_path = _repo_path(args.gpu_telemetry_path)
    if not telemetry_path.exists():
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text("", encoding="utf-8")
    rows = _read_jsonl(raw_path)
    resumed_from_checkpoint = checkpoint_path.exists() or bool(rows)
    resume_plan = build_resume_plan(
        run_id=B7R1_RUN_ID,
        prompt_rows=runner_rows,
        checkpoint_path=checkpoint_path,
        partial_raw_jsonl_path=raw_path,
    )
    gold_rows = load_gold_records("data/scaleup_2000_full")
    gold_by_prompt = {str(row.get("prompt_id") or ""): row for row in gold_rows}
    items_by_prompt = {str(row["prompt_id"]): _workload_item(row) for row in runner_rows}
    completed_ids = {str(row.get("prompt_id")) for row in rows}
    fatal_engine_errors = sum(is_fatal_engine_error(row.get("error_message")) for row in rows)
    serving_restarts = 0
    fatal_stop_reason = ""
    checkpoint_warnings: list[dict[str, Any]] = []
    stop_event = threading.Event()
    telemetry_errors: list[str] = []
    telemetry_thread = threading.Thread(
        target=_telemetry_loop,
        kwargs={
            "path": telemetry_path,
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
                run_id=B7R1_RUN_ID,
                expected_count=len(runner_rows),
                result_rows=rows,
                raw_output_path=args.output_path,
                failed_output_path=args.failed_rows_path,
            )
            written_checkpoint = write_checkpoint(checkpoint, checkpoint_path)
            if written_checkpoint != checkpoint_path:
                checkpoint_warnings.append(
                    {
                        "warning": "checkpoint_primary_replace_failed",
                        "fallback_checkpoint_path": str(written_checkpoint),
                        "row_count": len(rows),
                    }
                )
            row_error = row.get("error_message")
            if is_fatal_engine_error(row_error):
                fatal_engine_errors += 1
                health_after_failure = check_server_readiness(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model_name=B7_MODEL_ID,
                    timeout_seconds=min(args.timeout_seconds, 30.0),
                )
                fatal_stop_reason = (
                    f"fatal_engine_error_at_prompt={prompt_id}; "
                    f"health_after_failure={health_after_failure.message}"
                )
                break
            if is_backend_connection_failure(row_error) and fatal_engine_errors:
                fatal_stop_reason = f"backend_unreachable_after_fatal_engine_error_at={prompt_id}"
                break
            if len(rows) % profile.health_check_every_n_requests == 0:
                health = check_server_readiness(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model_name=B7_MODEL_ID,
                    timeout_seconds=min(args.timeout_seconds, 30.0),
                )
                if not health.reachable or health.model_available is False:
                    fatal_engine_errors += 1
                    fatal_stop_reason = f"periodic_health_check_failed_at_row={len(rows)}"
                    break
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
                print(f"B7R1 progress: {len(rows)}/{len(runner_rows)} prompts")
    finally:
        stop_event.set()
        telemetry_thread.join(timeout=max(args.telemetry_interval_seconds + 5.0, 6.0))

    measured_wall_seconds = time.perf_counter() - run_started
    completed_at = utc_now()
    status = (
        "completed"
        if len(rows) >= len(runner_rows)
        else "failed"
        if fatal_stop_reason
        else "partial"
    )
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
        rows=rows,
        output_path=args.output_path,
        wall_seconds=_project_wall_seconds(rows, measured_wall_seconds),
        gpu_summary=gpu_summary,
        telemetry_errors=telemetry_errors,
        artifact_verification=preliminary_verification,
    )
    load_and_cache = build_b7_load_and_cache_report(
        rows=rows,
        traffic_profile=B7_TRAFFIC_PROFILE,
        concurrency=B7_CONCURRENCY,
        request_arrival_mode=B7_REQUEST_ARRIVAL_MODE,
    )
    projection = build_b7_runtime_projection(
        measured_prompt_count=len(rows),
        measured_wall_seconds=_project_wall_seconds(rows, measured_wall_seconds),
        mean_latency_ms=float(summary.get("mean_e2e_latency_ms") or 0.0),
        p50_latency_ms=float(summary.get("p50_e2e_latency_ms") or 0.0),
        p95_latency_ms=float(summary.get("p95_e2e_latency_ms") or 0.0),
    )
    _write_json(args.runtime_projection_path, projection)
    artifact_verification = _verify_current_backup(args)
    quality_gate = classify_b7_quality_gate(
        summary=summary,
        per_vertical_quality=per_vertical,
        completed_count=len(rows),
        expected_count=len(runner_rows),
        artifact_sync_verified=bool(artifact_verification.get("passed")),
        telemetry_sample_count=int(gpu_summary.get("sample_count") or 0),
    )
    stability_audit = build_vllm_stability_audit(
        result_rows=rows,
        telemetry_rows=[sample.to_dict() for sample in gpu_samples],
        eval_report={"summary": summary},
        expected_count=len(runner_rows),
    )
    peak_vram_mb = float(gpu_summary.get("peak_memory_used_mb") or 0.0)
    stability_gate = classify_b7r1_stability_gate(
        completed_count=len(rows),
        expected_count=len(runner_rows),
        success_count=int(summary.get("request_success_count") or 0),
        fatal_engine_errors=fatal_engine_errors,
        cascading_backend_failure=bool(
            cast(dict[str, Any], stability_audit["cascading_failure"]).get(
                "cascading_failure_observed"
            )
        ),
        safety_violation_count=float(summary.get("safety_violation_count") or 0.0),
        artifact_sync_complete=bool(artifact_verification.get("passed")),
        manifest_valid=True,
        checkpoint_valid=checkpoint_path.exists()
        or (_repo_path(str(args.checkpoint_path) + ".warning.json")).exists(),
        peak_vram_mb=peak_vram_mb,
        peak_vram_threshold_mb=float(profile.peak_vram_safe_threshold_mb),
        quality_passed=bool(quality_gate["passed"]),
    )
    report.update(
        {
            "block": "B7R1",
            "experiment": "model2_3b_vllm_controlled_1000_stability_repair",
            "config_id": B7R1_CONFIG_ID,
            "run_id": B7R1_RUN_ID,
            "serving_profile": profile.to_dict(),
            "serving_profile_vllm_args": profile.vllm_server_args(),
            "status": stability_gate["status"],
            "quality_gate": quality_gate,
            "stability_gate": stability_gate,
            "stability_audit": stability_audit,
            "runtime_projection": projection,
            "load_profile_report": load_and_cache["load_profile"],
            "cache_readiness": load_and_cache["cache_readiness"],
            "serving_restarts": serving_restarts,
            "fatal_engine_errors": fatal_engine_errors,
            "fatal_stop_reason": fatal_stop_reason,
            "checkpoint_warnings": checkpoint_warnings,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resume_plan": resume_plan.to_dict(),
        }
    )
    _write_json(args.eval_report_path, report)
    _write_csv(args.eval_summary_path, [{"vertical": "all", **summary}, *per_vertical])
    b7_report_path = _repo_path(B7_EVAL_REPORT)
    b7_report = (
        json.loads(b7_report_path.read_text(encoding="utf-8")) if b7_report_path.exists() else None
    )
    comparison = _build_comparison(b7_report, report)
    _write_json(args.comparison_path, comparison)
    final_sync = _sync_current_artifacts(args=args, event="run_end")
    sync_events.append(final_sync)
    final_verification = _verify_current_backup(args)
    summary["artifact_sync_verified"] = bool(final_verification.get("passed"))
    quality_gate = classify_b7_quality_gate(
        summary=summary,
        per_vertical_quality=per_vertical,
        completed_count=len(rows),
        expected_count=len(runner_rows),
        artifact_sync_verified=bool(final_verification.get("passed")),
        telemetry_sample_count=int(gpu_summary.get("sample_count") or 0),
    )
    stability_gate = classify_b7r1_stability_gate(
        completed_count=len(rows),
        expected_count=len(runner_rows),
        success_count=int(summary.get("request_success_count") or 0),
        fatal_engine_errors=fatal_engine_errors,
        cascading_backend_failure=bool(
            cast(dict[str, Any], stability_audit["cascading_failure"]).get(
                "cascading_failure_observed"
            )
        ),
        safety_violation_count=float(summary.get("safety_violation_count") or 0.0),
        artifact_sync_complete=bool(final_verification.get("passed")),
        manifest_valid=True,
        checkpoint_valid=checkpoint_path.exists()
        or (_repo_path(str(args.checkpoint_path) + ".warning.json")).exists(),
        peak_vram_mb=peak_vram_mb,
        peak_vram_threshold_mb=float(profile.peak_vram_safe_threshold_mb),
        quality_passed=bool(quality_gate["passed"]),
    )
    report["status"] = stability_gate["status"]
    report["quality_gate"] = quality_gate
    report["stability_gate"] = stability_gate
    report["artifact_backup_verification"] = final_verification
    report["summary"] = summary
    _write_json(args.eval_report_path, report)
    _write_csv(args.eval_summary_path, [{"vertical": "all", **summary}, *per_vertical])
    comparison = _build_comparison(b7_report, report)
    _write_json(args.comparison_path, comparison)
    artifact_sync_report = {
        "block": "B7R1",
        "run_id": B7R1_RUN_ID,
        "backup_root": args.backup_root,
        "artifact_sync_enabled": True,
        "sync_events": sync_events,
        "final_verification": final_verification,
        "success": bool(final_verification.get("passed")),
    }
    _write_json(args.artifact_sync_report_path, artifact_sync_report)
    readiness = {
        "block": "B7R1",
        "status": stability_gate["status"],
        "preflight": preflight,
        "server_readiness": server_readiness.to_dict(),
        "completed_prompts": len(rows),
        "expected_prompts": len(runner_rows),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "quality_gate": quality_gate,
        "stability_gate": stability_gate,
        "artifact_sync_report": artifact_sync_report,
        "artifact_backup_verification": final_verification,
        "gpu_telemetry_summary": gpu_summary,
        "runtime_projection": projection,
        "benchmark_execution_readiness": stability_gate["benchmark_execution_readiness"],
        "next_api_load_probe_allowed": stability_gate["next_api_load_probe_allowed"],
        "rtx3070_qwen3b_suitability": stability_gate["rtx3070_qwen3b_suitability"],
        "runpod_readiness_claimed": False,
    }
    _write_json(args.readiness_report_path, readiness)
    return readiness


def main() -> int:
    """CLI entry point."""

    args = build_parser().parse_args()
    try:
        result = run_b7r1(args)
    except Exception as exc:  # noqa: BLE001
        print(f"B7R1 stability repair failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
