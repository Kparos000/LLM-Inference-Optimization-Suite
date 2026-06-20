"""Dry-run artifact sync and long-run recovery simulation."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.artifact_sync import (  # noqa: E402
    ArtifactSpec,
    ArtifactSyncConfig,
    build_artifact_specs,
    should_incremental_sync,
    sync_artifacts,
    verify_backup,
)
from inference_bench.checkpoint_resume import (  # noqa: E402
    append_unique_jsonl_rows,
    build_resume_plan,
    build_resume_report,
    checkpoint_from_rows,
    read_jsonl_rows,
    write_checkpoint,
    write_jsonl_rows,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    hash_existing_paths,
    utc_now,
    write_run_manifest,
)

RUN_ID = "long_run_recovery_dry_run"
RAW_ROOT = ROOT / "results" / "raw"
PROCESSED_ROOT = ROOT / "results" / "processed"
BACKUP_ROOT = ROOT / "backups"
WORKLOAD_PATH = RAW_ROOT / f"{RUN_ID}_workload.jsonl"
RAW_JSONL_PATH = RAW_ROOT / f"{RUN_ID}_results.jsonl"
FAILED_JSONL_PATH = RAW_ROOT / f"{RUN_ID}_failed_rows.jsonl"
CHECKPOINT_PATH = RAW_ROOT / f"{RUN_ID}_checkpoint.json"
MANIFEST_PATH = RAW_ROOT / f"{RUN_ID}_manifest.json"
TELEMETRY_PATH = RAW_ROOT / f"{RUN_ID}_telemetry.jsonl"
LOG_PATH = RAW_ROOT / f"{RUN_ID}.log"
REPORT_PATH = PROCESSED_ROOT / f"{RUN_ID}_report.json"
SUMMARY_PATH = PROCESSED_ROOT / f"{RUN_ID}_summary.csv"


def _fake_prompt_rows(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "prompt_id": f"dry_run_prompt_{index:03d}",
            "vertical": "airline",
            "question": f"Dry-run prompt {index}",
        }
        for index in range(1, count + 1)
    ]


def _result_row(prompt_row: dict[str, object], *, success: bool = True) -> dict[str, object]:
    prompt_id = str(prompt_row["prompt_id"])
    return {
        "run_id": RUN_ID,
        "prompt_id": prompt_id,
        "success": success,
        "error_message": None if success else "simulated failure persisted",
        "output_text": f"Simulated output for {prompt_id}" if success else "",
    }


def _write_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def _remove_previous_outputs() -> None:
    for path in (
        WORKLOAD_PATH,
        RAW_JSONL_PATH,
        FAILED_JSONL_PATH,
        CHECKPOINT_PATH,
        MANIFEST_PATH,
        TELEMETRY_PATH,
        LOG_PATH,
        REPORT_PATH,
        SUMMARY_PATH,
    ):
        if path.exists():
            path.unlink()
    backup_run_root = BACKUP_ROOT / RUN_ID
    if backup_run_root.exists():
        shutil.rmtree(backup_run_root)


def _write_manifest(*, status: str, completed_count: int, failed_count: int) -> None:
    now = utc_now()
    expected_count = 20
    manifest = RunManifest(
        run_id=RUN_ID,
        timestamp_utc=now,
        backend="dry_run",
        model_alias="model2_3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        memory_mode="mm2_hybrid_top5",
        split="dry_run_20",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(WORKLOAD_PATH.relative_to(ROOT)),
        output_path=str(RAW_JSONL_PATH.relative_to(ROOT)),
        max_records=expected_count,
        git_commit=current_git_commit(ROOT),
        command="python scripts/phase4/test_long_run_recovery_dry_run.py",
        status=status,
        start_time=now,
        end_time=now if status in {"completed", "failed"} else None,
        error_count=failed_count,
        telemetry_path=str(TELEMETRY_PATH.relative_to(ROOT)),
        config_id="phase1e_long_run_recovery_dry_run",
        vertical="airline",
        runtime="dry_run",
        engine="dry_run",
        backend_type="local_compute",
        hardware="developer_workstation",
        provider="local",
        concurrency=1,
        traffic_profile="offline_throughput",
        prompt_count=expected_count,
        dataset_workload_hash=hash_existing_paths([WORKLOAD_PATH]),
        config_hash=hash_existing_paths(
            [
                ROOT / "configs" / "runtime_engines.yaml",
                ROOT / "configs" / "load_profiles.yaml",
            ]
        ),
        started_at=now,
        updated_at=now,
        completed_at=now if status == "completed" else None,
        completed_count=completed_count,
        failed_count=failed_count,
        expected_count=expected_count,
        artifact_paths={
            "raw_jsonl": str(RAW_JSONL_PATH.relative_to(ROOT)),
            "failed_rows": str(FAILED_JSONL_PATH.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "telemetry": str(TELEMETRY_PATH.relative_to(ROOT)),
            "log": str(LOG_PATH.relative_to(ROOT)),
        },
    )
    write_run_manifest(manifest, MANIFEST_PATH)


def _artifact_specs(*, include_processed_reports: bool = False) -> list[ArtifactSpec]:
    processed_reports = (
        [REPORT_PATH.relative_to(ROOT), SUMMARY_PATH.relative_to(ROOT)]
        if include_processed_reports
        else None
    )
    specs = build_artifact_specs(
        raw_jsonl=RAW_JSONL_PATH.relative_to(ROOT),
        manifest=MANIFEST_PATH.relative_to(ROOT),
        telemetry=TELEMETRY_PATH.relative_to(ROOT),
        processed_reports=processed_reports,
        logs=[LOG_PATH.relative_to(ROOT)],
    )
    specs.append(
        ArtifactSpec(
            path=str(FAILED_JSONL_PATH.relative_to(ROOT)),
            category="failed_rows",
            required=True,
        )
    )
    specs.append(
        ArtifactSpec(
            path=str(CHECKPOINT_PATH.relative_to(ROOT)),
            category="checkpoint",
            required=True,
        )
    )
    return specs


def run_dry_run() -> dict[str, object]:
    """Run the deterministic 20-prompt long-run recovery dry run."""

    _remove_previous_outputs()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_rows = _fake_prompt_rows()
    write_jsonl_rows(WORKLOAD_PATH, prompt_rows, append=False)
    write_jsonl_rows(TELEMETRY_PATH, [{"run_id": RUN_ID, "event": "started"}], append=False)
    _write_log("dry-run initialized")

    sync_config = ArtifactSyncConfig(
        run_id=RUN_ID,
        backup_root=str(BACKUP_ROOT.relative_to(ROOT)),
        incremental_every_n_requests=10,
    )
    sync_events: list[dict[str, object]] = []

    first_batch = [_result_row(row) for row in prompt_rows[:10]]
    append_unique_jsonl_rows(RAW_JSONL_PATH, first_batch)
    write_jsonl_rows(FAILED_JSONL_PATH, [], append=False)
    checkpoint = checkpoint_from_rows(
        run_id=RUN_ID,
        expected_count=len(prompt_rows),
        result_rows=first_batch,
        raw_output_path=RAW_JSONL_PATH.relative_to(ROOT),
        failed_output_path=FAILED_JSONL_PATH.relative_to(ROOT),
    )
    write_checkpoint(checkpoint, CHECKPOINT_PATH)
    _write_manifest(status="partial", completed_count=10, failed_count=0)
    sync_events.append(
        sync_artifacts(
            specs=_artifact_specs(),
            config=sync_config,
            event="start",
            repo_root=ROOT,
        )
    )
    if should_incremental_sync(10, sync_config.incremental_every_n_requests):
        sync_events.append(
            sync_artifacts(
                specs=_artifact_specs(),
                config=sync_config,
                event="incremental_10",
                repo_root=ROOT,
            )
        )
    _write_log("simulated interruption after 10 rows")

    resume_plan = build_resume_plan(
        run_id=RUN_ID,
        prompt_rows=prompt_rows,
        checkpoint_path=CHECKPOINT_PATH,
        partial_raw_jsonl_path=RAW_JSONL_PATH,
    )
    second_batch: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    for prompt_row in prompt_rows[10:]:
        success = prompt_row["prompt_id"] != "dry_run_prompt_020"
        row = _result_row(prompt_row, success=success)
        second_batch.append(row)
        if not success:
            failed_rows.append(row)
    append_unique_jsonl_rows(RAW_JSONL_PATH, second_batch)
    append_unique_jsonl_rows(FAILED_JSONL_PATH, failed_rows)
    all_rows = read_jsonl_rows(RAW_JSONL_PATH)
    checkpoint = checkpoint_from_rows(
        run_id=RUN_ID,
        expected_count=len(prompt_rows),
        result_rows=all_rows,
        raw_output_path=RAW_JSONL_PATH.relative_to(ROOT),
        failed_output_path=FAILED_JSONL_PATH.relative_to(ROOT),
    )
    write_checkpoint(checkpoint, CHECKPOINT_PATH)
    _write_manifest(status="completed", completed_count=19, failed_count=1)
    _write_log("resume completed remaining 10 rows")

    sync_events.append(
        sync_artifacts(
            specs=_artifact_specs(),
            config=sync_config,
            event="end",
            repo_root=ROOT,
        )
    )
    verification = verify_backup(
        specs=_artifact_specs(),
        config=sync_config,
        repo_root=ROOT,
    )
    duplicate_ids = sorted(
        {
            row["prompt_id"]
            for row in all_rows
            if [item["prompt_id"] for item in all_rows].count(row["prompt_id"]) > 1
        }
    )
    report: dict[str, object] = {
        "run_id": RUN_ID,
        "status": "PASSED"
        if verification["passed"] and len(all_rows) == 20 and not duplicate_ids
        else "FAILED",
        "simulated_prompt_count": len(prompt_rows),
        "initial_written_count": 10,
        "resume_pending_count": resume_plan.pending_count,
        "resume_skipped_count": resume_plan.skipped_count,
        "resumed_written_count": len(second_batch),
        "final_row_count": len(all_rows),
        "successful_row_count": 19,
        "failed_row_count": 1,
        "duplicate_prompt_ids": duplicate_ids,
        "resume_report": build_resume_report(resume_plan, CHECKPOINT_PATH.relative_to(ROOT)),
        "sync_events": sync_events,
        "backup_verification": verification,
        "artifacts": {
            "workload": str(WORKLOAD_PATH.relative_to(ROOT)),
            "raw_jsonl": str(RAW_JSONL_PATH.relative_to(ROOT)),
            "failed_rows": str(FAILED_JSONL_PATH.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "telemetry": str(TELEMETRY_PATH.relative_to(ROOT)),
            "log": str(LOG_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
            "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            "backup_root": str((BACKUP_ROOT / RUN_ID).relative_to(ROOT)),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_id",
                "status",
                "simulated_prompt_count",
                "final_row_count",
                "failed_row_count",
                "backup_passed",
                "backup_completeness_score",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": RUN_ID,
                "status": report["status"],
                "simulated_prompt_count": len(prompt_rows),
                "final_row_count": len(all_rows),
                "failed_row_count": 1,
                "backup_passed": verification["passed"],
                "backup_completeness_score": verification["backup_completeness_score"],
            }
        )
    processed_sync = sync_artifacts(
        specs=[
            ArtifactSpec(
                path=str(REPORT_PATH.relative_to(ROOT)),
                category="processed_report",
                required=True,
            ),
            ArtifactSpec(
                path=str(SUMMARY_PATH.relative_to(ROOT)),
                category="processed_report",
                required=True,
            ),
        ],
        config=sync_config,
        event="processed_reports",
        repo_root=ROOT,
    )
    sync_events.append(processed_sync)
    report["sync_events"] = sync_events
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sync_artifacts(
        specs=[
            ArtifactSpec(
                path=str(REPORT_PATH.relative_to(ROOT)),
                category="processed_report",
                required=True,
            )
        ],
        config=sync_config,
        event="processed_report_final",
        repo_root=ROOT,
    )
    return report


def main() -> int:
    report = run_dry_run()
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
