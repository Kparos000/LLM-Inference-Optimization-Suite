from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from inference_bench.artifact_sync import (
    ArtifactSpec,
    ArtifactSyncConfig,
    build_artifact_specs,
    should_incremental_sync,
    sync_artifacts,
)


def test_local_artifact_sync_copies_configured_files(tmp_path: Path) -> None:
    raw = tmp_path / "results" / "raw" / "run.jsonl"
    manifest = tmp_path / "results" / "raw" / "manifest.json"
    telemetry = tmp_path / "results" / "raw" / "telemetry.jsonl"
    report = tmp_path / "results" / "processed" / "report.json"
    log = tmp_path / "logs" / "run.log"
    for path, text in (
        (raw, '{"prompt_id":"p1"}\n'),
        (manifest, '{"status":"partial"}\n'),
        (telemetry, '{"event":"sample"}\n'),
        (report, '{"status":"ok"}\n'),
        (log, "started\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    specs = build_artifact_specs(
        raw_jsonl=raw.relative_to(tmp_path),
        manifest=manifest.relative_to(tmp_path),
        telemetry=telemetry.relative_to(tmp_path),
        processed_reports=[report.relative_to(tmp_path)],
        logs=[log.relative_to(tmp_path)],
    )
    result = sync_artifacts(
        specs=specs,
        config=ArtifactSyncConfig(run_id="run-1", backup_root=str(tmp_path / "backups")),
        event="start",
        repo_root=tmp_path,
    )

    assert result["success"] is True
    synced_artifacts = cast(list[dict[str, Any]], result["synced_artifacts"])
    assert len(synced_artifacts) == 5
    for artifact in synced_artifacts:
        assert Path(str(artifact["backup_path"])).exists()
        assert int(artifact["size_bytes"]) > 0


def test_missing_required_artifact_fails_sync(tmp_path: Path) -> None:
    result = sync_artifacts(
        specs=[ArtifactSpec(path="missing.jsonl", category="raw_jsonl", required=True)],
        config=ArtifactSyncConfig(run_id="run-1", backup_root=str(tmp_path / "backups")),
        event="failure",
        repo_root=tmp_path,
    )

    assert result["success"] is False
    assert result["missing_required"] == ["missing.jsonl"]


def test_incremental_sync_schedule() -> None:
    assert should_incremental_sync(10, 10) is True
    assert should_incremental_sync(11, 10) is False
    assert should_incremental_sync(10, None) is False
