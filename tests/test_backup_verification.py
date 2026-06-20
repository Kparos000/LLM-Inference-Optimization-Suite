from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from inference_bench.artifact_sync import (
    ArtifactSyncConfig,
    build_artifact_specs,
    sync_artifacts,
    verify_backup,
)


def _write_artifacts(root: Path, *, manifest_status: str = "completed") -> tuple[Path, Path]:
    raw = root / "results" / "raw" / "run.jsonl"
    manifest = root / "results" / "raw" / "manifest.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"prompt_id":"p1"}\n{"prompt_id":"p2"}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "status": manifest_status,
                "completed_count": 2,
                "failed_count": 0,
                "expected_count": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return raw, manifest


def test_backup_verification_passes_hash_and_manifest_checks(tmp_path: Path) -> None:
    raw, manifest = _write_artifacts(tmp_path)
    specs = build_artifact_specs(
        raw_jsonl=raw.relative_to(tmp_path), manifest=manifest.relative_to(tmp_path)
    )
    config = ArtifactSyncConfig(run_id="run-1", backup_root=str(tmp_path / "backups"))
    sync_artifacts(specs=specs, config=config, event="end", repo_root=tmp_path)

    verification = verify_backup(specs=specs, config=config, repo_root=tmp_path)

    assert verification["passed"] is True
    assert verification["backup_completeness_score"] == 1.0


def test_backup_verification_detects_hash_mismatch(tmp_path: Path) -> None:
    raw, manifest = _write_artifacts(tmp_path)
    specs = build_artifact_specs(
        raw_jsonl=raw.relative_to(tmp_path), manifest=manifest.relative_to(tmp_path)
    )
    config = ArtifactSyncConfig(run_id="run-1", backup_root=str(tmp_path / "backups"))
    sync_artifacts(specs=specs, config=config, event="end", repo_root=tmp_path)
    backup_raw = tmp_path / "backups" / "run-1" / raw.relative_to(tmp_path)
    backup_raw.write_text('{"prompt_id":"tampered"}\n', encoding="utf-8")

    verification = verify_backup(specs=specs, config=config, repo_root=tmp_path)

    assert verification["passed"] is False
    checks = cast(list[dict[str, Any]], verification["checks"])
    assert any(check["hash_ok"] is False for check in checks if "hash_ok" in check)


def test_backup_verification_detects_completed_manifest_row_mismatch(tmp_path: Path) -> None:
    raw, manifest = _write_artifacts(tmp_path)
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_count": 1,
                "failed_count": 0,
                "expected_count": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    specs = build_artifact_specs(
        raw_jsonl=raw.relative_to(tmp_path), manifest=manifest.relative_to(tmp_path)
    )
    config = ArtifactSyncConfig(run_id="run-1", backup_root=str(tmp_path / "backups"))
    sync_artifacts(specs=specs, config=config, event="end", repo_root=tmp_path)

    verification = verify_backup(specs=specs, config=config, repo_root=tmp_path)

    assert verification["passed"] is False
    checks = cast(list[dict[str, Any]], verification["checks"])
    assert any(
        check["name"] == "manifest_status_matches_completed_rows" and check["passed"] is False
        for check in checks
    )
