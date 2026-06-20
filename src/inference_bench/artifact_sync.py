"""Local artifact synchronization and backup verification."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inference_bench.checkpoint_resume import read_jsonl_rows
from inference_bench.run_manifest import file_sha256, utc_now


@dataclass(frozen=True)
class ArtifactSpec:
    """One source artifact that can be synchronized to backup storage."""

    path: str
    category: str
    required: bool = True
    non_zero_required: bool = True

    def __post_init__(self) -> None:
        if not self.path.strip():
            msg = "artifact path must not be empty"
            raise ValueError(msg)
        if not self.category.strip():
            msg = "artifact category must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class ArtifactSyncConfig:
    """Local artifact sync configuration.

    The interface intentionally keeps a provider field so S3/R2/GDrive backends
    can be added later without changing call sites.
    """

    run_id: str
    backup_root: str = "backups"
    provider: str = "local"
    incremental_every_n_requests: int | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            msg = "run_id must not be empty"
            raise ValueError(msg)
        if not self.backup_root.strip():
            msg = "backup_root must not be empty"
            raise ValueError(msg)
        if self.provider != "local":
            msg = "only local artifact sync provider is implemented"
            raise ValueError(msg)
        if self.incremental_every_n_requests is not None and self.incremental_every_n_requests <= 0:
            msg = "incremental_every_n_requests must be > 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class SyncedArtifact:
    """One copied artifact and its source/backup hashes."""

    category: str
    source_path: str
    backup_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable payload."""

        return asdict(self)


def should_incremental_sync(request_count: int, every_n_requests: int | None) -> bool:
    """Return whether an incremental sync should run at request_count."""

    if every_n_requests is None:
        return False
    if request_count < 0:
        msg = "request_count must be >= 0"
        raise ValueError(msg)
    return request_count > 0 and request_count % every_n_requests == 0


def _backup_path_for(source_path: Path, *, repo_root: Path, config: ArtifactSyncConfig) -> Path:
    try:
        relative_path = source_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        relative_path = Path(source_path.name)
    return Path(config.backup_root) / config.run_id / relative_path


def build_artifact_specs(
    *,
    raw_jsonl: str | Path,
    manifest: str | Path,
    telemetry: str | Path | None = None,
    processed_reports: list[str | Path] | None = None,
    logs: list[str | Path] | None = None,
) -> list[ArtifactSpec]:
    """Build common artifact specs for long-run backup."""

    specs = [
        ArtifactSpec(path=str(raw_jsonl), category="raw_jsonl", required=True),
        ArtifactSpec(path=str(manifest), category="manifest", required=True),
    ]
    if telemetry is not None:
        specs.append(ArtifactSpec(path=str(telemetry), category="telemetry", required=False))
    for report_path in processed_reports or []:
        specs.append(
            ArtifactSpec(path=str(report_path), category="processed_report", required=False)
        )
    for log_path in logs or []:
        specs.append(ArtifactSpec(path=str(log_path), category="log", required=False))
    return specs


def sync_artifacts(
    *,
    specs: list[ArtifactSpec],
    config: ArtifactSyncConfig,
    event: str,
    repo_root: str | Path = ".",
) -> dict[str, object]:
    """Copy configured artifacts into the local backup root."""

    root = Path(repo_root)
    synced: list[SyncedArtifact] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for spec in specs:
        source_path = Path(spec.path)
        if not source_path.is_absolute():
            source_path = root / source_path
        if not source_path.exists():
            target = missing_required if spec.required else missing_optional
            target.append(spec.path)
            continue
        backup_path = _backup_path_for(source_path, repo_root=root, config=config)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, backup_path)
        synced.append(
            SyncedArtifact(
                category=spec.category,
                source_path=str(source_path),
                backup_path=str(backup_path),
                size_bytes=backup_path.stat().st_size,
                sha256=file_sha256(backup_path),
            )
        )
    return {
        "run_id": config.run_id,
        "provider": config.provider,
        "backup_root": config.backup_root,
        "event": event,
        "synced_at": utc_now(),
        "synced_artifacts": [artifact.to_dict() for artifact in synced],
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "success": not missing_required,
    }


def _manifest_row_status_matches(
    *,
    manifest_payload: dict[str, Any],
    raw_rows: list[dict[str, Any]],
) -> bool:
    status = str(manifest_payload.get("status") or "")
    expected = manifest_payload.get("expected_count")
    completed = int(manifest_payload.get("completed_count") or 0)
    failed = int(manifest_payload.get("failed_count") or manifest_payload.get("error_count") or 0)
    observed = completed + failed
    if len(raw_rows) != observed:
        return False
    if status == "completed" and expected is not None:
        return observed == int(expected)
    if status == "partial" and expected is not None:
        return observed < int(expected)
    return status in {"initialized", "running", "partial", "completed", "failed", "planned"}


def verify_backup(
    *,
    specs: list[ArtifactSpec],
    config: ArtifactSyncConfig,
    repo_root: str | Path = ".",
) -> dict[str, object]:
    """Verify local backup completeness, hashes, and manifest row accounting."""

    root = Path(repo_root)
    checks: list[dict[str, object]] = []
    manifest_backup: Path | None = None
    raw_backup: Path | None = None
    for spec in specs:
        source_path = Path(spec.path)
        if not source_path.is_absolute():
            source_path = root / source_path
        backup_path = _backup_path_for(source_path, repo_root=root, config=config)
        source_exists = source_path.exists()
        backup_exists = backup_path.exists()
        size_ok = (not spec.non_zero_required) or (backup_exists and backup_path.stat().st_size > 0)
        hash_ok = (
            source_exists
            and backup_exists
            and source_path.is_file()
            and backup_path.is_file()
            and file_sha256(source_path) == file_sha256(backup_path)
        )
        passed = backup_exists and size_ok and hash_ok
        if spec.required:
            checks.append(
                {
                    "name": f"{spec.category}:{spec.path}",
                    "required": True,
                    "passed": passed,
                    "backup_path": str(backup_path),
                    "source_exists": source_exists,
                    "backup_exists": backup_exists,
                    "size_ok": size_ok,
                    "hash_ok": hash_ok,
                }
            )
        else:
            checks.append(
                {
                    "name": f"{spec.category}:{spec.path}",
                    "required": False,
                    "passed": (not source_exists and not backup_exists) or passed,
                    "backup_path": str(backup_path),
                    "source_exists": source_exists,
                    "backup_exists": backup_exists,
                    "size_ok": size_ok,
                    "hash_ok": hash_ok if source_exists else True,
                }
            )
        if spec.category == "manifest" and backup_exists:
            manifest_backup = backup_path
        if spec.category == "raw_jsonl" and backup_exists:
            raw_backup = backup_path

    manifest_status_matches = False
    if manifest_backup is not None and raw_backup is not None:
        manifest_payload = json.loads(manifest_backup.read_text(encoding="utf-8"))
        if isinstance(manifest_payload, dict):
            manifest_status_matches = _manifest_row_status_matches(
                manifest_payload=manifest_payload,
                raw_rows=read_jsonl_rows(raw_backup),
            )
    checks.append(
        {
            "name": "manifest_status_matches_completed_rows",
            "required": True,
            "passed": manifest_status_matches,
            "backup_path": str(manifest_backup) if manifest_backup is not None else None,
        }
    )
    passed_count = sum(1 for check in checks if check["passed"])
    completeness_score = passed_count / len(checks) if checks else 0.0
    return {
        "run_id": config.run_id,
        "backup_root": config.backup_root,
        "verified_at": utc_now(),
        "checks": checks,
        "passed_count": passed_count,
        "check_count": len(checks),
        "backup_completeness_score": round(completeness_score, 6),
        "passed": all(bool(check["passed"]) for check in checks if check["required"]),
    }
