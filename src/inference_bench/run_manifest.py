"""Run manifest support for Phase 4 execution plumbing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def current_git_commit(repo_root: str | Path = ".") -> str:
    """Return the current git commit hash, or ``unknown`` if unavailable."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be an integer >= 0"
        raise ValueError(msg)


def _validate_optional_non_negative_int(value: int | None, field_name: str) -> None:
    if value is not None:
        _validate_non_negative_int(value, field_name)


def _validate_optional_non_empty_string(value: str | None, field_name: str) -> None:
    if value is not None:
        _validate_non_empty_string(value, field_name)


def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 hash for a local artifact."""

    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_existing_paths(paths: list[str | Path]) -> str:
    """Return one deterministic hash over existing path names and bytes."""

    digest = sha256()
    for raw_path in sorted(str(path) for path in paths):
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        digest.update(raw_path.replace("\\", "/").encode("utf-8"))
        digest.update(file_sha256(path).encode("utf-8"))
    return digest.hexdigest()


VALID_MANIFEST_STATUSES = {
    "planned",
    "initialized",
    "running",
    "partial",
    "completed",
    "failed",
}


@dataclass(frozen=True)
class RunManifest:
    """Metadata describing one benchmark or smoke-test run."""

    run_id: str
    timestamp_utc: str
    backend: str
    model_alias: str
    model_id: str
    memory_mode: str
    split: str
    ablation_mode: str
    input_workload_path: str
    output_path: str
    max_records: int | None
    git_commit: str
    command: str
    status: str
    start_time: str
    end_time: str | None
    error_count: int
    telemetry_path: str | None = None
    telemetry_summary_path: str | None = None
    profiling_enabled: bool = False
    profiling_mode: str = "disabled"
    profiler_output_path: str | None = None
    profiling_metadata: dict[str, object] | None = None
    config_id: str | None = None
    vertical: str | None = None
    runtime: str | None = None
    engine: str | None = None
    backend_type: str | None = None
    hardware: str | None = None
    provider: str | None = None
    concurrency: int | None = None
    traffic_profile: str | None = None
    prompt_count: int | None = None
    dataset_workload_hash: str | None = None
    config_hash: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    completed_count: int | None = None
    failed_count: int | None = None
    expected_count: int | None = None
    artifact_paths: dict[str, str] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "timestamp_utc",
            "backend",
            "model_alias",
            "model_id",
            "memory_mode",
            "split",
            "ablation_mode",
            "input_workload_path",
            "output_path",
            "git_commit",
            "command",
            "status",
            "start_time",
        ):
            _validate_non_empty_string(str(getattr(self, field_name)), field_name)
        if self.max_records is not None:
            _validate_non_negative_int(self.max_records, "max_records")
        _validate_non_negative_int(self.error_count, "error_count")
        for field_name in ("telemetry_path", "telemetry_summary_path"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_non_empty_string(value, field_name)
        if not isinstance(self.profiling_enabled, bool):
            msg = "profiling_enabled must be boolean"
            raise ValueError(msg)
        _validate_non_empty_string(self.profiling_mode, "profiling_mode")
        if self.profiling_mode not in {"disabled", "pytorch", "nsys", "ncu"}:
            msg = "profiling_mode must be disabled, pytorch, nsys, or ncu"
            raise ValueError(msg)
        if self.profiling_enabled and self.profiling_mode == "disabled":
            msg = "profiling_mode must not be disabled when profiling_enabled is true"
            raise ValueError(msg)
        if self.profiler_output_path is not None:
            _validate_non_empty_string(self.profiler_output_path, "profiler_output_path")
        if self.profiling_enabled and self.profiler_output_path is None:
            msg = "profiler_output_path is required when profiling is enabled"
            raise ValueError(msg)
        if self.status not in VALID_MANIFEST_STATUSES:
            msg = "status must be one of: " + ", ".join(sorted(VALID_MANIFEST_STATUSES))
            raise ValueError(msg)
        for field_name in (
            "config_id",
            "vertical",
            "runtime",
            "engine",
            "backend_type",
            "hardware",
            "provider",
            "traffic_profile",
            "dataset_workload_hash",
            "config_hash",
            "started_at",
            "updated_at",
            "completed_at",
        ):
            _validate_optional_non_empty_string(
                getattr(self, field_name),
                field_name,
            )
        for field_name in (
            "concurrency",
            "prompt_count",
            "completed_count",
            "failed_count",
            "expected_count",
        ):
            _validate_optional_non_negative_int(getattr(self, field_name), field_name)
        if self.artifact_paths is not None:
            for key, value in self.artifact_paths.items():
                _validate_non_empty_string(key, "artifact_paths key")
                _validate_non_empty_string(value, f"artifact_paths[{key}]")
        if self.status == "completed" and self.completed_at is None and self.end_time is None:
            msg = "completed manifests require completed_at or end_time"
            raise ValueError(msg)
        if self.status == "completed" and self.expected_count is not None:
            observed = (self.completed_count or 0) + (self.failed_count or 0)
            if observed < self.expected_count:
                msg = "completed manifests cannot be partial"
                raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest payload."""

        payload: dict[str, Any] = asdict(self)
        payload["started_at"] = self.started_at or self.start_time
        payload["updated_at"] = self.updated_at or self.timestamp_utc
        payload["completed_at"] = self.completed_at or self.end_time
        return payload


def write_run_manifest(manifest: RunManifest, output_path: str | Path) -> Path:
    """Write a run manifest JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_run_manifest(path: str | Path) -> dict[str, Any]:
    """Read a run manifest payload from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "Run manifest must be a JSON object"
        raise ValueError(msg)
    return payload
