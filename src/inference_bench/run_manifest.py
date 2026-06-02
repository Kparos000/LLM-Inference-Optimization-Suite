"""Run manifest support for Phase 4 execution plumbing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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
        if self.status not in {"planned", "running", "completed", "failed"}:
            msg = "status must be one of: planned, running, completed, failed"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest payload."""

        return asdict(self)


def write_run_manifest(manifest: RunManifest, output_path: str | Path) -> Path:
    """Write a run manifest JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
