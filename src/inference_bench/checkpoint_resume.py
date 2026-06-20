"""Checkpoint and resume helpers for long inference runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from inference_bench.run_manifest import utc_now


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _prompt_id(row: dict[str, Any], field_name: str = "prompt_id") -> str:
    raw = row.get(field_name)
    if not isinstance(raw, str) or not raw.strip():
        msg = f"row missing non-empty {field_name}"
        raise ValueError(msg)
    return raw


@dataclass(frozen=True)
class ResumePlan:
    """Prompt-level resume plan for one run."""

    run_id: str
    expected_count: int
    completed_prompt_ids: tuple[str, ...]
    pending_prompt_ids: tuple[str, ...]
    duplicate_prompt_ids: tuple[str, ...]
    skipped_count: int
    pending_count: int
    resume_from_partial_raw: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable resume plan."""

        return asdict(self)


@dataclass(frozen=True)
class CheckpointState:
    """Durable checkpoint state for completed and failed prompt IDs."""

    run_id: str
    expected_count: int
    completed_prompt_ids: tuple[str, ...]
    failed_prompt_ids: tuple[str, ...] = ()
    status: str = "initialized"
    updated_at: str = ""
    raw_output_path: str | None = None
    failed_output_path: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.run_id, "run_id")
        if self.expected_count < 0:
            msg = "expected_count must be >= 0"
            raise ValueError(msg)
        if self.status not in {"initialized", "running", "partial", "completed", "failed"}:
            msg = "checkpoint status is invalid"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable checkpoint payload."""

        payload = asdict(self)
        payload["completed_count"] = len(self.completed_prompt_ids)
        payload["failed_count"] = len(self.failed_prompt_ids)
        return payload


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL objects from a path, returning an empty list if absent."""

    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                msg = f"JSONL row in {jsonl_path} must be an object"
                raise ValueError(msg)
            rows.append(payload)
    return rows


def write_jsonl_rows(path: str | Path, rows: list[dict[str, Any]], *, append: bool = True) -> Path:
    """Write JSONL rows to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output_path


def prompt_ids_from_rows(
    rows: list[dict[str, Any]], *, field_name: str = "prompt_id"
) -> tuple[str, ...]:
    """Return prompt IDs from JSONL-style rows."""

    return tuple(_prompt_id(row, field_name) for row in rows)


def duplicate_prompt_ids(
    rows: list[dict[str, Any]],
    *,
    field_name: str = "prompt_id",
) -> tuple[str, ...]:
    """Return prompt IDs appearing more than once, preserving sorted order."""

    seen: set[str] = set()
    duplicates: set[str] = set()
    for prompt_id in prompt_ids_from_rows(rows, field_name=field_name):
        if prompt_id in seen:
            duplicates.add(prompt_id)
        seen.add(prompt_id)
    return tuple(sorted(duplicates))


def append_unique_jsonl_rows(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    allow_duplicate_prompt_ids: bool = False,
    field_name: str = "prompt_id",
) -> Path:
    """Append rows while preventing duplicate prompt IDs by default."""

    existing_rows = read_jsonl_rows(path)
    existing_ids = set(prompt_ids_from_rows(existing_rows, field_name=field_name))
    new_ids = list(prompt_ids_from_rows(rows, field_name=field_name))
    duplicate_new = sorted({prompt_id for prompt_id in new_ids if new_ids.count(prompt_id) > 1})
    duplicate_existing = sorted(prompt_id for prompt_id in new_ids if prompt_id in existing_ids)
    if not allow_duplicate_prompt_ids and (duplicate_new or duplicate_existing):
        msg = (
            "duplicate prompt_id rows are not allowed; "
            f"existing={duplicate_existing}; new={duplicate_new}"
        )
        raise ValueError(msg)
    return write_jsonl_rows(path, rows, append=True)


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Load checkpoint state from JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "checkpoint must be a JSON object"
        raise ValueError(msg)
    return CheckpointState(
        run_id=str(payload["run_id"]),
        expected_count=int(payload["expected_count"]),
        completed_prompt_ids=tuple(str(item) for item in payload.get("completed_prompt_ids", [])),
        failed_prompt_ids=tuple(str(item) for item in payload.get("failed_prompt_ids", [])),
        status=str(payload.get("status", "partial")),
        updated_at=str(payload.get("updated_at", "")),
        raw_output_path=(
            str(payload["raw_output_path"]) if payload.get("raw_output_path") is not None else None
        ),
        failed_output_path=(
            str(payload["failed_output_path"])
            if payload.get("failed_output_path") is not None
            else None
        ),
    )


def write_checkpoint(state: CheckpointState, path: str | Path) -> Path:
    """Write checkpoint state atomically where possible."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def build_resume_plan(
    *,
    run_id: str,
    prompt_rows: list[dict[str, Any]],
    checkpoint_path: str | Path | None = None,
    partial_raw_jsonl_path: str | Path | None = None,
    allow_duplicate_prompt_ids: bool = False,
    field_name: str = "prompt_id",
) -> ResumePlan:
    """Build a deterministic resume plan from checkpoint and partial raw JSONL."""

    _non_empty(run_id, "run_id")
    source_duplicates = duplicate_prompt_ids(prompt_rows, field_name=field_name)
    if source_duplicates and not allow_duplicate_prompt_ids:
        msg = f"duplicate prompt IDs in workload: {', '.join(source_duplicates)}"
        raise ValueError(msg)

    completed_ids: set[str] = set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.run_id != run_id:
            msg = f"checkpoint run_id {checkpoint.run_id} does not match {run_id}"
            raise ValueError(msg)
        completed_ids.update(checkpoint.completed_prompt_ids)
        completed_ids.update(checkpoint.failed_prompt_ids)

    resume_from_partial_raw = False
    raw_duplicates: tuple[str, ...] = ()
    if partial_raw_jsonl_path is not None and Path(partial_raw_jsonl_path).exists():
        raw_rows = read_jsonl_rows(partial_raw_jsonl_path)
        raw_duplicates = duplicate_prompt_ids(raw_rows, field_name=field_name)
        if raw_duplicates and not allow_duplicate_prompt_ids:
            msg = f"duplicate prompt IDs in partial raw JSONL: {', '.join(raw_duplicates)}"
            raise ValueError(msg)
        completed_ids.update(prompt_ids_from_rows(raw_rows, field_name=field_name))
        resume_from_partial_raw = bool(raw_rows)

    prompt_ids = prompt_ids_from_rows(prompt_rows, field_name=field_name)
    pending_ids = tuple(prompt_id for prompt_id in prompt_ids if prompt_id not in completed_ids)
    completed_known = tuple(
        sorted(prompt_id for prompt_id in completed_ids if prompt_id in set(prompt_ids))
    )
    return ResumePlan(
        run_id=run_id,
        expected_count=len(prompt_rows),
        completed_prompt_ids=completed_known,
        pending_prompt_ids=pending_ids,
        duplicate_prompt_ids=tuple(sorted(set(source_duplicates) | set(raw_duplicates))),
        skipped_count=len(completed_known),
        pending_count=len(pending_ids),
        resume_from_partial_raw=resume_from_partial_raw,
    )


def checkpoint_from_rows(
    *,
    run_id: str,
    expected_count: int,
    result_rows: list[dict[str, Any]],
    raw_output_path: str | Path | None = None,
    failed_output_path: str | Path | None = None,
    field_name: str = "prompt_id",
) -> CheckpointState:
    """Create checkpoint state from result rows."""

    completed: list[str] = []
    failed: list[str] = []
    for row in result_rows:
        prompt_id = _prompt_id(row, field_name)
        success = row.get("success")
        if success is False or str(success).lower() == "false":
            failed.append(prompt_id)
        else:
            completed.append(prompt_id)
    observed = len(completed) + len(failed)
    status = "completed" if observed >= expected_count else "partial"
    return CheckpointState(
        run_id=run_id,
        expected_count=expected_count,
        completed_prompt_ids=tuple(sorted(set(completed))),
        failed_prompt_ids=tuple(sorted(set(failed))),
        status=status,
        updated_at=utc_now(),
        raw_output_path=str(raw_output_path) if raw_output_path is not None else None,
        failed_output_path=str(failed_output_path) if failed_output_path is not None else None,
    )


def build_resume_report(plan: ResumePlan, checkpoint_path: str | Path | None) -> dict[str, object]:
    """Return a clear resume report for logs and processed reports."""

    return {
        "run_id": plan.run_id,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "expected_count": plan.expected_count,
        "completed_count": len(plan.completed_prompt_ids),
        "pending_count": plan.pending_count,
        "skipped_count": plan.skipped_count,
        "resume_from_partial_raw": plan.resume_from_partial_raw,
        "duplicate_prompt_ids": list(plan.duplicate_prompt_ids),
        "pending_prompt_ids": list(plan.pending_prompt_ids),
    }
