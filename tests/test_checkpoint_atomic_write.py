from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from inference_bench.checkpoint_resume import CheckpointState, write_checkpoint


def test_checkpoint_write_falls_back_after_permission_errors(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = CheckpointState(
        run_id="run-1",
        expected_count=2,
        completed_prompt_ids=("p1",),
        failed_prompt_ids=(),
        status="partial",
    )
    checkpoint_path = tmp_path / "checkpoint.json"

    def always_locked(self: Path, target: str | Path) -> Path:
        raise PermissionError("simulated lock")

    monkeypatch.setattr(Path, "replace", always_locked)

    written = write_checkpoint(
        state,
        checkpoint_path,
        max_replace_attempts=2,
        retry_base_seconds=0.0,
    )

    assert written != checkpoint_path
    assert written.exists()
    assert written.name.startswith("checkpoint.json.fallback-")
    warning_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".warning.json")
    assert warning_path.exists()
    assert "primary_checkpoint_replace_failed" in warning_path.read_text(encoding="utf-8")


def test_checkpoint_write_retries_then_replaces(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    state = CheckpointState(
        run_id="run-1",
        expected_count=1,
        completed_prompt_ids=("p1",),
        status="completed",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    original_replace = Path.replace
    calls = {"count": 0}

    def locked_once(self: Path, target: str | Path) -> Path:
        calls["count"] += 1
        if calls["count"] == 1:
            raise PermissionError("simulated transient lock")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", locked_once)

    written = write_checkpoint(
        state,
        checkpoint_path,
        max_replace_attempts=3,
        retry_base_seconds=0.0,
    )

    assert written == checkpoint_path
    assert checkpoint_path.exists()
    assert calls["count"] == 2
