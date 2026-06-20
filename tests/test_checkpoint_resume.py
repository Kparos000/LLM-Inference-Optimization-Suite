from __future__ import annotations

from pathlib import Path

import pytest

from inference_bench.checkpoint_resume import (
    append_unique_jsonl_rows,
    build_resume_plan,
    checkpoint_from_rows,
    duplicate_prompt_ids,
    load_checkpoint,
    read_jsonl_rows,
    write_checkpoint,
    write_jsonl_rows,
)


def _prompt_rows(count: int = 20) -> list[dict[str, object]]:
    return [{"prompt_id": f"p{index:03d}", "question": f"q{index}"} for index in range(count)]


def _result_rows(prompt_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"prompt_id": row["prompt_id"], "success": True, "output": "ok"} for row in prompt_rows]


def test_resume_plan_uses_checkpoint_and_partial_raw_jsonl(tmp_path: Path) -> None:
    prompts = _prompt_rows()
    raw_path = tmp_path / "raw.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    first_results = _result_rows(prompts[:10])
    write_jsonl_rows(raw_path, first_results, append=False)
    checkpoint = checkpoint_from_rows(
        run_id="run-1",
        expected_count=20,
        result_rows=first_results[:5],
        raw_output_path=raw_path,
    )
    write_checkpoint(checkpoint, checkpoint_path)

    plan = build_resume_plan(
        run_id="run-1",
        prompt_rows=prompts,
        checkpoint_path=checkpoint_path,
        partial_raw_jsonl_path=raw_path,
    )

    assert plan.skipped_count == 10
    assert plan.pending_count == 10
    assert plan.pending_prompt_ids[0] == "p010"
    assert plan.resume_from_partial_raw is True


def test_append_unique_rows_prevents_duplicate_prompt_ids(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    append_unique_jsonl_rows(raw_path, [{"prompt_id": "p1", "success": True}])

    with pytest.raises(ValueError, match="duplicate prompt_id rows are not allowed"):
        append_unique_jsonl_rows(raw_path, [{"prompt_id": "p1", "success": True}])


def test_duplicate_prompt_ids_can_be_explicitly_allowed(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    append_unique_jsonl_rows(raw_path, [{"prompt_id": "p1", "success": True}])
    append_unique_jsonl_rows(
        raw_path,
        [{"prompt_id": "p1", "success": True}],
        allow_duplicate_prompt_ids=True,
    )

    assert len(read_jsonl_rows(raw_path)) == 2


def test_checkpoint_tracks_failed_prompt_ids(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint = checkpoint_from_rows(
        run_id="run-1",
        expected_count=2,
        result_rows=[
            {"prompt_id": "p1", "success": True},
            {"prompt_id": "p2", "success": False},
        ],
    )
    write_checkpoint(checkpoint, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path)

    assert loaded.status == "completed"
    assert loaded.completed_prompt_ids == ("p1",)
    assert loaded.failed_prompt_ids == ("p2",)


def test_duplicate_detector_reports_sorted_ids() -> None:
    assert duplicate_prompt_ids(
        [
            {"prompt_id": "p2"},
            {"prompt_id": "p1"},
            {"prompt_id": "p2"},
            {"prompt_id": "p1"},
        ]
    ) == ("p1", "p2")
