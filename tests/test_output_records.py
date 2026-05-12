import json
from pathlib import Path

import pytest

from inference_bench.output_records import (
    GenerationRecord,
    write_generation_records_jsonl,
)


def _record() -> GenerationRecord:
    return GenerationRecord(
        run_id="run-1",
        prompt_id="prompt-1",
        workload_name="smoke",
        backend="huggingface",
        model_name="model",
        optimization="hf_baseline",
        prompt="Write a short response.",
        generated_text="Short response.",
        input_tokens=5,
        output_tokens=2,
        success=True,
    )


def test_valid_generation_record() -> None:
    record = _record()

    assert record.to_dict()["prompt_id"] == "prompt-1"
    assert record.generated_text == "Short response."


def test_generation_record_rejects_empty_prompt_id_or_prompt() -> None:
    with pytest.raises(ValueError, match="prompt_id"):
        GenerationRecord(
            run_id="run-1",
            prompt_id="",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text="Text",
            input_tokens=1,
            output_tokens=1,
            success=True,
        )

    with pytest.raises(ValueError, match="prompt"):
        GenerationRecord(
            run_id="run-1",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="",
            generated_text="Text",
            input_tokens=1,
            output_tokens=1,
            success=True,
        )


def test_success_rejects_missing_generated_text() -> None:
    with pytest.raises(ValueError, match="generated_text"):
        GenerationRecord(
            run_id="run-1",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text=None,
            input_tokens=1,
            output_tokens=1,
            success=True,
        )


def test_failure_rejects_missing_error_message() -> None:
    with pytest.raises(ValueError, match="error_message"):
        GenerationRecord(
            run_id="run-1",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text=None,
            input_tokens=1,
            output_tokens=0,
            success=False,
        )


def test_write_generation_records_jsonl_writes_records(tmp_path: Path) -> None:
    output_path = write_generation_records_jsonl([_record()], tmp_path / "records.jsonl")

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["generated_text"] == "Short response."


def test_write_generation_records_jsonl_creates_empty_file(tmp_path: Path) -> None:
    output_path = write_generation_records_jsonl([], tmp_path / "records.jsonl")

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""
