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
        timestamp_utc="2026-05-12T04:00:00Z",
        prompt_id="prompt-1",
        workload_name="smoke",
        backend="huggingface",
        model_name="model",
        optimization="hf_baseline",
        prompt="Write a short response.",
        generated_text="Short response.",
        input_tokens=5,
        output_tokens=2,
        ttft_ms=20.0,
        tpot_ms=10.0,
        end_to_end_latency_ms=40.0,
        throughput_tokens_per_second=175.0,
        peak_memory_mb=None,
        estimated_cost_usd=0.0,
        success=True,
    )


def test_valid_generation_record() -> None:
    record = _record()

    assert record.to_dict()["prompt_id"] == "prompt-1"
    assert record.generated_text == "Short response."
    assert record.generation_contract_valid is False
    assert record.evidence_ids == []


def test_generation_record_rejects_empty_prompt_id_or_prompt() -> None:
    with pytest.raises(ValueError, match="prompt_id"):
        GenerationRecord(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            prompt_id="",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text="Text",
            input_tokens=1,
            output_tokens=1,
            ttft_ms=None,
            tpot_ms=None,
            end_to_end_latency_ms=1.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=0.0,
            success=True,
        )

    with pytest.raises(ValueError, match="prompt"):
        GenerationRecord(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="",
            generated_text="Text",
            input_tokens=1,
            output_tokens=1,
            ttft_ms=None,
            tpot_ms=None,
            end_to_end_latency_ms=1.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=0.0,
            success=True,
        )


def test_success_rejects_missing_generated_text() -> None:
    with pytest.raises(ValueError, match="generated_text"):
        GenerationRecord(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text=None,
            input_tokens=1,
            output_tokens=1,
            ttft_ms=None,
            tpot_ms=None,
            end_to_end_latency_ms=1.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=0.0,
            success=True,
        )


def test_failure_rejects_missing_error_message() -> None:
    with pytest.raises(ValueError, match="error_message"):
        GenerationRecord(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text=None,
            input_tokens=1,
            output_tokens=0,
            ttft_ms=None,
            tpot_ms=None,
            end_to_end_latency_ms=1.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=0.0,
            success=False,
        )


def test_generation_record_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="end_to_end_latency_ms"):
        GenerationRecord(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text="Text",
            input_tokens=1,
            output_tokens=1,
            ttft_ms=None,
            tpot_ms=None,
            end_to_end_latency_ms=-1.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=0.0,
            success=True,
        )


def test_generation_record_rejects_negative_optional_metric_values() -> None:
    with pytest.raises(ValueError, match="ttft_ms"):
        GenerationRecord(
            run_id="run-1",
            timestamp_utc="2026-05-12T04:00:00Z",
            prompt_id="prompt-1",
            workload_name="smoke",
            backend="huggingface",
            model_name="model",
            optimization="hf_baseline",
            prompt="Prompt",
            generated_text="Text",
            input_tokens=1,
            output_tokens=1,
            ttft_ms=-0.1,
            tpot_ms=None,
            end_to_end_latency_ms=1.0,
            throughput_tokens_per_second=None,
            peak_memory_mb=None,
            estimated_cost_usd=0.0,
            success=True,
        )


def test_write_generation_records_jsonl_writes_records(tmp_path: Path) -> None:
    output_path = write_generation_records_jsonl([_record()], tmp_path / "records.jsonl")

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["generated_text"] == "Short response."
    assert row["timestamp_utc"] == "2026-05-12T04:00:00Z"
    assert row["end_to_end_latency_ms"] == 40.0
    assert row["throughput_tokens_per_second"] == 175.0


def test_write_generation_records_jsonl_creates_empty_file(tmp_path: Path) -> None:
    output_path = write_generation_records_jsonl([], tmp_path / "records.jsonl")

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == ""
