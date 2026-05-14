import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import inference_bench.cli as cli
import inference_bench.runners.openai_load_runner as load_runner
from inference_bench.cli import app
from inference_bench.output_records import GenerationRecord
from inference_bench.runners.openai_load_runner import run_openai_compatible_load_benchmark
from inference_bench.schema import BenchmarkResult, WorkloadItem


def _write_workload(path: Path, prompt_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for prompt_id in prompt_ids:
            file.write(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "workload_name": "short_chat",
                        "prompt": f"Summarize status for {prompt_id}.",
                        "metadata": {"synthetic": "true"},
                    }
                )
                + "\n"
            )


def _result_for_item(item: WorkloadItem) -> BenchmarkResult:
    return BenchmarkResult(
        run_id="load-run",
        timestamp_utc="2026-05-14T00:00:00Z",
        backend="vllm",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        optimization="vllm_baseline",
        workload_name=item.workload_name,
        prompt_id=item.prompt_id,
        input_tokens=4,
        output_tokens=6,
        ttft_ms=10.0,
        tpot_ms=2.0,
        end_to_end_latency_ms=50.0,
        throughput_tokens_per_second=100.0,
        peak_memory_mb=None,
        estimated_cost_usd=0.0,
        success=True,
        error_message=None,
    )


def _generation_for_result(result: BenchmarkResult, item: WorkloadItem) -> GenerationRecord:
    return GenerationRecord(
        run_id=result.run_id,
        timestamp_utc=result.timestamp_utc,
        prompt_id=result.prompt_id,
        workload_name=result.workload_name,
        backend=result.backend,
        model_name=result.model_name,
        optimization=result.optimization,
        prompt=item.prompt,
        generated_text=f"Generated response for {item.prompt_id}.",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        ttft_ms=result.ttft_ms,
        tpot_ms=result.tpot_ms,
        end_to_end_latency_ms=result.end_to_end_latency_ms,
        throughput_tokens_per_second=result.throughput_tokens_per_second,
        peak_memory_mb=result.peak_memory_mb,
        estimated_cost_usd=result.estimated_cost_usd,
        success=result.success,
        error_message=result.error_message,
    )


async def _fake_run_load_benchmark_async(
    *,
    workload_items: list[WorkloadItem],
    config: load_runner.OpenAIConcurrencyConfig,
    run_id: str,
    backend: str,
    optimization: str,
) -> tuple[list[BenchmarkResult], list[GenerationRecord]]:
    del config, run_id, backend, optimization
    results = [_result_for_item(item) for item in workload_items]
    generation_records = [
        _generation_for_result(result, item)
        for result, item in zip(results, workload_items, strict=True)
    ]
    return results, generation_records


def _patch_no_server_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(load_runner, "require_openai_dependency", lambda: None)
    monkeypatch.setattr(
        load_runner,
        "_run_load_benchmark_async",
        _fake_run_load_benchmark_async,
    )


def test_chunk_size_validation_rejects_non_positive_value(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    _write_workload(workload_path, ["prompt-1"])

    with pytest.raises(ValueError, match="chunk_size"):
        run_openai_compatible_load_benchmark(
            workload_path=workload_path,
            output_path=tmp_path / "results.csv",
            generation_output_path=None,
            model="Qwen/Qwen2.5-0.5B-Instruct",
            chunk_size=0,
        )


def test_chunked_run_writes_checkpoint_csv_jsonl_and_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_no_server_calls(monkeypatch)
    workload_path = tmp_path / "workload.jsonl"
    output_path = tmp_path / "results.csv"
    generation_path = tmp_path / "generations.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    metadata_path = tmp_path / "metadata.json"
    log_path = tmp_path / "run.log"
    _write_workload(workload_path, ["prompt-1", "prompt-2", "prompt-3"])

    results = run_openai_compatible_load_benchmark(
        workload_path=workload_path,
        output_path=output_path,
        generation_output_path=generation_path,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        run_metadata_path=metadata_path,
        chunk_size=2,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )

    assert [result.prompt_id for result in results] == ["prompt-1", "prompt-2", "prompt-3"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["run_id"] == "openai-load-run"
    assert checkpoint["chunk_size"] == 2
    assert checkpoint["total_prompts"] == 3
    assert checkpoint["completed_prompt_ids"] == ["prompt-1", "prompt-2", "prompt-3"]
    assert checkpoint["success_count"] == 3
    assert checkpoint["failure_count"] == 0
    assert checkpoint["run_metadata_path"] == str(metadata_path)
    assert checkpoint["log_path"] == str(log_path)

    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["prompt_id"] for row in rows] == ["prompt-1", "prompt-2", "prompt-3"]
    assert output_path.read_text(encoding="utf-8").count("run_id,timestamp_utc") == 1
    assert len(generation_path.read_text(encoding="utf-8").splitlines()) == 3
    assert "processed=3/3" in log_path.read_text(encoding="utf-8")
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["total_requests"] == 3


def test_resume_skips_completed_prompt_ids_and_appends_without_duplicate_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_no_server_calls(monkeypatch)
    workload_path = tmp_path / "workload.jsonl"
    output_path = tmp_path / "results.csv"
    generation_path = tmp_path / "generations.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    _write_workload(workload_path, ["prompt-1", "prompt-2"])

    run_openai_compatible_load_benchmark(
        workload_path=workload_path,
        output_path=output_path,
        generation_output_path=generation_path,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        chunk_size=1,
        checkpoint_path=checkpoint_path,
        max_prompts=1,
    )

    resumed_results = run_openai_compatible_load_benchmark(
        workload_path=workload_path,
        output_path=output_path,
        generation_output_path=generation_path,
        model="Qwen/Qwen2.5-0.5B-Instruct",
        chunk_size=1,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert [result.prompt_id for result in resumed_results] == ["prompt-2"]
    with output_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["prompt_id"] for row in rows] == ["prompt-1", "prompt-2"]
    assert output_path.read_text(encoding="utf-8").count("run_id,timestamp_utc") == 1
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["completed_prompt_ids"] == ["prompt-1", "prompt-2"]


def test_cli_openai_load_run_accepts_resumable_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_openai_compatible_load_benchmark(**kwargs: object) -> list[BenchmarkResult]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        cli,
        "run_openai_compatible_load_benchmark",
        fake_run_openai_compatible_load_benchmark,
    )

    result = CliRunner().invoke(
        app,
        [
            "openai-load-run",
            "--workload-path",
            "data/prompts/smoke_workload.jsonl",
            "--output-path",
            str(tmp_path / "results.csv"),
            "--generation-output-path",
            str(tmp_path / "generations.jsonl"),
            "--chunk-size",
            "100",
            "--checkpoint-path",
            str(tmp_path / "checkpoint.json"),
            "--resume",
            "--log-path",
            str(tmp_path / "run.log"),
            "--progress-interval",
            "50",
        ],
    )

    assert result.exit_code == 0
    assert captured["chunk_size"] == 100
    assert captured["checkpoint_path"] == str(tmp_path / "checkpoint.json")
    assert captured["resume"] is True
    assert captured["log_path"] == str(tmp_path / "run.log")
    assert captured["progress_interval"] == 50
    assert "Checkpoint path:" in result.output
    assert "Log path:" in result.output
