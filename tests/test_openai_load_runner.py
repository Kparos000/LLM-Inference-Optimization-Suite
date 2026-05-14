import json
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

import inference_bench.cli as cli
import inference_bench.runners.openai_load_runner as load_runner
from inference_bench.cli import app
from inference_bench.runners.openai_load_runner import (
    OpenAIConcurrencyConfig,
    build_run_metadata,
    require_openai_dependency,
    write_run_metadata,
)
from inference_bench.schema import BenchmarkResult


def _result(
    *,
    prompt_id: str,
    input_tokens: int = 10,
    output_tokens: int = 20,
    success: bool = True,
) -> BenchmarkResult:
    return BenchmarkResult(
        run_id="load-run",
        timestamp_utc="2026-05-14T00:00:00Z",
        backend="vllm",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        optimization="vllm_baseline",
        workload_name="short_chat",
        prompt_id=prompt_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=10.0,
        tpot_ms=2.0,
        end_to_end_latency_ms=100.0,
        throughput_tokens_per_second=300.0,
        peak_memory_mb=None,
        estimated_cost_usd=0.0,
        success=success,
        error_message=None if success else "request failed",
    )


def test_openai_concurrency_config_valid_case() -> None:
    config = OpenAIConcurrencyConfig(concurrency=4)

    assert config.base_url == "http://localhost:8000/v1"
    assert config.api_key == "EMPTY"
    assert config.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.concurrency == 4
    assert config.stream is True


def test_openai_concurrency_config_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        OpenAIConcurrencyConfig(concurrency=0)


def test_openai_concurrency_config_rejects_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAIConcurrencyConfig(base_url="")


def test_openai_concurrency_config_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAIConcurrencyConfig(model="")


def test_openai_concurrency_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAIConcurrencyConfig(timeout_seconds=0)


def test_missing_openai_dependency_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> ModuleType:
        if name == "openai":
            raise ImportError(name)
        return import_module(name)

    monkeypatch.setattr(load_runner, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match=r"\.\[openai,dev\]"):
        require_openai_dependency()


def test_cli_openai_load_run_accepts_concurrency_option(
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
            str(tmp_path / "load_results.csv"),
            "--generation-output-path",
            str(tmp_path / "load_generations.jsonl"),
            "--run-metadata-path",
            str(tmp_path / "load_metadata.json"),
            "--concurrency",
            "3",
            "--max-prompts",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["concurrency"] == 3
    assert captured["run_metadata_path"] == str(tmp_path / "load_metadata.json")
    assert "Concurrency: 3" in result.output
    assert "Run metadata path:" in result.output


def test_run_metadata_writer_writes_expected_keys(tmp_path: Path) -> None:
    metadata = build_run_metadata(
        results=[
            _result(prompt_id="prompt-1", input_tokens=5, output_tokens=8),
            _result(prompt_id="prompt-2", input_tokens=7, output_tokens=12),
            _result(prompt_id="prompt-3", input_tokens=11, output_tokens=0, success=False),
        ],
        run_id="load-run",
        workload_path="data/prompts/scaled/short_chat_100.jsonl",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="vllm",
        optimization="vllm_baseline",
        concurrency=4,
        max_prompts=100,
        max_new_tokens=64,
        stream=True,
        started_at_utc="2026-05-14T00:00:00+00:00",
        ended_at_utc="2026-05-14T00:00:10+00:00",
        wall_clock_seconds=10.0,
    )
    output_path = tmp_path / "metadata.json"

    written_path = write_run_metadata(metadata, output_path)

    assert written_path == output_path
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "load-run"
    assert loaded["workload_path"] == "data/prompts/scaled/short_chat_100.jsonl"
    assert loaded["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert loaded["backend"] == "vllm"
    assert loaded["optimization"] == "vllm_baseline"
    assert loaded["concurrency"] == 4
    assert loaded["max_prompts"] == 100
    assert loaded["max_new_tokens"] == 64
    assert loaded["stream"] is True
    assert loaded["started_at_utc"] == "2026-05-14T00:00:00+00:00"
    assert loaded["ended_at_utc"] == "2026-05-14T00:00:10+00:00"
    assert loaded["wall_clock_seconds"] == 10.0
    assert loaded["total_requests"] == 3
    assert loaded["success_count"] == 2
    assert loaded["failure_count"] == 1
    assert loaded["total_input_tokens"] == 23
    assert loaded["total_output_tokens"] == 20
    assert loaded["aggregate_requests_per_second"] == 0.3
    assert loaded["aggregate_output_tokens_per_second"] == 2.0


def test_cli_openai_load_run_surfaces_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> ModuleType:
        if name == "openai":
            raise ImportError(name)
        return import_module(name)

    monkeypatch.setattr(load_runner, "import_module", fake_import_module)

    result = CliRunner().invoke(
        app,
        [
            "openai-load-run",
            "--workload-path",
            "data/prompts/smoke_workload.jsonl",
            "--output-path",
            "results/raw/test_openai_load_results.csv",
            "--generation-output-path",
            "results/raw/test_openai_load_generations.jsonl",
            "--max-prompts",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "Missing optional OpenAI dependency" in result.output
    assert 'python -m pip install -e ".[openai,dev]"' in result.output
