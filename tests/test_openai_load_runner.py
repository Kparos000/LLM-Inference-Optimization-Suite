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
    require_openai_dependency,
)
from inference_bench.schema import BenchmarkResult


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
            "--concurrency",
            "3",
            "--max-prompts",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert captured["concurrency"] == 3
    assert "Concurrency: 3" in result.output


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
