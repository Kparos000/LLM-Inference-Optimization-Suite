from importlib import import_module
from types import ModuleType

import pytest
from typer.testing import CliRunner

import inference_bench.runners.openai_compatible_runner as openai_runner
from inference_bench.cli import app
from inference_bench.runners.openai_compatible_runner import (
    OpenAICompatibleRunnerConfig,
    require_openai_dependency,
)


def test_openai_compatible_runner_config_valid_case() -> None:
    config = OpenAICompatibleRunnerConfig()

    assert config.base_url == "http://localhost:8000/v1"
    assert config.api_key == "EMPTY"
    assert config.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.stream is True


def test_openai_compatible_runner_config_rejects_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleRunnerConfig(base_url="")


def test_openai_compatible_runner_config_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleRunnerConfig(model="")


def test_openai_compatible_runner_config_rejects_non_positive_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        OpenAICompatibleRunnerConfig(max_new_tokens=0)


def test_openai_compatible_runner_config_rejects_negative_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        OpenAICompatibleRunnerConfig(temperature=-0.1)


def test_openai_compatible_runner_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAICompatibleRunnerConfig(timeout_seconds=0)


def test_missing_openai_dependency_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> ModuleType:
        if name == "openai":
            raise ImportError(name)
        return import_module(name)

    monkeypatch.setattr(openai_runner, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match=r"\.\[openai,dev\]"):
        require_openai_dependency()


def test_cli_openai_compatible_run_surfaces_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> ModuleType:
        if name == "openai":
            raise ImportError(name)
        return import_module(name)

    monkeypatch.setattr(openai_runner, "import_module", fake_import_module)

    result = CliRunner().invoke(
        app,
        [
            "openai-compatible-run",
            "--workload-path",
            "data/prompts/smoke_workload.jsonl",
            "--output-path",
            "results/raw/test_openai_compatible_results.csv",
            "--generation-output-path",
            "results/raw/test_openai_compatible_generations.jsonl",
            "--max-prompts",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "Missing optional OpenAI dependency" in result.output
    assert 'python -m pip install -e ".[openai,dev]"' in result.output
