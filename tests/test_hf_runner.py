import types

import pytest
from typer.testing import CliRunner

from inference_bench import cli
from inference_bench.cli import app
from inference_bench.runners import hf_runner
from inference_bench.runners.hf_runner import (
    HF_EXTRA_INSTALL_MESSAGE,
    HuggingFaceRunnerConfig,
    require_hf_dependencies,
)


def test_hugging_face_runner_config_valid_case() -> None:
    config = HuggingFaceRunnerConfig(model_id="Qwen/Qwen2.5-0.5B-Instruct")

    assert config.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.device == "auto"
    assert config.max_new_tokens == 64


def test_hugging_face_runner_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        HuggingFaceRunnerConfig(model_id="")


def test_hugging_face_runner_config_rejects_non_positive_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        HuggingFaceRunnerConfig(model_id="model", max_new_tokens=0)


def test_hugging_face_runner_config_rejects_negative_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        HuggingFaceRunnerConfig(model_id="model", temperature=-0.1)


def test_require_hf_dependencies_raises_runtime_error_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_module_stub(module_name: str) -> types.ModuleType:
        if module_name in {"torch", "transformers"}:
            raise ImportError
        return types.ModuleType(module_name)

    monkeypatch.setattr(hf_runner, "import_module", import_module_stub)

    with pytest.raises(
        RuntimeError, match="Missing optional Hugging Face dependencies"
    ) as exc_info:
        require_hf_dependencies()

    assert HF_EXTRA_INSTALL_MESSAGE in str(exc_info.value)


def test_cli_hf_run_surfaces_dependency_error_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_hf_benchmark_stub(*args: object, **kwargs: object) -> object:
        raise RuntimeError(
            f"Missing optional Hugging Face dependencies. {HF_EXTRA_INSTALL_MESSAGE}"
        )

    monkeypatch.setattr(cli, "run_hf_benchmark", run_hf_benchmark_stub)

    result = CliRunner().invoke(app, ["hf-run"])

    assert result.exit_code == 1
    assert "Missing optional Hugging Face dependencies" in result.output
    assert HF_EXTRA_INSTALL_MESSAGE in result.output
