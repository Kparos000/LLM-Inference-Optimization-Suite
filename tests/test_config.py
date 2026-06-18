from pathlib import Path

import pytest
from typer.testing import CliRunner

from inference_bench.cli import app
from inference_bench.config import (
    ExperimentConfig,
    ModelConfig,
    ProjectConfig,
    WorkloadConfig,
    load_project_config,
    load_yaml_file,
)


def test_loads_default_project_config() -> None:
    config = load_project_config()

    assert "qwen2_5_0_5b_instruct" in config.models
    assert "qwen2_5_1_5b_instruct" in config.models
    assert "qwen2_5_3b_instruct" in config.models
    assert "qwen2_5_7b_instruct" in config.models
    assert "qwen2_5_32b_instruct" in config.models
    assert "ministral_3b_2512_api" in config.models
    assert "llama_3_2_3b_instruct_api" in config.models
    assert "llama_3_1_8b_instruct_api" in config.models
    assert "mistral_small_3_2_24b_instruct_api" in config.models
    assert "future_large_model_placeholder" in config.models
    assert config.workloads["smoke"].path == "data/prompts/smoke_workload.jsonl"
    assert (
        config.workloads["structured_output_smoke"].path
        == "data/prompts/structured_output_smoke.jsonl"
    )
    assert config.workloads["short_chat"].path == "data/prompts/short_chat.jsonl"
    assert config.workloads["code_helpdesk"].path == "data/prompts/code_helpdesk.jsonl"
    assert config.workloads["long_context"].path == "data/prompts/long_context.jsonl"
    assert config.workloads["shared_prefix"].path == "data/prompts/shared_prefix.jsonl"
    assert config.experiments["mock_smoke"].backend == "mock"
    assert config.experiments["mock_structured_output_smoke"].workload == "structured_output_smoke"
    assert config.experiments["mock_short_chat"].workload == "short_chat"
    assert config.experiments["mock_code_helpdesk"].workload == "code_helpdesk"
    assert config.experiments["mock_long_context"].workload == "long_context"
    assert config.experiments["mock_shared_prefix"].workload == "shared_prefix"


def test_model_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        ModelConfig(name="Model", provider="provider", model_id="")


def test_workload_config_rejects_empty_path() -> None:
    with pytest.raises(ValueError, match="path"):
        WorkloadConfig(name="smoke", path="")


def test_experiment_config_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        ExperimentConfig(
            name="experiment",
            backend="mock",
            model="model",
            optimization="none",
            workload="smoke",
            output_path="results/raw/results.csv",
            concurrency=0,
        )


def test_project_config_rejects_unknown_model_reference() -> None:
    workload = WorkloadConfig(name="smoke", path="data/prompts/smoke_workload.jsonl")
    experiment = ExperimentConfig(
        name="experiment",
        backend="mock",
        model="missing-model",
        optimization="none",
        workload="smoke",
        output_path="results/raw/results.csv",
    )

    with pytest.raises(ValueError, match="unknown model"):
        ProjectConfig(
            models={},
            workloads={"smoke": workload},
            experiments={"experiment": experiment},
        )


def test_missing_config_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml_file(tmp_path / "missing.yaml")


def test_cli_validate_config_succeeds_with_default_config() -> None:
    result = CliRunner().invoke(app, ["validate-config"])

    assert result.exit_code == 0
    assert "Configuration valid" in result.output
    assert "Models loaded: 10" in result.output
    assert "Model aliases loaded: 12" in result.output
    assert "Workloads loaded: 6" in result.output
    assert "Experiments loaded: 6" in result.output
    assert "mock_smoke" in result.output
    assert "mock_structured_output_smoke" in result.output
    assert "mock_short_chat" in result.output
    assert "mock_code_helpdesk" in result.output
    assert "mock_long_context" in result.output
    assert "mock_shared_prefix" in result.output
