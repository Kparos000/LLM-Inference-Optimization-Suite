"""YAML-backed benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)


def _validate_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        msg = f"{field_name} must be > 0"
        raise ValueError(msg)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a benchmark model entry."""

    name: str
    provider: str
    model_id: str
    parameter_count: int | None = None
    default_dtype: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.provider, "provider")
        _validate_non_empty_string(self.model_id, "model_id")
        _validate_optional_positive_int(self.parameter_count, "parameter_count")


@dataclass(frozen=True)
class WorkloadConfig:
    """Configuration for a benchmark workload entry."""

    name: str
    path: str
    description: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.path, "path")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a reproducible benchmark experiment."""

    name: str
    backend: str
    model: str
    optimization: str
    workload: str
    output_path: str
    max_prompts: int | None = None
    concurrency: int = 1

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.backend, "backend")
        _validate_non_empty_string(self.model, "model")
        _validate_non_empty_string(self.optimization, "optimization")
        _validate_non_empty_string(self.workload, "workload")
        _validate_non_empty_string(self.output_path, "output_path")
        _validate_optional_positive_int(self.max_prompts, "max_prompts")
        if self.concurrency <= 0:
            msg = "concurrency must be > 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class ProjectConfig:
    """Complete project configuration across models, workloads, and experiments."""

    models: dict[str, ModelConfig]
    workloads: dict[str, WorkloadConfig]
    experiments: dict[str, ExperimentConfig]

    def __post_init__(self) -> None:
        for experiment_key, experiment in self.experiments.items():
            if experiment.model not in self.models:
                msg = f"Experiment '{experiment_key}' references unknown model '{experiment.model}'"
                raise ValueError(msg)
            if experiment.workload not in self.workloads:
                msg = (
                    f"Experiment '{experiment_key}' references unknown workload "
                    f"'{experiment.workload}'"
                )
                raise ValueError(msg)


def load_yaml_file(path: str | Path) -> dict[str, object]:
    """Load a YAML mapping from disk."""

    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

    try:
        with yaml_path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in {yaml_path}: {exc}"
        raise ValueError(msg) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        msg = f"Expected top-level mapping in {yaml_path}"
        raise ValueError(msg)
    if not all(isinstance(key, str) for key in loaded):
        msg = f"Expected string keys in {yaml_path}"
        raise ValueError(msg)
    return cast(dict[str, object], loaded)


def _load_config_mapping(path: str | Path) -> dict[str, dict[str, Any]]:
    raw_config = load_yaml_file(path)
    mapping: dict[str, dict[str, Any]] = {}
    for key, value in raw_config.items():
        if not isinstance(value, dict):
            msg = f"Config entry '{key}' in {path} must be a mapping"
            raise ValueError(msg)
        mapping[key] = cast(dict[str, Any], value)
    return mapping


def load_models_config(path: str | Path) -> dict[str, ModelConfig]:
    """Load model configurations from YAML."""

    models: dict[str, ModelConfig] = {}
    for key, value in _load_config_mapping(path).items():
        try:
            models[key] = ModelConfig(**value)
        except TypeError as exc:
            msg = f"Invalid model config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid model config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return models


def load_workloads_config(path: str | Path) -> dict[str, WorkloadConfig]:
    """Load workload configurations from YAML."""

    workloads: dict[str, WorkloadConfig] = {}
    for key, value in _load_config_mapping(path).items():
        try:
            workloads[key] = WorkloadConfig(**value)
        except TypeError as exc:
            msg = f"Invalid workload config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid workload config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return workloads


def load_experiments_config(path: str | Path) -> dict[str, ExperimentConfig]:
    """Load experiment configurations from YAML."""

    experiments: dict[str, ExperimentConfig] = {}
    for key, value in _load_config_mapping(path).items():
        try:
            experiments[key] = ExperimentConfig(**value)
        except TypeError as exc:
            msg = f"Invalid experiment config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid experiment config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return experiments


def load_project_config(
    models_path: str | Path = "configs/models.yaml",
    workloads_path: str | Path = "configs/workloads.yaml",
    experiments_path: str | Path = "configs/experiments.yaml",
) -> ProjectConfig:
    """Load and validate the complete benchmark project configuration."""

    try:
        return ProjectConfig(
            models=load_models_config(models_path),
            workloads=load_workloads_config(workloads_path),
            experiments=load_experiments_config(experiments_path),
        )
    except ValueError as exc:
        msg = f"Invalid project config: {exc}"
        raise ValueError(msg) from exc
