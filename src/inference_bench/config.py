"""YAML-backed benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

MODEL_ALIASES_KEY = "model_aliases"
DEPRECATED_MODEL_ALIASES_KEY = "deprecated_model_aliases"
DEFAULT_MEMORY_MODES_PATH = "configs/memory_modes.yaml"
DEFAULT_VECTOR_STORES_PATH = "configs/vector_stores.yaml"


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not value.strip():
        msg = f"{field_name} must not be empty"
        raise ValueError(msg)


def _validate_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None and value <= 0:
        msg = f"{field_name} must be > 0"
        raise ValueError(msg)


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be >= 0"
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
    execution_target: str | None = None
    allowed_backends: list[str] = field(default_factory=list)
    access_type: str | None = None
    requires_hf_token: bool = False
    requires_license_acceptance: bool = False
    intended_role: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.provider, "provider")
        _validate_non_empty_string(self.model_id, "model_id")
        _validate_optional_positive_int(self.parameter_count, "parameter_count")
        if self.execution_target is not None:
            _validate_non_empty_string(self.execution_target, "execution_target")
        if self.access_type is not None:
            _validate_non_empty_string(self.access_type, "access_type")
        if self.intended_role is not None:
            _validate_non_empty_string(self.intended_role, "intended_role")
        if not isinstance(self.allowed_backends, list):
            msg = "allowed_backends must be a list"
            raise ValueError(msg)
        for backend in self.allowed_backends:
            _validate_non_empty_string(backend, "allowed_backends entry")
        if not isinstance(self.requires_hf_token, bool):
            msg = "requires_hf_token must be boolean"
            raise ValueError(msg)
        if not isinstance(self.requires_license_acceptance, bool):
            msg = "requires_license_acceptance must be boolean"
            raise ValueError(msg)


@dataclass(frozen=True)
class MemoryModeConfig:
    """Configuration for a Phase 3 memory/context mode."""

    description: str
    requires_retrieval: bool
    retrieval_type: str
    top_k: int
    requires_compression: bool
    requires_agentic_workflow: bool
    max_context_tokens: int
    expected_stage: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.description, "description")
        _validate_non_empty_string(self.retrieval_type, "retrieval_type")
        _validate_non_empty_string(self.expected_stage, "expected_stage")
        _validate_non_negative_int(self.top_k, "top_k")
        _validate_non_negative_int(self.max_context_tokens, "max_context_tokens")

        for field_name in (
            "requires_retrieval",
            "requires_compression",
            "requires_agentic_workflow",
        ):
            if not isinstance(getattr(self, field_name), bool):
                msg = f"{field_name} must be boolean"
                raise ValueError(msg)


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for a local vector store used by retrieval workloads."""

    provider: str
    mode: str
    storage_path: str
    collection_prefix: str
    distance: str
    embedding_backend: str
    embedding_model: str
    batch_size: int = 64

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.provider, "provider")
        _validate_non_empty_string(self.mode, "mode")
        _validate_non_empty_string(self.storage_path, "storage_path")
        _validate_non_empty_string(self.collection_prefix, "collection_prefix")
        _validate_non_empty_string(self.distance, "distance")
        _validate_non_empty_string(self.embedding_backend, "embedding_backend")
        _validate_non_empty_string(self.embedding_model, "embedding_model")
        if self.provider != "qdrant":
            msg = "vector store provider must be 'qdrant'"
            raise ValueError(msg)
        if self.mode != "local":
            msg = "only local vector store mode is supported"
            raise ValueError(msg)
        if self.distance.lower() != "cosine":
            msg = "only cosine vector distance is supported"
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be > 0"
            raise ValueError(msg)


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
    model_aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for alias, target in self.model_aliases.items():
            _validate_non_empty_string(alias, "model alias")
            _validate_non_empty_string(target, f"model alias '{alias}' target")
            if alias in self.models:
                msg = f"Model alias '{alias}' conflicts with a canonical model key"
                raise ValueError(msg)
            if target not in self.models:
                msg = f"Model alias '{alias}' references unknown model '{target}'"
                raise ValueError(msg)

        for experiment_key, experiment in self.experiments.items():
            if experiment.model not in self.models and experiment.model not in self.model_aliases:
                msg = f"Experiment '{experiment_key}' references unknown model '{experiment.model}'"
                raise ValueError(msg)
            if experiment.workload not in self.workloads:
                msg = (
                    f"Experiment '{experiment_key}' references unknown workload "
                    f"'{experiment.workload}'"
                )
                raise ValueError(msg)

    def resolve_model_key(self, model_key_or_alias: str) -> str:
        """Resolve a canonical model key from either an old key or public alias."""

        _validate_non_empty_string(model_key_or_alias, "model_key_or_alias")
        resolved_key = self.model_aliases.get(model_key_or_alias, model_key_or_alias)
        if resolved_key not in self.models:
            msg = f"Unknown model or alias '{model_key_or_alias}'"
            raise KeyError(msg)
        return resolved_key

    def resolve_model_config(self, model_key_or_alias: str) -> ModelConfig:
        """Return the model config for either a canonical key or public alias."""

        return self.models[self.resolve_model_key(model_key_or_alias)]


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

    raw_config = load_yaml_file(path)
    models: dict[str, ModelConfig] = {}
    for key, value in raw_config.items():
        if key in {MODEL_ALIASES_KEY, DEPRECATED_MODEL_ALIASES_KEY}:
            continue
        if not isinstance(value, dict):
            msg = f"Config entry '{key}' in {path} must be a mapping"
            raise ValueError(msg)
        try:
            models[key] = ModelConfig(**cast(dict[str, Any], value))
        except TypeError as exc:
            msg = f"Invalid model config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid model config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return models


def load_model_aliases_config(path: str | Path) -> dict[str, str]:
    """Load model aliases from the models YAML file."""

    raw_config = load_yaml_file(path)
    aliases: dict[str, str] = {}
    for aliases_key in (MODEL_ALIASES_KEY, DEPRECATED_MODEL_ALIASES_KEY):
        raw_aliases = raw_config.get(aliases_key, {})
        if raw_aliases is None:
            continue
        if not isinstance(raw_aliases, dict):
            msg = f"Config entry '{aliases_key}' in {path} must be a mapping"
            raise ValueError(msg)

        for alias, target in raw_aliases.items():
            if not isinstance(alias, str) or not isinstance(target, str):
                msg = f"Model aliases in {path} must contain only string keys and values"
                raise ValueError(msg)
            _validate_non_empty_string(alias, "model alias")
            _validate_non_empty_string(target, f"model alias '{alias}' target")
            if alias in aliases:
                msg = f"Duplicate model alias '{alias}' in {path}"
                raise ValueError(msg)
            aliases[alias] = target
    return aliases


def load_memory_modes_config(
    path: str | Path = DEFAULT_MEMORY_MODES_PATH,
) -> dict[str, MemoryModeConfig]:
    """Load Phase 3 memory-mode configurations from YAML."""

    memory_modes: dict[str, MemoryModeConfig] = {}
    for key, value in _load_config_mapping(path).items():
        try:
            memory_modes[key] = MemoryModeConfig(**value)
        except TypeError as exc:
            msg = f"Invalid memory mode config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid memory mode config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return memory_modes


def resolve_memory_mode(
    memory_mode: str,
    path: str | Path = DEFAULT_MEMORY_MODES_PATH,
) -> MemoryModeConfig:
    """Return one configured memory mode or raise a clear validation error."""

    memory_modes = load_memory_modes_config(path)
    if memory_mode not in memory_modes:
        msg = f"Unknown memory mode '{memory_mode}' in {path}"
        raise ValueError(msg)
    return memory_modes[memory_mode]


def load_vector_stores_config(
    path: str | Path = DEFAULT_VECTOR_STORES_PATH,
) -> dict[str, VectorStoreConfig]:
    """Load vector store configurations from YAML."""

    vector_stores: dict[str, VectorStoreConfig] = {}
    for key, value in _load_config_mapping(path).items():
        try:
            vector_stores[key] = VectorStoreConfig(**value)
        except TypeError as exc:
            msg = f"Invalid vector store config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = f"Invalid vector store config '{key}' in {path}: {exc}"
            raise ValueError(msg) from exc
    return vector_stores


def resolve_vector_store(
    vector_store: str,
    path: str | Path = DEFAULT_VECTOR_STORES_PATH,
) -> VectorStoreConfig:
    """Return one configured vector store or raise a clear validation error."""

    vector_stores = load_vector_stores_config(path)
    if vector_store not in vector_stores:
        msg = f"Unknown vector store '{vector_store}' in {path}"
        raise ValueError(msg)
    return vector_stores[vector_store]


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
        models = load_models_config(models_path)
        model_aliases = load_model_aliases_config(models_path)
        return ProjectConfig(
            models=models,
            workloads=load_workloads_config(workloads_path),
            experiments=load_experiments_config(experiments_path),
            model_aliases=model_aliases,
        )
    except ValueError as exc:
        msg = f"Invalid project config: {exc}"
        raise ValueError(msg) from exc
