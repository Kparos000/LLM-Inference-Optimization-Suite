"""Production runtime and engine compatibility registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import ModelConfig, load_project_config, load_yaml_file

RuntimeStatus = Literal["ready", "dry_run_ready", "planned", "deprecated"]
BackendType = Literal["local_compute", "api_provider", "self_hosted_gpu"]

DEFAULT_RUNTIME_REGISTRY_PATH = "configs/runtime_engines.yaml"
API_EXECUTION_TARGETS = {"openrouter_api", "hf_inference_provider_api"}
SELF_HOSTED_BACKEND_TYPES = {"local_compute", "self_hosted_gpu"}


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{field_name} must be boolean"
        raise ValueError(msg)
    return value


def _string_list(values: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        msg = f"{field_name} must be a list"
        raise ValueError(msg)
    parsed: list[str] = []
    for value in values:
        parsed.append(_non_empty(value, f"{field_name} entry"))
    return tuple(parsed)


@dataclass(frozen=True)
class RuntimeEngineConfig:
    """One production runtime/engine capability entry."""

    runtime: str
    engine: str
    backend_type: BackendType
    status: RuntimeStatus
    planned_engine: bool
    smoke_tested: bool
    live_run_supported: bool
    backend_routes: tuple[str, ...]
    supported_model_providers: tuple[str, ...]
    supported_execution_targets: tuple[str, ...]
    supported_hardware_types: tuple[str, ...]
    notes: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.runtime, "runtime")
        _non_empty(self.engine, "engine")
        if self.backend_type not in {"local_compute", "api_provider", "self_hosted_gpu"}:
            msg = "backend_type must be local_compute, api_provider, or self_hosted_gpu"
            raise ValueError(msg)
        if self.status not in {"ready", "dry_run_ready", "planned", "deprecated"}:
            msg = "status must be ready, dry_run_ready, planned, or deprecated"
            raise ValueError(msg)
        _boolean(self.planned_engine, "planned_engine")
        _boolean(self.smoke_tested, "smoke_tested")
        _boolean(self.live_run_supported, "live_run_supported")
        if not self.backend_routes:
            msg = "backend_routes must not be empty"
            raise ValueError(msg)
        if not self.supported_model_providers:
            msg = "supported_model_providers must not be empty"
            raise ValueError(msg)
        if not self.supported_execution_targets:
            msg = "supported_execution_targets must not be empty"
            raise ValueError(msg)
        if not self.supported_hardware_types:
            msg = "supported_hardware_types must not be empty"
            raise ValueError(msg)
        if self.planned_engine and self.status != "planned":
            msg = "planned_engine entries must use status planned"
            raise ValueError(msg)
        if self.status == "planned" and self.live_run_supported:
            msg = "planned engines must not support live runs"
            raise ValueError(msg)
        if self.status == "planned" and self.smoke_tested:
            msg = "planned engines must not be marked smoke_tested"
            raise ValueError(msg)
        if self.backend_type == "api_provider" and set(self.supported_hardware_types) != {
            "provider_managed"
        }:
            msg = "api_provider runtimes must use provider_managed hardware"
            raise ValueError(msg)
        if self.backend_type in SELF_HOSTED_BACKEND_TYPES and "provider_managed" in set(
            self.supported_hardware_types
        ):
            msg = "self-hosted runtimes must not use provider_managed hardware"
            raise ValueError(msg)
        if self.notes is not None:
            _non_empty(self.notes, "notes")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable runtime entry."""

        return asdict(self)


@dataclass(frozen=True)
class RuntimeSelection:
    """Resolved runtime for one model/backend/hardware request."""

    model_alias: str
    canonical_model_key: str
    model_id: str
    model_provider: str
    execution_target: str
    runtime: str
    engine: str
    backend_type: BackendType
    backend_route: str
    hardware_type: str
    provider: str
    status: RuntimeStatus
    smoke_tested: bool
    live_run_allowed: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable runtime selection."""

        return asdict(self)


def load_runtime_registry(
    path: str | Path = DEFAULT_RUNTIME_REGISTRY_PATH,
) -> dict[str, RuntimeEngineConfig]:
    """Load and validate production runtime/engine capabilities."""

    loaded = load_yaml_file(path)
    registry: dict[str, RuntimeEngineConfig] = {}
    for key, value in loaded.items():
        if not isinstance(value, dict):
            msg = f"Runtime registry entry '{key}' must be a mapping"
            raise ValueError(msg)
        payload = dict(cast(dict[str, Any], value))
        for list_key in (
            "backend_routes",
            "supported_model_providers",
            "supported_execution_targets",
            "supported_hardware_types",
        ):
            payload[list_key] = _string_list(payload.get(list_key), list_key)
        try:
            engine = RuntimeEngineConfig(**payload)
        except (TypeError, ValueError) as exc:
            msg = f"Invalid runtime registry entry '{key}': {exc}"
            raise ValueError(msg) from exc
        if key != engine.runtime:
            msg = f"Runtime registry key '{key}' must match runtime '{engine.runtime}'"
            raise ValueError(msg)
        registry[key] = engine
    return registry


def _route_for_model(model: ModelConfig, runtime: RuntimeEngineConfig) -> str:
    routes = set(runtime.backend_routes)
    if model.execution_target == "openrouter_api":
        return "openrouter"
    if model.execution_target == "hf_inference_provider_api":
        return "hf_inference_provider"
    if "openai_compatible_vllm" in routes:
        return "openai_compatible_vllm"
    if "sglang_openai_compatible" in routes:
        return "sglang_openai_compatible"
    return runtime.backend_routes[0]


def _backend_tokens(runtime: RuntimeEngineConfig, backend_route: str) -> set[str]:
    tokens = {runtime.engine, backend_route, *runtime.backend_routes}
    return tokens | {f"{token}_optional" for token in tokens}


def runtime_supports_model(
    *,
    model: ModelConfig,
    runtime: RuntimeEngineConfig,
    backend_route: str | None = None,
    hardware_type: str | None = None,
) -> bool:
    """Return whether a runtime is compatible with one model and route."""

    execution_target = model.execution_target or ""
    if model.provider not in set(runtime.supported_model_providers):
        return False
    if execution_target not in set(runtime.supported_execution_targets):
        return False
    route = backend_route or _route_for_model(model, runtime)
    if route not in set(runtime.backend_routes):
        return False
    if hardware_type is not None and hardware_type not in set(runtime.supported_hardware_types):
        return False
    allowed_backend_tokens = set(model.allowed_backends)
    if allowed_backend_tokens and allowed_backend_tokens.isdisjoint(
        _backend_tokens(runtime, route)
    ):
        return False
    if execution_target in API_EXECUTION_TARGETS:
        return runtime.backend_type == "api_provider"
    if runtime.backend_type == "api_provider":
        return False
    return True


def select_runtime_for_model(
    *,
    model_alias: str,
    runtime: str,
    hardware_type: str | None = None,
    backend_route: str | None = None,
    live_run: bool = True,
    models_path: str | Path = "configs/models.yaml",
    registry_path: str | Path = DEFAULT_RUNTIME_REGISTRY_PATH,
) -> RuntimeSelection:
    """Resolve and validate one model/runtime pairing.

    This is a configuration guard only. It does not run inference, start servers,
    contact APIs, or allocate GPU resources.
    """

    project = load_project_config(models_path=models_path)
    registry = load_runtime_registry(registry_path)
    if runtime not in registry:
        msg = f"Unknown runtime '{runtime}'"
        raise ValueError(msg)
    runtime_config = registry[runtime]
    canonical_key = project.resolve_model_key(model_alias)
    model = project.models[canonical_key]
    route = backend_route or _route_for_model(model, runtime_config)
    hardware = hardware_type or (
        "provider_managed"
        if runtime_config.backend_type == "api_provider"
        else runtime_config.supported_hardware_types[0]
    )

    if not runtime_supports_model(
        model=model,
        runtime=runtime_config,
        backend_route=route,
        hardware_type=hardware,
    ):
        msg = (
            f"Runtime '{runtime}' is not compatible with model alias '{model_alias}' "
            f"on route '{route}' and hardware '{hardware}'"
        )
        raise ValueError(msg)
    if live_run and not runtime_config.live_run_supported:
        msg = f"Runtime '{runtime}' is not selectable for live runs"
        raise ValueError(msg)
    if live_run and runtime_config.planned_engine and not runtime_config.smoke_tested:
        msg = f"Runtime '{runtime}' is planned and has not been smoke-tested"
        raise ValueError(msg)
    if live_run and runtime_config.status not in {"ready", "dry_run_ready"}:
        msg = f"Runtime '{runtime}' status is {runtime_config.status}, not live-ready"
        raise ValueError(msg)

    provider = model.provider
    if runtime_config.backend_type == "api_provider":
        provider = route
    return RuntimeSelection(
        model_alias=model_alias,
        canonical_model_key=canonical_key,
        model_id=model.model_id,
        model_provider=model.provider,
        execution_target=model.execution_target or "",
        runtime=runtime_config.runtime,
        engine=runtime_config.engine,
        backend_type=runtime_config.backend_type,
        backend_route=route,
        hardware_type=hardware,
        provider=provider,
        status=runtime_config.status,
        smoke_tested=runtime_config.smoke_tested,
        live_run_allowed=runtime_config.live_run_supported
        and runtime_config.status in {"ready", "dry_run_ready"}
        and not (runtime_config.planned_engine and not runtime_config.smoke_tested),
    )


def build_engine_compatibility_rows(
    *,
    models_path: str | Path = "configs/models.yaml",
    registry_path: str | Path = DEFAULT_RUNTIME_REGISTRY_PATH,
) -> list[dict[str, object]]:
    """Return compatibility rows for public and deprecated model aliases."""

    project = load_project_config(models_path=models_path)
    registry = load_runtime_registry(registry_path)
    rows: list[dict[str, object]] = []
    active_aliases = [alias for alias in project.model_aliases if alias.startswith("model")]
    for alias in active_aliases:
        canonical_key = project.resolve_model_key(alias)
        model = project.models[canonical_key]
        for runtime_key, runtime_config in registry.items():
            route = _route_for_model(model, runtime_config)
            compatible = runtime_supports_model(
                model=model,
                runtime=runtime_config,
                backend_route=route,
            )
            rows.append(
                {
                    "model_alias": alias,
                    "canonical_model_key": canonical_key,
                    "model_provider": model.provider,
                    "execution_target": model.execution_target,
                    "runtime": runtime_key,
                    "engine": runtime_config.engine,
                    "backend_type": runtime_config.backend_type,
                    "backend_route": route,
                    "status": runtime_config.status,
                    "planned_engine": runtime_config.planned_engine,
                    "smoke_tested": runtime_config.smoke_tested,
                    "compatible": compatible,
                    "live_run_allowed": (
                        compatible
                        and runtime_config.live_run_supported
                        and runtime_config.status in {"ready", "dry_run_ready"}
                        and not (runtime_config.planned_engine and not runtime_config.smoke_tested)
                    ),
                }
            )
    return rows
