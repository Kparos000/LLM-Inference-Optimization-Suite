"""Serving-profile registry for constrained GPU stability experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import load_project_config, load_yaml_file
from inference_bench.runtime_registry import select_runtime_for_model

ServingProfileStatus = Literal["ready", "unstable_observed", "planned", "deprecated"]

DEFAULT_SERVING_PROFILES_PATH = "configs/serving_profiles.yaml"


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field_name} must be an integer >= 0"
        raise ValueError(msg)
    return value


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        msg = f"{field_name} must be boolean"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class ServingProfile:
    """One concrete serving configuration for a model/runtime/hardware pairing."""

    profile_id: str
    status: ServingProfileStatus
    engine: str
    model_alias: str
    model_id: str
    hardware: str
    backend_type: str
    provider: str
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    enforce_eager: bool
    disable_custom_all_reduce: bool
    dtype: str
    trust_remote_code: bool
    smoke_tested: bool
    live_run_allowed: bool
    health_check_every_n_requests: int
    restart_on_fatal_engine_error: bool
    max_serving_restarts: int
    peak_vram_safe_threshold_mb: int
    notes: str

    def __post_init__(self) -> None:
        for field_name in (
            "profile_id",
            "status",
            "engine",
            "model_alias",
            "model_id",
            "hardware",
            "backend_type",
            "provider",
            "dtype",
            "notes",
        ):
            _non_empty(getattr(self, field_name), field_name)
        if self.status not in {"ready", "unstable_observed", "planned", "deprecated"}:
            msg = "status must be ready, unstable_observed, planned, or deprecated"
            raise ValueError(msg)
        if not 0.0 < float(self.gpu_memory_utilization) <= 1.0:
            msg = "gpu_memory_utilization must be in (0, 1]"
            raise ValueError(msg)
        _positive_int(self.max_model_len, "max_model_len")
        _positive_int(self.max_num_seqs, "max_num_seqs")
        _positive_int(self.max_num_batched_tokens, "max_num_batched_tokens")
        _positive_int(self.health_check_every_n_requests, "health_check_every_n_requests")
        _non_negative_int(self.max_serving_restarts, "max_serving_restarts")
        _positive_int(self.peak_vram_safe_threshold_mb, "peak_vram_safe_threshold_mb")
        _bool(self.enforce_eager, "enforce_eager")
        _bool(self.disable_custom_all_reduce, "disable_custom_all_reduce")
        _bool(self.trust_remote_code, "trust_remote_code")
        _bool(self.smoke_tested, "smoke_tested")
        _bool(self.live_run_allowed, "live_run_allowed")
        _bool(self.restart_on_fatal_engine_error, "restart_on_fatal_engine_error")
        if self.status == "planned" and self.live_run_allowed:
            msg = "planned serving profiles cannot allow live runs"
            raise ValueError(msg)
        if self.status in {"unstable_observed", "deprecated"} and self.live_run_allowed:
            msg = f"{self.status} serving profiles cannot allow live runs"
            raise ValueError(msg)
        if self.engine == "vllm" and self.disable_custom_all_reduce is False:
            # Valid for old profiles, but the B7R1 safe profile must opt out.
            return

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable profile."""

        return asdict(self)

    def vllm_server_args(self) -> list[str]:
        """Return vLLM OpenAI-server arguments represented by this profile."""

        if self.engine != "vllm":
            msg = "only vLLM serving arguments are implemented"
            raise ValueError(msg)
        args = [
            "--model",
            self.model_id,
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.disable_custom_all_reduce:
            args.append("--disable-custom-all-reduce")
        if self.trust_remote_code:
            args.append("--trust-remote-code")
        return args


def load_serving_profiles(
    path: str | Path = DEFAULT_SERVING_PROFILES_PATH,
) -> dict[str, ServingProfile]:
    """Load and validate serving profiles."""

    loaded = load_yaml_file(path)
    profiles: dict[str, ServingProfile] = {}
    for key, value in loaded.items():
        if not isinstance(value, dict):
            msg = f"Serving profile '{key}' must be a mapping"
            raise ValueError(msg)
        try:
            profile = ServingProfile(**cast(dict[str, Any], value))
        except (TypeError, ValueError) as exc:
            msg = f"Invalid serving profile '{key}': {exc}"
            raise ValueError(msg) from exc
        if key != profile.profile_id:
            msg = f"Serving profile key '{key}' must match profile_id '{profile.profile_id}'"
            raise ValueError(msg)
        validate_serving_profile(profile)
        profiles[key] = profile
    return profiles


def validate_serving_profile(
    profile: ServingProfile,
    *,
    models_path: str | Path = "configs/models.yaml",
    runtime_registry_path: str | Path = "configs/runtime_engines.yaml",
) -> None:
    """Validate profile model/runtime compatibility without starting a server."""

    project = load_project_config(models_path=models_path)
    model = project.resolve_model_config(profile.model_alias)
    if model.model_id != profile.model_id:
        msg = (
            f"Serving profile '{profile.profile_id}' model_id {profile.model_id!r} "
            f"does not match alias '{profile.model_alias}' ({model.model_id!r})"
        )
        raise ValueError(msg)
    live_run = profile.status == "ready" and profile.live_run_allowed
    selection = select_runtime_for_model(
        model_alias=profile.model_alias,
        runtime=profile.engine,
        hardware_type=profile.hardware,
        backend_route="openai_compatible_vllm" if profile.engine == "vllm" else None,
        live_run=live_run,
        models_path=models_path,
        registry_path=runtime_registry_path,
    )
    if selection.backend_type != profile.backend_type:
        msg = "serving profile backend_type does not match runtime selection"
        raise ValueError(msg)
    if live_run and not selection.live_run_allowed:
        msg = f"serving profile '{profile.profile_id}' is not live-run compatible"
        raise ValueError(msg)


def select_serving_profile(
    profile_id: str,
    *,
    path: str | Path = DEFAULT_SERVING_PROFILES_PATH,
    live_run: bool = True,
) -> ServingProfile:
    """Select one serving profile and enforce live-run gating."""

    profiles = load_serving_profiles(path)
    if profile_id not in profiles:
        msg = f"Unknown serving profile '{profile_id}'"
        raise ValueError(msg)
    profile = profiles[profile_id]
    if live_run and (profile.status != "ready" or not profile.live_run_allowed):
        msg = f"Serving profile '{profile_id}' is not live-run ready"
        raise ValueError(msg)
    return profile
