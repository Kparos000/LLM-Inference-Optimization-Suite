"""RunPod calibration profile validation and manifest generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import load_yaml_file
from inference_bench.gpu_price_registry import (
    DEFAULT_GPU_PRICE_REGISTRY_PATH,
    get_gpu_price,
)
from inference_bench.run_manifest import current_git_commit, utc_now
from inference_bench.runtime_registry import select_runtime_for_model

DEFAULT_RUNPOD_CALIBRATION_PROFILES_PATH = "configs/runpod_calibration_profiles.yaml"
SUPPORTED_CALIBRATION_PROMPT_COUNTS = (100, 200)

CalibrationVerdict = Literal[
    "READY_FOR_A100_CALIBRATION",
    "READY_FOR_H100_CALIBRATION",
    "READY_FOR_L40S_CALIBRATION",
    "CALIBRATION_NOT_READY",
]


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _string_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{field_name} must be a non-empty list"
        raise ValueError(msg)
    parsed: list[str] = []
    for item in value:
        parsed.append(_non_empty(item, f"{field_name} entry"))
    return tuple(parsed)


def _int_list(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        msg = f"{field_name} must be a non-empty list"
        raise ValueError(msg)
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            msg = f"{field_name} entries must be positive integers"
            raise ValueError(msg)
        parsed.append(item)
    return tuple(parsed)


def _optional_float(value: object, field_name: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        msg = f"{field_name} must be numeric or null"
        raise ValueError(msg)
    parsed = float(str(value))
    if parsed < 0:
        msg = f"{field_name} must be >= 0 when provided"
        raise ValueError(msg)
    return parsed


@dataclass(frozen=True)
class RunPodCalibrationProfile:
    """One planned RunPod calibration profile."""

    profile_id: str
    gpu_name: str
    provider: str
    vram_gb: float
    hourly_price: float | None
    runtime: str
    engine: str
    hardware_type: str
    backend_type: str
    concurrency_list: tuple[int, ...]
    model_aliases: tuple[str, ...]
    memory_modes: tuple[str, ...]
    prompt_counts: tuple[int, ...]
    traffic_profile: str
    artifact_sync_required: bool
    checkpoint_resume_required: bool
    manifest_required: bool
    notes: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "profile_id",
            "gpu_name",
            "provider",
            "runtime",
            "engine",
            "hardware_type",
            "backend_type",
            "traffic_profile",
        ):
            _non_empty(str(getattr(self, field_name)), field_name)
        if self.provider != "runpod":
            msg = "calibration profile provider must be runpod"
            raise ValueError(msg)
        if self.vram_gb <= 0:
            msg = "vram_gb must be > 0"
            raise ValueError(msg)
        if self.hourly_price is not None and self.hourly_price < 0:
            msg = "hourly_price must be >= 0 when provided"
            raise ValueError(msg)
        if not set(self.prompt_counts).issubset(set(SUPPORTED_CALIBRATION_PROMPT_COUNTS)):
            msg = "calibration prompt counts must be 100 or 200"
            raise ValueError(msg)
        for field_name in (
            "artifact_sync_required",
            "checkpoint_resume_required",
            "manifest_required",
        ):
            if not isinstance(getattr(self, field_name), bool):
                msg = f"{field_name} must be boolean"
                raise ValueError(msg)
        if self.notes is not None:
            _non_empty(self.notes, "notes")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable profile."""

        return asdict(self)


@dataclass(frozen=True)
class CalibrationManifest:
    """Manifest for a planned or measured calibration run."""

    run_id: str
    config_id: str
    git_commit: str
    profile_id: str
    gpu_name: str
    runtime: str
    engine: str
    model_alias: str
    memory_mode: str
    concurrency: int
    traffic_profile: str
    prompt_count: int
    started_at: str
    status: str
    artifact_paths: dict[str, str]
    cost_estimate: dict[str, object]

    def __post_init__(self) -> None:
        for field_name in (
            "run_id",
            "config_id",
            "git_commit",
            "profile_id",
            "gpu_name",
            "runtime",
            "engine",
            "model_alias",
            "memory_mode",
            "traffic_profile",
            "started_at",
            "status",
        ):
            _non_empty(str(getattr(self, field_name)), field_name)
        if self.prompt_count not in SUPPORTED_CALIBRATION_PROMPT_COUNTS:
            msg = "prompt_count must be 100 or 200 for calibration"
            raise ValueError(msg)
        if self.concurrency <= 0:
            msg = "concurrency must be > 0"
            raise ValueError(msg)
        for key, value in self.artifact_paths.items():
            _non_empty(key, "artifact path key")
            _non_empty(value, f"artifact_paths[{key}]")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest."""

        return asdict(self)


def load_runpod_calibration_profiles(
    path: str | Path = DEFAULT_RUNPOD_CALIBRATION_PROFILES_PATH,
) -> dict[str, RunPodCalibrationProfile]:
    """Load planned RunPod calibration profiles."""

    payload = load_yaml_file(path)
    profiles: dict[str, RunPodCalibrationProfile] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            msg = f"Calibration profile '{key}' must be a mapping"
            raise ValueError(msg)
        raw = cast(dict[str, Any], value)
        profile = RunPodCalibrationProfile(
            profile_id=_non_empty(raw.get("profile_id"), f"{key}.profile_id"),
            gpu_name=_non_empty(raw.get("gpu_name"), f"{key}.gpu_name"),
            provider=_non_empty(raw.get("provider"), f"{key}.provider"),
            vram_gb=float(str(raw.get("vram_gb"))),
            hourly_price=_optional_float(raw.get("hourly_price"), f"{key}.hourly_price"),
            runtime=_non_empty(raw.get("runtime"), f"{key}.runtime"),
            engine=_non_empty(raw.get("engine"), f"{key}.engine"),
            hardware_type=_non_empty(raw.get("hardware_type"), f"{key}.hardware_type"),
            backend_type=_non_empty(raw.get("backend_type"), f"{key}.backend_type"),
            concurrency_list=_int_list(raw.get("concurrency_list"), f"{key}.concurrency_list"),
            model_aliases=_string_list(raw.get("model_aliases"), f"{key}.model_aliases"),
            memory_modes=_string_list(raw.get("memory_modes"), f"{key}.memory_modes"),
            prompt_counts=_int_list(raw.get("prompt_counts"), f"{key}.prompt_counts"),
            traffic_profile=_non_empty(raw.get("traffic_profile"), f"{key}.traffic_profile"),
            artifact_sync_required=bool(raw.get("artifact_sync_required")),
            checkpoint_resume_required=bool(raw.get("checkpoint_resume_required")),
            manifest_required=bool(raw.get("manifest_required")),
            notes=str(raw["notes"]) if raw.get("notes") not in (None, "") else None,
        )
        if key != profile.profile_id:
            msg = f"Calibration profile key '{key}' must match profile_id '{profile.profile_id}'"
            raise ValueError(msg)
        profiles[key] = profile
    return profiles


def validate_calibration_profile(
    profile: RunPodCalibrationProfile,
    *,
    models_path: str | Path = "configs/models.yaml",
    runtime_registry_path: str | Path = "configs/runtime_engines.yaml",
    gpu_price_registry_path: str | Path = DEFAULT_GPU_PRICE_REGISTRY_PATH,
) -> dict[str, object]:
    """Validate a planned profile against runtime compatibility and price registry."""

    runtime_errors: list[str] = []
    for model_alias in profile.model_aliases:
        try:
            select_runtime_for_model(
                model_alias=model_alias,
                runtime=profile.runtime,
                hardware_type=profile.hardware_type,
                live_run=True,
                models_path=models_path,
                registry_path=runtime_registry_path,
            )
        except ValueError as exc:
            runtime_errors.append(str(exc))
    try:
        registered_price = get_gpu_price(
            profile.gpu_name,
            registry_path=gpu_price_registry_path,
        )
        gpu_registered = True
    except KeyError as exc:
        registered_price = None
        gpu_registered = False
        runtime_errors.append(str(exc))
    price_registered = registered_price is not None
    profile_price_matches = (
        profile.hourly_price == registered_price
        if profile.hourly_price is not None or registered_price is not None
        else True
    )
    return {
        "profile_id": profile.profile_id,
        "gpu_registered": gpu_registered,
        "gpu_price_registered": price_registered,
        "profile_price_matches_registry": profile_price_matches,
        "runtime_profile_valid": not runtime_errors and profile_price_matches,
        "runtime_errors": runtime_errors,
        "registered_hourly_price": registered_price,
    }


def _ready_verdict(profile_id: str) -> CalibrationVerdict:
    if profile_id.startswith("A100"):
        return "READY_FOR_A100_CALIBRATION"
    if profile_id.startswith("H100"):
        return "READY_FOR_H100_CALIBRATION"
    if profile_id.startswith("L40S"):
        return "READY_FOR_L40S_CALIBRATION"
    return "CALIBRATION_NOT_READY"


def calibration_readiness_verdict(
    *,
    profile: RunPodCalibrationProfile,
    artifact_sync_enabled: bool,
    checkpoint_resume_enabled: bool,
    manifest_enabled: bool,
    runtime_profile_valid: bool,
    gpu_price_registered: bool,
    backup_verification_dry_run_passed: bool = False,
) -> dict[str, object]:
    """Return deterministic readiness gates for one calibration profile."""

    checks = {
        "artifact_sync_enabled": artifact_sync_enabled,
        "checkpoint_resume_enabled": checkpoint_resume_enabled,
        "manifest_enabled": manifest_enabled,
        "gpu_price_registered": gpu_price_registered,
        "runtime_profile_valid": runtime_profile_valid,
        "backup_verification_dry_run_passed": backup_verification_dry_run_passed,
    }
    ready = all(checks.values())
    return {
        "profile_id": profile.profile_id,
        "gpu_name": profile.gpu_name,
        "ready": ready,
        "verdict": _ready_verdict(profile.profile_id) if ready else "CALIBRATION_NOT_READY",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def build_calibration_manifest(
    *,
    profile: RunPodCalibrationProfile,
    model_alias: str,
    memory_mode: str,
    concurrency: int,
    prompt_count: int,
    artifact_paths: dict[str, str],
    repo_root: str | Path = ".",
    status: str = "planned",
) -> CalibrationManifest:
    """Build a first-class calibration manifest for a 100/200 prompt run."""

    if model_alias not in profile.model_aliases:
        msg = f"model_alias '{model_alias}' is not in profile '{profile.profile_id}'"
        raise ValueError(msg)
    if memory_mode not in profile.memory_modes:
        msg = f"memory_mode '{memory_mode}' is not in profile '{profile.profile_id}'"
        raise ValueError(msg)
    if concurrency not in profile.concurrency_list:
        msg = f"concurrency '{concurrency}' is not in profile '{profile.profile_id}'"
        raise ValueError(msg)
    if prompt_count not in profile.prompt_counts:
        msg = f"prompt_count '{prompt_count}' is not in profile '{profile.profile_id}'"
        raise ValueError(msg)
    run_id = (
        f"{profile.profile_id.lower()}-{model_alias}-{memory_mode}-c{concurrency}-{prompt_count}"
    )
    return CalibrationManifest(
        run_id=run_id,
        config_id=profile.profile_id,
        git_commit=current_git_commit(repo_root),
        profile_id=profile.profile_id,
        gpu_name=profile.gpu_name,
        runtime=profile.runtime,
        engine=profile.engine,
        model_alias=model_alias,
        memory_mode=memory_mode,
        concurrency=concurrency,
        traffic_profile=profile.traffic_profile,
        prompt_count=prompt_count,
        started_at=utc_now(),
        status=status,
        artifact_paths=artifact_paths,
        cost_estimate={
            "gpu_hourly_cost": profile.hourly_price,
            "estimated_run_cost": None,
            "cost_blocked_reason": (
                None if profile.hourly_price is not None else "gpu_hourly_price_missing"
            ),
        },
    )


def write_calibration_manifest(
    manifest: CalibrationManifest,
    output_path: str | Path,
) -> Path:
    """Write a calibration manifest JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
