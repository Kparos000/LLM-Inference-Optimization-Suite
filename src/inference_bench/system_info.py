"""Lightweight hardware and system metadata capture."""

from __future__ import annotations

import json
import os
import platform as platform_module
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, cast


@dataclass(frozen=True)
class SystemInfo:
    """Reproducibility metadata for a benchmark environment."""

    timestamp_utc: str
    platform: str
    platform_release: str
    python_version: str
    processor: str
    cpu_count_logical: int | None
    cpu_count_physical: int | None
    total_ram_gb: float | None
    torch_version: str | None
    cuda_available: bool | None
    cuda_device_count: int | None
    cuda_device_names: list[str]
    transformers_version: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return asdict(self)


def _try_import(module_name: str) -> ModuleType | None:
    try:
        return import_module(module_name)
    except ImportError:
        return None


def _collect_cpu_count_physical() -> int | None:
    psutil = _try_import("psutil")
    if psutil is None:
        return None

    psutil_module = cast(Any, psutil)
    physical_count = psutil_module.cpu_count(logical=False)
    return physical_count if isinstance(physical_count, int) else None


def _collect_total_ram_gb() -> float | None:
    psutil = _try_import("psutil")
    if psutil is None:
        return None

    psutil_module = cast(Any, psutil)
    memory = psutil_module.virtual_memory()
    total_bytes = getattr(memory, "total", None)
    if not isinstance(total_bytes, int):
        return None

    return round(total_bytes / (1024**3), 2)


def _collect_torch_info() -> tuple[str | None, bool | None, int | None, list[str]]:
    torch = _try_import("torch")
    if torch is None:
        return None, None, None, []

    torch_version = getattr(torch, "__version__", None)
    if not isinstance(torch_version, str):
        torch_version = None

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return torch_version, None, None, []

    try:
        cuda_available = bool(cuda.is_available())
    except (AttributeError, RuntimeError):
        cuda_available = None

    try:
        cuda_device_count = int(cuda.device_count())
    except (AttributeError, RuntimeError):
        cuda_device_count = None

    cuda_device_names: list[str] = []
    if cuda_device_count:
        for device_index in range(cuda_device_count):
            try:
                device_name = cuda.get_device_name(device_index)
            except (AttributeError, RuntimeError):
                continue
            if isinstance(device_name, str):
                cuda_device_names.append(device_name)

    return torch_version, cuda_available, cuda_device_count, cuda_device_names


def _collect_transformers_version() -> str | None:
    transformers = _try_import("transformers")
    if transformers is None:
        return None

    version = getattr(transformers, "__version__", None)
    return version if isinstance(version, str) else None


def collect_system_info() -> SystemInfo:
    """Collect lightweight hardware and library metadata."""

    torch_version, cuda_available, cuda_device_count, cuda_device_names = _collect_torch_info()

    return SystemInfo(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform_module.system(),
        platform_release=platform_module.release(),
        python_version=sys.version.split()[0],
        processor=platform_module.processor(),
        cpu_count_logical=os.cpu_count(),
        cpu_count_physical=_collect_cpu_count_physical(),
        total_ram_gb=_collect_total_ram_gb(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        transformers_version=_collect_transformers_version(),
    )


def write_system_info_json(info: SystemInfo, output_path: str | Path) -> Path:
    """Write system metadata to a JSON artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = info.to_dict()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
