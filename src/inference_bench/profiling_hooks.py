"""Optional profiling hook metadata for production runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ProfilerMode = Literal["disabled", "pytorch", "nsys", "ncu"]


@dataclass(frozen=True)
class ProfilingConfig:
    """Run-level profiling configuration.

    Profiling is disabled by default and this module never imports or requires
    PyTorch, Nsight Systems, or Nsight Compute.
    """

    mode: ProfilerMode = "disabled"
    enabled: bool = False
    output_path: str | None = None
    schedule: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "pytorch", "nsys", "ncu"}:
            msg = "mode must be disabled, pytorch, nsys, or ncu"
            raise ValueError(msg)
        if self.mode == "disabled" and self.enabled:
            msg = "profiling cannot be enabled with disabled mode"
            raise ValueError(msg)
        if self.mode != "disabled" and not self.enabled:
            msg = "non-disabled profiling modes must set enabled=True"
            raise ValueError(msg)
        if self.enabled and not self.output_path:
            msg = "enabled profiling requires output_path"
            raise ValueError(msg)

    def to_manifest_metadata(self) -> dict[str, object]:
        """Return profiling metadata suitable for run manifests."""

        return asdict(self)


def disabled_profiling_metadata() -> dict[str, object]:
    """Return the default profiling metadata for normal runs."""

    return ProfilingConfig().to_manifest_metadata()


def build_profiling_config(
    *,
    mode: ProfilerMode = "disabled",
    output_path: str | None = None,
    schedule: str | None = None,
    notes: str | None = None,
) -> ProfilingConfig:
    """Build a profiling config without importing profiler dependencies."""

    return ProfilingConfig(
        mode=mode,
        enabled=mode != "disabled",
        output_path=output_path,
        schedule=schedule,
        notes=notes,
    )
