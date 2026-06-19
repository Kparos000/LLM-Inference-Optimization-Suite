"""Phase 4 pre-GPU readiness inspection and configuration utilities.

The checks in this module inspect committed contracts, reports, and wrappers.
They do not load a model, contact a serving endpoint, invoke an API, or run GPU
work.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import load_yaml_file
from inference_bench.telemetry import TELEMETRY_FIELDS

ReadinessStatus = Literal["PASS", "NOT_AVAILABLE", "FAIL"]
BackendStatus = Literal["ready", "dry_run_ready", "planned", "deprecated"]
CostModel = Literal["local_compute", "gpu_infra", "api_token"]

DEFAULT_BACKEND_MATRIX_PATH = "configs/backend_matrix.yaml"
DEFAULT_GPU_COSTS_PATH = "configs/gpu_costs.yaml"
DEFAULT_CONTEXT_ROOT = "data/generated/context_engineering"
DEFAULT_OUTPUT_ROOT = "data/generated/phase4"

REQUIRED_TELEMETRY_FIELDS = {
    "timestamp",
    "backend",
    "model",
    "memory_mode",
    "latency_ms",
    "ttft_ms",
    "tpot_ms",
    "throughput_tokens_per_second",
    "requests_per_second",
    "success",
    "error_type",
    "gpu_utilization",
    "gpu_memory",
    "gpu_cost",
    "runpod_cost",
}
UNMEASURED_METRIC_FAMILIES = {
    "latency_slo",
    "throughput_slo",
    "resource_slo",
    "api_cost_slo",
    "gpu_cost_slo",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


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


@dataclass(frozen=True)
class BackendConfig:
    """One serving backend capability entry."""

    backend_name: str
    endpoint_type: str
    requires_server: bool
    requires_gpu: bool
    supports_streaming: bool
    supports_ttft: bool
    supports_tpot: bool
    supports_batch: bool
    supports_concurrency: bool
    cost_model: CostModel
    status: BackendStatus

    def __post_init__(self) -> None:
        _non_empty(self.backend_name, "backend_name")
        _non_empty(self.endpoint_type, "endpoint_type")
        for field_name in (
            "requires_server",
            "requires_gpu",
            "supports_streaming",
            "supports_ttft",
            "supports_tpot",
            "supports_batch",
            "supports_concurrency",
        ):
            _boolean(getattr(self, field_name), field_name)
        if self.cost_model not in {"local_compute", "gpu_infra", "api_token"}:
            msg = "cost_model must be local_compute, gpu_infra, or api_token"
            raise ValueError(msg)
        if self.status not in {"ready", "dry_run_ready", "planned", "deprecated"}:
            msg = "status must be ready, dry_run_ready, planned, or deprecated"
            raise ValueError(msg)


@dataclass(frozen=True)
class GPUCostConfig:
    """Provider cost inputs populated immediately before a GPU run."""

    provider: str
    gpu_type: str | None
    hourly_price_usd: float | None
    region: str | None
    instance_id_optional: str | None
    measured_start_time: str | None
    measured_end_time: str | None
    cost_formula: str

    def __post_init__(self) -> None:
        _non_empty(self.provider, "provider")
        _non_empty(self.cost_formula, "cost_formula")
        if self.hourly_price_usd is not None and self.hourly_price_usd < 0:
            msg = "hourly_price_usd must be >= 0 when configured"
            raise ValueError(msg)
        for field_name in ("gpu_type", "region", "instance_id_optional"):
            value = getattr(self, field_name)
            if value is not None:
                _non_empty(value, field_name)

    @property
    def live_run_values_configured(self) -> bool:
        """Return whether all values required to calculate one run are present."""

        return all(
            value is not None
            for value in (
                self.gpu_type,
                self.hourly_price_usd,
                self.region,
                self.measured_start_time,
                self.measured_end_time,
            )
        )


@dataclass(frozen=True)
class Phase4ReadinessRow:
    """One readiness check and its disposition."""

    area: str
    artifact: str
    status: ReadinessStatus
    required_before_gpu: bool
    details: str
    next_action: str

    @property
    def blocks_pre_gpu_plumbing(self) -> bool:
        """Return whether this row prevents a clean pre-GPU handoff."""

        return self.required_before_gpu and self.status == "FAIL"

    def to_dict(self) -> dict[str, object]:
        """Return stable JSON/CSV data."""

        return asdict(self)


def load_backend_matrix(
    path: str | Path = DEFAULT_BACKEND_MATRIX_PATH,
) -> dict[str, BackendConfig]:
    """Load and validate serving backend capabilities."""

    loaded = load_yaml_file(path)
    backends: dict[str, BackendConfig] = {}
    for key, value in loaded.items():
        if not isinstance(value, dict):
            msg = f"Backend matrix entry '{key}' must be a mapping"
            raise ValueError(msg)
        try:
            backends[key] = BackendConfig(**cast(dict[str, Any], value))
        except (TypeError, ValueError) as exc:
            msg = f"Invalid backend matrix entry '{key}': {exc}"
            raise ValueError(msg) from exc
    return backends


def load_gpu_cost_configs(
    path: str | Path = DEFAULT_GPU_COSTS_PATH,
) -> dict[str, GPUCostConfig]:
    """Load and validate GPU provider cost placeholders."""

    loaded = load_yaml_file(path)
    configs: dict[str, GPUCostConfig] = {}
    for key, value in loaded.items():
        if not isinstance(value, dict):
            msg = f"GPU cost entry '{key}' must be a mapping"
            raise ValueError(msg)
        try:
            configs[key] = GPUCostConfig(**cast(dict[str, Any], value))
        except (TypeError, ValueError) as exc:
            msg = f"Invalid GPU cost entry '{key}': {exc}"
            raise ValueError(msg) from exc
    return configs


def _parse_utc_timestamp(value: str, field_name: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        msg = f"{field_name} must be an ISO-8601 timestamp"
        raise ValueError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"{field_name} must include a timezone"
        raise ValueError(msg)
    return parsed.astimezone(timezone.utc)


def calculate_elapsed_gpu_cost(
    *,
    hourly_price_usd: float,
    measured_start_time: str,
    measured_end_time: str,
) -> float:
    """Calculate infrastructure cost from elapsed wall time and hourly price."""

    if hourly_price_usd < 0:
        msg = "hourly_price_usd must be >= 0"
        raise ValueError(msg)
    start = _parse_utc_timestamp(measured_start_time, "measured_start_time")
    end = _parse_utc_timestamp(measured_end_time, "measured_end_time")
    elapsed_seconds = (end - start).total_seconds()
    if elapsed_seconds < 0:
        msg = "measured_end_time must not precede measured_start_time"
        raise ValueError(msg)
    return elapsed_seconds / 3600 * hourly_price_usd


def calculate_configured_gpu_cost(config: GPUCostConfig) -> float:
    """Calculate one configured GPU run or fail clearly on placeholders."""

    if not config.live_run_values_configured:
        msg = "GPU type, hourly price, region, and measured timestamps must be configured"
        raise ValueError(msg)
    assert config.hourly_price_usd is not None
    assert config.measured_start_time is not None
    assert config.measured_end_time is not None
    return calculate_elapsed_gpu_cost(
        hourly_price_usd=config.hourly_price_usd,
        measured_start_time=config.measured_start_time,
        measured_end_time=config.measured_end_time,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        msg = f"Required JSON artifact is missing or empty: {path}"
        raise FileNotFoundError(msg)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"Expected JSON object in {path}"
        raise ValueError(msg)
    return cast(dict[str, Any], loaded)


def _path_status(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _local_hf_status(repo_root: Path) -> tuple[ReadinessStatus, str]:
    report_candidates = (
        repo_root / "results/processed/phase4_generation_contract_hardened_eval_report.json",
        repo_root / "results/processed/phase4_hf_local_smoke_eval_report.json",
    )
    available = [path for path in report_candidates if _path_status(path)]
    script_path = repo_root / "scripts/phase4/run_local_hf_smoke.py"
    dry_run_capable = _path_status(script_path) and "--dry-run" in script_path.read_text(
        encoding="utf-8"
    )
    if available:
        return (
            "PASS",
            f"Local HF evaluation report is available at {available[0].relative_to(repo_root)}; "
            f"dry-run support present={dry_run_capable}.",
        )
    if dry_run_capable:
        return (
            "PASS",
            "Ignored local HF reports are absent, but the local runner has a clean-checkout-safe "
            "--dry-run path.",
        )
    return "FAIL", "No local HF report or dry-run-capable local HF smoke runner was found."


def _metric_family_statuses(slo_report: dict[str, Any]) -> dict[str, set[str]]:
    statuses: dict[str, set[str]] = {}
    raw_results = slo_report.get("results", [])
    if not isinstance(raw_results, list):
        return statuses
    for result in raw_results:
        if not isinstance(result, dict):
            continue
        family = str(result.get("metric_family") or "")
        status = str(result.get("status") or "")
        if family and status:
            statuses.setdefault(family, set()).add(status)
    return statuses


def _unmerged_git_paths(repo_root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["git_status_unavailable"]
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def inspect_phase4_readiness(
    *,
    repo_root: str | Path,
    context_root: str | Path = DEFAULT_CONTEXT_ROOT,
    backend_matrix_path: str | Path = DEFAULT_BACKEND_MATRIX_PATH,
    gpu_costs_path: str | Path = DEFAULT_GPU_COSTS_PATH,
) -> tuple[dict[str, Any], list[Phase4ReadinessRow]]:
    """Inspect Phase 4 prerequisites without executing inference or servers."""

    root = Path(repo_root).resolve()
    context = root / context_root
    promotion_manifest_path = context / "retrieval_source_of_truth_manifest.json"
    slo_report_path = context / "slo_readiness_report.json"
    generation_contract_path = root / "src/inference_bench/generation_contract.py"
    vllm_wrapper_path = root / "scripts/phase4/run_openai_compatible_smoke.py"
    sglang_wrapper_path = root / "scripts/phase4/run_sglang_compatible_smoke.py"
    backend_path = root / backend_matrix_path
    gpu_path = root / gpu_costs_path

    promotion_manifest: dict[str, Any] = {}
    promotion_status: ReadinessStatus = "FAIL"
    promotion_details = "Promoted retrieval source-of-truth manifest is missing."
    if _path_status(promotion_manifest_path):
        promotion_manifest = _read_json(promotion_manifest_path)
        promoted = (
            promotion_manifest.get("retrieval_promotion_status") == "PROMOTED"
            and promotion_manifest.get("retrieval_slo_status") == "PASS"
            and promotion_manifest.get("retrieval_ready_for_phase4") is True
        )
        promotion_status = "PASS" if promoted else "FAIL"
        promotion_details = (
            "Canonical retrieval manifest is promoted and ready for Phase 4."
            if promoted
            else "Retrieval manifest exists but is not promoted with passing SLOs."
        )

    slo_report: dict[str, Any] = {}
    family_statuses: dict[str, set[str]] = {}
    retrieval_slo_status: ReadinessStatus = "FAIL"
    retrieval_slo_details = "SLO readiness report is missing."
    if _path_status(slo_report_path):
        slo_report = _read_json(slo_report_path)
        family_statuses = _metric_family_statuses(slo_report)
        retrieval_statuses = family_statuses.get("retrieval_slo", set())
        retrieval_pass = (
            retrieval_statuses == {"PASS"}
            and int(slo_report.get("retrieval_slo_blocked_count", 1)) == 0
        )
        retrieval_slo_status = "PASS" if retrieval_pass else "FAIL"
        retrieval_slo_details = (
            "All promoted retrieval SLO rows pass."
            if retrieval_pass
            else f"Retrieval SLO statuses are {sorted(retrieval_statuses)}."
        )

    unmeasured_statuses = {
        family: family_statuses.get(family, set()) for family in UNMEASURED_METRIC_FAMILIES
    }
    unmeasured_are_not_available = all(
        statuses == {"NOT_AVAILABLE"} for statuses in unmeasured_statuses.values()
    )

    hf_status, hf_details = _local_hf_status(root)
    telemetry_missing = sorted(REQUIRED_TELEMETRY_FIELDS.difference(TELEMETRY_FIELDS))
    telemetry_status: ReadinessStatus = "PASS" if not telemetry_missing else "FAIL"

    backends: dict[str, BackendConfig] = {}
    backend_status: ReadinessStatus = "FAIL"
    backend_details = "Backend matrix is missing or invalid."
    try:
        backends = load_backend_matrix(backend_path)
        expected_backends = {
            "hf_local",
            "openai_compatible_vllm",
            "sglang_openai_compatible_future",
        }
        matrix_valid = expected_backends.issubset(backends)
        backend_status = "PASS" if matrix_valid else "FAIL"
        backend_details = (
            "HF, vLLM, and future SGLang backend capabilities are defined."
            if matrix_valid
            else "Backend matrix does not define every required backend."
        )
    except (FileNotFoundError, ValueError):
        pass

    gpu_configs: dict[str, GPUCostConfig] = {}
    gpu_config_status: ReadinessStatus = "FAIL"
    gpu_config_details = "GPU cost configuration is missing or invalid."
    gpu_values_status: ReadinessStatus = "NOT_AVAILABLE"
    try:
        gpu_configs = load_gpu_cost_configs(gpu_path)
        gpu_config_status = "PASS" if "runpod_default" in gpu_configs else "FAIL"
        gpu_config_details = (
            "RunPod cost inputs and elapsed-hour formula are defined."
            if gpu_config_status == "PASS"
            else "runpod_default is missing from GPU cost configuration."
        )
        if gpu_config_status == "PASS" and gpu_configs["runpod_default"].live_run_values_configured:
            gpu_values_status = "PASS"
    except (FileNotFoundError, ValueError):
        pass

    unmerged_paths = _unmerged_git_paths(root)
    git_status: ReadinessStatus = "PASS" if not unmerged_paths else "FAIL"
    git_details = (
        "No unmerged paths or unresolved conflict artifacts were found."
        if not unmerged_paths
        else f"Unresolved git paths: {', '.join(unmerged_paths)}"
    )

    rows = [
        Phase4ReadinessRow(
            "promoted_retrieval",
            str(promotion_manifest_path.relative_to(root)),
            promotion_status,
            True,
            promotion_details,
            "Keep this manifest as the retrieval source of truth.",
        ),
        Phase4ReadinessRow(
            "generation_contract",
            str(generation_contract_path.relative_to(root)),
            "PASS" if _path_status(generation_contract_path) else "FAIL",
            True,
            "Shared grounded generation contract is present.",
            "Reuse it across HF, vLLM, and SGLang output paths.",
        ),
        Phase4ReadinessRow(
            "local_hf_smoke",
            "results/processed/phase4_*hf*_eval_report.json or local dry-run",
            hf_status,
            True,
            hf_details,
            "Keep the local dry-run available in clean checkout.",
        ),
        Phase4ReadinessRow(
            "vllm_openai_wrapper",
            str(vllm_wrapper_path.relative_to(root)),
            "PASS" if _path_status(vllm_wrapper_path) else "FAIL",
            True,
            "OpenAI-compatible vLLM smoke wrapper is present.",
            "Run live only after provisioning a local or remote GPU server.",
        ),
        Phase4ReadinessRow(
            "sglang_openai_wrapper",
            str(sglang_wrapper_path.relative_to(root)),
            "PASS" if _path_status(sglang_wrapper_path) else "FAIL",
            True,
            "SGLang-compatible scaffold is present and supports dry-run.",
            "Validate against a live SGLang server in a later GPU block.",
        ),
        Phase4ReadinessRow(
            "telemetry_schema",
            "src/inference_bench/telemetry.py",
            telemetry_status,
            True,
            (
                "Request telemetry and nullable future GPU fields are defined."
                if not telemetry_missing
                else f"Missing telemetry fields: {', '.join(telemetry_missing)}"
            ),
            "Populate GPU fields during the first live GPU smoke.",
        ),
        Phase4ReadinessRow(
            "retrieval_slo",
            str(slo_report_path.relative_to(root)),
            retrieval_slo_status,
            True,
            retrieval_slo_details,
            "Do not use historical pre-repair retrieval reports for promotion decisions.",
        ),
        Phase4ReadinessRow(
            "latency_cost_resource_metrics",
            str(slo_report_path.relative_to(root)),
            "NOT_AVAILABLE" if unmeasured_are_not_available else "FAIL",
            False,
            (
                "Latency, throughput, resource, API cost, and GPU cost metrics are explicitly "
                "NOT_AVAILABLE before live serving."
                if unmeasured_are_not_available
                else f"Unexpected metric-family statuses: {unmeasured_statuses}"
            ),
            "Measure these during live HF/vLLM/SGLang and GPU runs.",
        ),
        Phase4ReadinessRow(
            "backend_matrix",
            str(backend_path.relative_to(root)),
            backend_status,
            True,
            backend_details,
            "Use the matrix to select only backends whose prerequisites are satisfied.",
        ),
        Phase4ReadinessRow(
            "gpu_cost_config",
            str(gpu_path.relative_to(root)),
            gpu_config_status,
            True,
            gpu_config_details,
            "Fill provider-specific GPU values immediately before the run.",
        ),
        Phase4ReadinessRow(
            "gpu_cost_values",
            str(gpu_path.relative_to(root)),
            gpu_values_status,
            False,
            (
                "Live GPU type, region, hourly price, and measured timestamps are configured."
                if gpu_values_status == "PASS"
                else "GPU price and run timestamps intentionally remain unset until provisioning."
            ),
            "Record the actual RunPod listing and timestamps for each GPU run.",
        ),
        Phase4ReadinessRow(
            "git_artifacts",
            "git diff --name-only --diff-filter=U",
            git_status,
            True,
            git_details,
            "Commit only intended Block 25 artifacts after verification.",
        ),
    ]
    blocking_rows = [row.area for row in rows if row.blocks_pre_gpu_plumbing]
    pre_gpu_ready = not blocking_rows
    report = {
        "generated_at_utc": utc_now(),
        "scope": "phase4_pre_gpu_plumbing_no_inference_no_gpu_no_api",
        "overall_status": "PRE_GPU_PLUMBING_READY" if pre_gpu_ready else "BLOCKED",
        "pre_gpu_plumbing_ready": pre_gpu_ready,
        "ready_for_gpu_provisioning": pre_gpu_ready,
        "ready_for_live_gpu_smoke": (pre_gpu_ready and gpu_values_status == "PASS"),
        "blocking_checks": blocking_rows,
        "promoted_retrieval": promotion_manifest,
        "slo_metric_family_statuses": {
            family: sorted(statuses) for family, statuses in sorted(family_statuses.items())
        },
        "backend_matrix": {key: asdict(config) for key, config in sorted(backends.items())},
        "gpu_cost_configs": {key: asdict(config) for key, config in sorted(gpu_configs.items())},
        "checks": [row.to_dict() for row in rows],
        "remaining_before_live_gpu_smoke": [
            "Choose and provision a GPU with enough memory for the selected model.",
            "Record the actual RunPod GPU type, region, and hourly price.",
            "Start vLLM or SGLang and verify its OpenAI-compatible /models endpoint.",
            "Run a five-request live smoke before any concurrency or scale test.",
            "Populate TTFT, TPOT, throughput, GPU utilization, memory, and cost telemetry.",
        ],
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_paid_api_call_triggered": True,
    }
    return report, rows


def write_phase4_readiness_artifacts(
    *,
    output_root: str | Path,
    report: dict[str, Any],
    rows: list[Phase4ReadinessRow],
) -> tuple[Path, Path]:
    """Write the Phase 4 readiness JSON report and CSV summary."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase4_readiness_report.json"
    summary_path = output / "phase4_readiness_summary.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "area",
        "artifact",
        "status",
        "required_before_gpu",
        "details",
        "next_action",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    return report_path, summary_path


def build_phase4_readiness_report(
    *,
    repo_root: str | Path,
    output_root: str | Path,
    context_root: str | Path = DEFAULT_CONTEXT_ROOT,
    backend_matrix_path: str | Path = DEFAULT_BACKEND_MATRIX_PATH,
    gpu_costs_path: str | Path = DEFAULT_GPU_COSTS_PATH,
) -> dict[str, Any]:
    """Inspect and write Phase 4 pre-GPU readiness artifacts."""

    report, rows = inspect_phase4_readiness(
        repo_root=repo_root,
        context_root=context_root,
        backend_matrix_path=backend_matrix_path,
        gpu_costs_path=gpu_costs_path,
    )
    write_phase4_readiness_artifacts(
        output_root=output_root,
        report=report,
        rows=rows,
    )
    return report
