"""Production deployment/readiness guardrails for larger benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from inference_bench.result_track_schema import validate_result_track_row

BackendType = Literal["local_compute", "api_provider", "self_hosted_gpu"]


@dataclass(frozen=True)
class ProductionRunReadinessInput:
    """Inputs required to judge whether a larger production-style run is allowed."""

    planned_prompt_count: int
    expected_prompt_count: int
    observed_prompt_count: int
    manifest_status: str
    backend_type: BackendType
    traffic_profile: str
    concurrency: int
    request_arrival_mode: str
    artifact_sync_configured: bool = False
    gpu_hourly_price_usd: float | None = None
    making_gpu_cost_claim: bool = False
    checkpoint_resume_supported: bool = False
    api_provider_load_probe_completed: bool = False
    result_track_rows: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "planned_prompt_count",
            "expected_prompt_count",
            "observed_prompt_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                msg = f"{field_name} must be an integer >= 0"
                raise ValueError(msg)
        if self.concurrency <= 0:
            msg = "concurrency must be > 0"
            raise ValueError(msg)
        if self.backend_type not in {"local_compute", "api_provider", "self_hosted_gpu"}:
            msg = "backend_type is invalid"
            raise ValueError(msg)
        if self.gpu_hourly_price_usd is not None and self.gpu_hourly_price_usd < 0:
            msg = "gpu_hourly_price_usd must be >= 0 when provided"
            raise ValueError(msg)


def _check(
    *,
    name: str,
    passed: bool,
    required: bool,
    evidence: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL" if required else "GAP",
        "required": required,
        "blocking": required and not passed,
        "evidence": evidence,
    }


def build_production_readiness_verdict(
    readiness: ProductionRunReadinessInput,
) -> dict[str, Any]:
    """Return deterministic production-readiness guardrail checks."""

    long_run = readiness.planned_prompt_count >= 1000
    large_api_run = (
        readiness.backend_type == "api_provider" and readiness.planned_prompt_count >= 1000
    )
    runpod_like_run = readiness.backend_type == "self_hosted_gpu" and long_run
    result_rows = readiness.result_track_rows or []
    result_schema_errors = [
        {"index": index, "errors": validate_result_track_row(row)}
        for index, row in enumerate(result_rows)
    ]
    invalid_result_rows = [row for row in result_schema_errors if row["errors"]]

    checks = [
        _check(
            name="load_metadata_present",
            passed=bool(
                readiness.traffic_profile
                and readiness.concurrency > 0
                and readiness.request_arrival_mode
            ),
            required=True,
            evidence=(
                f"profile={readiness.traffic_profile}; concurrency={readiness.concurrency}; "
                f"arrival={readiness.request_arrival_mode}"
            ),
        ),
        _check(
            name="artifact_sync_before_long_run",
            passed=(not runpod_like_run or readiness.artifact_sync_configured),
            required=runpod_like_run,
            evidence=(
                "Artifact sync configured"
                if readiness.artifact_sync_configured
                else "Artifact sync required before long self-hosted/RunPod runs"
            ),
        ),
        _check(
            name="gpu_hourly_price_required_before_gpu_cost_claim",
            passed=(
                not readiness.making_gpu_cost_claim or readiness.gpu_hourly_price_usd is not None
            ),
            required=readiness.making_gpu_cost_claim,
            evidence=(
                f"gpu_hourly_price_usd={readiness.gpu_hourly_price_usd}"
                if readiness.gpu_hourly_price_usd is not None
                else "GPU hourly price missing; cost claim blocked"
            ),
        ),
        _check(
            name="checkpoint_resume_required_for_1000_plus",
            passed=(not long_run or readiness.checkpoint_resume_supported),
            required=long_run,
            evidence=(
                "Checkpoint/resume supported"
                if readiness.checkpoint_resume_supported
                else "Checkpoint/resume required for 1,000+ prompt runs"
            ),
        ),
        _check(
            name="partial_runs_cannot_be_marked_complete",
            passed=not (
                readiness.manifest_status == "completed"
                and readiness.observed_prompt_count < readiness.expected_prompt_count
            ),
            required=True,
            evidence=(
                f"manifest_status={readiness.manifest_status}; "
                f"rows={readiness.observed_prompt_count}/{readiness.expected_prompt_count}"
            ),
        ),
        _check(
            name="api_provider_load_probe_required_before_large_api_runs",
            passed=(not large_api_run or readiness.api_provider_load_probe_completed),
            required=large_api_run,
            evidence=(
                "API provider load probe completed"
                if readiness.api_provider_load_probe_completed
                else "Large API run requires a provider load probe"
            ),
        ),
        _check(
            name="api_and_gpu_tracks_join_through_unified_result_schema",
            passed=bool(result_rows) and not invalid_result_rows,
            required=True,
            evidence=(
                f"{len(result_rows)} result-track rows validated"
                if result_rows and not invalid_result_rows
                else f"Invalid result-track rows: {invalid_result_rows}"
            ),
        ),
    ]
    blocking = [check for check in checks if check["blocking"]]
    return {
        "status": "READY" if not blocking else "NOT_READY",
        "checks": checks,
        "blocking_count": len(blocking),
        "long_run": long_run,
        "large_api_run": large_api_run,
        "runpod_like_run": runpod_like_run,
        "input": asdict(readiness),
    }
