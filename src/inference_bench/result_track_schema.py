"""Unified result-track schema helpers for local, API, and self-hosted GPU runs."""

from __future__ import annotations

from typing import Any

RESULT_TRACK_JOIN_KEYS = (
    "run_id",
    "config_id",
    "prompt_id",
    "vertical",
    "model_alias",
    "memory_mode",
    "runtime",
    "backend_type",
    "engine",
    "hardware",
    "provider",
    "concurrency",
)

VALID_BACKEND_TYPES = {"local_compute", "api_provider", "self_hosted_gpu"}
API_PROVIDER_RUNTIMES = {"api_provider_route"}
SELF_HOSTED_RUNTIMES = {"huggingface_transformers", "vllm", "sglang", "tensorrt_llm"}


def result_track_join_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return the stable join key for combined runtime result plots."""

    return tuple(str(row.get(key) or "") for key in RESULT_TRACK_JOIN_KEYS)


def validate_result_track_row(row: dict[str, Any]) -> list[str]:
    """Return schema errors for runtime result-track rows."""

    errors: list[str] = []
    for key in RESULT_TRACK_JOIN_KEYS:
        if row.get(key) in (None, ""):
            errors.append(f"missing:{key}")
    backend_type = str(row.get("backend_type") or "")
    runtime = str(row.get("runtime") or "")
    hardware = str(row.get("hardware") or "")
    if backend_type not in VALID_BACKEND_TYPES:
        errors.append("invalid:backend_type")
    if backend_type == "local_compute":
        if runtime not in SELF_HOSTED_RUNTIMES:
            errors.append("local_compute_requires_local_runtime")
        if row.get("api_cost_usd") is not None:
            errors.append("local_compute_must_not_claim_api_cost")
    if backend_type == "api_provider":
        if runtime not in API_PROVIDER_RUNTIMES:
            errors.append("api_provider_requires_api_runtime")
        if hardware != "provider_managed":
            errors.append("api_provider_requires_provider_managed_hardware")
        if row.get("api_provider") in (None, ""):
            errors.append("missing:api_provider")
        if row.get("api_cost_usd") is None:
            errors.append("missing:api_cost_usd")
        if row.get("gpu_telemetry_available") is not False:
            errors.append("api_provider_must_not_claim_gpu_telemetry")
    if backend_type == "self_hosted_gpu":
        if runtime not in SELF_HOSTED_RUNTIMES:
            errors.append("self_hosted_gpu_requires_self_hosted_runtime")
        if hardware == "provider_managed":
            errors.append("self_hosted_gpu_must_not_use_provider_managed_hardware")
        if row.get("gpu_hourly_price_usd") is None and row.get("gpu_cost_usd") is not None:
            errors.append("gpu_cost_requires_hourly_price")
        if row.get("api_cost_usd") is not None:
            errors.append("self_hosted_gpu_must_not_claim_api_cost")
    return errors


def track_description(backend_type: str) -> str:
    """Return the plain-language execution track description."""

    if backend_type == "api_provider":
        return (
            "API provider track: model5/model6/model7 through OpenRouter, Novita, or "
            "Hugging Face provider routes; API token pricing is available when "
            "configured, but provider GPU telemetry is unavailable."
        )
    if backend_type == "local_compute":
        return (
            "Local compute track: Hugging Face Transformers runs on the developer "
            "machine; local hardware telemetry may be partial and API token pricing "
            "does not apply."
        )
    if backend_type == "self_hosted_gpu":
        return (
            "Self-hosted GPU track: open-weight models on vLLM, SGLang, or RunPod; "
            "GPU telemetry and hourly infrastructure cost can be measured when "
            "configured, but API token pricing does not apply."
        )
    raise ValueError(f"Unknown backend_type: {backend_type}")
