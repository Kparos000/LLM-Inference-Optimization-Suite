"""Provider and pricing route decisions for the model5 gated API path."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inference_bench.api_priced_validation import AccessCheck
from inference_bench.api_pricing import (
    ApiPricingRegistryEntry,
    load_api_pricing_registry,
    load_manual_pricing_override_state,
    resolve_api_pricing,
)

HF_FEATHERLESS_DOCS_URL = (
    "https://huggingface.co/docs/inference-providers/main/providers/featherless-ai"
)
HF_CHAT_COMPLETION_DOCS_URL = (
    "https://huggingface.co/docs/inference-providers/en/tasks/chat-completion"
)
HF_ROUTER_MODEL_URL = "https://router.huggingface.co/v1/models/{model_id}"


@dataclass(frozen=True)
class ProviderRoute:
    """One provider route exposed by Hugging Face router metadata."""

    provider: str
    status: str
    chat_completion_supported: bool
    streaming_supported: bool
    capability_basis: str
    capability_source_urls: tuple[str, ...]


def request_router_metadata(model_id: str) -> dict[str, Any]:
    """Fetch public Hugging Face router metadata for one model."""

    url = HF_ROUTER_MODEL_URL.format(model_id=model_id)
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        msg = f"Router metadata request failed with HTTP {exc.code}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Router metadata request failed: {exc.reason}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = "Router metadata response must be a JSON object"
        raise RuntimeError(msg)
    return payload


def provider_routes_from_router_payload(payload: dict[str, Any]) -> list[ProviderRoute]:
    """Normalize provider routes without making a generation request."""

    model_payload = payload.get("data")
    if not isinstance(model_payload, dict):
        msg = "Hugging Face router metadata is missing the data object"
        raise ValueError(msg)
    providers = model_payload.get("providers", [])
    if not isinstance(providers, list):
        msg = "Hugging Face router metadata providers must be a list"
        raise ValueError(msg)

    routes: list[ProviderRoute] = []
    for provider_payload in providers:
        if not isinstance(provider_payload, dict):
            continue
        provider = str(provider_payload.get("provider") or "")
        if not provider:
            continue
        documented_featherless_route = provider == "featherless-ai"
        routes.append(
            ProviderRoute(
                provider=provider,
                status=str(provider_payload.get("status") or "unknown"),
                chat_completion_supported=documented_featherless_route,
                streaming_supported=documented_featherless_route,
                capability_basis=(
                    "Hugging Face documents Featherless chat completion and the "
                    "chat-completion API stream parameter; no paid request was sent."
                    if documented_featherless_route
                    else "Provider capability was not established by this audit."
                ),
                capability_source_urls=(
                    (HF_FEATHERLESS_DOCS_URL, HF_CHAT_COMPLETION_DOCS_URL)
                    if documented_featherless_route
                    else ()
                ),
            )
        )
    return routes


def _base_live_pricing_available(entry: ApiPricingRegistryEntry | None) -> bool:
    return bool(
        entry
        and entry.pricing_status == "detected"
        and entry.input_usd_per_1m_tokens is not None
        and entry.output_usd_per_1m_tokens is not None
    )


def build_model5_route_decision(
    *,
    pricing_config: str | Path,
    token_check: AccessCheck,
    model_access_check: AccessCheck,
    provider_routes: list[ProviderRoute],
    model_alias: str = "model5_gated",
) -> dict[str, Any]:
    """Build a secret-free, non-generating route and pricing decision."""

    registry = load_api_pricing_registry(pricing_config)
    registered = registry.get(model_alias)
    override = load_manual_pricing_override_state(model_alias, pricing_config)
    live_pricing_available = _base_live_pricing_available(registered)

    pricing_resolution_reason: str | None
    try:
        resolved = resolve_api_pricing(model_alias, pricing_config)
    except (FileNotFoundError, ValueError) as exc:
        resolved = None
        pricing_resolution_reason = str(exc)
    else:
        pricing_resolution_reason = None

    live_routes = [route for route in provider_routes if route.status == "live"]
    selected_route = live_routes[0] if live_routes else None
    reasons: list[str] = []
    if not token_check.available:
        reasons.append("HF_TOKEN is missing or invalid")
    if not model_access_check.available:
        reasons.append("Gated model repository access is unavailable")
    if selected_route is None:
        reasons.append("No live inference provider route is available")
    elif not selected_route.chat_completion_supported:
        reasons.append("Chat completion support is not established")
    if resolved is None:
        reasons.append(
            "No complete live token pricing or enabled audited manual token-price override exists"
        )

    costed_smoke_allowed = not reasons
    return {
        "model_alias": model_alias,
        "hf_token_present_and_valid": token_check.available,
        "hf_token_status_code": token_check.status_code,
        "model_access_granted": model_access_check.available,
        "model_access_status_code": model_access_check.status_code,
        "providers": [asdict(route) for route in provider_routes],
        "selected_provider": selected_route.provider if selected_route else None,
        "provider_live": selected_route is not None,
        "chat_completion_supported": bool(
            selected_route and selected_route.chat_completion_supported
        ),
        "streaming_supported": bool(selected_route and selected_route.streaming_supported),
        "streaming_support_live_tested": False,
        "live_pricing_available": live_pricing_available,
        "manual_override_present": override.present,
        "manual_override_enabled": override.enabled,
        "manual_override_configured": override.present and override.enabled,
        "manual_override_provider": override.provider,
        "manual_override_source_url": override.pricing_source_url,
        "manual_override_notes": override.notes,
        "pricing_resolved": resolved is not None,
        "resolved_pricing_status": resolved.pricing_status if resolved else None,
        "resolved_provider": resolved.provider if resolved else None,
        "resolved_input_usd_per_1m_tokens": (
            resolved.input_usd_per_1m_tokens if resolved else None
        ),
        "resolved_output_usd_per_1m_tokens": (
            resolved.output_usd_per_1m_tokens if resolved else None
        ),
        "pricing_resolution_reason": pricing_resolution_reason,
        "costed_smoke_allowed": costed_smoke_allowed,
        "pricing_route_decision": "allow" if costed_smoke_allowed else "block",
        "blocking_reasons": reasons,
        "no_generation_request_sent": True,
        "no_paid_api_call_sent": True,
    }


def audit_model5_route(
    *,
    model_id: str,
    pricing_config: str | Path,
    hf_token: str,
    token_checker: Callable[[str], AccessCheck],
    access_checker: Callable[[str, str], AccessCheck],
    metadata_fetcher: Callable[[str], dict[str, Any]],
    model_alias: str = "model5_gated",
) -> dict[str, Any]:
    """Run access and metadata checks without sending generation."""

    token_check = token_checker(hf_token)
    model_access_check = (
        access_checker(model_id, hf_token)
        if token_check.available
        else AccessCheck(
            available=False,
            status_code=None,
            error_type="Skipped",
            error_message="Model access check skipped because HF_TOKEN is unavailable",
        )
    )
    router_error: str | None = None
    try:
        router_payload = metadata_fetcher(model_id)
        provider_routes = provider_routes_from_router_payload(router_payload)
    except (RuntimeError, ValueError) as exc:
        router_error = str(exc)
        provider_routes = []

    decision = build_model5_route_decision(
        pricing_config=pricing_config,
        token_check=token_check,
        model_access_check=model_access_check,
        provider_routes=provider_routes,
        model_alias=model_alias,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_alias": model_alias,
        "model_id": model_id,
        "router_metadata_error": router_error,
        **decision,
        "secret_values_recorded": False,
        "gpu_work_triggered": False,
        "retrieval_modified": False,
        "evaluator_modified": False,
    }


def write_model5_route_audit(
    report: dict[str, Any],
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Write the model5 pricing route JSON and one-row CSV summary."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase4_model5_pricing_route_report.json"
    summary_path = output / "phase4_model5_pricing_route_summary.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model_alias": report["model_alias"],
        "model_id": report["model_id"],
        "hf_token_present_and_valid": report["hf_token_present_and_valid"],
        "model_access_granted": report["model_access_granted"],
        "selected_provider": report["selected_provider"],
        "provider_live": report["provider_live"],
        "chat_completion_supported": report["chat_completion_supported"],
        "streaming_supported": report["streaming_supported"],
        "streaming_support_live_tested": report["streaming_support_live_tested"],
        "live_pricing_available": report["live_pricing_available"],
        "manual_override_present": report["manual_override_present"],
        "manual_override_enabled": report["manual_override_enabled"],
        "manual_override_configured": report["manual_override_configured"],
        "pricing_resolved": report["pricing_resolved"],
        "resolved_pricing_status": report["resolved_pricing_status"],
        "pricing_route_decision": report["pricing_route_decision"],
        "costed_smoke_allowed": report["costed_smoke_allowed"],
        "blocking_reasons": " | ".join(report["blocking_reasons"]),
    }
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return report_path, summary_path
