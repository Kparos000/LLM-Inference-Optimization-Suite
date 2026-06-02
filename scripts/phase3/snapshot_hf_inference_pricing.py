"""Snapshot Hugging Face Inference Provider pricing for gated API models.

This script queries Hugging Face router model metadata. It does not run model
inference and does not call LLM generation APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.config import load_project_config  # noqa: E402

ROUTER_MODELS_URL = "https://router.huggingface.co/v1/models"


def utc_now() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def request_json(url: str) -> dict[str, Any]:
    """GET JSON from an HTTP endpoint."""

    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code} from {url}: {body[:500]}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Could not query {url}: {exc.reason}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"Expected JSON object from {url}"
        raise RuntimeError(msg)
    return payload


def model_metadata_url(model_id: str) -> str:
    """Return the Hugging Face router metadata URL for one model."""

    return f"{ROUTER_MODELS_URL}/{model_id}"


def provider_pricing_fields(provider: dict[str, Any]) -> tuple[float | None, float | None]:
    """Extract input/output pricing from one provider payload."""

    pricing = provider.get("pricing")
    if not isinstance(pricing, dict):
        return None, None
    input_price = pricing.get("input")
    output_price = pricing.get("output")
    if not isinstance(input_price, int | float) or not isinstance(output_price, int | float):
        return None, None
    return float(input_price), float(output_price)


def live_priced_providers(model_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return live providers that expose input and output token prices."""

    providers = model_payload.get("providers", [])
    if not isinstance(providers, list):
        return []
    priced: list[dict[str, Any]] = []
    for provider in providers:
        if not isinstance(provider, dict):
            continue
        input_price, output_price = provider_pricing_fields(provider)
        if input_price is None or output_price is None:
            continue
        if provider.get("status") != "live":
            continue
        priced.append(provider)
    return priced


def select_provider(providers: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a priced provider deterministically."""

    if not providers:
        msg = "No priced providers available"
        raise ValueError(msg)
    return sorted(
        providers,
        key=lambda provider: (
            float(provider["pricing"]["input"]) + float(provider["pricing"]["output"]),
            str(provider.get("provider") or ""),
        ),
    )[0]


def entry_from_provider(
    *,
    model_alias: str,
    model_id: str,
    provider: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Build a pricing config entry from a selected provider."""

    input_price, output_price = provider_pricing_fields(provider)
    if input_price is None or output_price is None:
        msg = f"Provider {provider.get('provider')} is missing pricing fields"
        raise ValueError(msg)
    first_token_latency_ms = provider.get("first_token_latency_ms")
    latency_seconds = (
        float(first_token_latency_ms) / 1000
        if isinstance(first_token_latency_ms, int | float)
        else None
    )
    throughput = provider.get("throughput")
    return {
        "model_alias": model_alias,
        "model_id": model_id,
        "provider": str(provider.get("provider") or ""),
        "provider_status": str(provider.get("status") or "unknown"),
        "input_cost_per_1m_tokens_usd": input_price,
        "output_cost_per_1m_tokens_usd": output_price,
        "context_length": provider.get("context_length"),
        "latency_seconds_if_available": latency_seconds,
        "throughput_tokens_per_second_if_available": float(throughput)
        if isinstance(throughput, int | float)
        else None,
        "supports_tools_if_available": provider.get("supports_tools"),
        "supports_structured_output_if_available": provider.get("supports_structured_output"),
        "pricing_snapshot_timestamp_utc": timestamp,
        "pricing_source_url": model_metadata_url(model_id),
        "selected_for_experiment": True,
    }


def model_size_between_1b_and_8b(model_id: str) -> bool:
    """Heuristic filter for model IDs that name a 1B-8B model size."""

    import re

    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)b(?![a-z])", model_id.lower())
    if not match:
        return False
    size_b = float(match.group(1))
    return 1.0 <= size_b <= 8.0


def discover_candidate_alternatives(limit: int = 12) -> list[dict[str, Any]]:
    """Discover candidate priced 1B-8B models from the router model listing."""

    try:
        listing = request_json(ROUTER_MODELS_URL)
    except RuntimeError:
        return []
    raw_models = listing.get("data", [])
    if not isinstance(raw_models, list):
        return []
    candidates: list[dict[str, Any]] = []
    for model in raw_models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id") or "")
        if not model_size_between_1b_and_8b(model_id):
            continue
        providers = live_priced_providers(model)
        if not providers:
            continue
        selected = select_provider(providers)
        input_price, output_price = provider_pricing_fields(selected)
        candidates.append(
            {
                "model_id": model_id,
                "provider": selected.get("provider"),
                "input_cost_per_1m_tokens_usd": input_price,
                "output_cost_per_1m_tokens_usd": output_price,
                "context_length": selected.get("context_length"),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_pricing_yaml(
    path: str | Path, entries: dict[str, dict[str, Any]], timestamp: str
) -> Path:
    """Write pricing snapshot YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pricing_source_url": ROUTER_MODELS_URL,
        "snapshot_timestamp_utc": timestamp,
        "models": entries,
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return output_path


def snapshot_pricing(
    *,
    model_aliases: list[str],
    output_path: str | Path,
    report_path: str | Path,
) -> tuple[dict[str, Any], bool]:
    """Snapshot pricing for model aliases and write report/config."""

    timestamp = utc_now()
    project_config = load_project_config()
    entries: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []

    for model_alias in model_aliases:
        model_config = project_config.resolve_model_config(model_alias)
        model_id = model_config.model_id
        payload = request_json(model_metadata_url(model_id))
        model_payload = payload.get("data")
        if not isinstance(model_payload, dict):
            failures.append(
                {
                    "model_alias": model_alias,
                    "model_id": model_id,
                    "failure": "router metadata missing data object",
                }
            )
            continue
        providers = model_payload.get("providers", [])
        provider_names = [
            str(provider.get("provider") or "")
            for provider in providers
            if isinstance(provider, dict)
        ]
        priced = live_priced_providers(model_payload)
        inspected.append(
            {
                "model_alias": model_alias,
                "model_id": model_id,
                "providers_found": provider_names,
                "priced_provider_count": len(priced),
            }
        )
        if not priced:
            failures.append(
                {
                    "model_alias": model_alias,
                    "model_id": model_id,
                    "providers_found": provider_names,
                    "missing_pricing_fields": ["pricing.input", "pricing.output"],
                    "failure": "no live provider exposed both input and output pricing",
                }
            )
            continue
        selected = select_provider(priced)
        entries[model_alias] = entry_from_provider(
            model_alias=model_alias,
            model_id=model_id,
            provider=selected,
            timestamp=timestamp,
        )

    success = not failures
    report = {
        "generated_at_utc": timestamp,
        "pricing_source_url": ROUTER_MODELS_URL,
        "model_aliases_requested": model_aliases,
        "pricing_snapshot_written": success,
        "output_path": str(output_path),
        "entries": entries,
        "inspected_models": inspected,
        "failures": failures,
        "candidate_alternative_gated_or_api_priced_models_1b_to_8b": []
        if success
        else discover_candidate_alternatives(),
        "no_model_inference_triggered": True,
        "no_gpu_work_triggered": True,
        "no_external_llm_api_generation_triggered": True,
    }
    write_json(report_path, report)
    if success:
        write_pricing_yaml(output_path, entries, timestamp)
    return report, success


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Snapshot Hugging Face Inference Provider token pricing."
    )
    parser.add_argument("--models", nargs="+", default=["model5_gated", "model6_gated"])
    parser.add_argument("--output", default="configs/api_pricing.yaml")
    parser.add_argument(
        "--report",
        default="data/generated/context_engineering/hf_api_pricing_snapshot_report.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run pricing snapshot."""

    args = build_parser().parse_args(argv)
    report, success = snapshot_pricing(
        model_aliases=args.models,
        output_path=args.output,
        report_path=args.report,
    )
    if success:
        for alias, entry in report["entries"].items():
            print(
                f"{alias}: provider={entry['provider']} "
                f"input=${entry['input_cost_per_1m_tokens_usd']}/1M "
                f"output=${entry['output_cost_per_1m_tokens_usd']}/1M"
            )
        print(f"Pricing snapshot written: {args.output}")
        print(f"Report written: {args.report}")
        return 0

    print("Pricing snapshot failed; report written with details.", file=sys.stderr)
    for failure in report["failures"]:
        print(
            f"- {failure['model_alias']} ({failure['model_id']}): {failure['failure']}",
            file=sys.stderr,
        )
    print(f"Report: {args.report}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
