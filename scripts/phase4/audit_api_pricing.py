"""Audit live and registered pricing for gated API models without generation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_pricing import (  # noqa: E402
    load_api_pricing_registry,
    resolve_api_pricing,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.model_registry import write_model_registry_artifacts  # noqa: E402
from inference_bench.openrouter_api import fetch_openrouter_model_metadata  # noqa: E402


def _load_snapshot_module() -> ModuleType:
    path = REPO_ROOT / "scripts/phase3/snapshot_hf_inference_pricing.py"
    spec = importlib.util.spec_from_file_location("_block29_pricing_snapshot", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load pricing metadata client from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build pricing audit CLI."""

    parser = argparse.ArgumentParser(
        description="Audit provider pricing without running model generation."
    )
    parser.add_argument("--models", nargs="+", default=["model5_gated", "model6_gated"])
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--output-root", default="results/processed")
    return parser


def audit_pricing(
    *,
    model_aliases: list[str],
    pricing_config: str | Path,
) -> dict[str, Any]:
    """Compare live router metadata with the local pricing registry."""

    snapshot: ModuleType | None = None
    config = load_project_config()
    registry = load_api_pricing_registry(pricing_config)
    rows: list[dict[str, Any]] = []
    for alias in model_aliases:
        model = config.resolve_model_config(alias)
        metadata_url = ""
        providers_found: list[str] = []
        detected_provider: str | None = None
        detected_input: float | None = None
        detected_output: float | None = None
        live_error: str | None = None
        if model.provider == "openrouter":
            metadata_url = "https://openrouter.ai/api/v1/models"
            try:
                metadata = fetch_openrouter_model_metadata(model.model_id)
                providers_found = ["openrouter"]
                detected_provider = "openrouter"
                detected_input = metadata.input_usd_per_1m_tokens
                detected_output = metadata.output_usd_per_1m_tokens
            except (RuntimeError, ValueError) as exc:
                live_error = str(exc)
        else:
            if snapshot is None:
                snapshot = _load_snapshot_module()
            metadata_url = snapshot.model_metadata_url(model.model_id)
            try:
                payload = snapshot.request_json(metadata_url)
                model_payload = payload.get("data")
                if not isinstance(model_payload, dict):
                    raise RuntimeError("Router metadata missing data object")
                providers = model_payload.get("providers", [])
                providers_found = [
                    str(provider.get("provider") or "")
                    for provider in providers
                    if isinstance(provider, dict)
                ]
                priced = snapshot.live_priced_providers(model_payload)
                if priced:
                    selected = snapshot.select_provider(priced)
                    detected_provider = str(selected.get("provider") or "")
                    detected_input, detected_output = snapshot.provider_pricing_fields(selected)
            except RuntimeError as exc:
                live_error = str(exc)
        registered = registry.get(alias)
        resolution_reason: str | None
        try:
            resolved = resolve_api_pricing(alias, pricing_config)
        except (FileNotFoundError, ValueError) as exc:
            resolved = None
            resolution_reason = str(exc)
        else:
            resolution_reason = None
        rows.append(
            {
                "model_alias": alias,
                "model_id": model.model_id,
                "providers_found": providers_found,
                "live_detected_provider": detected_provider,
                "live_input_usd_per_1m_tokens": detected_input,
                "live_output_usd_per_1m_tokens": detected_output,
                "registry_provider": registered.provider if registered else None,
                "registry_pricing_status": (
                    registered.pricing_status if registered else "unavailable"
                ),
                "registry_input_usd_per_1m_tokens": (
                    registered.input_usd_per_1m_tokens if registered else None
                ),
                "registry_output_usd_per_1m_tokens": (
                    registered.output_usd_per_1m_tokens if registered else None
                ),
                "registry_pricing_source": (registered.pricing_source if registered else None),
                "manual_override_used": bool(
                    registered and registered.pricing_status == "manual_override"
                ),
                "runnable_for_costed_smoke": resolved is not None,
                "resolved_provider": resolved.provider if resolved else None,
                "resolution_reason": resolution_reason,
                "live_query_error": live_error,
                "pricing_source_url": metadata_url,
                "live_pricing_matches_registry": (
                    registered is not None
                    and detected_input is not None
                    and detected_output is not None
                    and registered.input_usd_per_1m_tokens is not None
                    and registered.output_usd_per_1m_tokens is not None
                    and math.isclose(
                        detected_input,
                        registered.input_usd_per_1m_tokens,
                        rel_tol=1e-9,
                    )
                    and math.isclose(
                        detected_output,
                        registered.output_usd_per_1m_tokens,
                        rel_tol=1e-9,
                    )
                ),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pricing_config": str(pricing_config),
        "models": rows,
        "no_model_inference_triggered": True,
        "no_paid_generation_api_call_triggered": True,
        "no_gpu_work_triggered": True,
        "selection_order": model_aliases,
    }


def write_audit(
    report: dict[str, Any],
    output_root: str | Path,
) -> tuple[Path, Path]:
    """Write pricing audit JSON and CSV."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase4_api_pricing_audit_report.json"
    summary_path = output / "phase4_api_pricing_audit_summary.csv"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report["models"]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report_path, summary_path


def main(argv: list[str] | None = None) -> int:
    """Run pricing audit."""

    args = build_parser().parse_args(argv)
    try:
        report = audit_pricing(
            model_aliases=args.models,
            pricing_config=args.pricing_config,
        )
        report_path, summary_path = write_audit(report, args.output_root)
        registry_outputs = write_model_registry_artifacts(
            output_root=args.output_root,
            pricing_path=args.pricing_config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Pricing audit failed: {exc}", file=sys.stderr)
        return 1
    for row in report["models"]:
        print(
            f"{row['model_alias']}: status={row['registry_pricing_status']} "
            f"runnable={row['runnable_for_costed_smoke']} "
            f"provider={row['resolved_provider'] or 'none'}"
        )
    print(f"Pricing audit report: {report_path}")
    print(f"Pricing audit summary: {summary_path}")
    for label, path in registry_outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
