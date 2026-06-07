"""Public model alias registry and switch-report generation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_bench.api_pricing import (
    load_api_pricing_registry,
    resolve_api_pricing,
)
from inference_bench.api_routes import resolve_api_provider_route
from inference_bench.config import (
    DEPRECATED_MODEL_ALIASES_KEY,
    MODEL_ALIASES_KEY,
    load_project_config,
    load_yaml_file,
)


def build_model_alias_rows(
    *,
    models_path: str | Path = "configs/models.yaml",
    pricing_path: str | Path = "configs/api_pricing.yaml",
) -> list[dict[str, Any]]:
    """Build a human-readable alias table in configured order."""

    config = load_project_config(models_path=models_path)
    raw = load_yaml_file(models_path)
    pricing_registry = load_api_pricing_registry(pricing_path)
    rows: list[dict[str, Any]] = []
    for alias_group, deprecated in (
        (MODEL_ALIASES_KEY, False),
        (DEPRECATED_MODEL_ALIASES_KEY, True),
    ):
        aliases = raw.get(alias_group, {})
        if not isinstance(aliases, dict):
            continue
        for alias, canonical_key in aliases.items():
            if not isinstance(alias, str) or not isinstance(canonical_key, str):
                continue
            model = config.models[canonical_key]
            pricing = pricing_registry.get(alias)
            api_key_env: str | None = None
            if pricing is not None and pricing.pricing_status != "unavailable":
                route = resolve_api_provider_route(
                    model=model,
                    pricing=resolve_api_pricing(alias, pricing_path),
                )
                api_key_env = route.api_key_env
            rows.append(
                {
                    "model_alias": alias,
                    "canonical_key": canonical_key,
                    "name": model.name,
                    "model_id": model.model_id,
                    "provider": model.provider,
                    "execution_target": model.execution_target,
                    "intended_role": model.intended_role,
                    "active_public_alias": not deprecated,
                    "deprecated_alias": deprecated,
                    "pricing_status": pricing.pricing_status if pricing else "not_applicable",
                    "input_usd_per_1m_tokens": (
                        pricing.input_usd_per_1m_tokens if pricing else None
                    ),
                    "output_usd_per_1m_tokens": (
                        pricing.output_usd_per_1m_tokens if pricing else None
                    ),
                    "api_key_env": api_key_env,
                }
            )
    return rows


def build_model_registry_report(
    *,
    models_path: str | Path = "configs/models.yaml",
    pricing_path: str | Path = "configs/api_pricing.yaml",
) -> dict[str, Any]:
    """Build the complete public alias registry report."""

    rows = build_model_alias_rows(models_path=models_path, pricing_path=pricing_path)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models_path": str(models_path),
        "pricing_path": str(pricing_path),
        "public_alias_order": [row["model_alias"] for row in rows if row["active_public_alias"]],
        "aliases": rows,
        "secret_values_recorded": False,
        "paid_api_call_triggered": False,
    }


def build_model5_switch_report(
    registry_report: dict[str, Any],
) -> dict[str, Any]:
    """Describe the active model5 switch while retaining the old route."""

    rows = registry_report["aliases"]
    model5 = next(row for row in rows if row["model_alias"] == "model5_gated")
    old_model5 = next(row for row in rows if row["model_alias"] == "old_model5_llama_3_2_3b")
    model6 = next(row for row in rows if row["model_alias"] == "model6_gated")
    return {
        "generated_at_utc": registry_report["generated_at_utc"],
        "switch_status": "READY_FOR_EXPLICIT_TINY_OPENROUTER_SMOKE",
        "old_active_model": old_model5,
        "new_active_model": model5,
        "model6_gated": model6,
        "switch_reason": (
            "The previous Featherless route exposed no complete per-token price. "
            "Ministral 3B has auditable OpenRouter input and output token rates."
        ),
        "live_smoke_requires": "OPENROUTER_API_KEY",
        "paid_api_call_triggered": False,
        "raw_paid_output_generated": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model_registry_artifacts(
    *,
    output_root: str | Path,
    models_path: str | Path = "configs/models.yaml",
    pricing_path: str | Path = "configs/api_pricing.yaml",
) -> dict[str, Path]:
    """Write the requested registry and model5 switch JSON/CSV artifacts."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    registry = build_model_registry_report(
        models_path=models_path,
        pricing_path=pricing_path,
    )
    switch = build_model5_switch_report(registry)
    registry_report = output / "phase4_model_registry_report.json"
    registry_summary = output / "phase4_model_registry_summary.csv"
    switch_report = output / "phase4_model5_switch_report.json"
    switch_summary = output / "phase4_model5_switch_summary.csv"
    registry_report.write_text(
        json.dumps(registry, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(registry_summary, registry["aliases"])
    switch_report.write_text(
        json.dumps(switch, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        switch_summary,
        [
            {
                "switch_status": switch["switch_status"],
                "model_alias": switch["new_active_model"]["model_alias"],
                "canonical_key": switch["new_active_model"]["canonical_key"],
                "model_id": switch["new_active_model"]["model_id"],
                "provider": switch["new_active_model"]["provider"],
                "pricing_status": switch["new_active_model"]["pricing_status"],
                "input_usd_per_1m_tokens": switch["new_active_model"]["input_usd_per_1m_tokens"],
                "output_usd_per_1m_tokens": switch["new_active_model"]["output_usd_per_1m_tokens"],
                "api_key_env": switch["new_active_model"]["api_key_env"],
                "paid_api_call_triggered": switch["paid_api_call_triggered"],
            }
        ],
    )
    return {
        "registry_report": registry_report,
        "registry_summary": registry_summary,
        "switch_report": switch_report,
        "switch_summary": switch_summary,
    }
