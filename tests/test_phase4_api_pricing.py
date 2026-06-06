from pathlib import Path

import pytest

from inference_bench.api_pricing import (
    load_api_pricing_registry,
    resolve_api_pricing,
)


def test_pricing_registry_loads_detected_and_unavailable_entries() -> None:
    registry = load_api_pricing_registry("configs/api_pricing.yaml")

    assert registry["model5_gated"].pricing_status == "unavailable"
    assert registry["model5_gated"].input_usd_per_1m_tokens is None
    assert registry["model6_gated"].pricing_status == "detected"
    assert registry["model6_gated"].input_usd_per_1m_tokens == pytest.approx(0.02)


def test_manual_override_is_used_when_detected_pricing_is_unavailable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        """
models:
  model5_gated:
    model_id: meta-llama/Llama-3.2-3B-Instruct
    provider: featherless-ai
    input_usd_per_1m_tokens: null
    output_usd_per_1m_tokens: null
    pricing_source: hugging_face_router_metadata
    pricing_source_url: https://example.test/live
    pricing_last_checked: "2026-06-06T00:00:00Z"
    pricing_status: unavailable
    notes: no complete live price
manual_overrides:
  model5_gated:
    provider: approved-provider
    input_usd_per_1m_tokens: 0.1
    output_usd_per_1m_tokens: 0.2
    pricing_source: reviewed_provider_price_page
    pricing_source_url: https://example.test/manual
    pricing_last_checked: "2026-06-06T00:01:00Z"
    notes: reviewed manual override
""".strip()
        + "\n",
        encoding="utf-8",
    )

    pricing = resolve_api_pricing("model5_gated", path)

    assert pricing.pricing_status == "manual_override"
    assert pricing.provider == "approved-provider"
    assert pricing.input_usd_per_1m_tokens == pytest.approx(0.1)
    assert pricing.output_usd_per_1m_tokens == pytest.approx(0.2)


def test_detected_pricing_takes_precedence_over_manual_override(tmp_path: Path) -> None:
    path = tmp_path / "pricing.yaml"
    path.write_text(
        """
models:
  model6_gated:
    model_id: meta-llama/Llama-3.1-8B-Instruct
    provider: detected-provider
    input_usd_per_1m_tokens: 0.02
    output_usd_per_1m_tokens: 0.05
    pricing_source: hugging_face_router_metadata
    pricing_source_url: https://example.test/live
    pricing_last_checked: "2026-06-06T00:00:00Z"
    pricing_status: detected
manual_overrides:
  model6_gated:
    provider: override-provider
    input_usd_per_1m_tokens: 1.0
    output_usd_per_1m_tokens: 2.0
    pricing_source: manual
    pricing_source_url: https://example.test/manual
    pricing_last_checked: "2026-06-06T00:01:00Z"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    pricing = resolve_api_pricing("model6_gated", path)

    assert pricing.provider == "detected-provider"
    assert pricing.pricing_status == "detected"


def test_missing_pricing_blocks_costed_run() -> None:
    with pytest.raises(ValueError, match="Missing API pricing"):
        resolve_api_pricing("model5_gated", "configs/api_pricing.yaml")
