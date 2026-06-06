import json
from pathlib import Path

import pytest

from inference_bench.api_priced_validation import AccessCheck
from inference_bench.api_pricing import (
    load_manual_pricing_override_state,
    resolve_api_pricing,
)
from inference_bench.model5_pricing_routing import (
    audit_model5_route,
    build_model5_route_decision,
    provider_routes_from_router_payload,
    write_model5_route_audit,
)


def _router_payload() -> dict[str, object]:
    return {
        "data": {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "providers": [
                {
                    "provider": "featherless-ai",
                    "status": "live",
                }
            ],
        }
    }


def _write_pricing(
    path: Path,
    *,
    live: bool = False,
    override: bool = False,
) -> Path:
    live_input = "0.03" if live else "null"
    live_output = "0.06" if live else "null"
    live_status = "detected" if live else "unavailable"
    override_block = (
        """
manual_overrides:
  model5_gated:
    enabled: true
    provider: audited-provider
    model_id: meta-llama/Llama-3.2-3B-Instruct
    input_usd_per_1m_tokens: 0.10
    output_usd_per_1m_tokens: 0.20
    pricing_source_url: https://example.test/audited-price
    pricing_status: manual_override
    last_checked: "2026-06-06"
    notes: reviewed exact token rates
"""
        if override
        else "manual_overrides: {}\n"
    )
    path.write_text(
        f"""
models:
  model5_gated:
    model_id: meta-llama/Llama-3.2-3B-Instruct
    provider: live-provider
    input_usd_per_1m_tokens: {live_input}
    output_usd_per_1m_tokens: {live_output}
    pricing_source_url: https://example.test/live
    pricing_last_checked: "2026-06-06T00:00:00Z"
    pricing_status: {live_status}
{override_block}""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_model5_route_audit_works_without_paid_call(tmp_path: Path) -> None:
    pricing = _write_pricing(tmp_path / "pricing.yaml", override=True)
    calls = {"token": 0, "access": 0, "metadata": 0}

    def token_checker(_token: str) -> AccessCheck:
        calls["token"] += 1
        return AccessCheck(True, 200)

    def access_checker(_model_id: str, _token: str) -> AccessCheck:
        calls["access"] += 1
        return AccessCheck(True, 200)

    def metadata_fetcher(_model_id: str) -> dict[str, object]:
        calls["metadata"] += 1
        return _router_payload()

    report = audit_model5_route(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        pricing_config=pricing,
        hf_token="fixture-secret",
        token_checker=token_checker,
        access_checker=access_checker,
        metadata_fetcher=metadata_fetcher,
    )
    report_path, summary_path = write_model5_route_audit(report, tmp_path / "out")

    assert calls == {"token": 1, "access": 1, "metadata": 1}
    assert report["model_access_granted"] is True
    assert report["selected_provider"] == "featherless-ai"
    assert report["chat_completion_supported"] is True
    assert report["streaming_supported"] is True
    assert report["costed_smoke_allowed"] is True
    assert report["no_generation_request_sent"] is True
    assert report["no_paid_api_call_sent"] is True
    assert "fixture-secret" not in report_path.read_text(encoding="utf-8")
    assert summary_path.is_file()


def test_manual_pricing_override_works(tmp_path: Path) -> None:
    pricing = _write_pricing(tmp_path / "pricing.yaml", override=True)

    resolved = resolve_api_pricing("model5_gated", pricing)
    override = load_manual_pricing_override_state("model5_gated", pricing)

    assert override.present is True
    assert override.enabled is True
    assert resolved.pricing_status == "manual_override"
    assert resolved.provider == "audited-provider"
    assert resolved.input_usd_per_1m_tokens == pytest.approx(0.10)
    assert resolved.output_usd_per_1m_tokens == pytest.approx(0.20)


def test_missing_pricing_blocks_run(tmp_path: Path) -> None:
    pricing = _write_pricing(tmp_path / "pricing.yaml")
    decision = build_model5_route_decision(
        pricing_config=pricing,
        token_check=AccessCheck(True, 200),
        model_access_check=AccessCheck(True, 200),
        provider_routes=provider_routes_from_router_payload(_router_payload()),
    )

    assert decision["pricing_resolved"] is False
    assert decision["costed_smoke_allowed"] is False
    assert any(
        "No complete live token pricing" in reason for reason in decision["blocking_reasons"]
    )


def test_live_pricing_beats_manual_override(tmp_path: Path) -> None:
    pricing = _write_pricing(tmp_path / "pricing.yaml", live=True, override=True)

    resolved = resolve_api_pricing("model5_gated", pricing)

    assert resolved.pricing_status == "detected"
    assert resolved.provider == "live-provider"
    assert resolved.input_usd_per_1m_tokens == pytest.approx(0.03)


def test_checked_in_disabled_override_does_not_enable_costed_run() -> None:
    override = load_manual_pricing_override_state(
        "model5_gated",
        "configs/api_pricing.yaml",
    )

    assert override.present is True
    assert override.enabled is False
    with pytest.raises(ValueError, match="Missing API pricing"):
        resolve_api_pricing("model5_gated", "configs/api_pricing.yaml")


def test_route_report_is_json_serializable(tmp_path: Path) -> None:
    pricing = _write_pricing(tmp_path / "pricing.yaml", override=True)
    decision = build_model5_route_decision(
        pricing_config=pricing,
        token_check=AccessCheck(True, 200),
        model_access_check=AccessCheck(True, 200),
        provider_routes=provider_routes_from_router_payload(_router_payload()),
    )

    assert json.loads(json.dumps(decision))["costed_smoke_allowed"] is True
