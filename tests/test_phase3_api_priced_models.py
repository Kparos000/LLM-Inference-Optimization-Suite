import os
import subprocess
import sys
from pathlib import Path

import pytest

from inference_bench.api_pricing import (
    ApiPricingEntry,
    estimate_api_cost_from_pricing,
    load_api_pricing_config,
    resolve_api_pricing,
)
from inference_bench.config import load_project_config


def write_pricing_config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "pricing_source_url: https://router.huggingface.co/v1/models",
                "snapshot_timestamp_utc: '2026-06-01T00:00:00+00:00'",
                "models:",
                "  model5_gated:",
                "    model_alias: model5_gated",
                "    model_id: meta-llama/Llama-3.2-3B-Instruct",
                "    provider: test-provider",
                "    provider_status: live",
                "    input_cost_per_1m_tokens_usd: 0.10",
                "    output_cost_per_1m_tokens_usd: 0.20",
                "    context_length: 8192",
                "    latency_seconds_if_available: 0.5",
                "    throughput_tokens_per_second_if_available: 100.0",
                "    supports_tools_if_available: false",
                "    supports_structured_output_if_available: true",
                "    pricing_snapshot_timestamp_utc: '2026-06-01T00:00:00+00:00'",
                "    pricing_source_url: https://router.huggingface.co/v1/models/meta-llama/Llama-3.2-3B-Instruct",
                "    selected_for_experiment: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_old_qwen_aliases_still_resolve() -> None:
    config = load_project_config()

    assert config.resolve_model_config("model1_0_5b").model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.resolve_model_config("model2_1_5b").model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.resolve_model_config("model3_7b").model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.resolve_model_config("model4_32b").model_id == "Qwen/Qwen2.5-32B-Instruct"


def test_new_gated_and_large_aliases_resolve() -> None:
    config = load_project_config()

    assert (
        config.resolve_model_config("model5_gated").model_id == "meta-llama/Llama-3.2-3B-Instruct"
    )
    assert (
        config.resolve_model_config("model6_gated").model_id == "meta-llama/Llama-3.1-8B-Instruct"
    )
    assert config.resolve_model_config("model7_large_placeholder").model_id == (
        "placeholder/large-model"
    )
    assert config.resolve_model_config("model5_large_placeholder").model_id == (
        "placeholder/large-model"
    )


def test_api_model_metadata_indicates_gated_access_and_hf_token_requirement() -> None:
    config = load_project_config()

    for alias in ("model5_gated", "model6_gated"):
        model_config = config.resolve_model_config(alias)
        assert model_config.access_type == "gated"
        assert model_config.requires_hf_token is True
        assert model_config.requires_license_acceptance is True
        assert model_config.execution_target == "hf_inference_provider_api"
        assert "hf_inference_provider" in model_config.allowed_backends


def test_model7_large_placeholder_is_last_public_alias() -> None:
    config = load_project_config()
    public_aliases = [
        alias
        for alias in config.model_aliases
        if alias.startswith("model") and alias != "model5_large_placeholder"
    ]

    assert public_aliases[:7] == [
        "model1_0_5b",
        "model2_1_5b",
        "model3_7b",
        "model4_32b",
        "model5_gated",
        "model6_gated",
        "model7_large_placeholder",
    ]


def test_pricing_config_loads() -> None:
    entries = load_api_pricing_config("configs/api_pricing.yaml")

    assert entries == {}


def test_cost_calculation_works_with_fixture_pricing() -> None:
    pricing = ApiPricingEntry(
        model_alias="model5_gated",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        provider="test-provider",
        provider_status="live",
        input_cost_per_1m_tokens_usd=0.10,
        output_cost_per_1m_tokens_usd=0.20,
        pricing_snapshot_timestamp_utc="2026-06-01T00:00:00+00:00",
        pricing_source_url="https://router.huggingface.co/v1/models/meta-llama/Llama-3.2-3B-Instruct",
    )

    cost = estimate_api_cost_from_pricing(input_tokens=1000, output_tokens=500, pricing=pricing)

    assert cost["input_cost_usd"] == pytest.approx(0.0001)
    assert cost["output_cost_usd"] == pytest.approx(0.0001)
    assert cost["total_api_cost_usd"] == pytest.approx(0.0002)


def test_cost_calculation_refuses_missing_pricing(tmp_path: Path) -> None:
    empty_config = tmp_path / "api_pricing.yaml"
    empty_config.write_text("models: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing API pricing"):
        resolve_api_pricing("model5_gated", empty_config)


def test_smoke_script_refuses_without_allow_paid_api_call(tmp_path: Path) -> None:
    pricing_path = write_pricing_config(tmp_path / "api_pricing.yaml")
    env = os.environ.copy()
    env["HF_TOKEN"] = "hf_test_secret_should_not_print"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase3/hf_api_tiny_smoke.py",
            "--model",
            "model5_gated",
            "--pricing-config",
            str(pricing_path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 1
    assert "allow-paid-api-call" in result.stdout
    assert "hf_test_secret_should_not_print" not in result.stdout
    assert "hf_test_secret_should_not_print" not in result.stderr


def test_smoke_script_never_prints_hf_token_when_pricing_missing(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HF_TOKEN"] = "hf_another_secret_should_not_print"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase3/hf_api_tiny_smoke.py",
            "--model",
            "model5_gated",
            "--pricing-config",
            str(tmp_path / "missing.yaml"),
            "--allow-paid-api-call",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "hf_another_secret_should_not_print" not in result.stdout
    assert "hf_another_secret_should_not_print" not in result.stderr


def test_config_validation_still_passes() -> None:
    subprocess.run(
        ["inference-bench", "validate-config"],
        check=True,
        capture_output=True,
        text=True,
    )
