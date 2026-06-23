from __future__ import annotations

from typer.testing import CliRunner

from inference_bench.calibration_manifest import load_runpod_calibration_profiles
from inference_bench.cli import app
from inference_bench.config import load_project_config
from inference_bench.gpu_price_registry import load_gpu_price_registry
from inference_bench.load_profiles import load_sequence_buckets, load_traffic_profiles
from inference_bench.optimization_negative_rules import load_optimization_negative_rules
from inference_bench.result_track_schema import RESULT_TRACK_JOIN_KEYS, validate_result_track_row
from inference_bench.runtime_registry import load_runtime_registry
from inference_bench.serving_profiles import load_serving_profiles
from inference_bench.slo import SLO_METRIC_FAMILIES, SLO_VERTICALS, load_slo_config
from inference_bench.slo_profiles import load_slo_profiles


def test_validate_config_cli_covers_production_config_files() -> None:
    result = CliRunner().invoke(app, ["validate-config"])

    assert result.exit_code == 0, result.output
    expected_lines = [
        "Configuration valid",
        "Models loaded: 10",
        "Model aliases loaded: 12",
        "Runtime engines loaded: 5",
        "Serving profiles loaded: 2",
        "Sequence length buckets loaded: 6 input, 6 output",
        "Traffic profiles loaded: 4",
        "Optimization negative-rule groups loaded: 8",
        "SLO targets loaded: 5 verticals, 7 metric families",
        "SLO profiles loaded: 1",
        "GPU price registry loaded: 26 GPUs",
        "RunPod calibration profiles loaded: 3",
        "Result track schema join keys loaded: 12",
    ]
    for expected in expected_lines:
        assert expected in result.output


def test_direct_config_loaders_cover_all_production_registries() -> None:
    project = load_project_config()
    runtime_registry = load_runtime_registry()
    serving_profiles = load_serving_profiles()
    sequence_buckets = load_sequence_buckets()
    traffic_profiles = load_traffic_profiles()
    negative_rules = load_optimization_negative_rules()
    slo_config = load_slo_config()
    slo_profiles = load_slo_profiles()
    gpu_prices = load_gpu_price_registry()
    calibration_profiles = load_runpod_calibration_profiles()

    assert "model2_3b" in project.model_aliases
    assert project.resolve_model_key("model2_1_5b") == "qwen2_5_1_5b_instruct"
    assert runtime_registry["tensorrt_llm"].status == "planned"
    assert runtime_registry["tensorrt_llm"].live_run_supported is False
    assert serving_profiles["remote_rtx3070_qwen3b_baseline_b7"].status == "unstable_observed"
    assert serving_profiles["remote_rtx3070_qwen3b_safe_v1"].status == "ready"
    assert len(sequence_buckets["input"]) == 6
    assert len(sequence_buckets["output"]) == 6
    assert set(traffic_profiles) == {
        "online_low_latency",
        "office_hours_bursty",
        "offline_throughput",
        "custom",
    }
    assert set(negative_rules) == {
        "quantization",
        "prefix_caching",
        "speculative_decoding",
        "tensor_parallelism",
        "disaggregated_prefill",
        "context_compression",
        "concurrency_increase",
        "stronger_model_escalation",
    }
    assert set(slo_config["verticals"]) == set(SLO_VERTICALS)
    assert len(SLO_METRIC_FAMILIES) == 7
    assert slo_profiles["default_profile"] in slo_profiles["profiles"]
    assert len(gpu_prices) == 26
    assert set(calibration_profiles) == {
        "A100_SXM_CALIBRATION",
        "H100_SXM_CALIBRATION",
        "L40S_CALIBRATION",
    }


def test_result_track_schema_smoke_row_is_validated_by_config_gate() -> None:
    assert RESULT_TRACK_JOIN_KEYS == (
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
    errors = validate_result_track_row(
        {
            "run_id": "run-1",
            "config_id": "cfg-1",
            "prompt_id": "airline-1",
            "vertical": "airline",
            "model_alias": "model2_3b",
            "memory_mode": "mm2_hybrid_top5",
            "runtime": "vllm",
            "backend_type": "self_hosted_gpu",
            "engine": "vllm",
            "hardware": "remote_rtx3070",
            "provider": "huggingface",
            "concurrency": 1,
            "api_cost_usd": None,
            "gpu_cost_usd": None,
            "gpu_hourly_price_usd": None,
        }
    )

    assert errors == []
