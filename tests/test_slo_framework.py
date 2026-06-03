from pathlib import Path

import pytest

from inference_bench.retrieval_quality_gate import build_retrieval_quality_gate_report
from inference_bench.retrieval_root_cause import load_slo_targets
from inference_bench.slo import (
    SLO_METRIC_FAMILIES,
    SLO_VERTICALS,
    build_slo_readiness_report,
    evaluate_metric,
    evaluate_metric_family,
    load_slo_config,
    validate_slo_config,
)


def test_slo_config_loads() -> None:
    config = load_slo_config("configs/slo_targets.yaml")

    assert config["version"] == 1
    assert set(config["verticals"]) == set(SLO_VERTICALS)


def test_all_five_verticals_exist() -> None:
    config = load_slo_config("configs/slo_targets.yaml")

    assert sorted(config["verticals"]) == sorted(
        ["airline", "retail", "healthcare_admin", "finance", "research_ai"]
    )


def test_every_vertical_has_same_metric_families() -> None:
    config = load_slo_config("configs/slo_targets.yaml")
    families_by_vertical = {
        vertical: set(payload)
        for vertical, payload in config["verticals"].items()
        if vertical in SLO_VERTICALS
    }

    assert all(families == set(SLO_METRIC_FAMILIES) for families in families_by_vertical.values())


def test_invalid_config_fails() -> None:
    config = load_slo_config("configs/slo_targets.yaml")
    broken = dict(config)
    broken["verticals"] = dict(config["verticals"])
    broken["verticals"].pop("finance")

    with pytest.raises(ValueError, match="missing verticals"):
        validate_slo_config(broken)


def test_pass_warn_blocked_not_available_logic_works() -> None:
    passing = evaluate_metric(
        vertical="finance",
        metric_family="retrieval_slo",
        metric_name="final_recall_at_5_min",
        target=0.90,
        observed=0.91,
    )
    warning = evaluate_metric(
        vertical="finance",
        metric_family="retrieval_slo",
        metric_name="final_recall_at_5_min",
        target=0.90,
        observed=0.82,
    )
    blocked = evaluate_metric(
        vertical="finance",
        metric_family="retrieval_slo",
        metric_name="final_recall_at_5_min",
        target=0.90,
        observed=0.50,
    )
    missing = evaluate_metric(
        vertical="finance",
        metric_family="latency_slo",
        metric_name="ttft_p95_ms_max",
        target=2500,
        observed=None,
    )

    assert passing.status == "PASS"
    assert warning.status == "WARN"
    assert blocked.status == "BLOCKED"
    assert missing.status == "NOT_AVAILABLE"


def test_retrieval_slo_comparison_works() -> None:
    config = load_slo_config("configs/slo_targets.yaml")
    rows = evaluate_metric_family(
        config=config,
        vertical="airline",
        metric_family="retrieval_slo",
        observations={
            "candidate_recall_at_20_min": 0.95,
            "candidate_recall_at_50_min": 0.96,
            "final_recall_at_5_min": 0.91,
            "mrr_min": 0.90,
        },
    )

    assert {row.status for row in rows} == {"PASS"}


def test_latency_slo_comparison_works() -> None:
    result = evaluate_metric(
        vertical="airline",
        metric_family="latency_slo",
        metric_name="ttft_p95_ms_max",
        target=1000,
        observed=1200,
    )

    assert result.status == "BLOCKED"
    assert result.gap == -200


def test_api_token_cost_slo_comparison_works() -> None:
    result = evaluate_metric(
        vertical="retail",
        metric_family="api_cost_slo",
        metric_name="api_cost_per_request_usd_max",
        target=0.05,
        observed=0.052,
    )

    assert result.status == "WARN"


def test_gpu_cost_slo_comparison_works() -> None:
    cost_result = evaluate_metric(
        vertical="research_ai",
        metric_family="gpu_cost_slo",
        metric_name="tokens_per_gpu_dollar_min",
        target=100000.0,
        observed=125000.0,
    )
    required_result = evaluate_metric(
        vertical="research_ai",
        metric_family="gpu_cost_slo",
        metric_name="gpu_hourly_price_usd_required",
        target=True,
        observed=False,
    )

    assert cost_result.status == "PASS"
    assert required_result.status == "BLOCKED"


def test_current_retrieval_report_produces_blocked_status() -> None:
    config = load_slo_config("configs/slo_targets.yaml")
    retrieval_report = Path("data/generated/context_engineering/retrieval_evaluation_report.json")

    report, rows = build_slo_readiness_report(
        slo_config=config,
        retrieval_report_path=retrieval_report,
        quality_gate_report_path="data/generated/context_engineering/retrieval_quality_gate_report.json",
    )

    assert retrieval_report.exists()
    assert report["inference_scaling_blocked_by_retrieval_slos"] is True
    assert any(
        row["metric_family"] == "retrieval_slo" and row["status"] == "BLOCKED" for row in rows
    )


def test_future_inference_metrics_are_not_available_without_reports() -> None:
    config = load_slo_config("configs/slo_targets.yaml")

    report, rows = build_slo_readiness_report(slo_config=config)

    assert report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
    assert any(row["metric_family"] == "latency_slo" for row in rows)
    assert all(row["status"] == "NOT_AVAILABLE" for row in rows)


def test_legacy_retrieval_targets_still_load_for_root_cause_scripts() -> None:
    targets = load_slo_targets("configs/slo_targets.yaml")

    assert targets["prompt_text_only"] == 0.70
    assert targets["prompt_plus_metadata"] == 0.80
    assert targets["finance_prompt_text_only"] == 0.65


def test_existing_quality_gate_still_blocks_below_slo_fixture() -> None:
    retrieval_rows = [
        {
            "split": "final_10000",
            "ablation_mode": "prompt_text_only",
            "memory_mode": "mm2_hybrid_top5",
            "vertical": "finance",
            "record_count": 10,
            "recall_at_5": 0.2,
        }
    ]
    compression_rows = [
        {
            "split": "final_10000",
            "record_count": 10,
            "token_reduction_pct": 0.1,
            "recall_loss": 0.0,
        }
    ]

    report, _summary_rows = build_retrieval_quality_gate_report(retrieval_rows, compression_rows)

    assert report["quality_gate_status"] == "BLOCKED"
    assert report["no_model_inference_triggered"] is True
