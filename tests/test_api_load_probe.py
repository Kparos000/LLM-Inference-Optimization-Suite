from __future__ import annotations

from pathlib import Path
from typing import cast

from inference_bench.api_load_probe import (
    build_api_load_probe_plan,
    build_framework_only_api_probe_report,
    summarize_api_probe_results,
    write_api_probe_artifacts,
)


def _row(
    *,
    model_alias: str = "model6_gated",
    concurrency: int = 1,
    success: bool = True,
    streaming_stable: bool = True,
    http_429_count: int = 0,
    http_5xx_count: int = 0,
    timeout_count: int = 0,
    retry_count: int = 0,
) -> dict[str, object]:
    return {
        "model_alias": model_alias,
        "concurrency": concurrency,
        "success": success,
        "streaming_stable": streaming_stable,
        "ttft_ms": 120.0,
        "tpot_ms": 11.0,
        "latency_ms": 1500.0,
        "requests_per_second": 1.0,
        "tokens_per_second": 80.0,
        "http_429_count": http_429_count,
        "http_5xx_count": http_5xx_count,
        "timeout_count": timeout_count,
        "retry_count": retry_count,
        "provider_throttling_count": http_429_count,
    }


def test_api_load_probe_plan_covers_gated_models_and_concurrency_levels() -> None:
    plan = build_api_load_probe_plan()

    assert plan["live_probe_executed"] is False
    assert plan["supported_model_aliases"] == ["model5_gated", "model6_gated", "model7_gated"]
    assert plan["concurrency_levels"] == [1, 2, 4, 8, 16]
    assert len(cast(list[object], plan["planned_rows"])) == 15


def test_framework_only_probe_is_blocked_without_live_requests() -> None:
    report = build_framework_only_api_probe_report()

    assert report["status"] == "API_PROBE_BLOCKED"
    assert report["blocked_reason"] == "live_api_probe_not_executed"
    assert report["no_live_requests_were_sent"] is True


def test_api_probe_pass_warning_and_blocked_verdicts() -> None:
    passed = summarize_api_probe_results([_row() for _ in range(10)], live_probe_executed=True)
    warned = summarize_api_probe_results(
        [_row(http_429_count=1) for _ in range(1)] + [_row() for _ in range(19)],
        live_probe_executed=True,
    )
    blocked = summarize_api_probe_results(
        [_row(timeout_count=1, success=False)] + [_row() for _ in range(20)],
        live_probe_executed=True,
    )

    assert passed["verdict"] == "API_PROBE_PASSED"
    assert warned["verdict"] == "API_PROBE_WARNING"
    assert blocked["verdict"] == "API_PROBE_BLOCKED"


def test_api_probe_artifact_writer(tmp_path: Path) -> None:
    report = build_framework_only_api_probe_report()
    report_path, summary_path = write_api_probe_artifacts(
        report=report,
        report_path=tmp_path / "api_load_probe_report.json",
        summary_path=tmp_path / "api_load_probe_summary.csv",
    )

    assert report_path.exists()
    assert summary_path.exists()
    assert "API_PROBE_BLOCKED" in report_path.read_text(encoding="utf-8")
    assert "live_probe_executed" in summary_path.read_text(encoding="utf-8")
