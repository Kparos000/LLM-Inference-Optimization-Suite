import json
from pathlib import Path

from inference_bench.model5_streaming_validation import (
    fallback_allowed_after_primary_execution,
    write_blocked_model5_artifacts,
)


def test_fallback_requires_actual_primary_execution_failure() -> None:
    assert (
        fallback_allowed_after_primary_execution(
            primary_execution_attempted=False,
            primary_success_count=0,
        )
        is False
    )
    assert (
        fallback_allowed_after_primary_execution(
            primary_execution_attempted=True,
            primary_success_count=0,
        )
        is True
    )
    assert (
        fallback_allowed_after_primary_execution(
            primary_execution_attempted=True,
            primary_success_count=1,
        )
        is False
    )


def test_blocked_artifacts_do_not_fabricate_cost_or_metrics(tmp_path: Path) -> None:
    outputs = write_blocked_model5_artifacts(
        raw_path=tmp_path / "raw" / "results.jsonl",
        processed_root=tmp_path / "processed",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        provider="featherless-ai",
        planned_prompt_count=5,
        reason="pricing unavailable",
        pricing_status="unavailable",
        manual_override_configured=False,
    )

    raw = json.loads(outputs["raw"].read_text(encoding="utf-8"))
    cost = json.loads(outputs["cost_report"].read_text(encoding="utf-8"))
    latency = json.loads(outputs["latency_report"].read_text(encoding="utf-8"))

    assert raw["execution_attempted"] is False
    assert raw["paid_api_call_triggered"] is False
    assert raw["total_cost_usd"] is None
    assert cost["total_cost_usd"] is None
    assert cost["cost_per_request_usd"] is None
    assert latency["ttft_ms"] is None
    assert latency["streaming_success_count"] == 0
