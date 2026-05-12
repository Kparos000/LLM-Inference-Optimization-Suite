import pytest

from inference_bench.metrics.latency import (
    calculate_end_to_end_latency_ms,
    calculate_tpot_ms,
    calculate_ttft_ms,
    summarize_latency_ms,
)


def test_calculates_ttft_ms() -> None:
    assert calculate_ttft_ms(1.0, 1.125) == 125.0


def test_rejects_invalid_timestamp_ordering() -> None:
    with pytest.raises(ValueError, match="end timestamp"):
        calculate_end_to_end_latency_ms(2.0, 1.0)


def test_tpot_returns_none_for_insufficient_output_tokens() -> None:
    assert calculate_tpot_ms(1.0, 2.0, 1) is None
    assert calculate_tpot_ms(1.0, 2.0, 0) is None


def test_calculates_tpot_for_multi_token_output() -> None:
    assert calculate_tpot_ms(1.0, 1.5, 6) == 100.0


def test_summarizes_latency_ms() -> None:
    summary = summarize_latency_ms([10.0, 20.0, 30.0, 40.0])

    assert summary["count"] == 4.0
    assert summary["min"] == 10.0
    assert summary["max"] == 40.0
    assert summary["p50"] == 25.0


def test_empty_latency_summary_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_latency_ms([])
