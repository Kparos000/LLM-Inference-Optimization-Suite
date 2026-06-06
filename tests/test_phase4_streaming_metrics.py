import pytest

from inference_bench.streaming_metrics import (
    TimedStreamChunk,
    calculate_streaming_metrics,
)


def test_streaming_ttft_and_itl_are_captured_from_fixture_chunks() -> None:
    chunks = [
        TimedStreamChunk(
            100.0,
            {"choices": [{"delta": {"content": "hello"}}]},
        ),
        TimedStreamChunk(
            130.0,
            {"choices": [{"delta": {"content": " world"}}]},
        ),
        TimedStreamChunk(
            170.0,
            {"choices": [{"delta": {"content": "!"}}]},
        ),
        TimedStreamChunk(
            180.0,
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            },
        ),
    ]

    metrics = calculate_streaming_metrics(
        chunks,
        e2e_latency_ms=190.0,
        prompt="fixture prompt",
    )

    assert metrics.generated_text == "hello world!"
    assert metrics.ttft_ms == 100.0
    assert metrics.itl_p50_ms == 35.0
    assert metrics.itl_p95_ms == pytest.approx(39.5)
    assert metrics.itl_p99_ms == pytest.approx(39.9)
    assert metrics.tpot_ms == 45.0
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 3
    assert metrics.token_count_source == "provider_usage"


def test_streaming_unavailable_does_not_fake_ttft() -> None:
    metrics = calculate_streaming_metrics(
        [
            TimedStreamChunk(
                50.0,
                {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 0}},
            )
        ],
        e2e_latency_ms=60.0,
        prompt="fixture",
    )

    assert metrics.streaming_available is False
    assert metrics.ttft_ms is None
    assert metrics.tpot_ms is None
    assert metrics.itl_p50_ms is None


def test_streaming_fallback_token_source_is_explicit() -> None:
    metrics = calculate_streaming_metrics(
        [
            TimedStreamChunk(
                25.0,
                {"choices": [{"delta": {"content": "one two"}}]},
            )
        ],
        e2e_latency_ms=40.0,
        prompt="input words",
    )

    assert metrics.token_count_source == "whitespace_fallback"
    assert metrics.input_tokens == 2
    assert metrics.output_tokens == 2
