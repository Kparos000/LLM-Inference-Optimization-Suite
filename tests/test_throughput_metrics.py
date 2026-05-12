import pytest

from inference_bench.metrics.throughput import (
    calculate_requests_per_second,
    calculate_tokens_per_second,
)


def test_calculates_token_throughput() -> None:
    assert calculate_tokens_per_second(100, 2.0) == 50.0


def test_calculates_request_throughput() -> None:
    assert calculate_requests_per_second(8, 4.0) == 2.0


def test_rejects_invalid_elapsed_seconds() -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        calculate_tokens_per_second(100, 0.0)


def test_zero_count_returns_zero_throughput() -> None:
    assert calculate_requests_per_second(0, 2.0) == 0.0
