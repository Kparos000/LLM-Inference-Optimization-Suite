"""Throughput metric calculations."""

from __future__ import annotations


def _validate_non_negative_count(value: int, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


def _validate_positive_elapsed_seconds(elapsed_seconds: float) -> None:
    if elapsed_seconds <= 0:
        msg = "elapsed_seconds must be > 0"
        raise ValueError(msg)


def calculate_tokens_per_second(total_tokens: int, elapsed_seconds: float) -> float:
    """Calculate token throughput."""

    _validate_non_negative_count(total_tokens, "total_tokens")
    _validate_positive_elapsed_seconds(elapsed_seconds)
    if total_tokens == 0:
        return 0.0
    return total_tokens / elapsed_seconds


def calculate_requests_per_second(total_requests: int, elapsed_seconds: float) -> float:
    """Calculate request throughput."""

    _validate_non_negative_count(total_requests, "total_requests")
    _validate_positive_elapsed_seconds(elapsed_seconds)
    if total_requests == 0:
        return 0.0
    return total_requests / elapsed_seconds
