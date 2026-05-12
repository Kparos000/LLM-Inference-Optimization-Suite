"""Latency metric calculations."""

from __future__ import annotations

from collections.abc import Sequence


def _validate_non_negative_timestamp(value: float, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


def _validate_timestamp_order(start_s: float, end_s: float) -> None:
    if end_s < start_s:
        msg = "end timestamp must not be earlier than start timestamp"
        raise ValueError(msg)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def calculate_ttft_ms(request_start_s: float, first_token_s: float) -> float:
    """Calculate time to first token in milliseconds."""

    _validate_non_negative_timestamp(request_start_s, "request_start_s")
    _validate_non_negative_timestamp(first_token_s, "first_token_s")
    _validate_timestamp_order(request_start_s, first_token_s)
    return (first_token_s - request_start_s) * 1000


def calculate_end_to_end_latency_ms(request_start_s: float, request_end_s: float) -> float:
    """Calculate end-to-end request latency in milliseconds."""

    _validate_non_negative_timestamp(request_start_s, "request_start_s")
    _validate_non_negative_timestamp(request_end_s, "request_end_s")
    _validate_timestamp_order(request_start_s, request_end_s)
    return (request_end_s - request_start_s) * 1000


def calculate_tpot_ms(
    first_token_s: float,
    request_end_s: float,
    output_tokens: int,
) -> float | None:
    """Calculate time per output token in milliseconds."""

    _validate_non_negative_timestamp(first_token_s, "first_token_s")
    _validate_non_negative_timestamp(request_end_s, "request_end_s")
    _validate_timestamp_order(first_token_s, request_end_s)
    if output_tokens <= 1:
        return None
    return (request_end_s - first_token_s) * 1000 / (output_tokens - 1)


def summarize_latency_ms(values: Sequence[float]) -> dict[str, float]:
    """Summarize latency values in milliseconds."""

    if not values:
        msg = "values must not be empty"
        raise ValueError(msg)
    if any(value < 0 for value in values):
        msg = "latency values must be >= 0"
        raise ValueError(msg)

    sorted_values = sorted(values)
    return {
        "count": float(len(sorted_values)),
        "mean": sum(sorted_values) / len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p50": _percentile(sorted_values, 0.50),
        "p90": _percentile(sorted_values, 0.90),
        "p95": _percentile(sorted_values, 0.95),
        "p99": _percentile(sorted_values, 0.99),
    }
