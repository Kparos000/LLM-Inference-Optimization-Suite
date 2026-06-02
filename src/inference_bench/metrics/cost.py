"""Cost metric calculations."""

from __future__ import annotations


def _validate_non_negative_int(value: int, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


def _validate_non_negative_float(value: float, field_name: str) -> None:
    if value < 0:
        msg = f"{field_name} must be >= 0"
        raise ValueError(msg)


def estimate_token_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_million_tokens: float,
    output_cost_per_million_tokens: float,
) -> float:
    """Estimate token usage cost in USD."""

    _validate_non_negative_int(input_tokens, "input_tokens")
    _validate_non_negative_int(output_tokens, "output_tokens")
    _validate_non_negative_float(
        input_cost_per_million_tokens,
        "input_cost_per_million_tokens",
    )
    _validate_non_negative_float(
        output_cost_per_million_tokens,
        "output_cost_per_million_tokens",
    )

    input_cost = input_tokens / 1_000_000 * input_cost_per_million_tokens
    output_cost = output_tokens / 1_000_000 * output_cost_per_million_tokens
    return input_cost + output_cost


def estimate_api_token_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m_tokens_usd: float,
    output_cost_per_1m_tokens_usd: float,
) -> dict[str, float]:
    """Estimate backend-aware API token cost in USD."""

    _validate_non_negative_int(input_tokens, "input_tokens")
    _validate_non_negative_int(output_tokens, "output_tokens")
    _validate_non_negative_float(
        input_cost_per_1m_tokens_usd,
        "input_cost_per_1m_tokens_usd",
    )
    _validate_non_negative_float(
        output_cost_per_1m_tokens_usd,
        "output_cost_per_1m_tokens_usd",
    )

    input_cost = input_tokens / 1_000_000 * input_cost_per_1m_tokens_usd
    output_cost = output_tokens / 1_000_000 * output_cost_per_1m_tokens_usd
    return {
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_api_cost_usd": input_cost + output_cost,
    }
