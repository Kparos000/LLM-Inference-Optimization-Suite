"""Cache-readiness metrics for production inference planning."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_.$:-]+", text.lower())


def _common_prefix_length(sequences: list[list[str]]) -> int:
    if not sequences:
        return 0
    minimum_length = min(len(sequence) for sequence in sequences)
    prefix = 0
    for index in range(minimum_length):
        value = sequences[0][index]
        if all(sequence[index] == value for sequence in sequences[1:]):
            prefix += 1
        else:
            break
    return prefix


@dataclass(frozen=True)
class CacheReadinessMetrics:
    """Estimated prefix/cache suitability for a workload slice."""

    request_count: int
    repeated_prefix_tokens: int
    shared_context_percentage: float
    prefix_reuse_potential: float
    kv_cache_pressure_estimate: float
    cacheability_score: float
    estimated_prefix_cache_benefit: float

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-serializable metrics."""

        return asdict(self)


def estimate_kv_cache_pressure(
    *,
    input_tokens: list[int],
    output_tokens: list[int],
    concurrency: int,
    context_window_tokens: int,
) -> float:
    """Estimate KV-cache pressure as active tokens over capacity."""

    if concurrency <= 0:
        msg = "concurrency must be > 0"
        raise ValueError(msg)
    if context_window_tokens <= 0:
        msg = "context_window_tokens must be > 0"
        raise ValueError(msg)
    if len(input_tokens) != len(output_tokens):
        msg = "input_tokens and output_tokens must have the same length"
        raise ValueError(msg)
    if not input_tokens:
        return 0.0
    mean_tokens = sum(i + o for i, o in zip(input_tokens, output_tokens, strict=True)) / len(
        input_tokens
    )
    return min(1.0, (mean_tokens * concurrency) / context_window_tokens)


def calculate_cache_readiness(
    *,
    prompts: list[str],
    context_blocks: list[list[str]] | None,
    input_tokens: list[int],
    output_tokens: list[int],
    concurrency: int,
    context_window_tokens: int = 4096,
) -> CacheReadinessMetrics:
    """Calculate deterministic cache-readiness metrics for a workload."""

    if len(prompts) != len(input_tokens) or len(prompts) != len(output_tokens):
        msg = "prompts, input_tokens, and output_tokens must have the same length"
        raise ValueError(msg)
    if context_blocks is not None and len(context_blocks) != len(prompts):
        msg = "context_blocks must match prompt count"
        raise ValueError(msg)
    tokenized = [_tokens(prompt) for prompt in prompts]
    common_prefix = _common_prefix_length(tokenized)
    mean_input_tokens = sum(input_tokens) / len(input_tokens) if input_tokens else 0.0
    repeated_prefix_tokens = common_prefix * max(len(prompts) - 1, 0)
    prefix_reuse_potential = (
        min(1.0, common_prefix / mean_input_tokens) if mean_input_tokens > 0 else 0.0
    )

    shared_context_percentage = 0.0
    if context_blocks:
        context_sets = [set(blocks) for blocks in context_blocks]
        union = set().union(*context_sets) if context_sets else set()
        intersection = set.intersection(*context_sets) if context_sets else set()
        shared_context_percentage = len(intersection) / len(union) if union else 0.0

    kv_pressure = estimate_kv_cache_pressure(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        context_window_tokens=context_window_tokens,
    )
    cacheability_score = min(
        1.0,
        max(
            0.0,
            0.55 * prefix_reuse_potential
            + 0.35 * shared_context_percentage
            + 0.10 * (1.0 - kv_pressure),
        ),
    )
    estimated_prefix_cache_benefit = min(
        1.0,
        cacheability_score * (1.0 - math.sqrt(kv_pressure) * 0.25),
    )
    return CacheReadinessMetrics(
        request_count=len(prompts),
        repeated_prefix_tokens=repeated_prefix_tokens,
        shared_context_percentage=round(shared_context_percentage, 6),
        prefix_reuse_potential=round(prefix_reuse_potential, 6),
        kv_cache_pressure_estimate=round(kv_pressure, 6),
        cacheability_score=round(cacheability_score, 6),
        estimated_prefix_cache_benefit=round(estimated_prefix_cache_benefit, 6),
    )
