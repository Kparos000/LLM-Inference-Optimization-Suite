"""Streaming OpenAI-compatible response parsing and latency metrics."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inference_bench.runners.mock_runner import count_whitespace_tokens


@dataclass(frozen=True)
class TimedStreamChunk:
    """One parsed server-sent event with its arrival time."""

    arrival_ms: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class StreamingMetrics:
    """Measured streaming output and token/latency data."""

    generated_text: str
    ttft_ms: float | None
    itl_p50_ms: float | None
    itl_p95_ms: float | None
    itl_p99_ms: float | None
    tpot_ms: float | None
    e2e_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_count_source: str
    content_chunk_count: int
    streaming_available: bool


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile."""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _delta_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _usage(payloads: Iterable[dict[str, Any]]) -> tuple[int | None, int | None]:
    for payload in payloads:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            return max(0, input_tokens), max(0, output_tokens)
    return None, None


def calculate_streaming_metrics(
    chunks: list[TimedStreamChunk],
    *,
    e2e_latency_ms: float,
    prompt: str,
) -> StreamingMetrics:
    """Calculate TTFT, inter-chunk latency, TPOT, and token counts."""

    content_events = [
        (chunk.arrival_ms, content)
        for chunk in chunks
        if (content := _delta_content(chunk.payload))
    ]
    generated_text = "".join(content for _, content in content_events)
    ttft_ms = content_events[0][0] if content_events else None
    arrival_times = [arrival for arrival, _ in content_events]
    inter_arrival = [
        arrival_times[index] - arrival_times[index - 1] for index in range(1, len(arrival_times))
    ]
    input_tokens, output_tokens = _usage(chunk.payload for chunk in chunks)
    token_source = "provider_usage"
    if input_tokens is None or output_tokens is None:
        input_tokens = count_whitespace_tokens(prompt)
        output_tokens = count_whitespace_tokens(generated_text)
        token_source = "whitespace_fallback"
    tpot_ms = None
    if ttft_ms is not None and output_tokens > 1:
        tpot_ms = max(0.0, e2e_latency_ms - ttft_ms) / (output_tokens - 1)
    return StreamingMetrics(
        generated_text=generated_text,
        ttft_ms=ttft_ms,
        itl_p50_ms=percentile(inter_arrival, 0.50),
        itl_p95_ms=percentile(inter_arrival, 0.95),
        itl_p99_ms=percentile(inter_arrival, 0.99),
        tpot_ms=tpot_ms,
        e2e_latency_ms=e2e_latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        token_count_source=token_source,
        content_chunk_count=len(content_events),
        streaming_available=bool(content_events),
    )


def request_streaming_chat_completion(
    *,
    hf_token: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    api_route: str,
    timeout_seconds: float = 120.0,
) -> StreamingMetrics:
    """Run one streaming request and measure server-sent event arrival times."""

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = Request(
        api_route,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.perf_counter()
    chunks: list[TimedStreamChunk] = []
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type") or "")
            if "text/event-stream" not in content_type.lower():
                msg = (
                    f"Streaming unavailable: response content type was {content_type or 'unknown'}"
                )
                raise RuntimeError(msg)
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    chunks.append(
                        TimedStreamChunk(
                            arrival_ms=(time.perf_counter() - started) * 1000,
                            payload=payload,
                        )
                    )
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code}: {body_text[:500]}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Network error: {exc.reason}"
        raise RuntimeError(msg) from exc
    e2e_latency_ms = (time.perf_counter() - started) * 1000
    return calculate_streaming_metrics(
        chunks,
        e2e_latency_ms=e2e_latency_ms,
        prompt=prompt,
    )
