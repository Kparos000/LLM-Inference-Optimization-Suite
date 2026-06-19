"""Production load profiles and sequence-length distribution helpers."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.config import load_yaml_file

DEFAULT_LOAD_PROFILES_PATH = "configs/load_profiles.yaml"
TrafficProfileName = Literal[
    "online_low_latency",
    "office_hours_bursty",
    "offline_throughput",
    "custom",
]
ArrivalMode = Literal["closed_loop", "jittered_poisson", "bursty_jittered", "custom"]


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{field_name} must be a non-empty string"
        raise ValueError(msg)
    return value


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{field_name} must be a positive integer"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class SequenceLengthBucket:
    """Inclusive/exclusive token bucket for ISL or OSL distributions."""

    id: str
    min_tokens: int
    max_tokens_exclusive: int | None

    def __post_init__(self) -> None:
        _non_empty(self.id, "id")
        if not isinstance(self.min_tokens, int) or isinstance(self.min_tokens, bool):
            msg = "min_tokens must be an integer"
            raise ValueError(msg)
        if self.min_tokens < 0:
            msg = "min_tokens must be >= 0"
            raise ValueError(msg)
        if self.max_tokens_exclusive is not None:
            if (
                not isinstance(self.max_tokens_exclusive, int)
                or isinstance(self.max_tokens_exclusive, bool)
                or self.max_tokens_exclusive <= self.min_tokens
            ):
                msg = "max_tokens_exclusive must be greater than min_tokens"
                raise ValueError(msg)

    def contains(self, token_count: int) -> bool:
        """Return whether token_count belongs in this bucket."""

        if token_count < self.min_tokens:
            return False
        if self.max_tokens_exclusive is None:
            return True
        return token_count < self.max_tokens_exclusive


@dataclass(frozen=True)
class TrafficProfile:
    """Configured request-arrival profile."""

    id: str
    description: str
    default_concurrency: int
    max_recommended_concurrency: int | None
    default_request_arrival_mode: ArrivalMode
    target: str

    def __post_init__(self) -> None:
        _non_empty(self.id, "id")
        _non_empty(self.description, "description")
        _positive_int(self.default_concurrency, "default_concurrency")
        if self.max_recommended_concurrency is not None:
            _positive_int(self.max_recommended_concurrency, "max_recommended_concurrency")
        if self.default_request_arrival_mode not in {
            "closed_loop",
            "jittered_poisson",
            "bursty_jittered",
            "custom",
        }:
            msg = "default_request_arrival_mode is invalid"
            raise ValueError(msg)
        _non_empty(self.target, "target")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable profile."""

        return asdict(self)


def load_sequence_buckets(
    path: str | Path = DEFAULT_LOAD_PROFILES_PATH,
) -> dict[str, tuple[SequenceLengthBucket, ...]]:
    """Load ISL and OSL bucket definitions."""

    payload = load_yaml_file(path)
    raw = payload.get("sequence_length_buckets")
    if not isinstance(raw, dict):
        msg = "load profile config must define sequence_length_buckets"
        raise ValueError(msg)
    buckets: dict[str, tuple[SequenceLengthBucket, ...]] = {}
    for family in ("input", "output"):
        entries = raw.get(family)
        if not isinstance(entries, list) or not entries:
            msg = f"sequence_length_buckets.{family} must be a non-empty list"
            raise ValueError(msg)
        parsed = tuple(SequenceLengthBucket(**cast(dict[str, Any], entry)) for entry in entries)
        buckets[family] = parsed
    return buckets


def load_traffic_profiles(
    path: str | Path = DEFAULT_LOAD_PROFILES_PATH,
) -> dict[str, TrafficProfile]:
    """Load configured production traffic profiles."""

    payload = load_yaml_file(path)
    raw = payload.get("traffic_profiles")
    if not isinstance(raw, dict):
        msg = "load profile config must define traffic_profiles"
        raise ValueError(msg)
    profiles: dict[str, TrafficProfile] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            msg = f"Traffic profile '{key}' must be a mapping"
            raise ValueError(msg)
        profiles[key] = TrafficProfile(id=key, **cast(dict[str, Any], value))
    return profiles


def bucket_token_count(token_count: int, buckets: tuple[SequenceLengthBucket, ...]) -> str:
    """Return the bucket ID for one token count."""

    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
        msg = "token_count must be an integer >= 0"
        raise ValueError(msg)
    for bucket in buckets:
        if bucket.contains(token_count):
            return bucket.id
    msg = f"No sequence length bucket covers {token_count} tokens"
    raise ValueError(msg)


def sequence_length_distribution(
    token_counts: list[int],
    buckets: tuple[SequenceLengthBucket, ...],
) -> dict[str, dict[str, float | int]]:
    """Build a bucketed token distribution."""

    total = len(token_counts)
    counts = {bucket.id: 0 for bucket in buckets}
    for token_count in token_counts:
        counts[bucket_token_count(token_count, buckets)] += 1
    return {
        bucket_id: {
            "count": count,
            "share": (count / total if total else 0.0),
        }
        for bucket_id, count in counts.items()
    }


def simulate_request_arrivals(
    *,
    request_count: int,
    arrival_mode: ArrivalMode,
    seed: int = 0,
    mean_interarrival_ms: float = 1000.0,
    burst_size: int = 4,
) -> list[float]:
    """Return deterministic request arrival offsets in milliseconds."""

    if request_count < 0:
        msg = "request_count must be >= 0"
        raise ValueError(msg)
    if mean_interarrival_ms <= 0:
        msg = "mean_interarrival_ms must be > 0"
        raise ValueError(msg)
    if burst_size <= 0:
        msg = "burst_size must be > 0"
        raise ValueError(msg)
    rng = random.Random(seed)
    arrivals: list[float] = []
    current_ms = 0.0
    for index in range(request_count):
        if index == 0:
            current_ms = 0.0
        elif arrival_mode == "closed_loop":
            current_ms += mean_interarrival_ms
        elif arrival_mode == "jittered_poisson":
            current_ms += rng.expovariate(1.0 / mean_interarrival_ms)
        elif arrival_mode == "bursty_jittered":
            if index % burst_size == 0:
                current_ms += rng.expovariate(1.0 / (mean_interarrival_ms * burst_size))
            else:
                current_ms += rng.uniform(1.0, mean_interarrival_ms * 0.05)
        elif arrival_mode == "custom":
            current_ms += mean_interarrival_ms
        else:
            msg = f"Unknown arrival_mode '{arrival_mode}'"
            raise ValueError(msg)
        arrivals.append(round(current_ms, 6))
    return arrivals


def build_load_profile_report(
    *,
    input_tokens: list[int],
    output_tokens: list[int],
    traffic_profile: str,
    concurrency: int,
    request_arrival_mode: ArrivalMode | None = None,
    seed: int = 0,
    config_path: str | Path = DEFAULT_LOAD_PROFILES_PATH,
) -> dict[str, object]:
    """Build the load metadata required for production benchmark reports."""

    if len(input_tokens) != len(output_tokens):
        msg = "input_tokens and output_tokens must have the same length"
        raise ValueError(msg)
    _positive_int(concurrency, "concurrency")
    buckets = load_sequence_buckets(config_path)
    profiles = load_traffic_profiles(config_path)
    if traffic_profile not in profiles:
        msg = f"Unknown traffic profile '{traffic_profile}'"
        raise ValueError(msg)
    profile = profiles[traffic_profile]
    arrival_mode = request_arrival_mode or profile.default_request_arrival_mode
    arrivals = simulate_request_arrivals(
        request_count=len(input_tokens),
        arrival_mode=arrival_mode,
        seed=seed,
    )
    return {
        "input_token_distribution": sequence_length_distribution(input_tokens, buckets["input"]),
        "output_token_distribution": sequence_length_distribution(output_tokens, buckets["output"]),
        "traffic_profile": profile.to_dict(),
        "concurrency": concurrency,
        "request_arrival_mode": arrival_mode,
        "request_arrival_offsets_ms": arrivals,
        "request_count": len(input_tokens),
    }
