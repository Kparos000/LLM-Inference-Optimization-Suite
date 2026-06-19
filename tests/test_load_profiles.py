from __future__ import annotations

from typing import Any, cast

from inference_bench.load_profiles import (
    build_load_profile_report,
    load_traffic_profiles,
    simulate_request_arrivals,
)


def test_configured_traffic_profiles_are_available() -> None:
    profiles = load_traffic_profiles()

    assert set(profiles) == {
        "online_low_latency",
        "office_hours_bursty",
        "offline_throughput",
        "custom",
    }
    assert profiles["online_low_latency"].default_request_arrival_mode == "jittered_poisson"
    assert profiles["offline_throughput"].default_request_arrival_mode == "closed_loop"


def test_jittered_request_arrivals_are_deterministic_and_nonuniform() -> None:
    first = simulate_request_arrivals(
        request_count=5,
        arrival_mode="jittered_poisson",
        seed=7,
        mean_interarrival_ms=100.0,
    )
    second = simulate_request_arrivals(
        request_count=5,
        arrival_mode="jittered_poisson",
        seed=7,
        mean_interarrival_ms=100.0,
    )

    assert first == second
    assert first[0] == 0.0
    assert len(set(round(first[index] - first[index - 1], 6) for index in range(1, 5))) > 1


def test_load_profile_report_includes_required_metadata() -> None:
    report = build_load_profile_report(
        input_tokens=[200, 900, 1500],
        output_tokens=[20, 80, 180],
        traffic_profile="office_hours_bursty",
        concurrency=4,
        seed=3,
    )
    profile = cast(dict[str, Any], report["traffic_profile"])

    assert profile["id"] == "office_hours_bursty"
    assert report["concurrency"] == 4
    assert report["request_arrival_mode"] == "bursty_jittered"
    assert report["request_count"] == 3
    assert "input_token_distribution" in report
    assert "output_token_distribution" in report
