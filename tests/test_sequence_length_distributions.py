from __future__ import annotations

from inference_bench.load_profiles import (
    bucket_token_count,
    load_sequence_buckets,
    sequence_length_distribution,
)


def test_isl_and_osl_bucket_boundaries_are_stable() -> None:
    buckets = load_sequence_buckets()

    assert bucket_token_count(0, buckets["input"]) == "isl_0_512"
    assert bucket_token_count(512, buckets["input"]) == "isl_512_1024"
    assert bucket_token_count(8192, buckets["input"]) == "isl_8192_plus"
    assert bucket_token_count(64, buckets["output"]) == "osl_64_128"
    assert bucket_token_count(1024, buckets["output"]) == "osl_1024_plus"


def test_sequence_length_distribution_reports_counts_and_shares() -> None:
    buckets = load_sequence_buckets()["input"]
    distribution = sequence_length_distribution([10, 700, 9000], buckets)

    assert distribution["isl_0_512"]["count"] == 1
    assert distribution["isl_512_1024"]["count"] == 1
    assert distribution["isl_8192_plus"]["count"] == 1
    assert distribution["isl_0_512"]["share"] == 1 / 3
