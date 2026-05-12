import pytest

from inference_bench.metrics.memory import (
    estimate_kv_cache_memory_mb,
    estimate_model_weight_memory_mb,
)


def test_estimates_model_weight_memory_mb() -> None:
    assert estimate_model_weight_memory_mb(1_048_576, 2.0) == 2.0


def test_estimates_kv_cache_memory_mb() -> None:
    memory_mb = estimate_kv_cache_memory_mb(
        batch_size=1,
        sequence_length=1024,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        bytes_per_element=2.0,
    )

    assert memory_mb == 2.0


def test_rejects_invalid_memory_inputs() -> None:
    with pytest.raises(ValueError, match="parameter_count"):
        estimate_model_weight_memory_mb(0, 2.0)

    with pytest.raises(ValueError, match="batch_size"):
        estimate_kv_cache_memory_mb(
            batch_size=0,
            sequence_length=1024,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
        )
