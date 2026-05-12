"""Memory estimate calculations."""

from __future__ import annotations


def _validate_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        msg = f"{field_name} must be > 0"
        raise ValueError(msg)


def _validate_positive_float(value: float, field_name: str) -> None:
    if value <= 0:
        msg = f"{field_name} must be > 0"
        raise ValueError(msg)


def estimate_model_weight_memory_mb(parameter_count: int, bytes_per_parameter: float) -> float:
    """Estimate model weight memory in MiB."""

    _validate_positive_int(parameter_count, "parameter_count")
    _validate_positive_float(bytes_per_parameter, "bytes_per_parameter")
    return parameter_count * bytes_per_parameter / 1024**2


def estimate_kv_cache_memory_mb(
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: float = 2.0,
) -> float:
    """Estimate KV cache memory in MiB."""

    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(sequence_length, "sequence_length")
    _validate_positive_int(num_layers, "num_layers")
    _validate_positive_int(num_kv_heads, "num_kv_heads")
    _validate_positive_int(head_dim, "head_dim")
    _validate_positive_float(bytes_per_element, "bytes_per_element")
    return (
        batch_size
        * sequence_length
        * num_layers
        * 2
        * num_kv_heads
        * head_dim
        * bytes_per_element
        / 1024**2
    )
