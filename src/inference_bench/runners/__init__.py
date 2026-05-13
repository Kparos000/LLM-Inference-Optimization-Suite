"""Inference runners for different backends."""

from inference_bench.runners.hf_runner import HuggingFaceRunnerConfig, run_hf_benchmark
from inference_bench.runners.openai_compatible_runner import (
    OpenAICompatibleRunnerConfig,
    run_openai_compatible_benchmark,
)

__all__ = [
    "HuggingFaceRunnerConfig",
    "OpenAICompatibleRunnerConfig",
    "run_hf_benchmark",
    "run_openai_compatible_benchmark",
]
