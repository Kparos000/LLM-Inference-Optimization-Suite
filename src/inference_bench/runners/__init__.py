"""Inference runners for different backends."""

from inference_bench.runners.hf_runner import HuggingFaceRunnerConfig, run_hf_benchmark

__all__ = ["HuggingFaceRunnerConfig", "run_hf_benchmark"]
