"""Hugging Face benchmark runner foundation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

from inference_bench.env import load_local_env
from inference_bench.metrics import (
    calculate_end_to_end_latency_ms,
    calculate_tokens_per_second,
)
from inference_bench.results import write_results_csv
from inference_bench.schema import BenchmarkResult
from inference_bench.workloads.loader import load_jsonl_workload

HF_EXTRA_INSTALL_MESSAGE = (
    'Install the Hugging Face extra with: python -m pip install -e ".[hf,dev]"'
)


@dataclass(frozen=True)
class HuggingFaceRunnerConfig:
    """Configuration for the Hugging Face local inference runner."""

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 64
    temperature: float = 0.0
    do_sample: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            msg = "model_id must not be empty"
            raise ValueError(msg)
        if self.max_new_tokens <= 0:
            msg = "max_new_tokens must be > 0"
            raise ValueError(msg)
        if self.temperature < 0:
            msg = "temperature must be >= 0"
            raise ValueError(msg)


def require_hf_dependencies() -> None:
    """Ensure optional Hugging Face runner dependencies are installed."""

    missing_packages: list[str] = []
    for package_name in ("torch", "transformers"):
        try:
            import_module(package_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        missing = ", ".join(missing_packages)
        msg = f"Missing optional Hugging Face dependencies: {missing}. {HF_EXTRA_INSTALL_MESSAGE}"
        raise RuntimeError(msg)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_device(torch: Any, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(torch: Any, dtype: str) -> Any | None:
    if dtype == "auto":
        return None
    dtype_value = getattr(torch, dtype, None)
    if dtype_value is None:
        msg = f"Unsupported torch dtype: {dtype}"
        raise ValueError(msg)
    return dtype_value


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _input_token_count(inputs: Any) -> int:
    input_ids = inputs["input_ids"]
    return int(input_ids.shape[-1])


def _build_failure_result(
    *,
    run_id: str,
    model_id: str,
    optimization: str,
    workload_name: str,
    prompt_id: str,
    input_tokens: int,
    request_start_s: float,
    error: Exception,
) -> BenchmarkResult:
    request_end_s = time.perf_counter()
    return BenchmarkResult(
        run_id=run_id,
        timestamp_utc=_utc_timestamp(),
        backend="huggingface",
        model_name=model_id,
        optimization=optimization,
        workload_name=workload_name,
        prompt_id=prompt_id,
        input_tokens=max(0, input_tokens),
        output_tokens=0,
        ttft_ms=None,
        tpot_ms=None,
        end_to_end_latency_ms=calculate_end_to_end_latency_ms(
            request_start_s,
            request_end_s,
        ),
        throughput_tokens_per_second=None,
        peak_memory_mb=None,
        estimated_cost_usd=0.0,
        success=False,
        error_message=str(error),
    )


def run_hf_benchmark(
    workload_path: str | Path,
    output_path: str | Path,
    model_id: str,
    run_id: str = "hf-run",
    optimization: str = "hf_baseline",
    max_new_tokens: int = 64,
    max_prompts: int | None = None,
) -> list[BenchmarkResult]:
    """Run a local Hugging Face causal language model benchmark."""

    config = HuggingFaceRunnerConfig(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
    )
    load_local_env()
    require_hf_dependencies()

    torch = import_module("torch")
    transformers = import_module("transformers")

    workload_items = load_jsonl_workload(workload_path)
    if max_prompts is not None:
        if max_prompts <= 0:
            msg = "max_prompts must be > 0"
            raise ValueError(msg)
        workload_items = workload_items[:max_prompts]

    device = _resolve_device(torch, config.device)
    model_kwargs: dict[str, Any] = {}
    torch_dtype = _resolve_torch_dtype(torch, config.dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_id,
        **model_kwargs,
    )
    if hasattr(model, "to"):
        model = model.to(device)
    model.eval()

    results: list[BenchmarkResult] = []
    for item in workload_items:
        request_start_s = time.perf_counter()
        input_tokens = 0
        try:
            inputs = tokenizer(item.prompt, return_tensors="pt")
            input_tokens = _input_token_count(inputs)
            inputs = _move_inputs_to_device(inputs, device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                )

            tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_length = int(generated_ids.shape[-1])
            output_tokens = max(0, generated_length - input_tokens)
            request_end_s = time.perf_counter()
            end_to_end_latency_ms = calculate_end_to_end_latency_ms(
                request_start_s,
                request_end_s,
            )
            elapsed_seconds = request_end_s - request_start_s

            results.append(
                BenchmarkResult(
                    run_id=run_id,
                    timestamp_utc=_utc_timestamp(),
                    backend="huggingface",
                    model_name=config.model_id,
                    optimization=optimization,
                    workload_name=item.workload_name,
                    prompt_id=item.prompt_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=None,
                    tpot_ms=(end_to_end_latency_ms / output_tokens if output_tokens > 0 else None),
                    end_to_end_latency_ms=end_to_end_latency_ms,
                    throughput_tokens_per_second=calculate_tokens_per_second(
                        input_tokens + output_tokens,
                        elapsed_seconds,
                    ),
                    peak_memory_mb=None,
                    estimated_cost_usd=0.0,
                    success=True,
                    error_message=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                _build_failure_result(
                    run_id=run_id,
                    model_id=config.model_id,
                    optimization=optimization,
                    workload_name=item.workload_name,
                    prompt_id=item.prompt_id,
                    input_tokens=input_tokens,
                    request_start_s=request_start_s,
                    error=exc,
                )
            )

    write_results_csv(results, output_path)
    return results
