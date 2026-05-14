"""Async OpenAI-compatible concurrency load runner."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from inference_bench.output_records import GenerationRecord, write_generation_records_jsonl
from inference_bench.results import write_results_csv
from inference_bench.runners.mock_runner import count_whitespace_tokens
from inference_bench.runners.openai_compatible_runner import (
    OPENAI_EXTRA_INSTALL_MESSAGE,
    _build_generation_record,
    _build_result,
    _extract_response_text,
    _extract_stream_delta,
)
from inference_bench.schema import BenchmarkResult, WorkloadItem
from inference_bench.workloads.loader import load_jsonl_workload


@dataclass(frozen=True)
class OpenAIConcurrencyConfig:
    """Configuration for OpenAI-compatible concurrent request execution."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    concurrency: int = 1
    max_new_tokens: int = 64
    max_prompts: int | None = None
    timeout_seconds: float = 120.0
    stream: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            msg = "base_url must not be empty"
            raise ValueError(msg)
        if not self.model.strip():
            msg = "model must not be empty"
            raise ValueError(msg)
        if self.concurrency <= 0:
            msg = "concurrency must be > 0"
            raise ValueError(msg)
        if self.max_new_tokens <= 0:
            msg = "max_new_tokens must be > 0"
            raise ValueError(msg)
        if self.max_prompts is not None and self.max_prompts <= 0:
            msg = "max_prompts must be > 0"
            raise ValueError(msg)
        if self.timeout_seconds <= 0:
            msg = "timeout_seconds must be > 0"
            raise ValueError(msg)


def require_openai_dependency() -> None:
    """Ensure the optional OpenAI client dependency is installed."""

    try:
        import_module("openai")
    except ImportError as exc:
        msg = f"Missing optional OpenAI dependency. {OPENAI_EXTRA_INSTALL_MESSAGE}"
        raise RuntimeError(msg) from exc


async def _run_one_prompt(
    *,
    client: Any,
    config: OpenAIConcurrencyConfig,
    item: WorkloadItem,
    run_id: str,
    backend: str,
    optimization: str,
    semaphore: asyncio.Semaphore,
) -> tuple[BenchmarkResult, GenerationRecord]:
    async with semaphore:
        input_tokens = count_whitespace_tokens(item.prompt)
        generated_text: str | None = None
        output_tokens = 0
        first_token_s: float | None = None
        request_start_s = time.perf_counter()

        try:
            response = await client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": item.prompt}],
                max_tokens=config.max_new_tokens,
                temperature=0.0,
                stream=config.stream,
            )

            if config.stream:
                chunks: list[str] = []
                async for chunk in response:
                    text_delta = _extract_stream_delta(chunk)
                    if text_delta:
                        chunks.append(text_delta)
                        if first_token_s is None:
                            first_token_s = time.perf_counter()
                generated_text = "".join(chunks)
            else:
                generated_text = _extract_response_text(response)

            request_end_s = time.perf_counter()
            output_tokens = count_whitespace_tokens(generated_text)
            result = _build_result(
                run_id=run_id,
                backend=backend,
                model=config.model,
                optimization=optimization,
                workload_name=item.workload_name,
                prompt_id=item.prompt_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_start_s=request_start_s,
                request_end_s=request_end_s,
                first_token_s=first_token_s,
                success=True,
                error_message=None,
            )
        except Exception as exc:  # noqa: BLE001
            request_end_s = time.perf_counter()
            result = _build_result(
                run_id=run_id,
                backend=backend,
                model=config.model,
                optimization=optimization,
                workload_name=item.workload_name,
                prompt_id=item.prompt_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_start_s=request_start_s,
                request_end_s=request_end_s,
                first_token_s=first_token_s,
                success=False,
                error_message=str(exc),
            )

        generation_record = _build_generation_record(
            result=result,
            prompt=item.prompt,
            generated_text=generated_text,
        )
        return result, generation_record


async def _run_load_benchmark_async(
    *,
    workload_items: list[WorkloadItem],
    config: OpenAIConcurrencyConfig,
    run_id: str,
    backend: str,
    optimization: str,
) -> tuple[list[BenchmarkResult], list[GenerationRecord]]:
    openai = cast(Any, import_module("openai"))
    client = openai.AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )
    semaphore = asyncio.Semaphore(config.concurrency)

    completed = await asyncio.gather(
        *[
            _run_one_prompt(
                client=client,
                config=config,
                item=item,
                run_id=run_id,
                backend=backend,
                optimization=optimization,
                semaphore=semaphore,
            )
            for item in workload_items
        ]
    )

    results = [result for result, _generation_record in completed]
    generation_records = [generation_record for _result, generation_record in completed]
    return results, generation_records


def run_openai_compatible_load_benchmark(
    workload_path: str | Path,
    output_path: str | Path,
    generation_output_path: str | Path | None,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    run_id: str = "openai-load-run",
    backend: str = "vllm",
    optimization: str = "vllm_baseline",
    concurrency: int = 1,
    max_new_tokens: int = 64,
    max_prompts: int | None = None,
    stream: bool = True,
    timeout_seconds: float = 120.0,
) -> list[BenchmarkResult]:
    """Run concurrent requests against an OpenAI-compatible chat completions endpoint."""

    config = OpenAIConcurrencyConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        concurrency=concurrency,
        max_new_tokens=max_new_tokens,
        max_prompts=max_prompts,
        timeout_seconds=timeout_seconds,
        stream=stream,
    )
    require_openai_dependency()

    workload_items = load_jsonl_workload(workload_path)
    if config.max_prompts is not None:
        workload_items = workload_items[: config.max_prompts]

    results, generation_records = asyncio.run(
        _run_load_benchmark_async(
            workload_items=workload_items,
            config=config,
            run_id=run_id,
            backend=backend,
            optimization=optimization,
        )
    )

    write_results_csv(results, output_path)
    if generation_output_path is not None:
        write_generation_records_jsonl(generation_records, generation_output_path)
    return results
