"""Async OpenAI-compatible concurrency load runner."""

from __future__ import annotations

import asyncio
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_metadata(
    *,
    results: list[BenchmarkResult],
    run_id: str,
    workload_path: str | Path,
    model: str,
    backend: str,
    optimization: str,
    concurrency: int,
    max_prompts: int | None,
    max_new_tokens: int,
    stream: bool,
    started_at_utc: str,
    ended_at_utc: str,
    wall_clock_seconds: float,
) -> dict[str, object]:
    """Build run-level metadata for a concurrent benchmark."""

    success_count = sum(1 for result in results if result.success)
    total_input_tokens = sum(result.input_tokens for result in results)
    total_output_tokens = sum(result.output_tokens for result in results)

    return {
        "run_id": run_id,
        "workload_path": str(workload_path),
        "model": model,
        "backend": backend,
        "optimization": optimization,
        "concurrency": concurrency,
        "max_prompts": max_prompts,
        "max_new_tokens": max_new_tokens,
        "stream": stream,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "wall_clock_seconds": wall_clock_seconds,
        "total_requests": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "aggregate_requests_per_second": (
            len(results) / wall_clock_seconds if wall_clock_seconds > 0 else None
        ),
        "aggregate_output_tokens_per_second": (
            total_output_tokens / wall_clock_seconds if wall_clock_seconds > 0 else None
        ),
    }


def write_run_metadata(metadata: dict[str, object], output_path: str | Path) -> Path:
    """Write run-level metadata JSON and return the output path."""

    metadata_path = Path(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def _append_results_csv(results: list[BenchmarkResult], output_path: str | Path) -> Path:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BenchmarkResult.csv_fieldnames())
        if write_header:
            writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    return csv_path


def _append_generation_records_jsonl(
    records: list[GenerationRecord],
    output_path: str | Path,
) -> Path:
    jsonl_path = Path(output_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    return jsonl_path


def _read_checkpoint(path: str | Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(Path(path).read_text(encoding="utf-8")))


def _checkpoint_int(checkpoint: dict[str, object], field_name: str) -> int:
    value = checkpoint.get(field_name, 0)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return 0


def _write_checkpoint(checkpoint: dict[str, object], path: str | Path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def _write_log_message(message: str, log_path: str | Path | None) -> None:
    if log_path is None:
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def _progress_message(
    *,
    processed: int,
    total_prompts: int,
    chunk_number: int,
    total_chunks: int,
    success_count: int,
    failure_count: int,
    elapsed_seconds: float,
    checkpoint_saved: bool,
) -> str:
    aggregate_requests_per_second = processed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return (
        f"processed={processed}/{total_prompts} "
        f"chunk={chunk_number}/{total_chunks} "
        f"success={success_count} "
        f"failure={failure_count} "
        f"elapsed_seconds={elapsed_seconds:.1f} "
        f"aggregate_requests_per_second={aggregate_requests_per_second:.1f} "
        f"checkpoint_saved={str(checkpoint_saved).lower()}"
    )


def _chunks(items: list[WorkloadItem], chunk_size: int) -> list[list[WorkloadItem]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _remove_fresh_run_artifacts(paths: list[str | Path | None]) -> None:
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.exists():
            path.unlink()


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
                item=item,
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
                item=item,
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
    run_metadata_path: str | Path | None = None,
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
    chunk_size: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    log_path: str | Path | None = None,
    progress_interval: int = 100,
) -> list[BenchmarkResult]:
    """Run concurrent requests against an OpenAI-compatible chat completions endpoint."""

    if chunk_size is not None and chunk_size <= 0:
        msg = "chunk_size must be > 0"
        raise ValueError(msg)
    if progress_interval <= 0:
        msg = "progress_interval must be > 0"
        raise ValueError(msg)

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

    if chunk_size is not None:
        return _run_chunked_openai_compatible_load_benchmark(
            workload_items=workload_items,
            workload_path=workload_path,
            output_path=output_path,
            generation_output_path=generation_output_path,
            run_metadata_path=run_metadata_path,
            model=config.model,
            run_id=run_id,
            backend=backend,
            optimization=optimization,
            concurrency=config.concurrency,
            max_prompts=config.max_prompts,
            max_new_tokens=config.max_new_tokens,
            stream=config.stream,
            chunk_size=chunk_size,
            checkpoint_path=checkpoint_path,
            resume=resume,
            log_path=log_path,
            progress_interval=progress_interval,
            config=config,
        )

    started_at_utc = _utc_now()
    wall_clock_start_s = time.perf_counter()
    results, generation_records = asyncio.run(
        _run_load_benchmark_async(
            workload_items=workload_items,
            config=config,
            run_id=run_id,
            backend=backend,
            optimization=optimization,
        )
    )
    wall_clock_seconds = time.perf_counter() - wall_clock_start_s
    ended_at_utc = _utc_now()

    write_results_csv(results, output_path)
    if generation_output_path is not None:
        write_generation_records_jsonl(generation_records, generation_output_path)
    if run_metadata_path is not None:
        metadata = build_run_metadata(
            results=results,
            run_id=run_id,
            workload_path=workload_path,
            model=config.model,
            backend=backend,
            optimization=optimization,
            concurrency=config.concurrency,
            max_prompts=config.max_prompts,
            max_new_tokens=config.max_new_tokens,
            stream=config.stream,
            started_at_utc=started_at_utc,
            ended_at_utc=ended_at_utc,
            wall_clock_seconds=wall_clock_seconds,
        )
        write_run_metadata(metadata, run_metadata_path)
    return results


def _run_chunked_openai_compatible_load_benchmark(
    *,
    workload_items: list[WorkloadItem],
    workload_path: str | Path,
    output_path: str | Path,
    generation_output_path: str | Path | None,
    run_metadata_path: str | Path | None,
    model: str,
    run_id: str,
    backend: str,
    optimization: str,
    concurrency: int,
    max_prompts: int | None,
    max_new_tokens: int,
    stream: bool,
    chunk_size: int,
    checkpoint_path: str | Path | None,
    resume: bool,
    log_path: str | Path | None,
    progress_interval: int,
    config: OpenAIConcurrencyConfig,
) -> list[BenchmarkResult]:
    total_prompts = len(workload_items)
    total_chunks = (total_prompts + chunk_size - 1) // chunk_size if total_prompts else 0
    checkpoint: dict[str, object] = {}
    completed_prompt_ids: set[str] = set()
    success_count = 0
    failure_count = 0
    started_at_utc = _utc_now()

    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = _read_checkpoint(checkpoint_path)
        raw_completed = checkpoint.get("completed_prompt_ids", [])
        if isinstance(raw_completed, list):
            completed_prompt_ids = {str(prompt_id) for prompt_id in raw_completed}
        success_count = _checkpoint_int(checkpoint, "success_count")
        failure_count = _checkpoint_int(checkpoint, "failure_count")
        started_at_utc = str(checkpoint.get("started_at_utc", started_at_utc))
    elif not resume:
        _remove_fresh_run_artifacts(
            [
                output_path,
                generation_output_path,
                checkpoint_path,
                run_metadata_path,
                log_path,
            ]
        )

    remaining_items = [
        item for item in workload_items if item.prompt_id not in completed_prompt_ids
    ]
    all_new_results: list[BenchmarkResult] = []
    wall_clock_start_s = time.perf_counter()

    for chunk_items in _chunks(remaining_items, chunk_size):
        chunk_results, chunk_generation_records = asyncio.run(
            _run_load_benchmark_async(
                workload_items=chunk_items,
                config=config,
                run_id=run_id,
                backend=backend,
                optimization=optimization,
            )
        )

        _append_results_csv(chunk_results, output_path)
        if generation_output_path is not None:
            _append_generation_records_jsonl(chunk_generation_records, generation_output_path)

        all_new_results.extend(chunk_results)
        for result in chunk_results:
            completed_prompt_ids.add(result.prompt_id)
            if result.success:
                success_count += 1
            else:
                failure_count += 1

        elapsed_seconds = time.perf_counter() - wall_clock_start_s
        checkpoint_saved = False
        last_updated_utc = _utc_now()
        if checkpoint_path is not None:
            checkpoint = {
                "run_id": run_id,
                "workload_path": str(workload_path),
                "model": model,
                "backend": backend,
                "optimization": optimization,
                "concurrency": concurrency,
                "chunk_size": chunk_size,
                "total_prompts": total_prompts,
                "completed_prompt_ids": sorted(completed_prompt_ids),
                "success_count": success_count,
                "failure_count": failure_count,
                "started_at_utc": started_at_utc,
                "last_updated_utc": last_updated_utc,
                "output_path": str(output_path),
                "generation_output_path": (
                    str(generation_output_path) if generation_output_path is not None else None
                ),
                "run_metadata_path": (
                    str(run_metadata_path) if run_metadata_path is not None else None
                ),
                "log_path": str(log_path) if log_path is not None else None,
            }
            _write_checkpoint(checkpoint, checkpoint_path)
            checkpoint_saved = True

        if run_metadata_path is not None:
            metadata = build_run_metadata(
                results=all_new_results,
                run_id=run_id,
                workload_path=workload_path,
                model=model,
                backend=backend,
                optimization=optimization,
                concurrency=concurrency,
                max_prompts=max_prompts,
                max_new_tokens=max_new_tokens,
                stream=stream,
                started_at_utc=started_at_utc,
                ended_at_utc=last_updated_utc,
                wall_clock_seconds=elapsed_seconds,
            )
            write_run_metadata(metadata, run_metadata_path)

        processed = len(completed_prompt_ids)
        chunk_number = (processed + chunk_size - 1) // chunk_size if chunk_size else 0
        message = _progress_message(
            processed=processed,
            total_prompts=total_prompts,
            chunk_number=chunk_number,
            total_chunks=total_chunks,
            success_count=success_count,
            failure_count=failure_count,
            elapsed_seconds=elapsed_seconds,
            checkpoint_saved=checkpoint_saved,
        )
        print(message)
        _write_log_message(message, log_path)

    return all_new_results
