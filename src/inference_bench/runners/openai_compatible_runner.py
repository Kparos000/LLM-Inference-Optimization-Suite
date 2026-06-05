"""OpenAI-compatible benchmark runner foundation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from inference_bench.generation_contract import (
    allowed_evidence_ids_from_aliases,
    parse_generation_contract,
)
from inference_bench.metrics import (
    calculate_end_to_end_latency_ms,
    calculate_tokens_per_second,
    calculate_tpot_ms,
    calculate_ttft_ms,
)
from inference_bench.output_records import GenerationRecord, write_generation_records_jsonl
from inference_bench.results import write_results_csv
from inference_bench.runners.mock_runner import count_whitespace_tokens
from inference_bench.schema import (
    BenchmarkResult,
    WorkloadItem,
    benchmark_metadata_from_workload_item,
    empty_benchmark_metadata,
)
from inference_bench.workloads.loader import load_jsonl_workload

OPENAI_EXTRA_INSTALL_MESSAGE = (
    'Install the OpenAI-compatible client extra with: python -m pip install -e ".[openai,dev]"'
)


@dataclass(frozen=True)
class OpenAICompatibleRunnerConfig:
    """Configuration for an OpenAI-compatible inference endpoint."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 64
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    stream: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            msg = "base_url must not be empty"
            raise ValueError(msg)
        if not self.model.strip():
            msg = "model must not be empty"
            raise ValueError(msg)
        if self.max_new_tokens <= 0:
            msg = "max_new_tokens must be > 0"
            raise ValueError(msg)
        if self.temperature < 0:
            msg = "temperature must be >= 0"
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


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_stream_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def _build_result(
    *,
    run_id: str,
    backend: str,
    model: str,
    optimization: str,
    workload_name: str,
    prompt_id: str,
    input_tokens: int,
    output_tokens: int,
    request_start_s: float,
    request_end_s: float,
    first_token_s: float | None,
    success: bool,
    error_message: str | None,
    item: WorkloadItem | None = None,
) -> BenchmarkResult:
    elapsed_seconds = request_end_s - request_start_s
    end_to_end_latency_ms = calculate_end_to_end_latency_ms(request_start_s, request_end_s)

    return BenchmarkResult(
        run_id=run_id,
        timestamp_utc=_utc_timestamp(),
        backend=backend,
        model_name=model,
        optimization=optimization,
        workload_name=workload_name,
        prompt_id=prompt_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        ttft_ms=(
            calculate_ttft_ms(request_start_s, first_token_s) if first_token_s is not None else None
        ),
        tpot_ms=(
            calculate_tpot_ms(first_token_s, request_end_s, output_tokens)
            if first_token_s is not None
            else (end_to_end_latency_ms / output_tokens if output_tokens > 0 else None)
        ),
        end_to_end_latency_ms=end_to_end_latency_ms,
        throughput_tokens_per_second=calculate_tokens_per_second(
            input_tokens + output_tokens,
            elapsed_seconds,
        ),
        peak_memory_mb=None,
        estimated_cost_usd=0.0,
        success=success,
        error_message=error_message,
        **(
            benchmark_metadata_from_workload_item(item)
            if item is not None
            else empty_benchmark_metadata()
        ),
    )


def _build_generation_record(
    *,
    result: BenchmarkResult,
    prompt: str,
    generated_text: str | None,
    item: WorkloadItem | None = None,
) -> GenerationRecord:
    metadata = item.metadata if item is not None else {}
    contract_parse = parse_generation_contract(
        generated_text or "",
        allowed_evidence_ids=(
            allowed_evidence_ids_from_aliases(metadata.get("citation_id_aliases")) or None
        ),
    )
    contract = contract_parse.contract
    return GenerationRecord(
        run_id=result.run_id,
        timestamp_utc=result.timestamp_utc,
        prompt_id=result.prompt_id,
        workload_name=result.workload_name,
        backend=result.backend,
        model_name=result.model_name,
        optimization=result.optimization,
        prompt=prompt,
        generated_text=generated_text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        ttft_ms=result.ttft_ms,
        tpot_ms=result.tpot_ms,
        end_to_end_latency_ms=result.end_to_end_latency_ms,
        throughput_tokens_per_second=result.throughput_tokens_per_second,
        peak_memory_mb=result.peak_memory_mb,
        estimated_cost_usd=result.estimated_cost_usd,
        success=result.success,
        error_message=result.error_message,
        workload_id=metadata.get("workload_id"),
        vertical=metadata.get("vertical"),
        memory_mode=metadata.get("memory_mode"),
        ablation_mode=metadata.get("ablation_mode"),
        expected_output_format=(
            item.expected_output if item is not None else metadata.get("expected_output_format")
        ),
        citation_id_aliases=metadata.get("citation_id_aliases"),
        generation_contract_valid=contract_parse.contract_valid,
        generation_contract_error=contract_parse.error,
        generation_contract_missing_fields=contract_parse.missing_fields,
        parse_error_type=contract_parse.parse_error_type,
        parse_repair_applied=contract_parse.parse_repair_applied,
        truncation_detected=contract_parse.truncation_detected,
        answer=contract.answer if contract else "",
        evidence_ids=contract.evidence_ids if contract else [],
        confidence=contract.confidence if contract else None,
        insufficient_evidence=contract.insufficient_evidence if contract else None,
        citation_notes=contract.citation_notes if contract else "",
    )


def run_openai_compatible_benchmark(
    workload_path: str | Path,
    output_path: str | Path,
    generation_output_path: str | Path | None,
    model: str,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    run_id: str = "openai-compatible-run",
    backend: str = "openai_compatible",
    optimization: str = "vllm_baseline",
    max_new_tokens: int = 64,
    max_prompts: int | None = None,
    stream: bool = True,
    timeout_seconds: float = 120.0,
) -> list[BenchmarkResult]:
    """Run a benchmark against an OpenAI-compatible chat completions endpoint."""

    config = OpenAICompatibleRunnerConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
        stream=stream,
    )
    require_openai_dependency()
    openai = cast(Any, import_module("openai"))
    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )

    workload_items = load_jsonl_workload(workload_path)
    if max_prompts is not None:
        if max_prompts <= 0:
            msg = "max_prompts must be > 0"
            raise ValueError(msg)
        workload_items = workload_items[:max_prompts]

    results: list[BenchmarkResult] = []
    generation_records: list[GenerationRecord] = []
    for item in workload_items:
        input_tokens = count_whitespace_tokens(item.prompt)
        generated_text: str | None = None
        output_tokens = 0
        first_token_s: float | None = None
        request_start_s = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": item.prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                stream=config.stream,
            )

            if config.stream:
                chunks: list[str] = []
                for chunk in response:
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

        results.append(result)
        if generation_output_path is not None:
            generation_records.append(
                _build_generation_record(
                    result=result,
                    prompt=item.prompt,
                    generated_text=generated_text,
                    item=item,
                )
            )

    write_results_csv(results, output_path)
    if generation_output_path is not None:
        write_generation_records_jsonl(generation_records, generation_output_path)
    return results
