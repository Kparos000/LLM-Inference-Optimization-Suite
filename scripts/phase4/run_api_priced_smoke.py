"""Run the five-prompt API-priced smoke with optional required streaming."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_priced_validation import (  # noqa: E402
    check_hf_token,
    select_api_model,
)
from inference_bench.api_pricing import estimate_api_cost_from_pricing  # noqa: E402
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.streaming_metrics import (  # noqa: E402
    request_streaming_chat_completion,
)
from inference_bench.stronger_model_validation import (  # noqa: E402
    load_and_validate_runner_input,
    write_jsonl,
)
from inference_bench.workloads.loader import load_jsonl_workload  # noqa: E402

DEFAULT_INPUT = "data/generated/phase4/api_priced_contract_runner_input.jsonl"
DEFAULT_OUTPUT = "results/raw/phase4_api_streaming_smoke_results.jsonl"
DEFAULT_API_ROUTE = "https://router.huggingface.co/v1/chat/completions"


def _load_non_streaming_client() -> ModuleType:
    path = REPO_ROOT / "scripts/phase3/hf_api_tiny_smoke.py"
    spec = importlib.util.spec_from_file_location("_block29_non_stream_client", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load non-streaming API client from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build the streaming API smoke parser."""

    parser = argparse.ArgumentParser(
        description="Run at most five priced API requests with measured streaming latency."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-alias", default="model5_gated")
    parser.add_argument("--fallback-model-alias", default="model6_gated")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--api-route", default=DEFAULT_API_ROUTE)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    parser.add_argument("--require-streaming", action="store_true")
    parser.add_argument("--allow-paid-api-call", action="store_true")
    return parser


def _base_row(
    *,
    item: Any,
    model_alias: str,
    model_id: str,
    provider: str,
    pricing: Any,
    stream: bool,
) -> dict[str, Any]:
    metadata = item.metadata
    return {
        "run_id": "phase4-api-streaming-smoke",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "hf_inference_provider",
        "model_alias": model_alias,
        "model_id": model_id,
        "provider": provider,
        "workload_name": item.workload_name,
        "prompt_id": item.prompt_id,
        "workload_id": str(metadata.get("workload_id") or ""),
        "vertical": str(metadata.get("vertical") or ""),
        "memory_mode": str(metadata.get("memory_mode") or ""),
        "ablation_mode": str(metadata.get("ablation_mode") or ""),
        "dataset_split": str(metadata.get("dataset_split") or ""),
        "expected_output_format": item.expected_output,
        "citation_id_aliases": str(metadata.get("citation_id_aliases") or "{}"),
        "gold_evidence_ids": str(metadata.get("gold_evidence_ids") or "[]"),
        "prompt": item.prompt,
        "stream_requested": stream,
        "pricing_status": pricing.pricing_status,
        "pricing_source": pricing.pricing_source,
        "pricing_source_url": pricing.pricing_source_url,
        "pricing_last_checked": pricing.pricing_last_checked,
        "input_usd_per_1m_tokens": pricing.input_usd_per_1m_tokens,
        "output_usd_per_1m_tokens": pricing.output_usd_per_1m_tokens,
        "paid_api_call_triggered": True,
        "no_gpu_experiment_triggered": True,
        "vllm_triggered": False,
        "sglang_triggered": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the guarded API smoke."""

    args = build_parser().parse_args(argv)
    if not args.allow_paid_api_call:
        print("Refusing paid API execution: pass --allow-paid-api-call.", file=sys.stderr)
        return 1
    if args.limit <= 0 or args.limit > 5:
        print("limit must be between 1 and 5.", file=sys.stderr)
        return 1
    if args.max_new_tokens <= 0 or args.max_new_tokens > 128:
        print("max-new-tokens must be between 1 and 128.", file=sys.stderr)
        return 1
    if args.require_streaming and not args.stream:
        print("--require-streaming cannot be combined with --no-stream.", file=sys.stderr)
        return 1
    load_and_validate_runner_input(args.input_path)
    items = load_jsonl_workload(args.input_path)[: args.limit]
    hf_token = os.environ.get("HF_TOKEN", "")
    token_check = check_hf_token(hf_token)
    if not token_check.available:
        print("HF_TOKEN is missing or invalid.", file=sys.stderr)
        return 1
    selected, attempts = select_api_model(
        config=load_project_config(),
        model_aliases=[args.model_alias, args.fallback_model_alias],
        pricing_config=args.pricing_config,
        hf_token=hf_token,
    )
    if selected is None:
        print(
            "No requested model has complete detected/manual pricing and model access.",
            file=sys.stderr,
        )
        for attempt in attempts:
            print(
                f"- {attempt['model_alias']}: {attempt.get('failure_reason') or 'unavailable'}",
                file=sys.stderr,
            )
        return 1
    max_cost = estimate_api_cost_from_pricing(
        input_tokens=sum(len(item.prompt.split()) for item in items),
        output_tokens=args.max_new_tokens * len(items),
        pricing=selected.pricing,
    )
    print(f"Selected model: {selected.model_alias}")
    print(f"Selected provider: {selected.pricing.provider}")
    print(f"Maximum token-cost estimate: ${max_cost['total_api_cost_usd']:.8f}")
    non_streaming = _load_non_streaming_client() if not args.stream else None
    rows: list[dict[str, Any]] = []
    for item in items:
        row = _base_row(
            item=item,
            model_alias=selected.model_alias,
            model_id=selected.model_id,
            provider=selected.pricing.provider,
            pricing=selected.pricing,
            stream=args.stream,
        )
        started = time.perf_counter()
        try:
            if args.stream:
                metrics = request_streaming_chat_completion(
                    hf_token=hf_token,
                    model_id=selected.provider_model_id,
                    prompt=item.prompt,
                    max_new_tokens=args.max_new_tokens,
                    api_route=args.api_route,
                    timeout_seconds=args.timeout_seconds,
                )
                if args.require_streaming and not metrics.streaming_available:
                    raise RuntimeError("Streaming unavailable: no content chunks were received")
                generated_text = metrics.generated_text
                input_tokens = metrics.input_tokens
                output_tokens = metrics.output_tokens
                e2e_ms = metrics.e2e_latency_ms
                metric_fields = {
                    "ttft_ms": metrics.ttft_ms,
                    "itl_p50_ms": metrics.itl_p50_ms,
                    "itl_p95_ms": metrics.itl_p95_ms,
                    "itl_p99_ms": metrics.itl_p99_ms,
                    "tpot_ms": metrics.tpot_ms,
                    "e2e_latency_ms": e2e_ms,
                    "latency_ms": e2e_ms,
                    "streaming_available": metrics.streaming_available,
                    "content_chunk_count": metrics.content_chunk_count,
                    "token_count_source": metrics.token_count_source,
                }
            else:
                if non_streaming is None:
                    raise RuntimeError("Non-streaming client was not loaded")
                payload, latency_ms = non_streaming.request_chat_completion(
                    hf_token=hf_token,
                    model_id=selected.provider_model_id,
                    prompt=item.prompt,
                    max_new_tokens=args.max_new_tokens,
                    api_route=args.api_route,
                )
                generated_text = str(non_streaming.extract_generated_text(payload))
                input_tokens, output_tokens = non_streaming.usage_tokens(
                    payload, item.prompt, generated_text
                )
                e2e_ms = float(latency_ms)
                metric_fields = {
                    "ttft_ms": None,
                    "itl_p50_ms": None,
                    "itl_p95_ms": None,
                    "itl_p99_ms": None,
                    "tpot_ms": None,
                    "e2e_latency_ms": e2e_ms,
                    "latency_ms": e2e_ms,
                    "streaming_available": False,
                    "content_chunk_count": 0,
                    "token_count_source": "provider_usage_or_whitespace_fallback",
                }
            cost = estimate_api_cost_from_pricing(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=selected.pricing,
            )
            row.update(metric_fields)
            row.update(
                {
                    "generated_text": generated_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_cost_usd": cost["input_cost_usd"],
                    "output_cost_usd": cost["output_cost_usd"],
                    "total_cost_usd": cost["total_api_cost_usd"],
                    "throughput_tokens_per_second": (
                        output_tokens / (e2e_ms / 1000) if e2e_ms > 0 else None
                    ),
                    "success": True,
                    "error_type": None,
                    "error_message": None,
                    "final_status": "answer",
                }
            )
            row.update(
                generation_contract_result_fields(
                    generated_text,
                    allowed_evidence_ids=(
                        allowed_evidence_ids_from_aliases(row["citation_id_aliases"]) or None
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000
            row.update(
                {
                    "generated_text": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_cost_usd": 0.0,
                    "output_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                    "ttft_ms": None,
                    "itl_p50_ms": None,
                    "itl_p95_ms": None,
                    "itl_p99_ms": None,
                    "tpot_ms": None,
                    "e2e_latency_ms": elapsed_ms,
                    "latency_ms": elapsed_ms,
                    "throughput_tokens_per_second": None,
                    "streaming_available": False,
                    "content_chunk_count": 0,
                    "token_count_source": "unavailable",
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "final_status": "failed_validation",
                }
            )
            row.update(generation_contract_result_fields(""))
        rows.append(row)
    write_jsonl(args.output_path, rows)
    success_count = sum(bool(row.get("success")) for row in rows)
    streaming_count = sum(bool(row.get("streaming_available")) for row in rows)
    print(f"Successful requests: {success_count}/{len(rows)}")
    print(f"Streaming responses: {streaming_count}/{len(rows)}")
    print(f"Output: {args.output_path}")
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
