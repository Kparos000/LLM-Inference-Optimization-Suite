"""Run the guarded five-prompt API-priced gated-model validation."""

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
    write_readiness_report,
)
from inference_bench.api_pricing import estimate_api_cost_from_pricing  # noqa: E402
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.stronger_model_validation import (  # noqa: E402
    build_promoted_runner_input,
    load_and_validate_runner_input,
    write_jsonl,
)
from inference_bench.workloads.loader import load_jsonl_workload  # noqa: E402

DEFAULT_WORKLOAD_PATH = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_RUNNER_INPUT_PATH = "data/generated/phase4/api_priced_contract_runner_input.jsonl"
DEFAULT_OUTPUT_PATH = "results/raw/phase4_api_priced_smoke_results.jsonl"
DEFAULT_READINESS_PATH = "results/processed/phase4_api_priced_readiness_report.json"
DEFAULT_API_ROUTE = "https://router.huggingface.co/v1/chat/completions"
MAX_PROMPTS = 5


def _load_api_module() -> ModuleType:
    path = REPO_ROOT / "scripts/phase3/hf_api_tiny_smoke.py"
    spec = importlib.util.spec_from_file_location("_block27_hf_api_client", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load Hugging Face API client from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build the Block 27 runner parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run exactly five promoted mm2 prompts through a priced HF Inference Provider. "
            "The command refuses missing token, pricing, or gated-model access."
        )
    )
    parser.add_argument("--workload-path", default=DEFAULT_WORKLOAD_PATH)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--readiness-report", default=DEFAULT_READINESS_PATH)
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--primary-model-alias", default="model5_gated")
    parser.add_argument("--fallback-model-alias", default="model6_gated")
    parser.add_argument("--api-route", default=DEFAULT_API_ROUTE)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--allow-paid-api-call", action="store_true")
    return parser


def _prepare_input(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_path) if args.input_path else Path(args.runner_input_path)
    if args.input_path is None:
        build_promoted_runner_input(
            workload_path=args.workload_path,
            output_path=input_path,
        )
    rows = load_and_validate_runner_input(input_path)
    if len(rows) != MAX_PROMPTS:
        msg = f"Block 27 requires exactly {MAX_PROMPTS} promoted records"
        raise ValueError(msg)
    return input_path


def _base_row(
    *,
    item: Any,
    model_alias: str,
    model_id: str,
    provider: str,
    api_route: str,
) -> dict[str, Any]:
    metadata = item.metadata
    return {
        "run_id": "phase4-api-priced-gated-model-smoke",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "hf_inference_provider",
        "provider": provider,
        "model_alias": model_alias,
        "model_id": model_id,
        "workload_name": item.workload_name,
        "prompt_id": item.prompt_id,
        "workload_id": str(metadata.get("workload_id") or ""),
        "vertical": str(metadata.get("vertical") or ""),
        "memory_mode": str(metadata.get("memory_mode") or ""),
        "ablation_mode": str(metadata.get("ablation_mode") or ""),
        "dataset_split": str(metadata.get("dataset_split") or ""),
        "expected_output_format": item.expected_output,
        "context_token_estimate": str(metadata.get("context_token_estimate") or "0"),
        "gold_evidence_ids": str(metadata.get("gold_evidence_ids") or "[]"),
        "selected_context_ids": str(metadata.get("selected_context_ids") or "[]"),
        "citation_id_aliases": str(metadata.get("citation_id_aliases") or "{}"),
        "prompt": item.prompt,
        "api_route": api_route,
        "execution_path": "hf_inference_provider_paid",
        "paid_api_call_triggered": True,
        "validation_measured": True,
        "no_gpu_experiment_triggered": True,
        "vllm_triggered": False,
        "sglang_triggered": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the API-priced smoke after all readiness gates pass."""

    args = build_parser().parse_args(argv)
    if not args.allow_paid_api_call:
        print("Refusing paid API execution: pass --allow-paid-api-call.", file=sys.stderr)
        return 1
    if args.max_new_tokens <= 0 or args.max_new_tokens > 256:
        print("max-new-tokens must be between 1 and 256.", file=sys.stderr)
        return 1
    input_path = _prepare_input(args)
    workload_count = len(load_jsonl_workload(input_path))
    hf_token = os.environ.get("HF_TOKEN", "")
    token_check = check_hf_token(hf_token)
    if not token_check.available:
        write_readiness_report(
            args.readiness_report,
            token_check=token_check,
            model_attempts=[],
            selected=None,
            workload_path=input_path,
            workload_count=workload_count,
            execution_status="STOPPED",
            stop_reason="HF_TOKEN is missing or invalid",
        )
        print("API-priced smoke stopped: HF_TOKEN is missing or invalid.", file=sys.stderr)
        return 1

    config = load_project_config()
    selected, attempts = select_api_model(
        config=config,
        model_aliases=[args.primary_model_alias, args.fallback_model_alias],
        pricing_config=args.pricing_config,
        hf_token=hf_token,
    )
    if selected is None:
        write_readiness_report(
            args.readiness_report,
            token_check=token_check,
            model_attempts=attempts,
            selected=None,
            workload_path=input_path,
            workload_count=workload_count,
            execution_status="STOPPED",
            stop_reason="No requested model had both captured pricing and gated-model access",
        )
        print(
            "API-priced smoke stopped: no model had both captured pricing and access.",
            file=sys.stderr,
        )
        return 1

    write_readiness_report(
        args.readiness_report,
        token_check=token_check,
        model_attempts=attempts,
        selected=selected,
        workload_path=input_path,
        workload_count=workload_count,
        execution_status="READY",
        stop_reason=None,
    )
    items = load_jsonl_workload(input_path)
    max_cost = estimate_api_cost_from_pricing(
        input_tokens=sum(len(item.prompt.split()) for item in items),
        output_tokens=args.max_new_tokens * len(items),
        pricing=selected.pricing,
    )
    print(f"Selected model alias: {selected.model_alias}")
    print(f"Selected provider: {selected.pricing.provider}")
    print(f"Maximum token-cost estimate: ${max_cost['total_api_cost_usd']:.8f}")

    api_module = _load_api_module()
    rows: list[dict[str, Any]] = []
    for item in items:
        row = _base_row(
            item=item,
            model_alias=selected.model_alias,
            model_id=selected.model_id,
            provider=selected.pricing.provider,
            api_route=args.api_route,
        )
        row.update(
            {
                "pricing_source_url": selected.pricing.pricing_source_url,
                "pricing_snapshot_timestamp_utc": (selected.pricing.pricing_snapshot_timestamp_utc),
                "input_cost_per_1m_tokens_usd": (selected.pricing.input_cost_per_1m_tokens_usd),
                "output_cost_per_1m_tokens_usd": (selected.pricing.output_cost_per_1m_tokens_usd),
                "ttft_ms": None,
            }
        )
        started = time.perf_counter()
        try:
            payload, latency_ms = api_module.request_chat_completion(
                hf_token=hf_token,
                model_id=selected.provider_model_id,
                prompt=item.prompt,
                max_new_tokens=args.max_new_tokens,
                api_route=args.api_route,
            )
            generated_text = str(api_module.extract_generated_text(payload))
            input_tokens, output_tokens = api_module.usage_tokens(
                payload,
                item.prompt,
                generated_text,
            )
            cost = estimate_api_cost_from_pricing(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                pricing=selected.pricing,
            )
            row.update(
                {
                    "generated_text": generated_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_cost_usd": cost["input_cost_usd"],
                    "output_cost_usd": cost["output_cost_usd"],
                    "total_cost_usd": cost["total_api_cost_usd"],
                    "latency_ms": latency_ms,
                    "end_to_end_latency_ms": latency_ms,
                    "throughput_tokens_per_second": (
                        output_tokens / (latency_ms / 1000) if latency_ms > 0 else None
                    ),
                    "success": True,
                    "error_type": None,
                    "error_message": None,
                    "final_status": "answer",
                    "generation_attempt_count": 1,
                }
            )
            row.update(
                generation_contract_result_fields(
                    generated_text,
                    allowed_evidence_ids=(
                        allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")) or None
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
                    "latency_ms": elapsed_ms,
                    "end_to_end_latency_ms": elapsed_ms,
                    "throughput_tokens_per_second": None,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "final_status": "failed_validation",
                    "generation_attempt_count": 1,
                }
            )
            row.update(generation_contract_result_fields(""))
        rows.append(row)
    write_jsonl(args.output_path, rows)
    success_count = sum(bool(row.get("success")) for row in rows)
    write_readiness_report(
        args.readiness_report,
        token_check=token_check,
        model_attempts=attempts,
        selected=selected,
        workload_path=input_path,
        workload_count=workload_count,
        execution_status="COMPLETE" if success_count == MAX_PROMPTS else "FAILED",
        stop_reason=None if success_count == MAX_PROMPTS else "One or more API requests failed",
    )
    print(f"Successful requests: {success_count}/{len(rows)}")
    print(f"Output path: {args.output_path}")
    print(f"Readiness report: {args.readiness_report}")
    return 0 if success_count == MAX_PROMPTS else 1


if __name__ == "__main__":
    raise SystemExit(main())
