"""Run stronger-model generation-contract validation or a safe gated API dry-run."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_pricing import (  # noqa: E402
    estimate_api_cost_from_pricing,
    resolve_api_pricing,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.stronger_model_validation import (  # noqa: E402
    build_promoted_runner_input,
    is_model_cached,
    load_and_validate_runner_input,
    write_jsonl,
)
from inference_bench.workloads.loader import load_jsonl_workload  # noqa: E402

DEFAULT_WORKLOAD_PATH = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_RUNNER_INPUT_PATH = "data/generated/phase4/stronger_model_contract_runner_input.jsonl"
DEFAULT_OUTPUT_PATH = "results/raw/phase4_stronger_model_contract_smoke.jsonl"
DEFAULT_API_ROUTE = "https://router.huggingface.co/v1/chat/completions"
MAX_PROMPTS = 5


def _load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load script module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build the stronger-model smoke CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the generation contract with model2_1_5b when cached, "
            "otherwise produce a gated API dry-run unless a paid call is explicitly allowed."
        )
    )
    parser.add_argument("--workload-path", default=DEFAULT_WORKLOAD_PATH)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--local-model-alias", default="model2_1_5b")
    parser.add_argument(
        "--fallback-model-alias",
        choices=("model5_gated", "model6_gated"),
        default="model5_gated",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "local_hf", "hf_api"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-contract-retries", type=int, default=1)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--api-route", default=DEFAULT_API_ROUTE)
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--allow-paid-api-call", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force the gated API path to remain a no-call schema dry-run.",
    )
    return parser


def _prepare_runner_input(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_path) if args.input_path else Path(args.runner_input_path)
    if args.input_path is None:
        build_promoted_runner_input(
            workload_path=args.workload_path,
            output_path=input_path,
        )
    load_and_validate_runner_input(input_path)
    return input_path


def _base_api_row(
    *,
    item: Any,
    model_alias: str,
    model_id: str,
    api_route: str,
    dry_run: bool,
) -> dict[str, Any]:
    metadata = item.metadata
    return {
        "run_id": (
            "phase4-stronger-model-hf-api-dry-run" if dry_run else "phase4-stronger-model-hf-api"
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": "hf_inference_provider",
        "optimization": "stronger_model_generation_contract",
        "model_alias": model_alias,
        "model_id": model_id,
        "model_name": model_id,
        "base_url": api_route,
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
        "no_gpu_experiment_triggered": True,
        "dry_run": dry_run,
    }


def _api_dry_run_rows(
    *,
    input_path: Path,
    model_alias: str,
    model_id: str,
    api_route: str,
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_jsonl_workload(input_path):
        row = _base_api_row(
            item=item,
            model_alias=model_alias,
            model_id=model_id,
            api_route=api_route,
            dry_run=True,
        )
        row.update(
            {
                "generated_text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0.0,
                "end_to_end_latency_ms": 0.0,
                "throughput_tokens_per_second": None,
                "success": False,
                "error_type": "DryRunOnly",
                "error_message": reason,
                "final_status": "failed_validation",
                "paid_api_call_triggered": False,
                "estimated_cost_usd": 0.0,
                "validation_measured": False,
                "execution_path": "hf_inference_provider_dry_run",
                "contract_retry_count": 0,
                "generation_attempt_count": 0,
            }
        )
        row.update(
            generation_contract_result_fields(
                "",
                allowed_evidence_ids=(
                    allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")) or None
                ),
            )
        )
        rows.append(row)
    return rows


def _run_paid_api(
    *,
    input_path: Path,
    model_alias: str,
    model_id: str,
    api_route: str,
    pricing_config: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        msg = "HF_TOKEN is required for the explicitly authorized paid API path"
        raise RuntimeError(msg)
    pricing = resolve_api_pricing(model_alias, pricing_config)
    api_module = _load_script_module(
        "_phase3_hf_api_tiny_smoke",
        REPO_ROOT / "scripts/phase3/hf_api_tiny_smoke.py",
    )
    items = load_jsonl_workload(input_path)
    estimated_input_tokens = sum(len(item.prompt.split()) for item in items)
    maximum_cost = estimate_api_cost_from_pricing(
        input_tokens=estimated_input_tokens,
        output_tokens=max_new_tokens * len(items),
        pricing=pricing,
    )
    print(f"Authorized API maximum token-cost estimate: ${maximum_cost['total_api_cost_usd']:.8f}.")

    rows: list[dict[str, Any]] = []
    for item in items:
        row = _base_api_row(
            item=item,
            model_alias=model_alias,
            model_id=model_id,
            api_route=api_route,
            dry_run=False,
        )
        started = time.perf_counter()
        try:
            payload, latency_ms = api_module.request_chat_completion(
                hf_token=hf_token,
                model_id=model_id,
                prompt=item.prompt,
                max_new_tokens=max_new_tokens,
                api_route=api_route,
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
                pricing=pricing,
            )
            row.update(
                {
                    "generated_text": generated_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "end_to_end_latency_ms": latency_ms,
                    "throughput_tokens_per_second": (
                        (input_tokens + output_tokens) / (latency_ms / 1000)
                        if latency_ms > 0
                        else None
                    ),
                    "success": True,
                    "error_type": None,
                    "error_message": None,
                    "final_status": "answer",
                    "paid_api_call_triggered": True,
                    "estimated_cost_usd": cost["total_api_cost_usd"],
                    "validation_measured": True,
                    "execution_path": "hf_inference_provider_paid",
                    "contract_retry_count": 0,
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
                    "input_tokens": len(item.prompt.split()),
                    "output_tokens": 0,
                    "latency_ms": elapsed_ms,
                    "end_to_end_latency_ms": elapsed_ms,
                    "throughput_tokens_per_second": None,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "final_status": "failed_validation",
                    "paid_api_call_triggered": True,
                    "estimated_cost_usd": None,
                    "validation_measured": True,
                    "execution_path": "hf_inference_provider_paid",
                    "contract_retry_count": 0,
                    "generation_attempt_count": 1,
                }
            )
            row.update(generation_contract_result_fields(""))
        rows.append(row)
    return rows


def _run_local(
    *,
    input_path: Path,
    output_path: Path,
    model_alias: str,
    max_new_tokens: int,
    max_contract_retries: int,
) -> list[dict[str, Any]]:
    local_module = _load_script_module(
        "_phase4_local_hf_smoke",
        REPO_ROOT / "scripts/phase4/run_local_hf_smoke.py",
    )
    rows, _manifest = local_module.run_smoke(
        input_path=input_path,
        output_path=output_path,
        model_alias=model_alias,
        limit=MAX_PROMPTS,
        max_new_tokens=max_new_tokens,
        max_contract_retries=max_contract_retries,
        dry_run=False,
        local_files_only=True,
        command="run_stronger_model_contract_smoke local_hf",
    )
    normalized: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(cast(dict[str, Any], raw_row))
        row["validation_measured"] = True
        row["execution_path"] = "local_hf"
        normalized.append(row)
    write_jsonl(output_path, normalized)
    return normalized


def main(argv: list[str] | None = None) -> int:
    """Run the stronger-model smoke or its guarded dry-run fallback."""

    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0 or args.max_new_tokens > 256:
        print("Stronger-model smoke failed: max-new-tokens must be between 1 and 256.")
        return 1
    try:
        input_path = _prepare_runner_input(args)
        config = load_project_config()
        local_model = config.resolve_model_config(args.local_model_alias)
        fallback_model = config.resolve_model_config(args.fallback_model_alias)
        local_cached = is_model_cached(
            local_model.model_id,
            cache_root=args.cache_root,
        )
        output_path = Path(args.output_path)
        rows: list[dict[str, Any]]
        if args.execution_mode in {"auto", "local_hf"} and local_cached and not args.dry_run:
            rows = _run_local(
                input_path=input_path,
                output_path=output_path,
                model_alias=args.local_model_alias,
                max_new_tokens=args.max_new_tokens,
                max_contract_retries=args.max_contract_retries,
            )
            selected_alias = args.local_model_alias
            status = "MEASURED"
        elif args.execution_mode == "local_hf":
            msg = (
                f"{local_model.model_id} is not fully cached. "
                "The local-only path refuses to download model weights."
            )
            raise RuntimeError(msg)
        elif args.allow_paid_api_call and not args.dry_run:
            rows = _run_paid_api(
                input_path=input_path,
                model_alias=args.fallback_model_alias,
                model_id=fallback_model.model_id,
                api_route=args.api_route,
                pricing_config=args.pricing_config,
                max_new_tokens=args.max_new_tokens,
            )
            write_jsonl(output_path, rows)
            selected_alias = args.fallback_model_alias
            status = "MEASURED"
        else:
            reason = f"{local_model.model_id} is not cached; paid API execution was not authorized."
            rows = _api_dry_run_rows(
                input_path=input_path,
                model_alias=args.fallback_model_alias,
                model_id=fallback_model.model_id,
                api_route=args.api_route,
                reason=reason,
            )
            write_jsonl(output_path, rows)
            selected_alias = args.fallback_model_alias
            status = "DRY_RUN_ONLY"
    except Exception as exc:  # noqa: BLE001
        print(f"Stronger-model smoke failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validation status: {status}")
    print(f"Selected model alias: {selected_alias}")
    print(f"Rows written: {len(rows)}")
    print(f"Output path: {args.output_path}")
    print(f"Paid API call triggered: {any(bool(row['paid_api_call_triggered']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
