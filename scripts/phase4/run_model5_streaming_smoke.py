"""Run Block 30B with strict model5 primary and execution-only fallback."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_priced_validation import (  # noqa: E402
    check_hf_token,
    check_model_access,
)
from inference_bench.api_pricing import resolve_api_pricing  # noqa: E402
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.env import load_local_env  # noqa: E402
from inference_bench.model5_pricing_routing import (  # noqa: E402
    audit_model5_route,
    request_router_metadata,
)
from inference_bench.model5_streaming_validation import (  # noqa: E402
    fallback_allowed_after_primary_execution,
    write_blocked_model5_artifacts,
)
from inference_bench.stronger_model_validation import (  # noqa: E402
    load_and_validate_runner_input,
    read_jsonl,
)

DEFAULT_INPUT = "data/generated/phase4/api_priced_contract_runner_input.jsonl"
DEFAULT_RAW = "results/raw/phase4_model5_streaming_smoke_results.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Build the Block 30B CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Run model5 streaming smoke; fallback only after an actual primary execution failure."
        )
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT)
    parser.add_argument("--output-path", default=DEFAULT_RAW)
    parser.add_argument("--processed-root", default="results/processed")
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--allow-paid-api-call", action="store_true")
    return parser


def _run(command: list[str]) -> int:
    return subprocess.run(command, check=False).returncode


def _run_one_model(
    *,
    model_alias: str,
    input_path: str,
    output_path: str,
    pricing_config: str,
    limit: int,
    max_new_tokens: int,
) -> int:
    return _run(
        [
            sys.executable,
            "scripts/phase4/run_api_priced_smoke.py",
            "--input-path",
            input_path,
            "--output-path",
            output_path,
            "--model-alias",
            model_alias,
            "--fallback-model-alias",
            model_alias,
            "--limit",
            str(limit),
            "--max-new-tokens",
            str(max_new_tokens),
            "--stream",
            "--require-streaming",
            "--pricing-config",
            pricing_config,
            "--allow-paid-api-call",
        ]
    )


def _finalize(
    *,
    raw_path: str,
    processed_root: str,
    dataset_root: str,
) -> int:
    eval_status = _run(
        [
            sys.executable,
            "scripts/phase4/evaluate_generation_outputs.py",
            "--results-path",
            raw_path,
            "--dataset-root",
            dataset_root,
            "--output-root",
            processed_root,
            "--report-name",
            "phase4_model5_streaming_eval_report.json",
            "--summary-name",
            "phase4_model5_streaming_eval_summary.csv",
        ]
    )
    if eval_status:
        return eval_status
    return _run(
        [
            sys.executable,
            "scripts/phase4/finalize_api_streaming_smoke.py",
            "--results-path",
            raw_path,
            "--eval-report",
            str(Path(processed_root) / "phase4_model5_streaming_eval_report.json"),
            "--output-root",
            processed_root,
            "--cost-report-name",
            "phase4_model5_streaming_cost_report.json",
            "--cost-summary-name",
            "phase4_model5_streaming_cost_summary.csv",
            "--latency-report-name",
            "phase4_model5_streaming_latency_report.json",
            "--grounding-report-name",
            "phase4_model5_streaming_grounding_report.json",
            "--grounding-summary-name",
            "phase4_model5_streaming_grounding_summary.csv",
        ]
    )


def _success_count(path: str | Path) -> int:
    output = Path(path)
    if not output.is_file():
        return 0
    return sum(bool(row.get("success")) for row in read_jsonl(output))


def main(argv: list[str] | None = None) -> int:
    """Run model5 or write an explicit preflight block."""

    args = build_parser().parse_args(argv)
    if not args.allow_paid_api_call:
        print("Refusing paid API execution: pass --allow-paid-api-call.", file=sys.stderr)
        return 1
    if args.limit != 5:
        print("Block 30B requires exactly five prompts.", file=sys.stderr)
        return 1
    if args.max_new_tokens <= 0 or args.max_new_tokens > 128:
        print("max-new-tokens must be between 1 and 128.", file=sys.stderr)
        return 1
    runner_rows = load_and_validate_runner_input(args.input_path)
    load_local_env()
    token = os.environ.get("HF_TOKEN", "")
    config = load_project_config()
    primary = config.resolve_model_config("model5_gated")
    route = audit_model5_route(
        model_id=primary.model_id,
        pricing_config=args.pricing_config,
        hf_token=token,
        token_checker=check_hf_token,
        access_checker=check_model_access,
        metadata_fetcher=request_router_metadata,
    )
    if not route["costed_smoke_allowed"]:
        reason = " | ".join(str(item) for item in route["blocking_reasons"])
        outputs = write_blocked_model5_artifacts(
            raw_path=args.output_path,
            processed_root=args.processed_root,
            model_id=primary.model_id,
            provider=route["selected_provider"],
            planned_prompt_count=len(runner_rows),
            reason=reason,
            pricing_status="unavailable",
            manual_override_configured=bool(route["manual_override_configured"]),
        )
        print(f"Model5 smoke blocked before execution: {reason}", file=sys.stderr)
        for label, path in outputs.items():
            print(f"{label}: {path}")
        return 2

    resolve_api_pricing("model5_gated", args.pricing_config)
    primary_status = _run_one_model(
        model_alias="model5_gated",
        input_path=args.input_path,
        output_path=args.output_path,
        pricing_config=args.pricing_config,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
    primary_success_count = _success_count(args.output_path)
    fallback_used = False
    if primary_status and fallback_allowed_after_primary_execution(
        primary_execution_attempted=True,
        primary_success_count=primary_success_count,
    ):
        fallback_used = True
        fallback_output = Path(args.output_path)
        if fallback_output.is_file():
            failed_path = fallback_output.with_name(
                f"{fallback_output.stem}_model5_failed{fallback_output.suffix}"
            )
            fallback_output.replace(failed_path)
        primary_status = _run_one_model(
            model_alias="model6_gated",
            input_path=args.input_path,
            output_path=args.output_path,
            pricing_config=args.pricing_config,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
        )
    if primary_status:
        print(
            f"Streaming smoke failed; fallback_used={fallback_used}.",
            file=sys.stderr,
        )
        return primary_status
    final_status = _finalize(
        raw_path=args.output_path,
        processed_root=args.processed_root,
        dataset_root=args.dataset_root,
    )
    if final_status:
        return final_status
    run_metadata: dict[str, Any] = {
        "fallback_used": fallback_used,
        "primary_model_alias": "model5_gated",
        "final_model_alias": "model6_gated" if fallback_used else "model5_gated",
    }
    metadata_path = Path(args.processed_root) / "phase4_model5_streaming_run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
