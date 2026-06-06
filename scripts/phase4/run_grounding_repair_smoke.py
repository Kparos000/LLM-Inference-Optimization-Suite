"""Run the five-prompt API smoke with one bounded citation-only repair."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_pricing import (  # noqa: E402
    estimate_api_cost_from_pricing,
    resolve_api_pricing,
)
from inference_bench.context_corpora import VERTICALS, benchmark_paths, read_jsonl  # noqa: E402
from inference_bench.env import load_local_env  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
    parse_generation_contract,
    render_citation_repair_prompt,
)
from inference_bench.grounding_repair import (  # noqa: E402
    citation_repair_decision,
    evaluate_result_row,
)
from inference_bench.streaming_metrics import (  # noqa: E402
    request_streaming_chat_completion,
)
from inference_bench.stronger_model_validation import (  # noqa: E402
    build_promoted_runner_input,
    write_jsonl,
)
from inference_bench.stronger_model_validation import (  # noqa: E402
    read_jsonl as read_result_jsonl,
)

DEFAULT_WORKLOAD = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_INPUT = "data/generated/phase4/grounding_repair_runner_input.jsonl"
DEFAULT_INITIAL_OUTPUT = "results/raw/phase4_grounding_repair_initial_results.jsonl"
DEFAULT_OUTPUT = "results/raw/phase4_grounding_repair_smoke_results.jsonl"
DEFAULT_API_ROUTE = "https://router.huggingface.co/v1/chat/completions"


def build_parser() -> argparse.ArgumentParser:
    """Build the grounding-repair smoke CLI."""

    parser = argparse.ArgumentParser(
        description="Run five API prompts and one optional citation-only retry per row."
    )
    parser.add_argument("--workload-path", default=DEFAULT_WORKLOAD)
    parser.add_argument("--input-path", default=DEFAULT_INPUT)
    parser.add_argument("--initial-output-path", default=DEFAULT_INITIAL_OUTPUT)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--api-route", default=DEFAULT_API_ROUTE)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--allow-paid-api-call", action="store_true")
    return parser


def load_gold_by_prompt(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    """Load promoted gold records keyed by prompt ID."""

    gold: dict[str, dict[str, Any]] = {}
    for vertical in VERTICALS:
        path = benchmark_paths(dataset_root, vertical)["gold"]
        for row in read_jsonl(path):
            prompt_id = str(row.get("prompt_id") or "")
            if prompt_id:
                gold[prompt_id] = row
    return gold


def run_initial_smoke(args: argparse.Namespace) -> int:
    """Run the existing guarded streaming path on rebuilt prompt input."""

    command = [
        sys.executable,
        "scripts/phase4/run_api_priced_smoke.py",
        "--input-path",
        args.input_path,
        "--output-path",
        args.initial_output_path,
        "--model-alias",
        "model5_gated",
        "--fallback-model-alias",
        "model6_gated",
        "--limit",
        "5",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--stream",
        "--require-streaming",
        "--pricing-config",
        args.pricing_config,
        "--api-route",
        args.api_route,
        "--allow-paid-api-call",
    ]
    return subprocess.run(command, check=False).returncode


def _repair_row(
    *,
    row: dict[str, Any],
    repair_prompt: str,
    pricing_config: str,
    api_route: str,
    hf_token: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run one streaming citation repair and return the merged final row."""

    result = dict(row)
    pricing = resolve_api_pricing(str(row["model_alias"]), pricing_config)
    provider_model_id = f"{row['model_id']}:{row['provider']}"
    metrics = request_streaming_chat_completion(
        hf_token=hf_token,
        model_id=provider_model_id,
        prompt=repair_prompt,
        max_new_tokens=max_new_tokens,
        api_route=api_route,
    )
    if not metrics.streaming_available:
        msg = "Citation repair streaming unavailable"
        raise RuntimeError(msg)
    repair_cost = estimate_api_cost_from_pricing(
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        pricing=pricing,
    )
    initial_parse = parse_generation_contract(
        str(row.get("generated_text") or ""),
        allowed_evidence_ids=(
            allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")) or None
        ),
    )
    repaired_parse = parse_generation_contract(
        metrics.generated_text,
        allowed_evidence_ids=(
            allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")) or None
        ),
    )
    initial_contract = initial_parse.contract
    repaired_contract = repaired_parse.contract
    result.update(
        {
            "initial_generated_text": row.get("generated_text"),
            "initial_evidence_ids": row.get("evidence_ids"),
            "initial_citation_notes": row.get("citation_notes"),
            "initial_input_tokens": row.get("input_tokens"),
            "initial_output_tokens": row.get("output_tokens"),
            "initial_total_cost_usd": row.get("total_cost_usd"),
            "initial_ttft_ms": row.get("ttft_ms"),
            "initial_e2e_latency_ms": row.get("e2e_latency_ms"),
            "citation_repair_attempted": True,
            "citation_repair_success": repaired_parse.contract_valid,
            "citation_repair_prompt": repair_prompt,
            "citation_repair_generated_text": metrics.generated_text,
            "citation_repair_input_tokens": metrics.input_tokens,
            "citation_repair_output_tokens": metrics.output_tokens,
            "citation_repair_total_cost_usd": repair_cost["total_api_cost_usd"],
            "citation_repair_ttft_ms": metrics.ttft_ms,
            "citation_repair_itl_p50_ms": metrics.itl_p50_ms,
            "citation_repair_itl_p95_ms": metrics.itl_p95_ms,
            "citation_repair_itl_p99_ms": metrics.itl_p99_ms,
            "citation_repair_tpot_ms": metrics.tpot_ms,
            "citation_repair_e2e_latency_ms": metrics.e2e_latency_ms,
            "citation_repair_answer_changed": bool(
                initial_contract
                and repaired_contract
                and initial_contract.answer != repaired_contract.answer
            ),
            "generated_text": (
                metrics.generated_text
                if repaired_parse.contract_valid
                else row.get("generated_text")
            ),
            "input_tokens": int(row.get("input_tokens") or 0) + metrics.input_tokens,
            "output_tokens": int(row.get("output_tokens") or 0) + metrics.output_tokens,
            "total_tokens": int(row.get("total_tokens") or 0) + metrics.total_tokens,
            "input_cost_usd": float(row.get("input_cost_usd") or 0.0)
            + repair_cost["input_cost_usd"],
            "output_cost_usd": float(row.get("output_cost_usd") or 0.0)
            + repair_cost["output_cost_usd"],
            "total_cost_usd": float(row.get("total_cost_usd") or 0.0)
            + repair_cost["total_api_cost_usd"],
            "latency_ms": float(row.get("latency_ms") or 0.0) + metrics.e2e_latency_ms,
            "e2e_latency_ms": float(row.get("e2e_latency_ms") or 0.0) + metrics.e2e_latency_ms,
            "generation_attempt_count": 2,
            "citation_repair_count": 1,
        }
    )
    result.update(
        generation_contract_result_fields(
            str(result["generated_text"]),
            allowed_evidence_ids=(
                allowed_evidence_ids_from_aliases(row.get("citation_id_aliases")) or None
            ),
        )
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run initial generation, bounded citation repair, and write final rows."""

    args = build_parser().parse_args(argv)
    if not args.allow_paid_api_call:
        print("Refusing paid API execution: pass --allow-paid-api-call.", file=sys.stderr)
        return 1
    if args.max_new_tokens <= 0 or args.max_new_tokens > 128:
        print("max-new-tokens must be between 1 and 128.", file=sys.stderr)
        return 1
    build_promoted_runner_input(
        workload_path=args.workload_path,
        output_path=args.input_path,
    )
    load_local_env()
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("HF_TOKEN is missing.", file=sys.stderr)
        return 1
    initial_status = run_initial_smoke(args)
    if initial_status:
        print("Initial API streaming smoke failed.", file=sys.stderr)
        return initial_status

    gold_by_prompt = load_gold_by_prompt(args.dataset_root)
    final_rows: list[dict[str, Any]] = []
    for initial_row in read_result_jsonl(args.initial_output_path):
        prompt_id = str(initial_row.get("prompt_id") or "")
        initial_evaluation = evaluate_result_row(initial_row, gold_by_prompt.get(prompt_id))
        decision = citation_repair_decision(
            evaluation=initial_evaluation,
            citation_aliases=initial_row.get("citation_id_aliases"),
        )
        row = dict(initial_row)
        row.update(
            {
                "initial_evidence_match": initial_evaluation["evidence_match"],
                "initial_groundedness": initial_evaluation["groundedness"],
                "citation_repair_attempted": False,
                "citation_repair_success": False,
                "citation_repair_count": 0,
                "citation_repair_reason": decision.reason,
                "citation_repair_missing_required_count": len(decision.missing_expected_ids),
                "citation_repair_missing_labels": list(decision.missing_evidence_labels),
                "citation_repair_missing_available_count": len(
                    decision.missing_ids_available_in_context
                ),
                "citation_repair_missing_absent_count": len(
                    decision.missing_ids_absent_from_context
                ),
                "generation_attempt_count": 1,
            }
        )
        if decision.should_retry:
            repair_prompt = render_citation_repair_prompt(
                original_prompt=str(row.get("prompt") or ""),
                previous_output=str(row.get("generated_text") or ""),
                allowed_evidence_ids=allowed_evidence_ids_from_aliases(
                    row.get("citation_id_aliases")
                ),
                missing_evidence_labels=decision.missing_evidence_labels,
            )
            try:
                row = _repair_row(
                    row=row,
                    repair_prompt=repair_prompt,
                    pricing_config=args.pricing_config,
                    api_route=args.api_route,
                    hf_token=hf_token,
                    max_new_tokens=args.max_new_tokens,
                )
                row["citation_repair_reason"] = decision.reason
            except Exception as exc:  # noqa: BLE001
                row["citation_repair_attempted"] = True
                row["citation_repair_error_type"] = type(exc).__name__
                row["citation_repair_error_message"] = str(exc)
        final_evaluation = evaluate_result_row(row, gold_by_prompt.get(prompt_id))
        row["final_evidence_match"] = final_evaluation["evidence_match"]
        row["final_groundedness"] = final_evaluation["groundedness"]
        if row.get("citation_repair_attempted"):
            row["citation_repair_success"] = final_evaluation["evidence_match"]
        final_rows.append(row)

    write_jsonl(args.output_path, final_rows)
    repair_count = sum(bool(row.get("citation_repair_attempted")) for row in final_rows)
    print(f"Initial requests: {len(final_rows)}")
    print(f"Citation repair requests: {repair_count}")
    print(f"Final output: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
