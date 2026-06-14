"""Compare contextual HF/API baselines with matched vLLM and SGLang smokes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PHASE4_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from run_remote_vllm_smoke import latency_summary_rows  # noqa: E402

from inference_bench.engine_comparison import (  # noqa: E402
    build_comparison_report,
    build_engine_row,
    read_csv_first,
    read_json,
    read_jsonl,
    write_comparison_artifacts,
)

DEFAULT_REPORT = "results/processed/a2_vllm_vs_sglang_comparison_report.json"
DEFAULT_SUMMARY = "results/processed/a2_vllm_vs_sglang_comparison_summary.csv"
MATCHED_SCOPE = "same_50_prompts_same_model_same_gpu_same_generation_settings"


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-results",
        default="results/raw/phase4_generation_contract_hardened_hf_smoke.jsonl",
    )
    parser.add_argument(
        "--hf-eval",
        default="results/processed/phase4_generation_contract_hardened_eval_report.json",
    )
    parser.add_argument(
        "--api-results",
        default="results/raw/phase4_api_priced_smoke_results.jsonl",
    )
    parser.add_argument(
        "--api-eval",
        default="results/processed/phase4_api_priced_smoke_eval_report.json",
    )
    parser.add_argument(
        "--vllm-results",
        default="results/raw/a1_remote_rtx3070_vllm_smoke_results.jsonl",
    )
    parser.add_argument(
        "--vllm-eval",
        default="results/processed/a1_remote_rtx3070_vllm_eval_summary.csv",
    )
    parser.add_argument(
        "--vllm-latency",
        default="results/processed/a1_remote_rtx3070_vllm_latency_summary.csv",
    )
    parser.add_argument(
        "--vllm-telemetry",
        default="results/processed/a1_remote_rtx3070_gpu_telemetry_summary.json",
    )
    parser.add_argument(
        "--sglang-results",
        default="results/raw/a2_remote_rtx3070_sglang_smoke_results.jsonl",
    )
    parser.add_argument(
        "--sglang-eval",
        default="results/processed/a2_remote_rtx3070_sglang_eval_summary.csv",
    )
    parser.add_argument(
        "--sglang-latency",
        default="results/processed/a2_remote_rtx3070_sglang_latency_summary.csv",
    )
    parser.add_argument(
        "--sglang-telemetry",
        default=("results/processed/a2_remote_rtx3070_sglang_gpu_telemetry_summary.json"),
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    return parser


def _report_summary(path: str | Path) -> dict[str, Any]:
    report = read_json(path)
    summary = report.get("summary", report)
    if not isinstance(summary, dict):
        msg = f"Evaluation report {path} does not contain an object summary"
        raise ValueError(msg)
    return summary


def _prompt_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row["prompt_id"]) for row in rows if row.get("prompt_id")}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Build and write all backend comparison artifacts."""

    hf_results = read_jsonl(args.hf_results)
    api_results = read_jsonl(args.api_results)
    vllm_results = read_jsonl(args.vllm_results)
    sglang_results = read_jsonl(args.sglang_results)
    rows = [
        build_engine_row(
            backend="local_hf",
            comparison_scope="contextual_five_prompt_different_hardware",
            result_rows=hf_results,
            evaluation_summary=_report_summary(args.hf_eval),
            latency_summary=latency_summary_rows(hf_results)[0],
        ),
        build_engine_row(
            backend="api",
            comparison_scope="contextual_five_prompt_remote_provider",
            result_rows=api_results,
            evaluation_summary=_report_summary(args.api_eval),
            latency_summary=latency_summary_rows(api_results)[0],
        ),
        build_engine_row(
            backend="vllm",
            comparison_scope=MATCHED_SCOPE,
            result_rows=vllm_results,
            evaluation_summary=read_csv_first(args.vllm_eval),
            latency_summary=read_csv_first(args.vllm_latency),
            telemetry_summary=read_json(args.vllm_telemetry),
        ),
        build_engine_row(
            backend="sglang",
            comparison_scope=MATCHED_SCOPE,
            result_rows=sglang_results,
            evaluation_summary=read_csv_first(args.sglang_eval),
            latency_summary=read_csv_first(args.sglang_latency),
            telemetry_summary=read_json(args.sglang_telemetry),
        ),
    ]
    report = build_comparison_report(
        rows=rows,
        prompt_ids_by_backend={
            "vllm": _prompt_ids(vllm_results),
            "sglang": _prompt_ids(sglang_results),
        },
    )
    write_comparison_artifacts(
        report_path=args.report_path,
        summary_path=args.summary_path,
        report=report,
        rows=rows,
    )
    return {
        "status": report["comparison_status"],
        "prompt_id_sets_match": report["prompt_id_sets_match"],
        "report_path": str(args.report_path),
        "summary_path": str(args.summary_path),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the comparison CLI."""

    args = build_parser().parse_args(argv)
    try:
        result = compare(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Engine comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
