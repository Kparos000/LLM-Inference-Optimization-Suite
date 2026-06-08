"""Compare measured model5, model6, and local Qwen smoke artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.model_smoke_comparison import (  # noqa: E402
    build_model5_comparison_report,
    read_json,
    read_jsonl,
    write_model5_comparison_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the offline comparison CLI."""

    parser = argparse.ArgumentParser(
        description="Compare existing model5, model6, and local Qwen smoke measurements."
    )
    parser.add_argument(
        "--model5-results",
        default="results/raw/phase4_model5_openrouter_streaming_smoke_results.jsonl",
    )
    parser.add_argument(
        "--model5-eval",
        default="results/processed/phase4_model5_openrouter_streaming_eval_report.json",
    )
    parser.add_argument(
        "--model5-cost",
        default="results/processed/phase4_model5_openrouter_streaming_cost_report.json",
    )
    parser.add_argument(
        "--model5-latency",
        default="results/processed/phase4_model5_openrouter_streaming_latency_report.json",
    )
    parser.add_argument(
        "--model6-results",
        default="results/raw/phase4_api_streaming_smoke_results.jsonl",
    )
    parser.add_argument(
        "--model6-eval",
        default="results/processed/phase4_api_streaming_smoke_eval_report.json",
    )
    parser.add_argument(
        "--model6-cost",
        default="results/processed/phase4_api_streaming_cost_report.json",
    )
    parser.add_argument(
        "--model6-latency",
        default="results/processed/phase4_api_streaming_latency_report.json",
    )
    parser.add_argument(
        "--local-results",
        default="results/raw/phase4_generation_contract_hardened_hf_smoke.jsonl",
    )
    parser.add_argument(
        "--local-eval",
        default="results/processed/phase4_generation_contract_hardened_eval_report.json",
    )
    parser.add_argument(
        "--report-path",
        default="results/processed/phase4_model5_vs_model6_api_comparison_report.json",
    )
    parser.add_argument(
        "--summary-path",
        default="results/processed/phase4_model5_vs_model6_api_comparison_summary.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Read measured files and write the comparison."""

    args = build_parser().parse_args(argv)
    try:
        report, rows = build_model5_comparison_report(
            model5_results=read_jsonl(args.model5_results),
            model5_eval=read_json(args.model5_eval),
            model5_cost=read_json(args.model5_cost),
            model5_latency=read_json(args.model5_latency),
            model6_results=read_jsonl(args.model6_results),
            model6_eval=read_json(args.model6_eval),
            model6_cost=read_json(args.model6_cost),
            model6_latency=read_json(args.model6_latency),
            local_results=read_jsonl(args.local_results),
            local_eval=read_json(args.local_eval),
        )
        report_path, summary_path = write_model5_comparison_artifacts(
            report_path=args.report_path,
            summary_path=args.summary_path,
            report=report,
            rows=rows,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Model smoke comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Model5 recommendation: {report['model5_final_benchmark_recommendation']}")
    print(f"Comparison report: {report_path}")
    print(f"Comparison summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
