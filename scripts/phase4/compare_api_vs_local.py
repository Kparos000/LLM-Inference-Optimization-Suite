"""Generate the Block 28 API-versus-local GPU-readiness comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_local_comparison import (  # noqa: E402
    build_comparison_report,
    write_comparison_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the Block 28 comparison parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare existing local Qwen 0.5B and API Llama 3.1 8B smoke artifacts "
            "without running inference, GPU work, or paid API calls."
        )
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
        "--api-results",
        default="results/raw/phase4_api_priced_smoke_results.jsonl",
    )
    parser.add_argument(
        "--api-eval",
        default="results/processed/phase4_api_priced_smoke_eval_report.json",
    )
    parser.add_argument(
        "--api-cost",
        default="results/processed/phase4_api_priced_cost_report.json",
    )
    parser.add_argument(
        "--retrieval-manifest",
        default="data/generated/context_engineering/retrieval_source_of_truth_manifest.json",
    )
    parser.add_argument("--gpu-costs", default="configs/gpu_costs.yaml")
    parser.add_argument(
        "--report-path",
        default="results/processed/phase4_api_vs_local_comparison_report.json",
    )
    parser.add_argument(
        "--summary-path",
        default="results/processed/phase4_api_vs_local_comparison_summary.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and write the offline comparison and decision gate."""

    args = build_parser().parse_args(argv)
    try:
        report, summary_rows = build_comparison_report(
            repo_root=REPO_ROOT,
            local_results_path=args.local_results,
            local_eval_path=args.local_eval,
            api_results_path=args.api_results,
            api_eval_path=args.api_eval,
            api_cost_path=args.api_cost,
            retrieval_manifest_path=args.retrieval_manifest,
            gpu_costs_path=args.gpu_costs,
        )
        report_path, summary_path = write_comparison_artifacts(
            report_path=args.report_path,
            summary_path=args.summary_path,
            report=report,
            summary_rows=summary_rows,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Block 28 comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"GPU readiness decision: {report['decision']}")
    print(f"Comparison report: {report_path}")
    print(f"Comparison summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
