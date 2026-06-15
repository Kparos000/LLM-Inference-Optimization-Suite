"""Build the frozen B4 context-aligned runner input and preflight reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.context_alignment_repair import (  # noqa: E402
    build_context_aligned_runner_input,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the B4 context repair CLI."""

    parser = argparse.ArgumentParser(
        description="Repair frozen B1 E1-E5 context alignment from promoted retrieval."
    )
    parser.add_argument(
        "--b1-runner-input",
        default="data/generated/phase4/b1_remote_rtx3070_runner_input.jsonl",
    )
    parser.add_argument(
        "--source-workload",
        default="data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl",
    )
    parser.add_argument(
        "--source-of-truth-manifest",
        default="data/generated/context_engineering/retrieval_source_of_truth_manifest.json",
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument(
        "--output",
        default="data/generated/phase4/b4_context_aligned_runner_input.jsonl",
    )
    parser.add_argument(
        "--report",
        default="results/processed/b4_context_alignment_report.json",
    )
    parser.add_argument(
        "--summary",
        default="results/processed/b4_context_alignment_summary.csv",
    )
    parser.add_argument(
        "--finance-examples",
        default="results/processed/b4_finance_context_alignment_examples.jsonl",
    )
    return parser


def main() -> int:
    """Run the offline context alignment preflight."""

    args = build_parser().parse_args()
    report = build_context_aligned_runner_input(
        b1_runner_input_path=ROOT / args.b1_runner_input,
        source_workload_path=ROOT / args.source_workload,
        source_of_truth_manifest_path=ROOT / args.source_of_truth_manifest,
        dataset_root=ROOT / args.dataset_root,
        context_root=ROOT / args.context_root,
        output_path=ROOT / args.output,
        report_path=ROOT / args.report,
        summary_path=ROOT / args.summary,
        finance_examples_path=ROOT / args.finance_examples,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "row_count": report["row_count"],
                "unrecoverable_row_count": report["unrecoverable_row_count"],
                "inference_allowed": report["inference_allowed"],
                "summary_rows": report["summary_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(report["inference_allowed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
