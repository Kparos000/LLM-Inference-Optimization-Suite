"""Build the B6 500-prompt context-aligned runner input and preflight reports."""

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
    build_b6_context_aligned_runner_input,
)

DEFAULT_SOURCE_WORKLOAD = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_MANIFEST = "data/generated/context_engineering/retrieval_source_of_truth_manifest.json"
DEFAULT_DATASET_ROOT = "data/scaleup_2000_full"
DEFAULT_CONTEXT_ROOT = "data/generated/context_engineering"
DEFAULT_OUTPUT = "data/generated/phase4/b6_context_aligned_500_runner_input.jsonl"
DEFAULT_REPORT = "results/processed/b6_context_alignment_preflight_report.json"
DEFAULT_SUMMARY = "results/processed/b6_context_alignment_preflight_summary.csv"
DEFAULT_EXAMPLES = "results/processed/b6_context_alignment_preflight_examples.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Build the B6 preflight CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build the B6 500-prompt context-aligned runner input."
    )
    parser.add_argument("--source-workload", default=DEFAULT_SOURCE_WORKLOAD)
    parser.add_argument("--source-of-truth-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--context-root", default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--examples", default=DEFAULT_EXAMPLES)
    parser.add_argument("--prompts-per-vertical", type=int, default=100)
    return parser


def main() -> int:
    """Run B6 input construction."""

    args = build_parser().parse_args()
    report = build_b6_context_aligned_runner_input(
        source_workload_path=ROOT / args.source_workload,
        source_of_truth_manifest_path=ROOT / args.source_of_truth_manifest,
        dataset_root=ROOT / args.dataset_root,
        context_root=ROOT / args.context_root,
        output_path=ROOT / args.output,
        report_path=ROOT / args.report,
        summary_path=ROOT / args.summary,
        examples_path=ROOT / args.examples,
        prompts_per_vertical=args.prompts_per_vertical,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
