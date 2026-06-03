"""Analyze retrieval root causes from existing Phase 3 reports.

This script does not run retrieval, model inference, GPU work, or API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.retrieval_root_cause import (  # noqa: E402
    build_retrieval_root_cause_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build an offline retrieval root-cause report from existing diagnostics."
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--slo-config", default="configs/slo_targets.yaml")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run root-cause analysis."""

    args = build_parser().parse_args(argv)
    report, summary_rows, examples = build_retrieval_root_cause_report(
        dataset_root=args.dataset_root,
        context_root=args.context_root,
        slo_config=args.slo_config,
        output_root=args.output_root,
    )
    output_root = Path(args.output_root)
    print(f"Retrieval root-cause report: {output_root / 'retrieval_root_cause_report.json'}")
    print(f"Retrieval root-cause summary: {output_root / 'retrieval_root_cause_summary.csv'}")
    print(f"Retrieval failure examples: {output_root / 'retrieval_failure_examples.jsonl'}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Failure examples: {len(examples)}")
    print(f"Primary blocker: {report['blocker_assessment']['primary_blocker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
