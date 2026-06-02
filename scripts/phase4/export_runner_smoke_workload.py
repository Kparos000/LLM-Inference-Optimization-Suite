"""Export Phase 3 WorkloadRecord JSONL into existing runner input JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.workload_adapter import export_runner_workload  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Export Phase 3 memory workload JSONL for existing runner CLIs."
    )
    parser.add_argument(
        "--workload-path",
        required=True,
        help="Path to a Phase 3 WorkloadRecord JSONL file.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where runner-compatible WorkloadItem JSONL should be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to export.",
    )
    parser.add_argument(
        "--report-path",
        default="data/generated/phase4/smoke_workload_export_report.json",
    )
    parser.add_argument(
        "--summary-path",
        default="data/generated/phase4/smoke_workload_export_summary.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run smoke workload export."""

    args = build_parser().parse_args(argv)
    report = export_runner_workload(
        workload_path=args.workload_path,
        output_path=args.output_path,
        limit=args.limit,
        report_path=args.report_path,
        summary_path=args.summary_path,
    )
    print(f"Runner workload rows written: {report['record_count']}")
    print(f"Output path: {report['output_path']}")
    print(f"Export report: {args.report_path}")
    print(f"Export summary: {args.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
