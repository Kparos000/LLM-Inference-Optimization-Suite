"""Build the Phase 3 readiness report.

This script only inspects artifacts and contracts. It does not run retrieval,
model inference, GPU work, or external API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.phase3_readiness import build_phase3_readiness_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build the Phase 3 readiness report without running inference."
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--workload-root", default="data/workloads")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run readiness report generation."""

    args = build_parser().parse_args(argv)
    report = build_phase3_readiness_report(
        dataset_root=args.dataset_root,
        context_root=args.context_root,
        workload_root=args.workload_root,
        output_root=args.output_root,
    )
    output_root = Path(args.output_root)
    print(f"Phase 3 readiness report: {output_root / 'phase3_readiness_report.json'}")
    print(f"Phase 3 readiness summary: {output_root / 'phase3_readiness_summary.csv'}")
    print(f"Ready for Phase 4 plumbing: {report['ready_for_phase4_plumbing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
