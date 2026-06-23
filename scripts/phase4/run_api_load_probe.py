"""Create the API load-probe framework artifacts without live provider calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.api_load_probe import (  # noqa: E402
    build_framework_only_api_probe_report,
    write_api_probe_artifacts,
)

DEFAULT_REPORT = "results/processed/api_load_probe_report.json"
DEFAULT_SUMMARY = "results/processed/api_load_probe_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the dry-run API probe CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Write API load-probe planning artifacts. This script does not send live API requests."
        )
    )
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    return parser


def main() -> int:
    """Write framework-only API load-probe artifacts."""

    args = build_parser().parse_args()
    report = build_framework_only_api_probe_report()
    write_api_probe_artifacts(
        report=report,
        report_path=ROOT / args.report,
        summary_path=ROOT / args.summary,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
