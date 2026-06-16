"""Write the B6 full-run AI engineering readiness audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.full_run_readiness_audit import (  # noqa: E402
    build_full_run_readiness_audit,
    write_full_run_readiness_artifacts,
)

DEFAULT_REPORT = "results/processed/b6_full_run_readiness_report.json"
DEFAULT_SUMMARY = "results/processed/b6_full_run_readiness_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the readiness audit CLI."""

    parser = argparse.ArgumentParser(
        description="Audit full-run and RunPod readiness after the B6 quality gate."
    )
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    return parser


def main() -> int:
    """Run the readiness audit."""

    args = build_parser().parse_args()
    report = build_full_run_readiness_audit(repo_root=args.repo_root)
    write_full_run_readiness_artifacts(
        report=report,
        report_path=ROOT / args.report,
        summary_path=ROOT / args.summary,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
