"""Evaluate production SLO readiness from available benchmark reports.

This script does not run inference, GPU work, or external API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.slo import (  # noqa: E402
    build_slo_readiness_report,
    load_slo_config,
    write_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate production SLO readiness from current generated reports."
    )
    parser.add_argument("--slo-config", default="configs/slo_targets.yaml")
    parser.add_argument(
        "--retrieval-report",
        default="data/generated/context_engineering/retrieval_evaluation_report.json",
    )
    parser.add_argument(
        "--quality-gate-report",
        default="data/generated/context_engineering/retrieval_quality_gate_report.json",
    )
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run SLO readiness evaluation."""

    args = build_parser().parse_args(argv)
    config = load_slo_config(args.slo_config)
    report, rows = build_slo_readiness_report(
        slo_config=config,
        retrieval_report_path=args.retrieval_report,
        quality_gate_report_path=args.quality_gate_report,
    )
    output_root = Path(args.output_root)
    report_path = write_json(output_root / "slo_readiness_report.json", report)
    summary_path = write_csv(output_root / "slo_readiness_summary.csv", rows)
    print(f"SLO readiness report: {report_path}")
    print(f"SLO readiness summary: {summary_path}")
    print(f"Overall status: {report['summary']['overall_status']}")
    print(
        "Inference scaling blocked by retrieval SLOs: "
        f"{report['inference_scaling_blocked_by_retrieval_slos']}"
    )
    print(f"Retrieval SLO blocked metrics: {report['retrieval_slo_blocked_count']}")
    print(f"NOT_AVAILABLE metrics: {report['not_available_metric_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
