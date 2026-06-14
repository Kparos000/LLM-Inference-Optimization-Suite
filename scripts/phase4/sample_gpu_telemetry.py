"""Poll local or remote nvidia-smi and write Phase 4 GPU telemetry reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.gpu_telemetry import (  # noqa: E402
    sample_gpu_telemetry,
    write_gpu_telemetry_csv,
    write_gpu_telemetry_summary,
)

DEFAULT_CSV_PATH = "results/processed/a1_remote_rtx3070_gpu_telemetry.csv"
DEFAULT_SUMMARY_PATH = "results/processed/a1_remote_rtx3070_gpu_telemetry_summary.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the telemetry CLI parser."""

    parser = argparse.ArgumentParser(description="Sample nvidia-smi GPU telemetry.")
    parser.add_argument("--ssh-host", default=None)
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--output-csv", default=DEFAULT_CSV_PATH)
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect and write GPU telemetry."""

    args = build_parser().parse_args(argv)
    samples = sample_gpu_telemetry(
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_seconds,
        ssh_host=args.ssh_host,
    )
    csv_path = write_gpu_telemetry_csv(args.output_csv, samples)
    summary_path = write_gpu_telemetry_summary(
        args.summary_json,
        samples,
        interval_seconds=args.interval_seconds,
        requested_duration_seconds=args.duration_seconds,
    )
    print(f"GPU telemetry samples: {len(samples)}")
    print(f"GPU telemetry CSV: {csv_path}")
    print(f"GPU telemetry summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
