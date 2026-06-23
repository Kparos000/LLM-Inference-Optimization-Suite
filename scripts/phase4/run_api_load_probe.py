"""Create API load-probe artifacts, optionally running a guarded live probe."""

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
    run_live_api_probe,
    write_api_probe_artifacts,
)

DEFAULT_REPORT = "results/processed/api_load_probe_report.json"
DEFAULT_SUMMARY = "results/processed/api_load_probe_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the API probe CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Write API load-probe artifacts. Live requests run only with "
            "--live-if-keys-present and configured provider keys."
        )
    )
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--live-if-keys-present",
        action="store_true",
        help="Run the small live probe when required API keys are present.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--prompt-count-per-model", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        type=int,
        action="append",
        default=None,
        help="Concurrency level to probe. May be supplied multiple times.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> int:
    """Write API load-probe artifacts."""

    args = build_parser().parse_args()
    concurrencies = args.concurrency if args.concurrency is not None else [1, 2, 4]
    report = (
        run_live_api_probe(
            concurrencies=concurrencies,
            prompt_count_per_model=args.prompt_count_per_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            env_path=ROOT / args.env_file,
        )
        if args.live_if_keys_present
        else build_framework_only_api_probe_report()
    )
    write_api_probe_artifacts(
        report=report,
        report_path=ROOT / args.report,
        summary_path=ROOT / args.summary,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
