"""Generate deterministic scaled synthetic benchmark workloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.workloads.scaled_generator import (  # noqa: E402
    DEFAULT_SCALED_WORKLOADS,
    generate_scaled_workloads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic workload JSONL files."
    )
    parser.add_argument("--output-dir", default="data/prompts/scaled")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workloads", nargs="*", default=list(DEFAULT_SCALED_WORKLOADS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written_paths = generate_scaled_workloads(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        workloads=args.workloads,
    )

    for path in written_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
