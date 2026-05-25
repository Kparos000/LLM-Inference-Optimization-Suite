"""Build Phase 3 normalized context corpora from promoted benchmark KB files.

This script performs context-corpus normalization only. It does not implement
retrieval scoring, dense embeddings, inference, GPU work, or harness changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.context_corpora import build_context_corpora  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build Phase 3 normalized context corpora from promoted benchmark KB files."
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the context corpus builder."""

    args = build_parser().parse_args(argv)
    result = build_context_corpora(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    report = result["report"]
    print(f"Registry: {report['registry_path']}")
    print(f"Corpora: {report['corpora_dir']}")
    for vertical, vertical_report in report["by_vertical"].items():
        print(
            f"{vertical}: context_rows={vertical_report['context_row_count']} "
            f"strategy={vertical_report['chunk_strategy']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
