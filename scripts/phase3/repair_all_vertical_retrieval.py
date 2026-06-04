"""Run all-vertical retrieval SLO repair audit and staged validation.

This script does not run inference, GPU work, or paid API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.vertical_retrieval_repair import (  # noqa: E402
    build_all_vertical_retrieval_repair,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit and repair all-vertical retrieval SLO readiness without inference, "
            "GPU work, or paid API calls."
        )
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--slo-config", default="configs/slo_targets.yaml")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    parser.add_argument("--stage-sizes", nargs="+", type=int, default=[250, 500])
    parser.add_argument(
        "--dense-backend",
        choices=["local_fallback", "qdrant_vector"],
        default="qdrant_vector",
    )
    parser.add_argument("--vector-store-config", default="configs/vector_stores.yaml")
    parser.add_argument("--vector-store-key", default="qdrant_local")
    parser.add_argument(
        "--require-dense-backend",
        action="store_true",
        help="Fail instead of falling back if the requested dense backend cannot be created.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run all-vertical retrieval repair."""

    args = build_parser().parse_args(argv)
    report = build_all_vertical_retrieval_repair(
        dataset_root=args.dataset_root,
        context_root=args.context_root,
        slo_config_path=args.slo_config,
        output_root=args.output_root,
        stage_sizes=args.stage_sizes,
        dense_backend=args.dense_backend,
        vector_store_config_path=args.vector_store_config,
        vector_store_key=args.vector_store_key,
        allow_dense_fallback=not args.require_dense_backend,
    )
    output_root = Path(args.output_root)
    print(
        "All-vertical retrieval repair report: "
        f"{output_root / 'all_vertical_retrieval_repair_report.json'}"
    )
    print(
        "All-vertical retrieval repair summary: "
        f"{output_root / 'all_vertical_retrieval_repair_summary.csv'}"
    )
    print(
        "All-vertical retrieval repair examples: "
        f"{output_root / 'all_vertical_retrieval_repair_examples.jsonl'}"
    )
    print(f"Stage sizes: {report['stage_sizes']}")
    print(f"Requested dense backend: {report['dense_backend_requested']}")
    print(f"Direct hint leakage detected: {report['direct_hint_leakage_detected_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
