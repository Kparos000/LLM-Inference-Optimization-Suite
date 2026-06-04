"""Build repaired retrieval dataset/gold alignment reports.

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

from inference_bench.retrieval_dataset_alignment import (  # noqa: E402
    build_retrieval_dataset_alignment_repair,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Repair generated retrieval dataset/gold alignment without modifying the "
            "promoted benchmark dataset."
        )
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--slo-config", default="configs/slo_targets.yaml")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    parser.add_argument("--stage-sizes", nargs="+", type=int, default=[500, 2000])
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
    """Run retrieval dataset/gold alignment repair."""

    args = build_parser().parse_args(argv)
    result = build_retrieval_dataset_alignment_repair(
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
    print(f"Alignment report: {output_root / 'retrieval_dataset_alignment_report.json'}")
    print(f"Alignment summary: {output_root / 'retrieval_dataset_alignment_summary.csv'}")
    print(f"Records needing repair: {output_root / 'retrieval_records_needing_repair.jsonl'}")
    print(
        f"Repaired validation report: {output_root / 'repaired_retrieval_validation_report.json'}"
    )
    print(
        f"Repaired validation summary: {output_root / 'repaired_retrieval_validation_summary.csv'}"
    )
    print(f"Promotion plan: {output_root / 'repaired_retrieval_promotion_plan.json'}")
    print(f"Promotion recommended: {result['promotion_plan']['promotion_recommended']}")
    print(f"Records needing repair: {result['alignment_report']['records_needing_repair_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
