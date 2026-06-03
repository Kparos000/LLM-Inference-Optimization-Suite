"""Build Finance prompt/gold repair and retrieval-impact reports.

This script does not run model inference, GPU work, or external API calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.finance_retrieval_repair import (  # noqa: E402
    build_finance_retrieval_repair,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Audit and repair Finance retrieval metadata without running inference, GPU work, "
            "or API calls."
        )
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
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
        help="Fail instead of falling back if the requested dense backend is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Finance retrieval repair reporting."""

    args = build_parser().parse_args(argv)
    report = build_finance_retrieval_repair(
        dataset_root=args.dataset_root,
        context_root=args.context_root,
        output_root=args.output_root,
        dense_backend=args.dense_backend,
        vector_store_config_path=args.vector_store_config,
        vector_store_key=args.vector_store_key,
        allow_dense_fallback=not args.require_dense_backend,
    )
    output_files = report["output_files"]
    print(f"Finance prompt quality report: {output_files['finance_prompt_quality_report']}")
    print(f"Finance gold quality report: {output_files['finance_gold_quality_report']}")
    print(
        f"Finance metadata enrichment report: {output_files['finance_metadata_enrichment_report']}"
    )
    print(
        "Finance retrieval repair impact report: "
        f"{output_files['finance_retrieval_repair_impact_report']}"
    )
    prompt_summary = report["prompt_quality_summary"]
    print(f"Prompts missing period: {prompt_summary['period_missing_count']}")
    print(f"Prompts missing metric: {prompt_summary['metric_missing_count']}")
    print(f"Prompts missing filing type: {prompt_summary['filing_type_missing_count']}")
    print(f"Prompts missing section: {prompt_summary['section_missing_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
