"""Run Block 16B Airline/Healthcare enrichment and Research AI validation.

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

from inference_bench.airline_healthcare_research_validation import (  # noqa: E402
    build_airline_healthcare_research_validation,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval-only Airline/Healthcare enrichment and Research AI scale "
            "validation without inference, GPU work, external APIs, gold IDs, or source IDs."
        )
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    parser.add_argument("--slo-config", default="configs/slo_targets.yaml")
    parser.add_argument("--stage-sizes", nargs="+", type=int, default=[250, 500])
    parser.add_argument("--research-stage-sizes", nargs="+", type=int, default=[500, 2000])
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
    """Run Block 16B validation."""

    args = build_parser().parse_args(argv)
    reports = build_airline_healthcare_research_validation(
        dataset_root=args.dataset_root,
        context_root=args.context_root,
        output_root=args.output_root,
        slo_config_path=args.slo_config,
        stage_sizes=args.stage_sizes,
        research_stage_sizes=args.research_stage_sizes,
        dense_backend=args.dense_backend,
        vector_store_config_path=args.vector_store_config,
        vector_store_key=args.vector_store_key,
        allow_dense_fallback=not args.require_dense_backend,
    )
    output_root = Path(args.output_root)
    print(
        "Airline/Healthcare enrichment report: "
        f"{output_root / 'airline_healthcare_enrichment_report.json'}"
    )
    print(
        "Airline/Healthcare enrichment summary: "
        f"{output_root / 'airline_healthcare_enrichment_summary.csv'}"
    )
    print(
        "Research AI scale validation report: "
        f"{output_root / 'research_ai_scale_validation_report.json'}"
    )
    print(
        "Research AI scale validation summary: "
        f"{output_root / 'research_ai_scale_validation_summary.csv'}"
    )
    enrichment = reports["airline_healthcare_enrichment"]
    research = reports["research_ai_scale_validation"]
    leakage_count = (
        enrichment["direct_hint_leakage_detected_count"]
        + research["direct_hint_leakage_detected_count"]
    )
    print(f"Direct hint leakage detected: {leakage_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
