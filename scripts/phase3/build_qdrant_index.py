"""Build a local Qdrant vector index for Phase 3 context corpora.

This script embeds normalized context records and persists local Qdrant
collections. It does not run model inference, GPU work, or external APIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.vector_store import build_qdrant_index  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build local Qdrant collections from Phase 3 context corpora."
    )
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--output-root", default="data/generated/context_engineering")
    parser.add_argument("--vector-store-config", default="configs/vector_stores.yaml")
    parser.add_argument("--vector-store-key", default="qdrant_local")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Qdrant index generation."""

    args = build_parser().parse_args(argv)
    result = build_qdrant_index(
        context_root=args.context_root,
        output_root=args.output_root,
        vector_store_config_path=args.vector_store_config,
        vector_store_key=args.vector_store_key,
    )
    for row in result.summary_rows:
        print(
            f"{row['vertical']}: collection={row['collection_name']} "
            f"indexed_chunks={row['indexed_chunks']} "
            f"embedding={row['embedding_backend_effective']}"
        )
    print(f"Qdrant index report: {Path(args.output_root) / 'qdrant_index_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
