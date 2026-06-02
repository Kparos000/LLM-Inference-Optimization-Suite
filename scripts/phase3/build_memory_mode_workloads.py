"""Build Phase 3 memory-mode workload files.

This script builds model-ready workload records for mm0-mm3. It does not run
model inference, GPU work, embeddings, or backend harnesses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.memory_workloads import (  # noqa: E402
    SUPPORTED_MEMORY_MODES,
    build_memory_mode_workloads,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Build Phase 3 memory-mode workload JSONL files for mm0-mm3."
    )
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--context-root", default="data/generated/context_engineering")
    parser.add_argument("--output-root", default="data/workloads")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["smoke_500", "controlled_2000", "final_10000"],
    )
    parser.add_argument(
        "--memory-modes",
        nargs="+",
        default=sorted(SUPPORTED_MEMORY_MODES),
    )
    parser.add_argument(
        "--dense-backend",
        choices=["local_fallback", "qdrant_vector"],
        default="local_fallback",
    )
    parser.add_argument(
        "--ablation-modes",
        nargs="+",
        default=["prompt_plus_source_hints"],
    )
    parser.add_argument("--vector-store-config", default="configs/vector_stores.yaml")
    parser.add_argument("--vector-store-key", default="qdrant_local")
    parser.add_argument(
        "--allow-dense-fallback",
        action="store_true",
        help="Allow local_fallback if the requested dense backend cannot be created.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run workload generation."""

    args = build_parser().parse_args(argv)
    result = build_memory_mode_workloads(
        dataset_root=args.dataset_root,
        context_root=args.context_root,
        output_root=args.output_root,
        splits=args.splits,
        memory_modes=args.memory_modes,
        dense_backend=args.dense_backend,
        ablation_modes=args.ablation_modes,
        vector_store_config_path=args.vector_store_config,
        vector_store_key=args.vector_store_key,
        allow_dense_fallback=args.allow_dense_fallback,
    )
    report = result.workload_build_report
    for split, split_payload in report["by_split"].items():
        for ablation_mode, ablation_payload in split_payload.items():
            for memory_mode, payload in ablation_payload.items():
                print(
                    f"{split}/{ablation_mode}/{memory_mode}: records={payload['record_count']} "
                    f"output={payload['output_path']}"
                )
    print(f"Workload build report: {Path(args.context_root) / 'workload_build_report.json'}")
    print(
        "Retrieval evaluation report: "
        f"{Path(args.context_root) / 'retrieval_evaluation_report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
