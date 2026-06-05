"""Generate the Phase 4 pre-GPU readiness report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.phase4_readiness import (  # noqa: E402
    DEFAULT_BACKEND_MATRIX_PATH,
    DEFAULT_CONTEXT_ROOT,
    DEFAULT_GPU_COSTS_PATH,
    DEFAULT_OUTPUT_ROOT,
    build_phase4_readiness_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the readiness CLI parser."""

    parser = argparse.ArgumentParser(
        description="Inspect Phase 4 plumbing readiness without running inference or GPU work."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--context-root", default=DEFAULT_CONTEXT_ROOT)
    parser.add_argument("--backend-matrix", default=DEFAULT_BACKEND_MATRIX_PATH)
    parser.add_argument("--gpu-costs", default=DEFAULT_GPU_COSTS_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness inspection."""

    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    report = build_phase4_readiness_report(
        repo_root=repo_root,
        output_root=output_root,
        context_root=args.context_root,
        backend_matrix_path=args.backend_matrix,
        gpu_costs_path=args.gpu_costs,
    )
    print(f"Phase 4 readiness status: {report['overall_status']}")
    print(f"Pre-GPU plumbing ready: {report['pre_gpu_plumbing_ready']}")
    print(f"Live GPU smoke ready: {report['ready_for_live_gpu_smoke']}")
    print(f"Report: {output_root / 'phase4_readiness_report.json'}")
    print(f"Summary: {output_root / 'phase4_readiness_summary.csv'}")
    return 0 if report["pre_gpu_plumbing_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
