"""Generate the controlled inference readiness audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.controlled_run_readiness import (  # noqa: E402
    inspect_controlled_inference_readiness,
    write_controlled_readiness_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the audit CLI parser."""

    parser = argparse.ArgumentParser(
        description="Audit controlled small-GPU smoke readiness without running a backend."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default="data/generated/phase4")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the audit and report its decision."""

    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    report, checks = inspect_controlled_inference_readiness(repo_root)
    report_path, summary_path = write_controlled_readiness_artifacts(
        output_root=output_root,
        report=report,
        checks=checks,
    )
    print(f"Controlled inference readiness: {report['readiness_status']}")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
