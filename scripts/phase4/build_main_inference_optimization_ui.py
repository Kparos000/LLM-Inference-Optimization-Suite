"""Build UI-facing optimization intelligence artifacts for Main_Inference_V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.main_inference_optimization_ui import write_ui_artifacts  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-root",
        default="experiments/main/main_inference_v1",
        help="Official Main_Inference_V1 artifact root.",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/main/main_inference_v1/processed",
        help="Directory where UI JSON artifacts are written.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = write_ui_artifacts(
        experiment_root=args.experiment_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": "UI_OPTIMIZATION_INTELLIGENCE_WRITTEN",
                "inference_executed": False,
                "optimized_result_created": False,
                "paths": {key: str(path) for key, path in paths.items()},
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
