"""Audit the B7 vLLM CUDA/CUBLAS failure mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.vllm_stability_audit import (  # noqa: E402
    audit_b7_artifacts,
    write_stability_audit_artifacts,
)

DEFAULT_RAW_RESULTS = "results/raw/b7_model2_3b_1000_results.jsonl"
DEFAULT_GPU_TELEMETRY = "results/raw/b7_model2_3b_1000_gpu_telemetry.jsonl"
DEFAULT_EVAL_REPORT = "results/processed/b7_model2_3b_1000_eval_report.json"
DEFAULT_REPORT = "results/processed/b7_vllm_cuda_failure_audit_report.json"
DEFAULT_SUMMARY = "results/processed/b7_vllm_cuda_failure_audit_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the audit CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", default=DEFAULT_RAW_RESULTS)
    parser.add_argument("--gpu-telemetry", default=DEFAULT_GPU_TELEMETRY)
    parser.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--expected-count", type=int, default=1000)
    return parser


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    """Run the audit and write JSON/CSV reports."""

    args = build_parser().parse_args()
    report = audit_b7_artifacts(
        raw_results_path=_repo_path(args.raw_results),
        telemetry_path=_repo_path(args.gpu_telemetry),
        eval_report_path=_repo_path(args.eval_report),
        expected_count=args.expected_count,
    )
    write_stability_audit_artifacts(
        report=report,
        report_path=_repo_path(args.report),
        summary_path=_repo_path(args.summary),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
