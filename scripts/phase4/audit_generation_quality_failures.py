"""Audit Phase B1 generation quality failures without running inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference_bench.generation_quality_audit import (  # noqa: E402
    build_generation_quality_audit,
    write_generation_quality_audit_artifacts,
)

DEFAULT_EVALUATION = "results/processed/b1_vllm_1_5b_quality_report.json"
DEFAULT_RESULTS = "results/raw/b1_remote_rtx3070_vllm_1_5b_results.jsonl"
DEFAULT_RUNNER_INPUT = "data/generated/phase4/b1_remote_rtx3070_runner_input.jsonl"
DEFAULT_REPORT = "results/processed/b3_generation_quality_audit_report.json"
DEFAULT_SUMMARY = "results/processed/b3_generation_quality_audit_summary.csv"
DEFAULT_FINANCE_EXAMPLES = "results/processed/b3_finance_failure_examples.jsonl"
DEFAULT_FAILURE_EXAMPLES = "results/processed/b3_quality_failure_examples.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object row: {path}")
            rows.append(cast(dict[str, Any], payload))
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the offline B3 audit CLI."""

    parser = argparse.ArgumentParser(
        description="Audit B1 generation quality failures from existing artifacts."
    )
    parser.add_argument("--evaluation-path", default=DEFAULT_EVALUATION)
    parser.add_argument("--results-path", default=DEFAULT_RESULTS)
    parser.add_argument("--runner-input-path", default=DEFAULT_RUNNER_INPUT)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    parser.add_argument("--finance-examples-path", default=DEFAULT_FINANCE_EXAMPLES)
    parser.add_argument("--failure-examples-path", default=DEFAULT_FAILURE_EXAMPLES)
    return parser


def main() -> None:
    """Run the deterministic audit and write all requested reports."""

    args = build_parser().parse_args()
    evaluation_report = _read_json(ROOT / args.evaluation_path)
    evaluation_rows = cast(list[dict[str, Any]], evaluation_report["evaluation_rows"])
    report = build_generation_quality_audit(
        evaluation_rows=evaluation_rows,
        result_rows=_read_jsonl(ROOT / args.results_path),
        runner_inputs=_read_jsonl(ROOT / args.runner_input_path),
    )
    outputs = write_generation_quality_audit_artifacts(
        report=report,
        report_path=ROOT / args.report_path,
        summary_path=ROOT / args.summary_path,
        finance_examples_path=ROOT / args.finance_examples_path,
        failure_examples_path=ROOT / args.failure_examples_path,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_row_count": report["failed_row_count"],
                "outputs": [str(path.relative_to(ROOT)) for path in outputs],
                "model_inference_triggered": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
