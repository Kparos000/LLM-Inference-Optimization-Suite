"""Evaluate Block 27 output and write measured API cost reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_priced_validation import (  # noqa: E402
    build_cost_report,
    write_cost_artifacts,
)

DEFAULT_RESULTS_PATH = "results/raw/phase4_api_priced_smoke_results.jsonl"
DEFAULT_EVAL_REPORT = "results/processed/phase4_api_priced_smoke_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/phase4_api_priced_smoke_eval_summary.csv"
DEFAULT_COST_REPORT = "results/processed/phase4_api_priced_cost_report.json"
DEFAULT_COST_SUMMARY = "results/processed/phase4_api_priced_cost_summary.csv"
DEFAULT_BASELINE_REPORT = "results/processed/phase4_generation_contract_hardened_eval_report.json"


def _load_evaluator() -> ModuleType:
    path = REPO_ROOT / "scripts/phase4/evaluate_generation_outputs.py"
    spec = importlib.util.spec_from_file_location("_block27_generation_evaluator", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load evaluator from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baseline_summary(path: str | Path) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.is_file():
        return None
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("summary"), dict):
        return None
    return cast(dict[str, Any], loaded["summary"])


def build_parser() -> argparse.ArgumentParser:
    """Build the Block 27 evaluator parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate API-priced generation output and calculate measured token cost."
    )
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--eval-report", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--cost-report", default=DEFAULT_COST_REPORT)
    parser.add_argument("--cost-summary", default=DEFAULT_COST_SUMMARY)
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate measured generation and write cost artifacts."""

    args = build_parser().parse_args(argv)
    evaluator = _load_evaluator()
    result_rows = evaluator.load_result_rows(args.results_path)
    if len(result_rows) != 5:
        print("Block 27 evaluation requires exactly five result rows.", file=sys.stderr)
        return 1
    generated_answers = [evaluator.result_row_to_generated_answer(row) for row in result_rows]
    evaluation_rows = evaluator.evaluate_generated_answers(
        generated_answers,
        evaluator.load_gold_records(args.dataset_root),
    )
    eval_report = Path(args.eval_report)
    eval_summary = Path(args.eval_summary)
    evaluator.write_report(
        results_path=args.results_path,
        output_root=eval_report.parent,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
        report_name=eval_report.name,
        summary_name=eval_summary.name,
    )
    loaded_eval_report = json.loads(eval_report.read_text(encoding="utf-8"))
    loaded_eval_report.update(
        {
            "generation_inference_executed": True,
            "paid_api_call_triggered": True,
            "model_alias": result_rows[0].get("model_alias"),
            "model_id": result_rows[0].get("model_id"),
            "provider": result_rows[0].get("provider"),
        }
    )
    eval_report.write_text(
        json.dumps(loaded_eval_report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cost_report = build_cost_report(
        result_rows=result_rows,
        evaluation_rows=cast(list[dict[str, Any]], evaluation_rows),
        baseline_summary=_baseline_summary(args.baseline_report),
    )
    write_cost_artifacts(
        report_path=args.cost_report,
        summary_path=args.cost_summary,
        report=cost_report,
    )
    print(f"Evaluation report: {eval_report}")
    print(f"Evaluation summary: {eval_summary}")
    print(f"Cost report: {args.cost_report}")
    print(f"Cost summary: {args.cost_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
