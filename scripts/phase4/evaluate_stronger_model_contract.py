"""Compare stronger-model contract results with the Block 24 0.5B baseline."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.stronger_model_validation import (  # noqa: E402
    COMPARISON_METRICS,
    load_baseline_metrics,
    read_jsonl,
    runtime_metrics,
    write_comparison_artifacts,
)

DEFAULT_RESULTS_PATH = "results/raw/phase4_stronger_model_contract_smoke.jsonl"
DEFAULT_BASELINE_REPORT = "results/processed/phase4_generation_contract_hardened_eval_report.json"
DEFAULT_REPORT_PATH = "results/processed/phase4_stronger_model_contract_eval_report.json"
DEFAULT_SUMMARY_PATH = "results/processed/phase4_stronger_model_contract_eval_summary.csv"


def _load_evaluator_module() -> ModuleType:
    path = REPO_ROOT / "scripts/phase4/evaluate_generation_outputs.py"
    spec = importlib.util.spec_from_file_location("_phase4_generation_evaluator", path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load evaluator module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Compare stronger-model generation-contract output with Block 24."
    )
    parser.add_argument("--results-path", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser


def _measured(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(bool(row.get("validation_measured")) for row in rows)


def _stronger_quality_metrics(
    *,
    evaluator: ModuleType,
    result_rows: list[dict[str, Any]],
    dataset_root: str | Path,
) -> tuple[dict[str, float | int | None], list[dict[str, Any]]]:
    generated_answers = [evaluator.result_row_to_generated_answer(row) for row in result_rows]
    evaluation_rows = evaluator.evaluate_generated_answers(
        generated_answers,
        evaluator.load_gold_records(dataset_root),
    )
    summary = evaluator.build_summary_rows(
        results_path="stronger_model",
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
    )[0]
    metrics: dict[str, float | int | None] = {metric: None for metric in COMPARISON_METRICS}
    for metric in COMPARISON_METRICS[:5]:
        value = summary.get(metric)
        if isinstance(value, int | float) and not isinstance(value, bool):
            metrics[metric] = float(value)
    metrics.update(runtime_metrics(result_rows))
    return metrics, cast(list[dict[str, Any]], evaluation_rows)


def main(argv: list[str] | None = None) -> int:
    """Write the stronger-model comparison report."""

    args = build_parser().parse_args(argv)
    try:
        result_rows = read_jsonl(args.results_path)
        if not result_rows:
            msg = "Stronger-model result file contains no rows"
            raise ValueError(msg)
        baseline = load_baseline_metrics(
            baseline_report_path=args.baseline_report,
        )
        model_alias = str(result_rows[0].get("model_alias") or "")
        model_id = str(result_rows[0].get("model_id") or "")
        execution_path = str(result_rows[0].get("execution_path") or "local_hf")
        if _measured(result_rows):
            evaluator = _load_evaluator_module()
            stronger, evaluation_rows = _stronger_quality_metrics(
                evaluator=evaluator,
                result_rows=result_rows,
                dataset_root=args.dataset_root,
            )
            status = "MEASURED"
            reason = None
        else:
            stronger = {metric: None for metric in COMPARISON_METRICS}
            evaluation_rows = []
            status = "DRY_RUN_ONLY"
            reason = str(
                result_rows[0].get("error_message")
                or "Stronger-model execution prerequisites were unavailable."
            )
        report_path, summary_path = write_comparison_artifacts(
            report_path=args.report_path,
            summary_path=args.summary_path,
            validation_status=status,
            model_alias=model_alias,
            model_id=model_id,
            execution_path=execution_path,
            baseline_metrics=baseline,
            stronger_metrics=stronger,
            evaluation_rows=evaluation_rows,
            reason=reason,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Stronger-model evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validation status: {status}")
    print(f"Evaluation report: {report_path}")
    print(f"Evaluation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
