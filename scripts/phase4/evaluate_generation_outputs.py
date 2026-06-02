"""Evaluate generated output rows against promoted gold/eval records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.context_corpora import VERTICALS, benchmark_paths, read_jsonl  # noqa: E402
from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402

DEFAULT_REPORT_NAME = "phase4_mock_smoke_eval_report.json"
DEFAULT_SUMMARY_NAME = "phase4_mock_smoke_eval_summary.csv"
HF_LOCAL_SMOKE_REPORT_NAME = "phase4_hf_local_smoke_eval_report.json"
HF_LOCAL_SMOKE_SUMMARY_NAME = "phase4_hf_local_smoke_eval_summary.csv"
OPENAI_COMPATIBLE_SMOKE_REPORT_NAME = "phase4_openai_compatible_smoke_eval_report.json"
OPENAI_COMPATIBLE_SMOKE_SUMMARY_NAME = "phase4_openai_compatible_smoke_eval_summary.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate generation result CSV/JSONL rows by joining to gold records."
    )
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--output-root", default="results/processed")
    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
    )
    parser.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
    )
    return parser


def resolve_output_names(
    *,
    results_path: str | Path,
    report_name: str,
    summary_name: str,
) -> tuple[str, str]:
    """Return output names for known Phase 4 smoke result paths."""

    stem = Path(results_path).stem
    if (
        stem == "phase4_hf_local_smoke_results"
        and report_name == DEFAULT_REPORT_NAME
        and summary_name == DEFAULT_SUMMARY_NAME
    ):
        return HF_LOCAL_SMOKE_REPORT_NAME, HF_LOCAL_SMOKE_SUMMARY_NAME
    if (
        stem == "phase4_openai_compatible_smoke_results"
        and report_name == DEFAULT_REPORT_NAME
        and summary_name == DEFAULT_SUMMARY_NAME
    ):
        return OPENAI_COMPATIBLE_SMOKE_REPORT_NAME, OPENAI_COMPATIBLE_SMOKE_SUMMARY_NAME
    return report_name, summary_name


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load result rows from CSV or JSONL."""

    result_path = Path(path)
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if result_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with result_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                row = json.loads(stripped_line)
                if not isinstance(row, dict):
                    msg = f"Invalid JSONL row at line {line_number}: expected object"
                    raise ValueError(msg)
                rows.append(row)
        return rows

    with result_path.open(encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def load_gold_records(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Load all promoted gold/eval records."""

    gold: list[dict[str, Any]] = []
    for vertical in VERTICALS:
        paths = benchmark_paths(dataset_root, vertical)
        for row in read_jsonl(paths["gold"]):
            row["vertical"] = row.get("vertical") or vertical
            gold.append(row)
    return gold


def _split_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item]
    return []


def result_row_to_generated_answer(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a runner result/generation row to evaluator input."""

    success_value = str(row.get("success") or "").strip().lower()
    success = success_value in {"true", "1", "yes"}
    generated_text = (
        row.get("generated_text")
        or row.get("output_text")
        or row.get("response")
        or row.get("answer")
        or ""
    )
    return {
        "prompt_id": str(row.get("prompt_id") or ""),
        "generated_text": str(generated_text),
        "final_status": str(
            row.get("final_status") or ("answer" if success else "failed_validation")
        ),
        "citations": _split_json_list(row.get("citations")),
        "expected_output_format": row.get("expected_output_format"),
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def build_summary_rows(
    *,
    results_path: str | Path,
    result_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build aggregate evaluation summary rows."""

    total = len(evaluation_rows)
    joined_count = sum(1 for row in evaluation_rows if row["joined"])
    format_valid_count = sum(1 for row in evaluation_rows if row["format_valid"])
    grounded_count = sum(1 for row in evaluation_rows if row["groundedness"])
    safety_violation_count = sum(1 for row in evaluation_rows if row["safety_violation"])
    memory_modes = sorted(
        {str(row.get("memory_mode")) for row in result_rows if row.get("memory_mode")}
    )
    verticals = sorted({str(row.get("vertical")) for row in result_rows if row.get("vertical")})
    joined_by_status = Counter(
        str(row["expected_status"]) for row in evaluation_rows if row["joined"]
    )

    return [
        {
            "results_path": str(results_path),
            "row_count": total,
            "joined_count": joined_count,
            "joined_rate": _rate(joined_count, total),
            "format_valid_count": format_valid_count,
            "format_valid_rate": _rate(format_valid_count, total),
            "grounded_count": grounded_count,
            "grounded_rate": _rate(grounded_count, total),
            "safety_violation_count": safety_violation_count,
            "safety_violation_rate": _rate(safety_violation_count, total),
            "memory_modes": ";".join(memory_modes),
            "verticals": ";".join(verticals),
            "expected_status_counts": json.dumps(
                dict(sorted(joined_by_status.items())),
                sort_keys=True,
            ),
        }
    ]


def write_report(
    *,
    results_path: str | Path,
    output_root: str | Path,
    result_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    report_name: str,
    summary_name: str,
) -> tuple[Path, Path]:
    """Write evaluation JSON and CSV outputs."""

    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_name
    summary_path = output_dir / summary_name
    summary_rows = build_summary_rows(
        results_path=results_path,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
    )

    report = {
        "results_path": str(results_path),
        "row_count": len(evaluation_rows),
        "summary": summary_rows[0],
        "evaluation_rows": evaluation_rows,
        "no_model_inference_triggered": True,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(summary_rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return report_path, summary_path


def main(argv: list[str] | None = None) -> int:
    """Run deterministic evaluator contract over generated output rows."""

    args = build_parser().parse_args(argv)
    result_rows = load_result_rows(args.results_path)
    generated_answers = [result_row_to_generated_answer(row) for row in result_rows]
    evaluation_rows = evaluate_generated_answers(
        generated_answers,
        load_gold_records(args.dataset_root),
    )
    report_name, summary_name = resolve_output_names(
        results_path=args.results_path,
        report_name=args.report_name,
        summary_name=args.summary_name,
    )
    report_path, summary_path = write_report(
        results_path=args.results_path,
        output_root=args.output_root,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
        report_name=report_name,
        summary_name=summary_name,
    )
    print(f"Evaluation rows written: {len(evaluation_rows)}")
    print(f"Evaluation report: {report_path}")
    print(f"Evaluation summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
