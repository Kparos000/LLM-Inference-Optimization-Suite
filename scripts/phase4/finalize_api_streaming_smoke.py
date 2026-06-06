"""Write cost, latency, and grounding diagnostics for the streaming smoke."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_priced_validation import (  # noqa: E402
    build_cost_report,
    write_cost_artifacts,
)
from inference_bench.grounding_diagnostics import (  # noqa: E402
    build_grounding_failure_report,
    write_grounding_failure_artifacts,
)
from inference_bench.stronger_model_validation import read_jsonl  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build finalizer CLI."""

    parser = argparse.ArgumentParser(
        description="Finalize measured streaming API cost, latency, and grounding reports."
    )
    parser.add_argument(
        "--results-path",
        default="results/raw/phase4_api_streaming_smoke_results.jsonl",
    )
    parser.add_argument(
        "--eval-report",
        default="results/processed/phase4_api_streaming_smoke_eval_report.json",
    )
    parser.add_argument("--output-root", default="results/processed")
    return parser


def _values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), int | float) and not isinstance(row.get(field), bool)
    ]


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main(argv: list[str] | None = None) -> int:
    """Write all post-evaluation streaming reports."""

    args = build_parser().parse_args(argv)
    try:
        result_rows = read_jsonl(args.results_path)
        eval_report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
        evaluation_rows = eval_report["evaluation_rows"]
        output = Path(args.output_root)
        output.mkdir(parents=True, exist_ok=True)
        cost_report = build_cost_report(
            result_rows=result_rows,
            evaluation_rows=evaluation_rows,
            baseline_summary=None,
        )
        write_cost_artifacts(
            report_path=output / "phase4_api_streaming_cost_report.json",
            summary_path=output / "phase4_api_streaming_cost_summary.csv",
            report=cost_report,
        )
        latency_report = {
            "row_count": len(result_rows),
            "streaming_success_count": sum(
                bool(row.get("streaming_available")) for row in result_rows
            ),
            "ttft_ms": _aggregate(_values(result_rows, "ttft_ms")),
            "itl_p50_ms": _aggregate(_values(result_rows, "itl_p50_ms")),
            "itl_p95_ms": _aggregate(_values(result_rows, "itl_p95_ms")),
            "itl_p99_ms": _aggregate(_values(result_rows, "itl_p99_ms")),
            "tpot_ms": _aggregate(_values(result_rows, "tpot_ms")),
            "e2e_latency_ms": _aggregate(_values(result_rows, "e2e_latency_ms")),
            "per_request": [
                {
                    key: row.get(key)
                    for key in (
                        "prompt_id",
                        "vertical",
                        "ttft_ms",
                        "itl_p50_ms",
                        "itl_p95_ms",
                        "itl_p99_ms",
                        "tpot_ms",
                        "e2e_latency_ms",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "token_count_source",
                    )
                }
                for row in result_rows
            ],
            "no_gpu_work_triggered": True,
        }
        (output / "phase4_api_streaming_latency_report.json").write_text(
            json.dumps(latency_report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        grounding_report = build_grounding_failure_report(
            evaluation_rows=evaluation_rows,
            result_rows=result_rows,
        )
        write_grounding_failure_artifacts(
            report_path=output / "phase4_grounding_failure_report.json",
            summary_path=output / "phase4_grounding_failure_summary.csv",
            report=grounding_report,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Streaming smoke finalization failed: {exc}", file=sys.stderr)
        return 1
    print(f"Cost report: {output / 'phase4_api_streaming_cost_report.json'}")
    print(f"Latency report: {output / 'phase4_api_streaming_latency_report.json'}")
    print(f"Grounding report: {output / 'phase4_grounding_failure_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
