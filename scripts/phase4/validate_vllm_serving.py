"""Validate the local vLLM OpenAI-compatible serving path on a tiny workload."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PHASE4_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from evaluate_generation_outputs import (  # noqa: E402
    build_summary_rows,
    load_gold_records,
    load_result_rows,
    result_row_to_generated_answer,
)
from run_openai_compatible_smoke import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    check_server_readiness,
    run_smoke,
    sanitize_command,
)

from inference_bench.evaluator_contract import evaluate_generated_answers  # noqa: E402
from inference_bench.telemetry import (  # noqa: E402
    TELEMETRY_FIELDS,
    BackendComparisonRow,
    build_backend_comparison_framework,
    telemetry_record_from_result_row,
    write_backend_comparison_csv,
    write_telemetry_json,
)

DEFAULT_INPUT_PATH = "data/generated/phase4/smoke_500_mm2_runner_input.jsonl"
DEFAULT_RAW_OUTPUT = "results/raw/phase4_vllm_validation.jsonl"
DEFAULT_REPORT = "results/processed/phase4_vllm_validation_report.json"
DEFAULT_SUMMARY = "results/processed/phase4_vllm_validation_summary.csv"
DEFAULT_MANIFEST = "results/raw/phase4_vllm_validation_manifest.json"
DEFAULT_TELEMETRY = "results/processed/phase4_vllm_validation_telemetry.json"
DEFAULT_BACKEND_COMPARISON = "results/processed/phase4_backend_comparison_framework.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Validate local vLLM serving through the OpenAI-compatible API."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--telemetry-path", default=DEFAULT_TELEMETRY)
    parser.add_argument("--backend-comparison-path", default=DEFAULT_BACKEND_COMPARISON)
    parser.add_argument("--dataset-root", default="data/scaleup_2000_full")
    parser.add_argument("--model-alias", default="model1_0_5b")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_summary(path: str | Path, row: dict[str, Any]) -> Path:
    """Write one validation summary row."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output_path


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _float_values(rows: list[dict[str, Any]], field_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(field_name)
        if raw_value in (None, ""):
            continue
        values.append(float(raw_value))
    return values


def _success_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("success")).lower() in {"true", "1", "yes"})


def empty_raw_output(path: str | Path) -> Path:
    """Create an empty raw validation JSONL when the live server is unavailable."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    return output_path


def backend_comparison_rows(
    *,
    model_name: str,
    memory_mode: str,
    validation_status: str,
    result_rows: list[dict[str, Any]],
) -> list[BackendComparisonRow]:
    """Build backend comparison framework rows with live vLLM metrics when present."""

    rows = build_backend_comparison_framework(model=model_name, memory_mode=memory_mode)
    if validation_status != "live_validated":
        return rows
    latency = _average(_float_values(result_rows, "latency_ms"))
    tpot = _average(_float_values(result_rows, "tpot_ms"))
    throughput = _average(_float_values(result_rows, "throughput_tokens_per_second"))
    requests = _average(
        [
            1000.0 / latency_ms
            for latency_ms in _float_values(result_rows, "latency_ms")
            if latency_ms > 0
        ]
    )
    return [
        rows[0],
        BackendComparisonRow(
            backend="vllm",
            status="validated",
            model=model_name,
            memory_mode=memory_mode,
            latency_ms=latency,
            ttft_ms=None,
            tpot_ms=tpot,
            throughput_tokens_per_second=throughput,
            requests_per_second=requests,
            notes="Live localhost OpenAI-compatible vLLM validation completed.",
        ),
        rows[2],
    ]


def build_unavailable_report(
    *,
    args: argparse.Namespace,
    message: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build report and summary when vLLM is not reachable."""

    empty_raw_output(args.output_path)
    comparison_rows = backend_comparison_rows(
        model_name=args.model_name,
        memory_mode="mm2_hybrid_top5",
        validation_status="server_unavailable",
        result_rows=[],
    )
    write_backend_comparison_csv(args.backend_comparison_path, comparison_rows)
    write_telemetry_json(args.telemetry_path, [])
    summary = {
        "validation_status": "server_unavailable",
        "server_reachable": False,
        "model_alias": args.model_alias,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "memory_mode": "mm2_hybrid_top5",
        "row_count": 0,
        "success_count": 0,
        "joined_count": 0,
        "joined_rate": 0.0,
        "format_valid_rate": 0.0,
        "grounded_rate": 0.0,
        "safety_violation_rate": 0.0,
        "avg_latency_ms": "",
        "avg_ttft_ms": "",
        "avg_tpot_ms": "",
        "avg_throughput_tokens_per_second": "",
        "avg_requests_per_second": "",
        "missing_gpu_metrics": "gpu_utilization;gpu_memory;gpu_cost;runpod_cost",
        "error_type": "server_unavailable",
    }
    report = {
        "validation_status": "server_unavailable",
        "server_readiness": {
            "reachable": False,
            "models_endpoint_supported": False,
            "model_available": None,
            "model_names": [],
            "message": message,
        },
        "raw_output_path": str(args.output_path),
        "evaluation_report_path": str(args.report_path),
        "summary_path": str(args.summary_path),
        "telemetry_path": str(args.telemetry_path),
        "backend_comparison_path": str(args.backend_comparison_path),
        "telemetry_schema": TELEMETRY_FIELDS,
        "backend_comparison_schema": [
            "HF",
            "vLLM",
            "SGLang (future)",
        ],
        "missing_gpu_metrics": [
            "gpu_utilization",
            "gpu_memory",
            "gpu_cost",
            "runpod_cost",
        ],
        "future_runpod_integration_points": [
            "pod hourly price",
            "pod runtime seconds",
            "GPU utilization and memory telemetry",
            "request and token counts per run",
        ],
        "future_sglang_integration_points": [
            "OpenAI-compatible SGLang server adapter",
            "SGLang-specific scheduler/cache metrics",
            "same telemetry and evaluator contracts",
        ],
        "no_paid_api_call_triggered": True,
        "no_gpu_rental_triggered": True,
        "no_retrieval_modified": True,
        "summary": summary,
        "evaluation_rows": [],
    }
    return report, summary


def build_live_report(
    *,
    args: argparse.Namespace,
    result_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate live vLLM result rows and build report/summary."""

    generated_answers = [result_row_to_generated_answer(row) for row in result_rows]
    evaluation_rows = evaluate_generated_answers(
        generated_answers,
        load_gold_records(args.dataset_root),
    )
    eval_summary = build_summary_rows(
        results_path=args.output_path,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
    )[0]
    memory_mode = str(result_rows[0].get("memory_mode") or "mm2_hybrid_top5") if result_rows else ""
    telemetry_records = [
        telemetry_record_from_result_row(
            row,
            backend="vllm",
            model=args.model_name,
            memory_mode=str(row.get("memory_mode") or memory_mode),
        )
        for row in result_rows
    ]
    write_telemetry_json(args.telemetry_path, telemetry_records)
    write_backend_comparison_csv(
        args.backend_comparison_path,
        backend_comparison_rows(
            model_name=args.model_name,
            memory_mode=memory_mode,
            validation_status="live_validated",
            result_rows=result_rows,
        ),
    )
    summary = {
        "validation_status": "live_validated",
        "server_reachable": True,
        "model_alias": args.model_alias,
        "model_name": args.model_name,
        "base_url": args.base_url,
        "memory_mode": memory_mode,
        "row_count": len(result_rows),
        "success_count": _success_count(result_rows),
        "joined_count": eval_summary["joined_count"],
        "joined_rate": eval_summary["joined_rate"],
        "format_valid_rate": eval_summary["format_valid_rate"],
        "grounded_rate": eval_summary["grounded_rate"],
        "safety_violation_rate": eval_summary["safety_violation_rate"],
        "avg_latency_ms": _average(_float_values(result_rows, "latency_ms")),
        "avg_ttft_ms": _average(_float_values(result_rows, "ttft_ms")),
        "avg_tpot_ms": _average(_float_values(result_rows, "tpot_ms")),
        "avg_throughput_tokens_per_second": _average(
            _float_values(result_rows, "throughput_tokens_per_second")
        ),
        "avg_requests_per_second": _average(
            [
                1000.0 / latency_ms
                for latency_ms in _float_values(result_rows, "latency_ms")
                if latency_ms > 0
            ]
        ),
        "missing_gpu_metrics": "gpu_utilization;gpu_memory;gpu_cost;runpod_cost",
        "error_type": "",
    }
    report = {
        "validation_status": "live_validated",
        "server_readiness": result_rows[0].get("server_readiness") if result_rows else None,
        "raw_output_path": str(args.output_path),
        "evaluation_report_path": str(args.report_path),
        "summary_path": str(args.summary_path),
        "telemetry_path": str(args.telemetry_path),
        "backend_comparison_path": str(args.backend_comparison_path),
        "telemetry_schema": TELEMETRY_FIELDS,
        "backend_comparison_schema": [
            "HF",
            "vLLM",
            "SGLang (future)",
        ],
        "missing_gpu_metrics": [
            "gpu_utilization",
            "gpu_memory",
            "gpu_cost",
            "runpod_cost",
        ],
        "future_runpod_integration_points": [
            "pod hourly price",
            "pod runtime seconds",
            "GPU utilization and memory telemetry",
            "request and token counts per run",
        ],
        "future_sglang_integration_points": [
            "OpenAI-compatible SGLang server adapter",
            "SGLang-specific scheduler/cache metrics",
            "same telemetry and evaluator contracts",
        ],
        "no_paid_api_call_triggered": True,
        "no_gpu_rental_triggered": True,
        "no_retrieval_modified": True,
        "summary": summary,
        "evaluation_rows": evaluation_rows,
    }
    return report, summary


def run_validation(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run live vLLM validation, or report unavailable server status."""

    try:
        check_server_readiness(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return build_unavailable_report(args=args, message=str(exc))

    run_smoke(
        input_path=args.input_path,
        output_path=args.output_path,
        model_alias=args.model_alias,
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
        dry_run=False,
        manifest_path=args.manifest_path,
        command=sanitize_command(sys.argv),
    )
    result_rows = load_result_rows(args.output_path)
    return build_live_report(args=args, result_rows=result_rows)


def main(argv: list[str] | None = None) -> int:
    """Run vLLM validation and write processed reports."""

    args = build_parser().parse_args(argv)
    report, summary = run_validation(args)
    report_path = write_json(args.report_path, report)
    summary_path = write_summary(args.summary_path, summary)
    print(f"vLLM validation status: {summary['validation_status']}")
    print(f"Rows evaluated: {summary['row_count']}")
    print(f"Raw output: {args.output_path}")
    print(f"Validation report: {report_path}")
    print(f"Validation summary: {summary_path}")
    print(f"Telemetry: {args.telemetry_path}")
    print(f"Backend comparison: {args.backend_comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
