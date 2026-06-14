"""Run the bounded LangGraph mm4 smoke and optional matched mm3 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PHASE4_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PHASE4_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE4_ROOT))

from evaluate_generation_outputs import load_result_rows  # noqa: E402
from run_openai_compatible_smoke import (  # noqa: E402
    DEFAULT_API_KEY,
    check_server_readiness,
)
from run_remote_vllm_smoke import (  # noqa: E402
    evaluate_result_rows,
    latency_summary_rows,
    sanitized_command,
    select_balanced_runner_items,
    write_csv_rows,
)

from inference_bench.agent_trace_schema import AgentTraceRecord  # noqa: E402
from inference_bench.agentic_comparison import (  # noqa: E402
    build_memory_mode_row,
    write_mm4_comparison,
)
from inference_bench.agents.langgraph_mm4 import (  # noqa: E402
    ModelGeneration,
    compile_mm4_graph,
    run_mm4_graph,
)
from inference_bench.agents.state import AgentState  # noqa: E402
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.context_corpora import VERTICALS  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    generation_contract_result_fields,
)
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.openai_compatible_runner import (  # noqa: E402
    run_openai_compatible_benchmark,
)
from inference_bench.streaming_metrics import (  # noqa: E402
    request_streaming_chat_completion,
)
from inference_bench.workload_adapter import (  # noqa: E402
    load_phase3_workload_records,
    write_runner_workload_jsonl,
)

DEFAULT_MM2_WORKLOAD = "data/workloads/smoke_500/prompt_plus_metadata/mm2_hybrid_top5.jsonl"
DEFAULT_MM3_WORKLOAD = (
    "data/workloads/smoke_500/prompt_plus_metadata/mm3_compressed_hybrid_top5.jsonl"
)
DEFAULT_OUTPUT = "results/raw/a6_mm4_agentic_smoke_results.jsonl"
DEFAULT_TRACE = "results/raw/a6_mm4_agentic_smoke_traces.jsonl"
DEFAULT_MANIFEST = "results/raw/a6_mm4_agentic_smoke_manifest.json"
DEFAULT_EVAL_REPORT = "results/processed/a6_mm4_agentic_eval_report.json"
DEFAULT_EVAL_SUMMARY = "results/processed/a6_mm4_agentic_eval_summary.csv"
DEFAULT_LATENCY_SUMMARY = "results/processed/a6_mm4_agentic_latency_summary.csv"
DEFAULT_AGENT_SUMMARY = "results/processed/a6_mm4_agentic_trace_summary.csv"
DEFAULT_MM3_RUNNER_INPUT = "data/generated/phase4/a6_mm3_runner_input.jsonl"
DEFAULT_MM3_RESULTS = "results/raw/a6_mm3_baseline_results.jsonl"
DEFAULT_MM3_METRICS = "results/raw/a6_mm3_baseline_metrics.csv"
DEFAULT_MM3_EVAL_REPORT = "results/processed/a6_mm3_baseline_eval_report.json"
DEFAULT_MM3_EVAL_SUMMARY = "results/processed/a6_mm3_baseline_eval_summary.csv"
DEFAULT_MM3_LATENCY = "results/processed/a6_mm3_baseline_latency_summary.csv"
DEFAULT_COMPARISON_REPORT = "results/processed/a6_mm4_vs_mm2_mm3_comparison_report.json"
DEFAULT_COMPARISON_SUMMARY = "results/processed/a6_mm4_vs_mm2_mm3_comparison_summary.csv"
TOTAL_PROMPTS = 50
MAX_NEW_TOKENS = 128


class StreamingGenerator:
    """Streaming OpenAI-compatible adapter used by the graph."""

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        api_route: str,
        max_new_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.api_route = api_route
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

    def __call__(self, prompt: str) -> ModelGeneration:
        metrics = request_streaming_chat_completion(
            api_key=self.api_key,
            model_id=self.model_id,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            api_route=self.api_route,
            timeout_seconds=self.timeout_seconds,
        )
        if not metrics.streaming_available:
            msg = "Streaming was required but no streamed content arrived"
            raise RuntimeError(msg)
        return ModelGeneration(
            text=metrics.generated_text,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            ttft_ms=metrics.ttft_ms,
            tpot_ms=metrics.tpot_ms,
            e2e_latency_ms=metrics.e2e_latency_ms,
            cost_usd=0.0,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the mm4 smoke CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-path", default=DEFAULT_MM2_WORKLOAD)
    parser.add_argument("--mm3-workload-path", default=DEFAULT_MM3_WORKLOAD)
    parser.add_argument("--model-alias", default="model1_0_5b")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-path", default=DEFAULT_TRACE)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--eval-report-path", default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--eval-summary-path", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--latency-summary-path", default=DEFAULT_LATENCY_SUMMARY)
    parser.add_argument("--agent-summary-path", default=DEFAULT_AGENT_SUMMARY)
    parser.add_argument("--mm3-runner-input-path", default=DEFAULT_MM3_RUNNER_INPUT)
    parser.add_argument("--mm3-results-path", default=DEFAULT_MM3_RESULTS)
    parser.add_argument("--mm3-metrics-path", default=DEFAULT_MM3_METRICS)
    parser.add_argument("--mm3-eval-report-path", default=DEFAULT_MM3_EVAL_REPORT)
    parser.add_argument("--mm3-eval-summary-path", default=DEFAULT_MM3_EVAL_SUMMARY)
    parser.add_argument("--mm3-latency-path", default=DEFAULT_MM3_LATENCY)
    parser.add_argument("--comparison-report-path", default=DEFAULT_COMPARISON_REPORT)
    parser.add_argument("--comparison-summary-path", default=DEFAULT_COMPARISON_SUMMARY)
    parser.add_argument("--run-mm3-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _source_question(record: Any) -> str:
    for field_name in ("question", "prompt", "request", "task"):
        value = record.source_prompt_record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    msg = f"No user question found for {record.prompt_id}"
    raise ValueError(msg)


def _selected_records(path: str | Path) -> list[Any]:
    selected_ids = {item.prompt_id for item in select_balanced_runner_items(path)}
    records = [
        record for record in load_phase3_workload_records(path) if record.prompt_id in selected_ids
    ]
    by_id = {record.prompt_id: record for record in records}
    ordered_ids = [item.prompt_id for item in select_balanced_runner_items(path)]
    return [by_id[prompt_id] for prompt_id in ordered_ids]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output


def _run_agentic_records(
    *,
    records: list[Any],
    generator: Any,
    backend: str,
    model_name: str,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = compile_mm4_graph(generator=generator)
    result_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        initial_state = AgentState(
            prompt_id=record.prompt_id,
            workload_id=record.workload_id.replace(
                record.memory_mode,
                "mm4_bounded_agentic",
            ),
            vertical=record.vertical,
            user_question=_source_question(record),
            task_type=str(record.source_prompt_record.get("task_type") or "unknown"),
            context_pool=[
                asdict(context) if not isinstance(context, dict) else dict(context)
                for context in record.context_records
            ],
            backend=backend,
            model_name=model_name,
            source_prompt_record=record.source_prompt_record,
        )
        started = time.perf_counter()
        final_state = run_mm4_graph(graph=graph, initial_state=initial_state)
        e2e_latency_ms = (time.perf_counter() - started) * 1000
        token_usage = final_state.token_usage
        generation_metrics = final_state.generation_metrics
        total_tokens = int(token_usage.get("total_tokens") or 0)
        row: dict[str, Any] = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model_name": model_name,
            "optimization": "mm4_bounded_agentic_langgraph",
            "workload_name": "smoke_500_mm4_bounded_agentic",
            "prompt_id": record.prompt_id,
            "workload_id": final_state.workload_id,
            "vertical": record.vertical,
            "memory_mode": "mm4_bounded_agentic",
            "ablation_mode": "prompt_plus_metadata",
            "expected_output_format": "generation_contract_json",
            "prompt": final_state.assembled_prompt,
            "generated_text": final_state.generated_answer,
            "citation_id_aliases": final_state.citation_id_aliases,
            "input_tokens": int(token_usage.get("input_tokens") or 0),
            "output_tokens": int(token_usage.get("output_tokens") or 0),
            "total_tokens": total_tokens,
            "comparison_input_tokens": int(token_usage.get("comparison_input_tokens") or 0),
            "comparison_output_tokens": int(token_usage.get("comparison_output_tokens") or 0),
            "comparison_token_count_source": "whitespace_normalized",
            "ttft_ms": generation_metrics.get("first_ttft_ms"),
            "tpot_ms": generation_metrics.get("tpot_ms"),
            "end_to_end_latency_ms": e2e_latency_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "throughput_tokens_per_second": (
                total_tokens / (e2e_latency_ms / 1000)
                if total_tokens and e2e_latency_ms > 0
                else None
            ),
            "total_cost_usd": None,
            "estimated_cost_usd": None,
            "success": final_state.final_status in {"answer", "insufficient_evidence", "escalate"},
            "error_message": final_state.escalation_reason or None,
            "final_status": final_state.final_status,
            "retrieval_rounds": final_state.retrieval_rounds,
            "repair_attempts": final_state.repair_attempts,
            "generation_attempts": final_state.generation_attempts,
            "tool_call_count": final_state.tool_call_count,
            "node_latencies": final_state.node_latencies,
            "validation_result": final_state.validation_result,
            "escalation_reason": final_state.escalation_reason,
            "contract_retry_count": final_state.repair_attempts,
        }
        row.update(
            generation_contract_result_fields(
                final_state.generated_answer,
                allowed_evidence_ids=final_state.allowed_evidence_ids,
            )
        )
        result_rows.append(row)
        trace = AgentTraceRecord.from_state(
            state=final_state,
            trace_id=f"{run_id}:{index:03d}:{record.prompt_id}",
            run_id=run_id,
        )
        trace_rows.append(trace.model_dump(mode="json"))
    return result_rows, trace_rows


def _agent_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    node_values: dict[str, list[float]] = {}
    for row in rows:
        for node, value in dict(row.get("node_latencies") or {}).items():
            node_values.setdefault(str(node), []).append(float(value))
    return {
        "row_count": len(rows),
        "mean_retrieval_rounds": sum(int(row.get("retrieval_rounds") or 0) for row in rows)
        / len(rows),
        "repair_rate": sum(int(row.get("repair_attempts") or 0) > 0 for row in rows) / len(rows),
        "escalation_rate": sum(
            str(row.get("final_status")) in {"escalate", "insufficient_evidence"} for row in rows
        )
        / len(rows),
        "mean_tool_call_count": sum(int(row.get("tool_call_count") or 0) for row in rows)
        / len(rows),
        "mean_node_latencies_json": json.dumps(
            {node: sum(values) / len(values) for node, values in sorted(node_values.items())},
            sort_keys=True,
        ),
    }


def _read_csv_first(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file), None)
    if row is None:
        msg = f"No CSV rows in {path}"
        raise ValueError(msg)
    return dict(row)


def _write_comparison(
    *,
    args: argparse.Namespace,
    mm4_rows: list[dict[str, Any]],
    mm4_eval: dict[str, Any],
    mm4_latency: dict[str, Any],
) -> None:
    mm2_rows = load_result_rows("results/raw/a1_remote_rtx3070_vllm_smoke_results.jsonl")
    mm2_eval = _read_csv_first("results/processed/a1_remote_rtx3070_vllm_eval_summary.csv")
    mm2_latency = _read_csv_first("results/processed/a1_remote_rtx3070_vllm_latency_summary.csv")
    if Path(args.mm3_results_path).exists():
        mm3_rows = load_result_rows(args.mm3_results_path)
        mm3_eval = _read_csv_first(args.mm3_eval_summary_path)
        mm3_latency = _read_csv_first(args.mm3_latency_path)
        mm3_status = "measured"
    else:
        mm3_rows = []
        mm3_eval = None
        mm3_latency = None
        mm3_status = "missing_not_estimated"
    comparison_rows = [
        build_memory_mode_row(
            memory_mode="mm2_hybrid_top5",
            result_rows=mm2_rows,
            evaluation_summary=mm2_eval,
            latency_summary=mm2_latency,
        ),
        build_memory_mode_row(
            memory_mode="mm3_compressed_hybrid_top5",
            result_rows=mm3_rows,
            evaluation_summary=mm3_eval,
            latency_summary=mm3_latency,
            measurement_status=mm3_status,
        ),
        build_memory_mode_row(
            memory_mode="mm4_bounded_agentic",
            result_rows=mm4_rows,
            evaluation_summary=mm4_eval,
            latency_summary=mm4_latency,
        ),
    ]
    write_mm4_comparison(
        report_path=args.comparison_report_path,
        summary_path=args.comparison_summary_path,
        rows=comparison_rows,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run dry validation or the bounded live smoke."""

    if args.max_new_tokens <= 0 or args.max_new_tokens > MAX_NEW_TOKENS:
        msg = f"max-new-tokens must be between 1 and {MAX_NEW_TOKENS}"
        raise ValueError(msg)
    model = load_project_config().resolve_model_config(args.model_alias)
    records = _selected_records(args.workload_path)
    if len(records) != TOTAL_PROMPTS:
        msg = f"mm4 smoke requires exactly {TOTAL_PROMPTS} records"
        raise RuntimeError(msg)
    if args.dry_run:
        return {
            "status": "dry_run",
            "record_count": len(records),
            "model_alias": args.model_alias,
            "model_id": model.model_id,
            "vertical_counts": {
                vertical: sum(record.vertical == vertical for record in records)
                for vertical in VERTICALS
            },
        }

    readiness = check_server_readiness(
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=model.model_id,
        timeout_seconds=args.timeout_seconds,
    )
    generator = StreamingGenerator(
        api_key=args.api_key,
        model_id=model.model_id,
        api_route=f"{args.base_url.rstrip('/')}/chat/completions",
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    run_id = "a6-mm4-bounded-agentic-smoke"
    started_at = utc_now()
    rows, traces = _run_agentic_records(
        records=records,
        generator=generator,
        backend="vllm_langgraph",
        model_name=model.model_id,
        run_id=run_id,
    )
    ended_at = utc_now()
    _write_jsonl(args.output_path, rows)
    _write_jsonl(args.trace_path, traces)
    _, eval_summary = evaluate_result_rows(
        result_rows=rows,
        output_path=args.output_path,
        eval_report_path=args.eval_report_path,
        eval_summary_path=args.eval_summary_path,
        block="A6",
        experiment="mm4_bounded_agentic_langgraph_smoke",
    )
    latency_rows = latency_summary_rows(rows)
    write_csv_rows(args.latency_summary_path, latency_rows)
    write_csv_rows(args.agent_summary_path, [_agent_summary(rows)])

    if args.run_mm3_baseline:
        mm3_items = select_balanced_runner_items(args.mm3_workload_path)
        write_runner_workload_jsonl(mm3_items, args.mm3_runner_input_path)
        run_openai_compatible_benchmark(
            workload_path=args.mm3_runner_input_path,
            output_path=args.mm3_metrics_path,
            generation_output_path=args.mm3_results_path,
            model=model.model_id,
            base_url=args.base_url,
            api_key=args.api_key,
            run_id="a6-mm3-compressed-baseline",
            backend="vllm",
            optimization="a6_mm3_compressed_baseline",
            max_new_tokens=args.max_new_tokens,
            max_prompts=TOTAL_PROMPTS,
            stream=True,
            timeout_seconds=args.timeout_seconds,
        )
        mm3_rows = load_result_rows(args.mm3_results_path)
        evaluate_result_rows(
            result_rows=mm3_rows,
            output_path=args.mm3_results_path,
            eval_report_path=args.mm3_eval_report_path,
            eval_summary_path=args.mm3_eval_summary_path,
            block="A6",
            experiment="mm3_compressed_hybrid_top5_baseline",
        )
        write_csv_rows(args.mm3_latency_path, latency_summary_rows(mm3_rows))

    _write_comparison(
        args=args,
        mm4_rows=rows,
        mm4_eval=eval_summary,
        mm4_latency=latency_rows[0],
    )
    manifest = RunManifest(
        run_id=run_id,
        timestamp_utc=ended_at,
        backend="vllm_langgraph",
        model_alias=args.model_alias,
        model_id=model.model_id,
        memory_mode="mm4_bounded_agentic",
        split="smoke_500",
        ablation_mode="prompt_plus_metadata",
        input_workload_path=str(args.workload_path),
        output_path=str(args.output_path),
        max_records=TOTAL_PROMPTS,
        git_commit=current_git_commit(REPO_ROOT),
        command=sanitized_command(sys.argv),
        status="completed",
        start_time=started_at,
        end_time=ended_at,
        error_count=sum(not bool(row.get("success")) for row in rows),
    )
    write_run_manifest(manifest, args.manifest_path)
    return {
        "status": "completed",
        "server_readiness": readiness.to_dict(),
        "row_count": len(rows),
        "evaluation_summary": eval_summary,
        "latency_summary": latency_rows[0],
        "agent_summary": _agent_summary(rows),
        "comparison_report": args.comparison_report_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the mm4 CLI."""

    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(args)
    except Exception as exc:  # noqa: BLE001
        print(f"mm4 smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
