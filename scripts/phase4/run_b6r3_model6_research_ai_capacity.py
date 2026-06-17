"""Run B6R3 model6 Research AI capacity validation on frozen replay rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PHASE4 = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from evaluate_generation_outputs import (  # noqa: E402
    build_summary_rows,
    load_gold_records,
)

from inference_bench.api_pricing import (  # noqa: E402
    estimate_api_cost_from_pricing,
    resolve_api_pricing,
)
from inference_bench.api_routes import api_key_for_route, resolve_api_provider_route  # noqa: E402
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.env import load_local_env  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    allowed_evidence_ids_from_aliases,
    generation_contract_result_fields,
)
from inference_bench.generation_contract_registry import (  # noqa: E402
    validate_and_map_contract_text,
)
from inference_bench.grounding_repair import evaluate_result_row  # noqa: E402
from inference_bench.research_ai_capacity_validation import (  # noqa: E402
    NormalizedResearchAiReplayItem,
    build_b6r3_manifest_payload,
    choose_b6r3_contract_id,
    completed_prompt_ids_from_jsonl,
    load_research_ai_capacity_replay,
    pending_replay_items,
    validate_b6r3_cli_limits,
)
from inference_bench.research_ai_contract_renderer import (  # noqa: E402
    RenderedResearchAiContract,
    render_research_ai_contract_item,
)
from inference_bench.research_ai_contract_selection import (  # noqa: E402
    B6R2_TARGET_THRESHOLDS,
)
from inference_bench.streaming_metrics import (  # noqa: E402
    StreamingMetrics,
    request_streaming_chat_completion,
)

DEFAULT_INPUT = "data/generated/phase4/b6r1_research_ai_failed_replay_input.jsonl"
DEFAULT_OUTPUT = "results/raw/b6r3_model6_research_ai_capacity_results.jsonl"
DEFAULT_REPORT = "results/processed/b6r3_model6_research_ai_capacity_report.json"
DEFAULT_SUMMARY = "results/processed/b6r3_model6_research_ai_capacity_summary.csv"
DEFAULT_COMPARISON = "results/processed/b6r3_model6_vs_b6r2_comparison.json"
DEFAULT_MANIFEST = "results/raw/b6r3_model6_research_ai_capacity_manifest.json"
RUN_ID = "b6r3-model6-research-ai-capacity"
EXPECTED_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    """Build the B6R3 CLI parser."""

    parser = argparse.ArgumentParser(description="Run B6R3 model6 Research AI replay.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-path", default=DEFAULT_COMPARISON)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST)
    parser.add_argument("--model-alias", default="model6_gated")
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true")
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    parser.set_defaults(stream=True)
    require_group = parser.add_mutually_exclusive_group()
    require_group.add_argument("--require-streaming", dest="require_streaming", action="store_true")
    require_group.add_argument(
        "--no-require-streaming",
        dest="require_streaming",
        action="store_false",
    )
    parser.set_defaults(require_streaming=True)
    parser.add_argument("--allow-paid-api-call", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("at least one CSV row is required")
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output = ROOT / Path(path)
    if not output.exists():
        return []
    rows: list[dict[str, Any]] = []
    with output.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = ROOT / Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _sanitized_command(argv: list[str]) -> str:
    sanitized = list(argv)
    for index, argument in enumerate(sanitized[:-1]):
        if argument in {"--api-key", "--hf-token"}:
            sanitized[index + 1] = "***"
    return " ".join([Path(sys.executable).name, *sanitized])


def _selected_model(model_alias: str) -> tuple[Any, Any, Any]:
    config = load_project_config()
    model = config.resolve_model_config(model_alias)
    if model.model_id != EXPECTED_MODEL_ID:
        raise RuntimeError(f"{model_alias} resolved to unexpected model {model.model_id}")
    pricing = resolve_api_pricing(model_alias)
    route = resolve_api_provider_route(model=model, pricing=pricing)
    return model, pricing, route


def _throughput(total_tokens: int, latency_ms: float) -> float | None:
    return total_tokens / (latency_ms / 1000.0) if latency_ms > 0 else None


def _non_streaming_chat_completion(
    *,
    api_key: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    api_route: str,
    timeout_seconds: float,
) -> StreamingMetrics:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": False,
    }
    request = Request(
        api_route,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    e2e_ms = (time.perf_counter() - started) * 1000.0
    choices = payload.get("choices") if isinstance(payload, dict) else None
    generated_text = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            generated_text = str(message.get("content") or "")
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if isinstance(usage, dict):
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        token_source = "provider_usage"
    else:
        input_tokens = len(prompt.split())
        output_tokens = len(generated_text.split())
        token_source = "whitespace_fallback"
    tpot = e2e_ms / max(output_tokens, 1) if output_tokens else None
    return StreamingMetrics(
        generated_text=generated_text,
        ttft_ms=None,
        itl_p50_ms=None,
        itl_p95_ms=None,
        itl_p99_ms=None,
        tpot_ms=tpot,
        e2e_latency_ms=e2e_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        token_count_source=token_source,
        content_chunk_count=0,
        streaming_available=False,
    )


def _request_with_backoff(
    *,
    api_key: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    api_route: str,
    timeout_seconds: float,
    stream: bool,
    max_attempts: int = 3,
) -> StreamingMetrics:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            if stream:
                return request_streaming_chat_completion(
                    api_key=api_key,
                    model_id=model_id,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    api_route=api_route,
                    timeout_seconds=timeout_seconds,
                )
            return _non_streaming_chat_completion(
                api_key=api_key,
                model_id=model_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                api_route=api_route,
                timeout_seconds=timeout_seconds,
            )
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(min(2**attempt, 8))
    if last_error is None:
        raise RuntimeError("API request failed without an exception")
    raise last_error


def _metric_fields(metrics: StreamingMetrics) -> dict[str, Any]:
    return {
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
        "token_count_source": metrics.token_count_source,
        "content_chunk_count": metrics.content_chunk_count,
        "streaming_available": metrics.streaming_available,
        "ttft_ms": metrics.ttft_ms,
        "itl_p50_ms": metrics.itl_p50_ms,
        "itl_p95_ms": metrics.itl_p95_ms,
        "itl_p99_ms": metrics.itl_p99_ms,
        "tpot_ms": metrics.tpot_ms,
        "end_to_end_latency_ms": metrics.e2e_latency_ms,
        "latency_ms": metrics.e2e_latency_ms,
        "throughput_tokens_per_second": _throughput(
            metrics.total_tokens,
            metrics.e2e_latency_ms,
        ),
    }


def _result_row(
    *,
    item: NormalizedResearchAiReplayItem,
    rendered: RenderedResearchAiContract,
    metrics: StreamingMetrics,
    model_alias: str,
    model_id: str,
    provider_model_id: str,
    provider: str,
    backend: str,
    api_route: str,
    pricing: Any,
    stream: bool,
) -> dict[str, Any]:
    aliases = rendered.item.metadata.get("citation_id_aliases")
    validation = validate_and_map_contract_text(
        text=metrics.generated_text,
        contract_id=rendered.requested_contract_id,
        allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        prompt_text=rendered.item.prompt,
        metadata=rendered.item.metadata,
    )
    generated_text = validation.common_text or metrics.generated_text
    cost = estimate_api_cost_from_pricing(
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        pricing=pricing,
    )
    row = {
        "run_id": RUN_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_id": "b6r3_model6_research_ai_capacity",
        "backend": backend,
        "backend_type": "api_provider",
        "engine": backend,
        "hardware": "api_provider",
        "provider": provider,
        "model_alias": model_alias,
        "model_id": model_id,
        "provider_model_id": provider_model_id,
        "api_route": api_route,
        "optimization": "b6r3_model6_capacity_validation",
        "workload_name": item.workload_name,
        "prompt_id": item.prompt_id,
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.vertical,
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": aliases,
        "gold_evidence_ids": item.metadata.get("gold_evidence_ids"),
        "prompt": rendered.item.prompt,
        "generated_text": generated_text,
        "raw_generated_text": metrics.generated_text,
        "stream_requested": stream,
        "success": True,
        "error_type": None,
        "error_message": None,
        "final_status": "answer",
        "b6r3_requested_research_ai_contract": rendered.requested_contract_id,
        "b6r3_effective_research_ai_contract": rendered.effective_contract_id,
        "b6r3_contract_validation": validation.to_dict(),
        "source_metadata": item.source_metadata,
        "pricing_status": pricing.pricing_status,
        "pricing_source": pricing.pricing_source,
        "pricing_source_url": pricing.pricing_source_url,
        "pricing_last_checked": pricing.pricing_last_checked,
        "input_usd_per_1m_tokens": pricing.input_usd_per_1m_tokens,
        "output_usd_per_1m_tokens": pricing.output_usd_per_1m_tokens,
        "input_cost_usd": cost["input_cost_usd"],
        "output_cost_usd": cost["output_cost_usd"],
        "total_cost_usd": cost["total_api_cost_usd"],
        "paid_api_call_triggered": True,
        "vllm_triggered": False,
        "sglang_triggered": False,
        **_metric_fields(metrics),
    }
    row.update(
        generation_contract_result_fields(
            generated_text,
            allowed_evidence_ids=allowed_evidence_ids_from_aliases(aliases),
        )
    )
    return row


def _failure_row(
    *,
    item: NormalizedResearchAiReplayItem,
    rendered: RenderedResearchAiContract | None,
    exc: Exception,
    elapsed_ms: float,
    model_alias: str,
    model_id: str,
    provider_model_id: str,
    provider: str,
    backend: str,
    api_route: str,
) -> dict[str, Any]:
    aliases = item.metadata.get("citation_id_aliases")
    row = {
        "run_id": RUN_ID,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_id": "b6r3_model6_research_ai_capacity",
        "backend": backend,
        "backend_type": "api_provider",
        "engine": backend,
        "hardware": "api_provider",
        "provider": provider,
        "model_alias": model_alias,
        "model_id": model_id,
        "provider_model_id": provider_model_id,
        "api_route": api_route,
        "optimization": "b6r3_model6_capacity_validation",
        "workload_name": item.workload_name,
        "prompt_id": item.prompt_id,
        "workload_id": item.metadata.get("workload_id"),
        "vertical": item.vertical,
        "memory_mode": item.metadata.get("memory_mode"),
        "ablation_mode": item.metadata.get("ablation_mode"),
        "expected_output_format": item.metadata.get("expected_output_format"),
        "citation_id_aliases": aliases,
        "gold_evidence_ids": item.metadata.get("gold_evidence_ids"),
        "prompt": rendered.item.prompt if rendered else item.prompt,
        "generated_text": "",
        "raw_generated_text": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "total_cost_usd": 0.0,
        "ttft_ms": None,
        "itl_p50_ms": None,
        "itl_p95_ms": None,
        "itl_p99_ms": None,
        "tpot_ms": None,
        "end_to_end_latency_ms": elapsed_ms,
        "latency_ms": elapsed_ms,
        "throughput_tokens_per_second": None,
        "streaming_available": False,
        "content_chunk_count": 0,
        "token_count_source": "unavailable",
        "stream_requested": True,
        "success": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "final_status": "failed_validation",
        "b6r3_requested_research_ai_contract": (
            rendered.requested_contract_id if rendered else None
        ),
        "b6r3_effective_research_ai_contract": (
            rendered.effective_contract_id if rendered else None
        ),
        "source_metadata": item.source_metadata,
        "paid_api_call_triggered": True,
        "vllm_triggered": False,
        "sglang_triggered": False,
    }
    row.update(generation_contract_result_fields(""))
    return row


def classify_b6r3_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Classify B6R3 model6 capacity gate."""

    checks = {
        "json_valid_rate": {
            "observed": float(summary.get("json_valid_rate") or 0.0),
            "threshold": B6R2_TARGET_THRESHOLDS["json_valid_rate"],
            "operator": ">=",
        },
        "generation_contract_valid_rate": {
            "observed": float(summary.get("generation_contract_valid_rate") or 0.0),
            "threshold": B6R2_TARGET_THRESHOLDS["generation_contract_valid_rate"],
            "operator": ">=",
        },
        "evidence_match_rate": {
            "observed": float(summary.get("evidence_match_rate") or 0.0),
            "threshold": B6R2_TARGET_THRESHOLDS["evidence_match_rate"],
            "operator": ">=",
        },
        "grounded_rate": {
            "observed": float(summary.get("grounded_rate") or 0.0),
            "threshold": B6R2_TARGET_THRESHOLDS["grounded_rate"],
            "operator": ">=",
        },
        "safety_violation_count": {
            "observed": float(summary.get("safety_violation_count") or 0.0),
            "threshold": 0.0,
            "operator": "==",
        },
        "truncation_rate": {
            "observed": float(summary.get("truncation_rate") or 0.0),
            "threshold": B6R2_TARGET_THRESHOLDS["truncation_rate"],
            "operator": "<=",
        },
    }
    failed: list[str] = []
    for metric, check in checks.items():
        operator = str(check["operator"])
        observed = float(check["observed"])
        threshold = float(check["threshold"])
        passed = (
            observed == threshold
            if operator == "=="
            else observed <= threshold
            if operator == "<="
            else observed >= threshold
        )
        check["passed"] = passed
        if not passed:
            failed.append(metric)
    return {
        "status": "B6R3_MODEL6_CAPACITY_PASSED" if not failed else "B6R3_MODEL6_CAPACITY_BLOCKED",
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "") and bool(row.get("success"))
    ]
    return sum(values) / len(values) if values else None


def _load_optional_json(path: str) -> dict[str, Any] | None:
    candidate = ROOT / path
    if not candidate.exists():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _comparison(
    *,
    b6r3_summary: dict[str, Any],
    b6r3_gate: dict[str, Any],
) -> dict[str, Any]:
    b6 = _load_optional_json("results/processed/b6_vllm_1_5b_500_eval_report.json")
    b6r1 = _load_optional_json("results/processed/b6r1_research_ai_strategy_comparison.json")
    b6r2 = _load_optional_json("results/processed/b6r2_research_ai_contract_selection_report.json")
    b6_research = None
    if b6 and isinstance(b6.get("per_vertical_quality"), list):
        b6_research = next(
            (row for row in b6["per_vertical_quality"] if row.get("vertical") == "research_ai"),
            None,
        )
    b6r1_best = None
    if b6r1 and isinstance(b6r1.get("strategy_summaries"), list):
        b6r1_best = max(
            b6r1["strategy_summaries"],
            key=lambda row: float(row.get("grounded_rate") or 0.0),
        )
    b6r2_best = None
    if b6r2 and isinstance(b6r2.get("candidate_summaries"), list):
        b6r2_best = max(
            b6r2["candidate_summaries"],
            key=lambda row: float(row.get("grounded_rate") or 0.0),
        )
    if bool(b6r3_gate.get("passed")):
        blocker = "qwen_1_5b_model_capacity_likely"
    elif b6r3_summary.get("grounded_rate") and float(b6r3_summary["grounded_rate"]) >= 0.80:
        blocker = "model_capacity_improved_but_contract_or_citation_selection_still_blocking"
    else:
        blocker = "unknown_or_not_confirmed"
    return {
        "b6_research_ai_full_vertical": b6_research,
        "b6r1_best_targeted_strategy": b6r1_best,
        "b6r2_best_targeted_contract": b6r2_best,
        "b6r3_model6_targeted_replay": b6r3_summary,
        "b6r3_gate": b6r3_gate,
        "likely_remaining_blocker": blocker,
        "do_not_overclaim": True,
    }


def _evaluate_rows(
    *,
    result_rows: list[dict[str, Any]],
    output_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_by_prompt = {
        str(row.get("prompt_id") or ""): row for row in load_gold_records("data/scaleup_2000_full")
    }
    evaluation_rows = [
        evaluate_result_row(row, gold_by_prompt.get(str(row["prompt_id"]))) for row in result_rows
    ]
    summary = build_summary_rows(
        results_path=ROOT / output_path,
        result_rows=result_rows,
        evaluation_rows=evaluation_rows,
    )[0]
    summary.update(
        {
            "mean_ttft_ms": _mean(result_rows, "ttft_ms"),
            "mean_tpot_ms": _mean(result_rows, "tpot_ms"),
            "mean_itl_p50_ms": _mean(result_rows, "itl_p50_ms"),
            "mean_itl_p95_ms": _mean(result_rows, "itl_p95_ms"),
            "mean_itl_p99_ms": _mean(result_rows, "itl_p99_ms"),
            "mean_e2e_latency_ms": _mean(result_rows, "end_to_end_latency_ms"),
            "input_tokens": sum(int(row.get("input_tokens") or 0) for row in result_rows),
            "output_tokens": sum(int(row.get("output_tokens") or 0) for row in result_rows),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in result_rows),
            "total_cost_usd": sum(float(row.get("total_cost_usd") or 0.0) for row in result_rows),
            "cost_per_request_usd": (
                sum(float(row.get("total_cost_usd") or 0.0) for row in result_rows)
                / len(result_rows)
                if result_rows
                else None
            ),
        }
    )
    return evaluation_rows, summary


def run_b6r3(args: argparse.Namespace) -> dict[str, Any]:
    """Run B6R3 dry-run or live paid API replay."""

    validate_b6r3_cli_limits(limit=args.limit, max_new_tokens=args.max_new_tokens)
    if args.require_streaming and not args.stream:
        raise RuntimeError("--require-streaming cannot be combined with --no-stream")
    items = load_research_ai_capacity_replay(ROOT / args.input_path, limit=args.limit)
    model, pricing, route = _selected_model(args.model_alias)
    if args.dry_run:
        payload = {
            "status": "dry_run",
            "model_alias": args.model_alias,
            "model_id": model.model_id,
            "provider": pricing.provider,
            "backend": route.backend,
            "input_path": args.input_path,
            "normalized_replay_row_count": len(items),
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "stream": args.stream,
            "require_streaming": args.require_streaming,
            "paid_api_call_triggered": False,
        }
        _write_json(args.report_path, payload)
        _write_csv(args.summary_path, [payload])
        return payload
    if not args.allow_paid_api_call:
        raise RuntimeError("Refusing paid API execution: pass --allow-paid-api-call")
    load_local_env()
    api_key = api_key_for_route(route, os.environ)
    completed_ids = completed_prompt_ids_from_jsonl(ROOT / args.output_path)
    pending = pending_replay_items(items, completed_prompt_ids=completed_ids)
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for item in pending:
        started = time.perf_counter()
        rendered: RenderedResearchAiContract | None = None
        try:
            contract_id = choose_b6r3_contract_id(item)
            rendered = render_research_ai_contract_item(
                item.to_workload_item(),
                requested_contract_id=contract_id,
                max_new_tokens=args.max_new_tokens,
            )
            metrics = _request_with_backoff(
                api_key=api_key,
                model_id=route.provider_model_id,
                prompt=rendered.item.prompt,
                max_new_tokens=args.max_new_tokens,
                api_route=route.chat_completions_url,
                timeout_seconds=args.timeout_seconds,
                stream=args.stream,
            )
            if args.require_streaming and not metrics.streaming_available:
                raise RuntimeError("Streaming unavailable: no content chunks were received")
            row = _result_row(
                item=item,
                rendered=rendered,
                metrics=metrics,
                model_alias=args.model_alias,
                model_id=model.model_id,
                provider_model_id=route.provider_model_id,
                provider=pricing.provider,
                backend=route.backend,
                api_route=route.chat_completions_url,
                pricing=pricing,
                stream=args.stream,
            )
        except Exception as exc:  # noqa: BLE001
            row = _failure_row(
                item=item,
                rendered=rendered,
                exc=exc,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                model_alias=args.model_alias,
                model_id=model.model_id,
                provider_model_id=route.provider_model_id,
                provider=pricing.provider,
                backend=route.backend,
                api_route=route.chat_completions_url,
            )
        _append_jsonl(args.output_path, row)
    result_rows = _read_jsonl(args.output_path)
    current_prompt_ids = {item.prompt_id for item in items}
    result_rows = [row for row in result_rows if str(row.get("prompt_id")) in current_prompt_ids]
    evaluation_rows, summary = _evaluate_rows(result_rows=result_rows, output_path=args.output_path)
    gate = classify_b6r3_gate(summary)
    comparison = _comparison(b6r3_summary=summary, b6r3_gate=gate)
    end_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = "completed" if len(result_rows) >= len(items) else "failed"
    manifest = build_b6r3_manifest_payload(
        run_id=RUN_ID,
        model_alias=args.model_alias,
        model_id=model.model_id,
        provider=pricing.provider,
        backend=route.backend,
        input_path=args.input_path,
        output_path=args.output_path,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        start_time=start_time,
        end_time=end_time,
        expected_count=len(items),
        completed_count=len(result_rows),
        error_count=sum(not bool(row.get("success")) for row in result_rows),
        total_cost_usd=float(summary.get("total_cost_usd") or 0.0),
        status=status,
        command=_sanitized_command(sys.argv),
    )
    _write_json(args.manifest_path, manifest)
    _write_json(
        args.report_path,
        {
            "block": "B6R3",
            "status": gate["status"],
            "quality_gate": gate,
            "model_alias": args.model_alias,
            "model_id": model.model_id,
            "provider": pricing.provider,
            "backend": route.backend,
            "normalized_replay_row_count": len(items),
            "row_count": len(result_rows),
            "summary": summary,
            "evaluation_rows": evaluation_rows,
            "comparison_path": args.comparison_path,
            "manifest_path": args.manifest_path,
            "evaluator_modified": False,
            "gold_data_modified": False,
            "promoted_retrieval_modified": False,
            "full_500_rerun_triggered": False,
            "vllm_triggered": False,
            "sglang_triggered": False,
            "runpod_triggered": False,
        },
    )
    _write_csv(args.summary_path, [summary])
    _write_json(args.comparison_path, comparison)
    return {
        "status": gate["status"],
        "normalized_replay_row_count": len(items),
        "row_count": len(result_rows),
        "summary": summary,
        "quality_gate": gate,
        "comparison_path": args.comparison_path,
        "report_path": args.report_path,
        "manifest_path": args.manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    try:
        result = run_b6r3(args)
    except Exception as exc:  # noqa: BLE001
        print(
            f"B6R3 model6 capacity validation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
