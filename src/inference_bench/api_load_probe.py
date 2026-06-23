"""API provider load probe planning, guarded live execution, and summaries."""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from inference_bench.api_pricing import estimate_api_cost_from_pricing, resolve_api_pricing
from inference_bench.api_routes import api_key_for_route, resolve_api_provider_route
from inference_bench.config import load_project_config
from inference_bench.runners.mock_runner import count_whitespace_tokens

SUPPORTED_API_PROBE_MODELS = ("model5_gated", "model6_gated", "model7_gated")
SUPPORTED_API_PROBE_CONCURRENCIES = (1, 2, 4, 8, 16)
API_PROBE_VERDICTS = (
    "API_PROBE_PASSED",
    "API_PROBE_WARNING",
    "API_PROBE_BLOCKED",
    "API_PROBE_SKIPPED_MISSING_KEYS",
)

ApiProbeVerdict = Literal[
    "API_PROBE_PASSED",
    "API_PROBE_WARNING",
    "API_PROBE_BLOCKED",
    "API_PROBE_SKIPPED_MISSING_KEYS",
]
DEFAULT_PROBE_PROMPTS = (
    "Answer with one concise sentence: what is inference latency?",
    "Return a short JSON object with field answer explaining token throughput.",
    "In one sentence, define time to first token.",
    "List two reasons API providers can throttle requests.",
    "Explain why streaming helps latency measurement in one sentence.",
    "Return a concise answer about retry handling for API calls.",
    "In one sentence, define tokens per second.",
    "Explain provider load probing in one concise sentence.",
    "Return a brief answer about timeout handling.",
    "In one sentence, explain why cost should be measured per successful answer.",
)


def _float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    return float(str(value))


def _int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(str(value))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def load_probe_environment(
    *,
    env_path: str | Path = ".env",
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Load environment variables plus simple KEY=VALUE entries from .env."""

    loaded = dict(base_environment or os.environ)
    path = Path(env_path)
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        cleaned = value.strip().strip('"').strip("'")
        if key and cleaned and key not in loaded:
            loaded[key] = cleaned
    return loaded


@dataclass(frozen=True)
class ApiLoadProbePlanRow:
    """One planned provider load-probe slice."""

    model_alias: str
    concurrency: int
    stream: bool = True
    live_probe_enabled: bool = False

    def __post_init__(self) -> None:
        if self.model_alias not in SUPPORTED_API_PROBE_MODELS:
            msg = f"Unsupported API probe model alias '{self.model_alias}'"
            raise ValueError(msg)
        if self.concurrency not in SUPPORTED_API_PROBE_CONCURRENCIES:
            msg = f"Unsupported API probe concurrency '{self.concurrency}'"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable plan row."""

        return asdict(self)


def build_api_load_probe_plan(
    *,
    model_aliases: Iterable[str] = SUPPORTED_API_PROBE_MODELS,
    concurrencies: Iterable[int] = SUPPORTED_API_PROBE_CONCURRENCIES,
    live_probe_enabled: bool = False,
) -> dict[str, object]:
    """Build the API-provider load-probe matrix without running requests."""

    rows = [
        ApiLoadProbePlanRow(
            model_alias=model_alias,
            concurrency=concurrency,
            live_probe_enabled=live_probe_enabled,
        ).to_dict()
        for model_alias in model_aliases
        for concurrency in concurrencies
    ]
    return {
        "probe_type": "api_provider_load_probe",
        "live_probe_executed": False,
        "live_probe_enabled": live_probe_enabled,
        "supported_model_aliases": list(SUPPORTED_API_PROBE_MODELS),
        "concurrency_levels": list(SUPPORTED_API_PROBE_CONCURRENCIES),
        "metrics": [
            "ttft_ms",
            "tpot_ms",
            "latency_ms",
            "streaming_stability",
            "requests_per_second",
            "tokens_per_second",
            "http_429_count",
            "http_5xx_count",
            "timeout_count",
            "retry_count",
            "provider_throttling_count",
            "total_api_cost_usd",
            "recommended_safe_concurrency",
        ],
        "planned_rows": rows,
    }


def classify_api_probe(summary: dict[str, object]) -> ApiProbeVerdict:
    """Classify observed API probe metrics into a deterministic verdict."""

    if not _bool(summary.get("live_probe_executed")):
        return "API_PROBE_BLOCKED"
    total = _int(summary.get("request_count"))
    if total <= 0:
        return "API_PROBE_BLOCKED"
    success_count = _int(summary.get("success_count"))
    if success_count <= 0:
        return "API_PROBE_BLOCKED"
    timeout_count = _int(summary.get("timeout_count"))
    http_5xx_count = _int(summary.get("http_5xx_count"))
    http_429_count = _int(summary.get("http_429_count"))
    throttling_count = _int(summary.get("provider_throttling_count"))
    retry_count = _int(summary.get("retry_count"))
    streaming_stability = _float(summary.get("streaming_stability_rate"))
    hard_error_rate = (timeout_count + http_5xx_count) / total
    throttle_rate = (http_429_count + throttling_count) / total
    if hard_error_rate > 0.02 or timeout_count > 0:
        return "API_PROBE_BLOCKED"
    if throttle_rate > 0.05 or retry_count > total * 0.1 or streaming_stability < 0.98:
        return "API_PROBE_WARNING"
    return "API_PROBE_PASSED"


def _extract_stream_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


def _status_code_from_exception(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_code = getattr(response, "status_code", None)
    return response_code if isinstance(response_code, int) else None


def _provider_error_counts(exc: Exception) -> dict[str, int]:
    status_code = _status_code_from_exception(exc)
    name = exc.__class__.__name__.lower()
    return {
        "http_429_count": 1 if status_code == 429 else 0,
        "http_5xx_count": 1 if status_code is not None and 500 <= status_code <= 599 else 0,
        "timeout_count": 1 if "timeout" in name else 0,
        "retry_count": 0,
        "provider_throttling_count": 1 if status_code == 429 else 0,
    }


def _probe_one_request(
    *,
    model_alias: str,
    model_id: str,
    provider: str,
    provider_model_id: str,
    base_url: str,
    api_key: str,
    prompt: str,
    prompt_index: int,
    concurrency: int,
    max_new_tokens: int,
    temperature: float,
    stream: bool,
    pricing_path: str | Path,
) -> dict[str, object]:
    openai = cast(Any, import_module("openai"))
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    input_tokens = count_whitespace_tokens(prompt)
    start = time.perf_counter()
    first_token: float | None = None
    try:
        response = client.chat.completions.create(
            model=provider_model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=stream,
        )
        chunks: list[str] = []
        if stream:
            for chunk in response:
                delta = _extract_stream_delta(chunk)
                if delta:
                    if first_token is None:
                        first_token = time.perf_counter()
                    chunks.append(delta)
        else:
            choices = getattr(response, "choices", None)
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    chunks.append(content)
            first_token = time.perf_counter()
        end = time.perf_counter()
        generated_text = "".join(chunks)
        output_tokens = count_whitespace_tokens(generated_text)
        elapsed_s = max(end - start, 1e-9)
        ttft_ms = (first_token - start) * 1000.0 if first_token is not None else None
        tpot_ms = (
            ((end - first_token) * 1000.0 / max(output_tokens, 1))
            if first_token is not None
            else None
        )
        pricing = resolve_api_pricing(model_alias, pricing_path)
        cost = estimate_api_cost_from_pricing(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        )
        return {
            "model_alias": model_alias,
            "model_id": model_id,
            "provider": provider,
            "concurrency": concurrency,
            "prompt_index": prompt_index,
            "success": True,
            "streaming_stable": bool(generated_text),
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "latency_ms": elapsed_s * 1000.0,
            "requests_per_second": 1.0 / elapsed_s,
            "tokens_per_second": (input_tokens + output_tokens) / elapsed_s,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost_usd": cost["total_api_cost_usd"],
            "http_429_count": 0,
            "http_5xx_count": 0,
            "timeout_count": 0,
            "retry_count": 0,
            "provider_throttling_count": 0,
            "error_type": "",
        }
    except Exception as exc:  # noqa: BLE001
        end = time.perf_counter()
        elapsed_s = max(end - start, 1e-9)
        counts = _provider_error_counts(exc)
        return {
            "model_alias": model_alias,
            "model_id": model_id,
            "provider": provider,
            "concurrency": concurrency,
            "prompt_index": prompt_index,
            "success": False,
            "streaming_stable": False,
            "ttft_ms": None,
            "tpot_ms": None,
            "latency_ms": elapsed_s * 1000.0,
            "requests_per_second": 0.0,
            "tokens_per_second": 0.0,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "total_cost_usd": 0.0,
            **counts,
            "error_type": exc.__class__.__name__,
        }


def _safe_concurrency(rows: list[dict[str, object]]) -> dict[str, int | None]:
    safe: dict[str, int | None] = {}
    for model_alias in sorted({str(row.get("model_alias") or "") for row in rows}):
        model_rows = [row for row in rows if row.get("model_alias") == model_alias]
        safe_concurrency: int | None = None
        for concurrency in sorted({_int(row.get("concurrency")) for row in model_rows}):
            slice_rows = [row for row in model_rows if _int(row.get("concurrency")) == concurrency]
            clean = bool(slice_rows) and all(_bool(row.get("success")) for row in slice_rows)
            clean = clean and sum(_int(row.get("http_429_count")) for row in slice_rows) == 0
            clean = clean and sum(_int(row.get("http_5xx_count")) for row in slice_rows) == 0
            clean = clean and sum(_int(row.get("timeout_count")) for row in slice_rows) == 0
            if clean:
                safe_concurrency = concurrency
        safe[model_alias] = safe_concurrency
    return safe


def run_live_api_probe(
    *,
    model_aliases: Iterable[str] = SUPPORTED_API_PROBE_MODELS,
    concurrencies: Iterable[int] = (1, 2, 4),
    prompt_count_per_model: int = 10,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    stream: bool = True,
    env_path: str | Path = ".env",
    pricing_path: str | Path = "configs/api_pricing.yaml",
    models_path: str | Path = "configs/models.yaml",
) -> dict[str, object]:
    """Run a small live API probe only when keys and pricing/routes exist."""

    if prompt_count_per_model <= 0:
        msg = "prompt_count_per_model must be > 0"
        raise ValueError(msg)
    if max_new_tokens <= 0:
        msg = "max_new_tokens must be > 0"
        raise ValueError(msg)
    environment = load_probe_environment(env_path=env_path)
    project = load_project_config(models_path=models_path)
    route_infos: list[dict[str, str]] = []
    missing_env_vars: set[str] = set()
    skipped_models: list[dict[str, object]] = []
    for alias in model_aliases:
        try:
            model = project.resolve_model_config(alias)
            pricing = resolve_api_pricing(alias, pricing_path)
            route = resolve_api_provider_route(model=model, pricing=pricing)
            api_key = api_key_for_route(route, environment)
        except ValueError as exc:
            reason = str(exc)
            if "required for provider" in reason:
                env_name = reason.split(" is required", 1)[0]
                missing_env_vars.add(env_name)
            skipped_models.append({"model_alias": alias, "reason": reason})
            continue
        route_infos.append(
            {
                "model_alias": alias,
                "model_id": model.model_id,
                "provider": route.provider,
                "provider_model_id": route.provider_model_id,
                "base_url": route.base_url,
                "api_key": api_key,
            }
        )
    if missing_env_vars:
        plan = build_api_load_probe_plan(
            model_aliases=model_aliases,
            concurrencies=concurrencies,
            live_probe_enabled=True,
        )
        return {
            "status": "API_PROBE_SKIPPED_MISSING_KEYS",
            "blocked_reason": "missing_api_keys",
            "required_env_vars": sorted(missing_env_vars),
            "skipped_models": skipped_models,
            "plan": plan,
            "summary": summarize_api_probe_results([], live_probe_executed=False),
            "no_live_requests_were_sent": True,
        }
    rows: list[dict[str, object]] = []
    prompts = list(DEFAULT_PROBE_PROMPTS[:prompt_count_per_model])
    while len(prompts) < prompt_count_per_model:
        prompts.append(DEFAULT_PROBE_PROMPTS[len(prompts) % len(DEFAULT_PROBE_PROMPTS)])
    for route_info in route_infos:
        for concurrency in concurrencies:
            if concurrency not in SUPPORTED_API_PROBE_CONCURRENCIES:
                msg = f"Unsupported API probe concurrency '{concurrency}'"
                raise ValueError(msg)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _probe_one_request,
                        model_alias=route_info["model_alias"],
                        model_id=route_info["model_id"],
                        provider=route_info["provider"],
                        provider_model_id=route_info["provider_model_id"],
                        base_url=route_info["base_url"],
                        api_key=route_info["api_key"],
                        prompt=prompt,
                        prompt_index=index,
                        concurrency=concurrency,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        stream=stream,
                        pricing_path=pricing_path,
                    )
                    for index, prompt in enumerate(prompts, start=1)
                ]
                for future in as_completed(futures):
                    rows.append(future.result())
    summary = summarize_api_probe_results(rows, live_probe_executed=bool(rows))
    safe_concurrency = _safe_concurrency(rows)
    return {
        "status": summary["verdict"],
        "blocked_reason": None if rows else "no_runnable_api_models",
        "plan": build_api_load_probe_plan(
            model_aliases=model_aliases,
            concurrencies=concurrencies,
            live_probe_enabled=True,
        ),
        "summary": summary,
        "rows": rows,
        "skipped_models": skipped_models,
        "recommended_safe_concurrency": safe_concurrency,
        "no_live_requests_were_sent": not rows,
    }


def summarize_api_probe_results(
    rows: list[dict[str, object]],
    *,
    live_probe_executed: bool,
) -> dict[str, object]:
    """Summarize already-collected API probe rows.

    This function is intentionally offline: callers pass rows from a future
    authorized probe, and no provider request is issued here.
    """

    model_counts = Counter(str(row.get("model_alias") or "") for row in rows)
    concurrency_counts = Counter(_int(row.get("concurrency")) for row in rows)
    successful_rows = [row for row in rows if _bool(row.get("success"))]
    ttft_values = [_float(row.get("ttft_ms")) for row in successful_rows if row.get("ttft_ms")]
    tpot_values = [_float(row.get("tpot_ms")) for row in successful_rows if row.get("tpot_ms")]
    latency_values = [
        _float(row.get("latency_ms")) for row in successful_rows if row.get("latency_ms")
    ]
    rps_values = [
        _float(row.get("requests_per_second"))
        for row in successful_rows
        if row.get("requests_per_second")
    ]
    tps_values = [
        _float(row.get("tokens_per_second"))
        for row in successful_rows
        if row.get("tokens_per_second")
    ]
    streaming_stable = sum(1 for row in rows if _bool(row.get("streaming_stable")))
    request_count = len(rows)
    summary: dict[str, object] = {
        "live_probe_executed": live_probe_executed,
        "request_count": request_count,
        "success_count": len(successful_rows),
        "model_counts": dict(model_counts),
        "concurrency_counts": dict(concurrency_counts),
        "mean_ttft_ms": _mean(ttft_values),
        "mean_tpot_ms": _mean(tpot_values),
        "mean_latency_ms": _mean(latency_values),
        "mean_requests_per_second": _mean(rps_values),
        "mean_tokens_per_second": _mean(tps_values),
        "http_429_count": sum(_int(row.get("http_429_count")) for row in rows),
        "http_5xx_count": sum(_int(row.get("http_5xx_count")) for row in rows),
        "timeout_count": sum(_int(row.get("timeout_count")) for row in rows),
        "retry_count": sum(_int(row.get("retry_count")) for row in rows),
        "provider_throttling_count": sum(
            _int(row.get("provider_throttling_count")) for row in rows
        ),
        "total_api_cost_usd": sum(_float(row.get("total_cost_usd")) for row in rows),
        "streaming_stability_rate": (
            streaming_stable / request_count if request_count > 0 else 0.0
        ),
    }
    summary["verdict"] = classify_api_probe(summary)
    return summary


def build_framework_only_api_probe_report() -> dict[str, object]:
    """Return a no-network API probe report used before live authorization."""

    plan = build_api_load_probe_plan()
    summary = summarize_api_probe_results([], live_probe_executed=False)
    return {
        "status": summary["verdict"],
        "blocked_reason": "live_api_probe_not_executed",
        "plan": plan,
        "summary": summary,
        "no_live_requests_were_sent": True,
    }


def _summary_rows(report: dict[str, object]) -> list[dict[str, object]]:
    summary = report.get("summary")
    summary_payload = summary if isinstance(summary, dict) else {}
    return [
        {
            "status": report.get("status"),
            "blocked_reason": report.get("blocked_reason"),
            "live_probe_executed": summary_payload.get("live_probe_executed"),
            "request_count": summary_payload.get("request_count"),
            "success_count": summary_payload.get("success_count"),
            "http_429_count": summary_payload.get("http_429_count"),
            "http_5xx_count": summary_payload.get("http_5xx_count"),
            "timeout_count": summary_payload.get("timeout_count"),
            "retry_count": summary_payload.get("retry_count"),
            "provider_throttling_count": summary_payload.get("provider_throttling_count"),
            "total_api_cost_usd": summary_payload.get("total_api_cost_usd"),
            "streaming_stability_rate": summary_payload.get("streaming_stability_rate"),
            "recommended_safe_concurrency": json.dumps(
                report.get("recommended_safe_concurrency") or {},
                ensure_ascii=True,
                sort_keys=True,
            ),
        }
    ]


def write_api_probe_artifacts(
    *,
    report: dict[str, object],
    report_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write API probe JSON and CSV artifacts."""

    json_path = Path(report_path)
    csv_path = Path(summary_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = _summary_rows(report)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
