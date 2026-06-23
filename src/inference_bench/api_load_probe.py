"""Framework-only API provider load probe planning and result summarization."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SUPPORTED_API_PROBE_MODELS = ("model5_gated", "model6_gated", "model7_gated")
SUPPORTED_API_PROBE_CONCURRENCIES = (1, 2, 4, 8, 16)
API_PROBE_VERDICTS = ("API_PROBE_PASSED", "API_PROBE_WARNING", "API_PROBE_BLOCKED")

ApiProbeVerdict = Literal["API_PROBE_PASSED", "API_PROBE_WARNING", "API_PROBE_BLOCKED"]


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
            "streaming_stability_rate": summary_payload.get("streaming_stability_rate"),
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
