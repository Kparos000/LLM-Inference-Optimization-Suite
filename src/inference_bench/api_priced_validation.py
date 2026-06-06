"""Readiness and cost reporting for the guarded API-priced smoke test."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from inference_bench.api_pricing import ApiPricingEntry, resolve_api_pricing
from inference_bench.config import ProjectConfig

HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_MODEL_CONFIG_URL = "https://huggingface.co/{model_id}/resolve/main/config.json"


@dataclass(frozen=True)
class AccessCheck:
    """Sanitized Hugging Face access-check result."""

    available: bool
    status_code: int | None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class SelectedApiModel:
    """Model and pricing selected after readiness checks."""

    model_alias: str
    model_id: str
    provider_model_id: str
    pricing: ApiPricingEntry


def _authorized_get(url: str, hf_token: str, *, timeout: int = 30) -> AccessCheck:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read(1)
            return AccessCheck(available=True, status_code=response.status)
    except HTTPError as exc:
        return AccessCheck(
            available=False,
            status_code=exc.code,
            error_type="HTTPError",
            error_message=f"HTTP {exc.code}",
        )
    except URLError as exc:
        return AccessCheck(
            available=False,
            status_code=None,
            error_type="URLError",
            error_message=str(exc.reason),
        )


def check_hf_token(hf_token: str) -> AccessCheck:
    """Validate an HF token without returning or logging the token."""

    if not hf_token.strip():
        return AccessCheck(
            available=False,
            status_code=None,
            error_type="MissingToken",
            error_message="HF_TOKEN is missing",
        )
    return _authorized_get(HF_WHOAMI_URL, hf_token)


def check_model_access(model_id: str, hf_token: str) -> AccessCheck:
    """Check gated repository access through a small model config request."""

    encoded_model_id = quote(model_id, safe="/")
    return _authorized_get(
        HF_MODEL_CONFIG_URL.format(model_id=encoded_model_id),
        hf_token,
    )


def select_api_model(
    *,
    config: ProjectConfig,
    model_aliases: list[str],
    pricing_config: str | Path,
    hf_token: str,
    access_checker: Callable[[str, str], AccessCheck] = check_model_access,
) -> tuple[SelectedApiModel | None, list[dict[str, Any]]]:
    """Select the first model with captured pricing and gated-model access."""

    attempts: list[dict[str, Any]] = []
    for alias in model_aliases:
        model = config.resolve_model_config(alias)
        attempt: dict[str, Any] = {
            "model_alias": alias,
            "model_id": model.model_id,
            "pricing_available": False,
            "model_access": False,
        }
        try:
            pricing = resolve_api_pricing(alias, pricing_config)
        except (FileNotFoundError, ValueError) as exc:
            attempt["failure_stage"] = "pricing"
            attempt["failure_reason"] = str(exc)
            attempts.append(attempt)
            continue
        attempt.update(
            {
                "pricing_available": True,
                "provider": pricing.provider,
                "input_cost_per_1m_tokens_usd": pricing.input_cost_per_1m_tokens_usd,
                "output_cost_per_1m_tokens_usd": pricing.output_cost_per_1m_tokens_usd,
                "pricing_source_url": pricing.pricing_source_url,
            }
        )
        access = access_checker(model.model_id, hf_token)
        attempt["model_access"] = access.available
        attempt["model_access_status_code"] = access.status_code
        if not access.available:
            attempt["failure_stage"] = "model_access"
            attempt["failure_reason"] = access.error_message
            attempts.append(attempt)
            continue
        provider_model_id = f"{model.model_id}:{pricing.provider}"
        attempts.append(attempt)
        return (
            SelectedApiModel(
                model_alias=alias,
                model_id=model.model_id,
                provider_model_id=provider_model_id,
                pricing=pricing,
            ),
            attempts,
        )
    return None, attempts


def write_readiness_report(
    path: str | Path,
    *,
    token_check: AccessCheck,
    model_attempts: list[dict[str, Any]],
    selected: SelectedApiModel | None,
    workload_path: str | Path,
    workload_count: int,
    execution_status: str,
    stop_reason: str | None,
) -> Path:
    """Write a secret-free API readiness report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "execution_status": execution_status,
        "hf_token_present_and_valid": token_check.available,
        "hf_token_status_code": token_check.status_code,
        "hf_token_error_type": token_check.error_type,
        "hf_token_error_message": token_check.error_message,
        "model_attempts": model_attempts,
        "selected_model_alias": selected.model_alias if selected else None,
        "selected_model_id": selected.model_id if selected else None,
        "selected_provider": selected.pricing.provider if selected else None,
        "pricing_detected": selected is not None,
        "pricing_source_url": selected.pricing.pricing_source_url if selected else None,
        "workload_path": str(workload_path),
        "workload_count": workload_count,
        "stop_reason": stop_reason,
        "secret_values_recorded": False,
        "gpu_work_triggered": False,
        "vllm_triggered": False,
        "sglang_triggered": False,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def build_cost_report(
    *,
    result_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate measured token, cost, latency, and quality values."""

    request_count = len(result_rows)
    success_count = sum(bool(row.get("success")) for row in result_rows)
    grounded_count = sum(bool(row.get("groundedness")) for row in evaluation_rows)
    input_tokens = sum(int(_number(row, "input_tokens")) for row in result_rows)
    output_tokens = sum(int(_number(row, "output_tokens")) for row in result_rows)
    input_cost = sum(_number(row, "input_cost_usd") for row in result_rows)
    output_cost = sum(_number(row, "output_cost_usd") for row in result_rows)
    total_cost = sum(_number(row, "total_cost_usd") for row in result_rows)
    latencies = [_number(row, "latency_ms") for row in result_rows if row.get("latency_ms")]
    throughputs = [
        _number(row, "throughput_tokens_per_second")
        for row in result_rows
        if row.get("throughput_tokens_per_second")
    ]
    quality = {
        "json_valid_rate": _rate(
            sum(bool(row.get("json_validity")) for row in evaluation_rows),
            len(evaluation_rows),
        ),
        "generation_contract_valid_rate": _rate(
            sum(bool(row.get("generation_contract_valid")) for row in evaluation_rows),
            len(evaluation_rows),
        ),
        "evidence_id_presence_rate": _rate(
            sum(bool(row.get("evidence_id_presence")) for row in evaluation_rows),
            len(evaluation_rows),
        ),
        "evidence_match_rate": _rate(
            sum(bool(row.get("evidence_match")) for row in evaluation_rows),
            len(evaluation_rows),
        ),
        "grounded_rate": _rate(grounded_count, len(evaluation_rows)),
        "safety_violation_rate": _rate(
            sum(bool(row.get("safety_violation")) for row in evaluation_rows),
            len(evaluation_rows),
        ),
    }
    baseline = baseline_summary or {}
    comparison: dict[str, dict[str, float | None]] = {}
    for metric in (
        "json_valid_rate",
        "generation_contract_valid_rate",
        "evidence_id_presence_rate",
        "evidence_match_rate",
        "grounded_rate",
    ):
        baseline_value = baseline.get(metric)
        baseline_number = (
            float(baseline_value)
            if isinstance(baseline_value, int | float) and not isinstance(baseline_value, bool)
            else None
        )
        comparison[metric] = {
            "qwen_0_5b": baseline_number,
            "api_priced_model": quality[metric],
            "delta": quality[metric] - baseline_number if baseline_number is not None else None,
        }
    first = result_rows[0] if result_rows else {}
    return {
        "execution_complete": request_count == 5 and success_count == 5,
        "model_alias": first.get("model_alias"),
        "model_id": first.get("model_id"),
        "provider": first.get("provider"),
        "pricing_source_url": first.get("pricing_source_url"),
        "pricing_source": first.get("pricing_source"),
        "pricing_status": first.get("pricing_status"),
        "pricing_snapshot_timestamp_utc": (
            first.get("pricing_last_checked") or first.get("pricing_snapshot_timestamp_utc")
        ),
        "input_cost_per_1m_tokens_usd": (
            first.get("input_usd_per_1m_tokens") or first.get("input_cost_per_1m_tokens_usd")
        ),
        "output_cost_per_1m_tokens_usd": (
            first.get("output_usd_per_1m_tokens") or first.get("output_cost_per_1m_tokens_usd")
        ),
        "request_count": request_count,
        "success_count": success_count,
        "error_count": request_count - success_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "cost_per_request_usd": total_cost / request_count if request_count else None,
        "cost_per_successful_answer_usd": total_cost / success_count if success_count else None,
        "cost_per_grounded_answer_usd": total_cost / grounded_count if grounded_count else None,
        "ttft_available": any(row.get("ttft_ms") is not None for row in result_rows),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "mean_throughput_tokens_per_second": (
            statistics.fmean(throughputs) if throughputs else None
        ),
        "quality": quality,
        "qwen_0_5b_comparison": comparison,
        "request_costs": [
            {
                "prompt_id": row.get("prompt_id"),
                "vertical": row.get("vertical"),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "total_tokens": row.get("total_tokens"),
                "input_cost_usd": row.get("input_cost_usd"),
                "output_cost_usd": row.get("output_cost_usd"),
                "total_cost_usd": row.get("total_cost_usd"),
                "latency_ms": row.get("latency_ms"),
                "ttft_ms": row.get("ttft_ms"),
                "success": row.get("success"),
                "error_type": row.get("error_type"),
            }
            for row in result_rows
        ],
        "gpu_work_triggered": False,
        "vllm_triggered": False,
        "sglang_triggered": False,
    }


def write_cost_artifacts(
    *,
    report_path: str | Path,
    summary_path: str | Path,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the API cost JSON report and aggregate CSV summary."""

    json_output = Path(report_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_output = Path(summary_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"quality", "qwen_0_5b_comparison", "request_costs"}
    }
    summary.update(report["quality"])
    with csv_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return json_output, csv_output


def access_check_dict(check: AccessCheck) -> dict[str, Any]:
    """Return an AccessCheck as a JSON-safe mapping."""

    return asdict(check)
