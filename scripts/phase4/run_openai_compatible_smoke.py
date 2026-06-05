"""Run a tiny OpenAI-compatible Phase 4 smoke test.

This script validates the vLLM/OpenAI-compatible execution path against
runner-adapted Phase 3 workload records. Dry-run mode does not contact a server.
Real mode requires a reachable OpenAI-compatible endpoint and never triggers
paid API calls by default when used with a local vLLM server and ``EMPTY`` key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.generation_contract import (  # noqa: E402
    generation_contract_result_fields,
)
from inference_bench.metrics import calculate_tokens_per_second  # noqa: E402
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.mock_runner import count_whitespace_tokens  # noqa: E402
from inference_bench.runners.openai_compatible_runner import (  # noqa: E402
    OpenAICompatibleRunnerConfig,
    _extract_response_text,
    require_openai_dependency,
)
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.workloads.loader import load_jsonl_workload  # noqa: E402

MAX_SMOKE_PROMPTS = 25
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"


@dataclass(frozen=True)
class ServerReadiness:
    """OpenAI-compatible server readiness status."""

    reachable: bool
    models_endpoint_supported: bool
    model_available: bool | None
    model_names: list[str]
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable status payload."""

        return {
            "reachable": self.reachable,
            "models_endpoint_supported": self.models_endpoint_supported,
            "model_available": self.model_available,
            "model_names": self.model_names,
            "message": self.message,
        }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Run a tiny OpenAI-compatible/vLLM smoke test.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional run manifest path. Defaults beside the raw output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate plumbing and write fixture output without contacting a server.",
    )
    return parser


def utc_timestamp() -> str:
    """Return an ISO UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def validate_limit(limit: int) -> None:
    """Validate smoke prompt limit."""

    if limit <= 0:
        msg = "limit must be > 0"
        raise ValueError(msg)
    if limit > MAX_SMOKE_PROMPTS:
        msg = f"limit must be <= {MAX_SMOKE_PROMPTS} for OpenAI-compatible smoke"
        raise ValueError(msg)


def validate_max_new_tokens(max_new_tokens: int) -> None:
    """Validate generation length."""

    if max_new_tokens <= 0:
        msg = "max_new_tokens must be > 0"
        raise ValueError(msg)
    if max_new_tokens > 128:
        msg = "max_new_tokens must be <= 128 for OpenAI-compatible smoke"
        raise ValueError(msg)


def load_smoke_items(path: str | Path, limit: int) -> list[WorkloadItem]:
    """Load and limit runner-compatible workload items."""

    validate_limit(limit)
    items = load_jsonl_workload(path)
    return items[:limit]


def resolve_model_id(model_alias: str) -> str:
    """Resolve a public or canonical model key to a model ID."""

    config = load_project_config()
    model_config = config.resolve_model_config(model_alias)
    if model_config.requires_hf_token or model_config.requires_license_acceptance:
        msg = (
            f"{model_alias} resolves to gated model {model_config.model_id}; "
            "OpenAI-compatible smoke requires a local/open model unless a future "
            "paid API script explicitly allows that path."
        )
        raise RuntimeError(msg)
    return model_config.model_id


def item_metadata(item: WorkloadItem, key: str, default: str = "") -> str:
    """Return one workload metadata value."""

    return str(item.metadata.get(key) or default)


def _model_names_from_models_response(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    names: list[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str) and model_id.strip():
            names.append(model_id)
    return sorted(set(names))


def check_server_readiness(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    timeout_seconds: float,
) -> ServerReadiness:
    """Probe the OpenAI-compatible server before issuing generation requests."""

    models_url = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 405, 501}:
            return ServerReadiness(
                reachable=True,
                models_endpoint_supported=False,
                model_available=None,
                model_names=[],
                message=(
                    f"Server reachable at {base_url}, but /models is not supported "
                    f"(HTTP {exc.code})."
                ),
            )
        msg = (
            f"OpenAI-compatible server at {base_url} rejected /models readiness "
            f"check with HTTP {exc.code}: {exc.reason}"
        )
        raise RuntimeError(msg) from exc
    except OSError as exc:
        msg = (
            f"OpenAI-compatible server is not reachable at {base_url}. "
            "Start vLLM with an OpenAI-compatible server, then rerun this smoke. "
            f"Underlying error: {exc}"
        )
        raise RuntimeError(msg) from exc

    try:
        payload = json.loads(response_body) if response_body.strip() else {}
    except json.JSONDecodeError as exc:
        msg = f"Server reachable at {base_url}, but /models did not return valid JSON."
        raise RuntimeError(msg) from exc

    model_names = _model_names_from_models_response(payload)
    if not model_names:
        return ServerReadiness(
            reachable=True,
            models_endpoint_supported=True,
            model_available=None,
            model_names=[],
            message="Server reachable and /models responded, but no model IDs were listed.",
        )
    model_available = model_name in model_names
    if not model_available:
        msg = (
            f"Server reachable at {base_url}, but model {model_name} was not in "
            f"/models response. Available models: {', '.join(model_names)}"
        )
        raise RuntimeError(msg)
    return ServerReadiness(
        reachable=True,
        models_endpoint_supported=True,
        model_available=True,
        model_names=model_names,
        message=f"Server reachable and model {model_name} is available.",
    )


def base_result_row(
    *,
    item: WorkloadItem,
    run_id: str,
    model_alias: str,
    model_id: str,
    model_name: str,
    base_url: str,
    dry_run: bool,
    readiness: ServerReadiness | None,
) -> dict[str, Any]:
    """Build common smoke result fields."""

    return {
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "backend": "openai_compatible",
        "optimization": "vllm_openai_compatible_smoke",
        "model_alias": model_alias,
        "model_id": model_id,
        "model_name": model_name,
        "base_url": base_url,
        "workload_name": item.workload_name,
        "prompt_id": item.prompt_id,
        "workload_id": item_metadata(item, "workload_id"),
        "vertical": item_metadata(item, "vertical"),
        "memory_mode": item_metadata(item, "memory_mode"),
        "ablation_mode": item_metadata(item, "ablation_mode", "none"),
        "dataset_split": item_metadata(item, "dataset_split"),
        "expected_output_format": item.expected_output
        or item_metadata(item, "expected_output_format", "text"),
        "context_token_estimate": item_metadata(item, "context_token_estimate", "0"),
        "gold_evidence_ids": item_metadata(item, "gold_evidence_ids", "[]"),
        "selected_context_ids": item_metadata(item, "selected_context_ids", "[]"),
        "citation_id_aliases": item_metadata(item, "citation_id_aliases", "{}"),
        "prompt": item.prompt,
        "estimated_cost_usd": 0.0,
        "paid_api_call_triggered": False,
        "no_gpu_experiment_triggered": True,
        "dry_run": dry_run,
        "server_readiness": readiness.to_dict() if readiness is not None else None,
    }


def validate_smoke_result_row(row: dict[str, Any]) -> None:
    """Validate the JSONL row shape used by OpenAI-compatible smoke."""

    required_fields = {
        "run_id",
        "timestamp_utc",
        "backend",
        "model_alias",
        "model_id",
        "model_name",
        "base_url",
        "prompt_id",
        "workload_id",
        "vertical",
        "memory_mode",
        "ablation_mode",
        "generated_text",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "success",
        "paid_api_call_triggered",
        "generation_contract_valid",
        "answer",
        "evidence_ids",
        "confidence",
        "insufficient_evidence",
        "citation_notes",
    }
    missing = sorted(field for field in required_fields if field not in row)
    if missing:
        msg = f"OpenAI-compatible smoke result row missing required fields: {missing}"
        raise ValueError(msg)
    if row["backend"] != "openai_compatible":
        msg = "backend must be openai_compatible"
        raise ValueError(msg)
    if int(row["input_tokens"]) < 0 or int(row["output_tokens"]) < 0:
        msg = "token counts must be >= 0"
        raise ValueError(msg)
    if float(row["latency_ms"]) < 0:
        msg = "latency_ms must be >= 0"
        raise ValueError(msg)
    if bool(row["paid_api_call_triggered"]):
        msg = "OpenAI-compatible local smoke must not trigger paid API calls"
        raise ValueError(msg)


def dry_run_result(
    *,
    item: WorkloadItem,
    run_id: str,
    model_alias: str,
    model_id: str,
    model_name: str,
    base_url: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build one dry-run smoke result without contacting a server."""

    started = time.perf_counter()
    generated_text = (
        f"DRY RUN: OpenAI-compatible smoke plumbing preserved metadata for {item.prompt_id}."
    )
    elapsed_seconds = time.perf_counter() - started
    input_tokens = count_whitespace_tokens(item.prompt)
    output_tokens = min(max_new_tokens, count_whitespace_tokens(generated_text))
    row = base_result_row(
        item=item,
        run_id=run_id,
        model_alias=model_alias,
        model_id=model_id,
        model_name=model_name,
        base_url=base_url,
        dry_run=True,
        readiness=None,
    )
    row.update(
        {
            "generated_text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(elapsed_seconds * 1000, 6),
            "end_to_end_latency_ms": round(elapsed_seconds * 1000, 6),
            "ttft_ms": None,
            "tpot_ms": None,
            "throughput_tokens_per_second": calculate_tokens_per_second(
                input_tokens + output_tokens,
                elapsed_seconds,
            ),
            "success": True,
            "error_type": None,
            "error_message": None,
            "final_status": "answer",
        }
    )
    row.update(generation_contract_result_fields(generated_text))
    validate_smoke_result_row(row)
    return row


def _usage_tokens(response: Any, prompt: str, generated_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    input_tokens = (
        int(prompt_tokens) if isinstance(prompt_tokens, int) else count_whitespace_tokens(prompt)
    )
    output_tokens = (
        int(completion_tokens)
        if isinstance(completion_tokens, int)
        else count_whitespace_tokens(generated_text)
    )
    return max(0, input_tokens), max(0, output_tokens)


def run_real_openai_compatible(
    *,
    items: list[WorkloadItem],
    run_id: str,
    model_alias: str,
    model_id: str,
    model_name: str,
    base_url: str,
    api_key: str,
    max_new_tokens: int,
    timeout_seconds: float,
    readiness: ServerReadiness,
) -> list[dict[str, Any]]:
    """Run real OpenAI-compatible generation over workload items."""

    require_openai_dependency()
    openai = cast(Any, import_module("openai"))
    config = OpenAICompatibleRunnerConfig(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
        stream=False,
    )
    client = openai.OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )

    rows: list[dict[str, Any]] = []
    for item in items:
        started = time.perf_counter()
        row = base_result_row(
            item=item,
            run_id=run_id,
            model_alias=model_alias,
            model_id=model_id,
            model_name=model_name,
            base_url=base_url,
            dry_run=False,
            readiness=readiness,
        )
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": item.prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                stream=False,
            )
            generated_text = _extract_response_text(response)
            elapsed_seconds = time.perf_counter() - started
            input_tokens, output_tokens = _usage_tokens(response, item.prompt, generated_text)
            row.update(
                {
                    "generated_text": generated_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": round(elapsed_seconds * 1000, 6),
                    "end_to_end_latency_ms": round(elapsed_seconds * 1000, 6),
                    "ttft_ms": None,
                    "tpot_ms": (
                        round((elapsed_seconds * 1000) / output_tokens, 6)
                        if output_tokens > 0
                        else None
                    ),
                    "throughput_tokens_per_second": calculate_tokens_per_second(
                        input_tokens + output_tokens,
                        elapsed_seconds,
                    ),
                    "success": True,
                    "error_type": None,
                    "error_message": None,
                    "final_status": "answer",
                }
            )
            row.update(generation_contract_result_fields(generated_text))
        except Exception as exc:  # noqa: BLE001
            elapsed_seconds = time.perf_counter() - started
            row.update(
                {
                    "generated_text": "",
                    "input_tokens": count_whitespace_tokens(item.prompt),
                    "output_tokens": 0,
                    "latency_ms": round(elapsed_seconds * 1000, 6),
                    "end_to_end_latency_ms": round(elapsed_seconds * 1000, 6),
                    "ttft_ms": None,
                    "tpot_ms": None,
                    "throughput_tokens_per_second": None,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "final_status": "failed_validation",
                }
            )
            row.update(generation_contract_result_fields(""))
        validate_smoke_result_row(row)
        rows.append(row)
    return rows


def write_jsonl_rows(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write smoke rows as JSONL."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def manifest_path_for_output(output_path: str | Path, manifest_path: str | None) -> Path:
    """Resolve run manifest path."""

    if manifest_path:
        return Path(manifest_path)
    return Path(output_path).with_name("phase4_openai_compatible_smoke_manifest.json")


def sanitize_command(argv: list[str]) -> str:
    """Return the command string without exposing an API key value."""

    sanitized: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            sanitized.append("***")
            skip_next = False
            continue
        sanitized.append(arg)
        if arg == "--api-key" and index + 1 < len(argv):
            skip_next = True
    return " ".join([Path(sys.executable).name, *sanitized])


def build_manifest(
    *,
    run_id: str,
    model_alias: str,
    model_id: str,
    input_path: str | Path,
    output_path: str | Path,
    limit: int,
    rows: list[dict[str, Any]],
    command: str,
    start_time: str,
    end_time: str | None,
    status: str,
) -> RunManifest:
    """Build run manifest from smoke rows."""

    first_row = rows[0] if rows else {}
    error_count = sum(1 for row in rows if not bool(row.get("success")))
    return RunManifest(
        run_id=run_id,
        timestamp_utc=utc_now(),
        backend="openai_compatible",
        model_alias=model_alias,
        model_id=model_id,
        memory_mode=str(first_row.get("memory_mode") or "unknown"),
        split=str(first_row.get("dataset_split") or "unknown"),
        ablation_mode=str(first_row.get("ablation_mode") or "unknown"),
        input_workload_path=str(input_path),
        output_path=str(output_path),
        max_records=limit,
        git_commit=current_git_commit(REPO_ROOT),
        command=command,
        status=status,
        start_time=start_time,
        end_time=end_time,
        error_count=error_count,
    )


def run_smoke(
    *,
    input_path: str | Path,
    output_path: str | Path,
    model_alias: str,
    model_name: str,
    base_url: str,
    api_key: str,
    limit: int,
    max_new_tokens: int,
    timeout_seconds: float,
    dry_run: bool = False,
    manifest_path: str | None = None,
    command: str = "run_openai_compatible_smoke",
) -> tuple[list[dict[str, Any]], Path]:
    """Run or dry-run OpenAI-compatible smoke and write outputs."""

    validate_limit(limit)
    validate_max_new_tokens(max_new_tokens)
    model_id = resolve_model_id(model_alias)
    run_id = (
        "phase4-openai-compatible-smoke-dry-run" if dry_run else "phase4-openai-compatible-smoke"
    )
    start_time = utc_now()
    items = load_smoke_items(input_path, limit)
    if dry_run:
        rows = [
            dry_run_result(
                item=item,
                run_id=run_id,
                model_alias=model_alias,
                model_id=model_id,
                model_name=model_name,
                base_url=base_url,
                max_new_tokens=max_new_tokens,
            )
            for item in items
        ]
    else:
        readiness = check_server_readiness(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=timeout_seconds,
        )
        rows = run_real_openai_compatible(
            items=items,
            run_id=run_id,
            model_alias=model_alias,
            model_id=model_id,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            max_new_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
            readiness=readiness,
        )
    output = write_jsonl_rows(rows, output_path)
    end_time = utc_now()
    manifest = build_manifest(
        run_id=run_id,
        model_alias=model_alias,
        model_id=model_id,
        input_path=input_path,
        output_path=output,
        limit=limit,
        rows=rows,
        command=command,
        start_time=start_time,
        end_time=end_time,
        status="completed",
    )
    written_manifest = write_run_manifest(
        manifest,
        manifest_path_for_output(output_path, manifest_path),
    )
    return rows, written_manifest


def main(argv: list[str] | None = None) -> int:
    """Run the OpenAI-compatible smoke CLI."""

    args = build_parser().parse_args(argv)
    command = sanitize_command(sys.argv)
    try:
        rows, manifest = run_smoke(
            input_path=args.input_path,
            output_path=args.output_path,
            model_alias=args.model_alias,
            model_name=args.model_name,
            base_url=args.base_url,
            api_key=args.api_key,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
            manifest_path=args.manifest_path,
            command=command,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"OpenAI-compatible smoke failed: {exc}", file=sys.stderr)
        return 1

    success_count = sum(1 for row in rows if bool(row["success"]))
    print(f"Rows written: {len(rows)}")
    print(f"Successful generations: {success_count}")
    print(f"Output path: {args.output_path}")
    print(f"Run manifest: {manifest}")
    if args.dry_run:
        print("Dry-run mode: server readiness check and model calls were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
