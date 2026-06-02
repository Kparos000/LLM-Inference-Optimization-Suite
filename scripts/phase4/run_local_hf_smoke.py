"""Run a small local Hugging Face Phase 4 smoke test.

This script runs local Transformers generation only. It does not call paid APIs,
does not use gated models, and enforces a hard prompt cap for smoke testing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.metrics import calculate_tokens_per_second  # noqa: E402
from inference_bench.run_manifest import (  # noqa: E402
    RunManifest,
    current_git_commit,
    utc_now,
    write_run_manifest,
)
from inference_bench.runners.hf_runner import (  # noqa: E402
    HuggingFaceRunnerConfig,
    _input_token_count,
    _move_inputs_to_device,
    _resolve_device,
    _resolve_torch_dtype,
    require_hf_dependencies,
)
from inference_bench.schema import WorkloadItem  # noqa: E402
from inference_bench.workloads.loader import load_jsonl_workload  # noqa: E402

MAX_SMOKE_PROMPTS = 25


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Run a local HF real inference smoke test.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-alias", default="model1_0_5b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional run manifest path. Defaults beside the raw output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate plumbing and write fixture output without loading a model.",
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
        msg = f"limit must be <= {MAX_SMOKE_PROMPTS} for local HF smoke"
        raise ValueError(msg)


def validate_max_new_tokens(max_new_tokens: int) -> None:
    """Validate generation length."""

    if max_new_tokens <= 0:
        msg = "max_new_tokens must be > 0"
        raise ValueError(msg)
    if max_new_tokens > 128:
        msg = "max_new_tokens must be <= 128 for local HF smoke"
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
            "local HF smoke requires an open, non-gated model."
        )
        raise RuntimeError(msg)
    if model_config.provider != "huggingface":
        msg = f"{model_alias} provider must be huggingface, got {model_config.provider}"
        raise RuntimeError(msg)
    return model_config.model_id


def item_metadata(item: WorkloadItem, key: str, default: str = "") -> str:
    """Return one workload metadata value."""

    return str(item.metadata.get(key) or default)


def base_result_row(
    *,
    item: WorkloadItem,
    run_id: str,
    model_alias: str,
    model_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Build common smoke result fields."""

    return {
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "backend": "huggingface_local",
        "model_alias": model_alias,
        "model_id": model_id,
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
        "prompt": item.prompt,
        "estimated_cost_usd": 0.0,
        "paid_api_call_triggered": False,
        "no_gpu_experiment_triggered": True,
        "dry_run": dry_run,
    }


def validate_smoke_result_row(row: dict[str, Any]) -> None:
    """Validate the JSONL row shape used by local HF smoke."""

    required_fields = {
        "run_id",
        "timestamp_utc",
        "backend",
        "model_alias",
        "model_id",
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
    }
    missing = sorted(field for field in required_fields if field not in row)
    if missing:
        msg = f"Smoke result row missing required fields: {missing}"
        raise ValueError(msg)
    if int(row["input_tokens"]) < 0 or int(row["output_tokens"]) < 0:
        msg = "token counts must be >= 0"
        raise ValueError(msg)
    if float(row["latency_ms"]) < 0:
        msg = "latency_ms must be >= 0"
        raise ValueError(msg)
    if bool(row["paid_api_call_triggered"]):
        msg = "local HF smoke must not trigger paid API calls"
        raise ValueError(msg)


def dry_run_result(
    *,
    item: WorkloadItem,
    run_id: str,
    model_alias: str,
    model_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build one dry-run smoke result without loading a model."""

    started = time.perf_counter()
    generated_text = (
        f"DRY RUN: local Hugging Face smoke plumbing preserved metadata for {item.prompt_id}."
    )
    latency_ms = (time.perf_counter() - started) * 1000
    input_tokens = len(item.prompt.split())
    output_tokens = min(max_new_tokens, len(generated_text.split()))
    row = base_result_row(
        item=item,
        run_id=run_id,
        model_alias=model_alias,
        model_id=model_id,
        dry_run=True,
    )
    row.update(
        {
            "generated_text": generated_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 6),
            "end_to_end_latency_ms": round(latency_ms, 6),
            "throughput_tokens_per_second": None,
            "success": True,
            "error_type": None,
            "error_message": None,
            "final_status": "answer",
        }
    )
    validate_smoke_result_row(row)
    return row


def run_real_local_hf(
    *,
    items: list[WorkloadItem],
    run_id: str,
    model_alias: str,
    model_id: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run real local Hugging Face generation over workload items."""

    require_hf_dependencies()
    torch = import_module("torch")
    transformers = import_module("transformers")
    config = HuggingFaceRunnerConfig(model_id=model_id, max_new_tokens=max_new_tokens)
    device = _resolve_device(torch, config.device)
    model_kwargs: dict[str, Any] = {}
    torch_dtype = _resolve_torch_dtype(torch, config.dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load local Hugging Face model {model_id}: {exc}"
        raise RuntimeError(msg) from exc

    if hasattr(model, "to"):
        model = model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    for item in items:
        started = time.perf_counter()
        row = base_result_row(
            item=item,
            run_id=run_id,
            model_alias=model_alias,
            model_id=model_id,
            dry_run=False,
        )
        input_tokens = 0
        try:
            inputs = tokenizer(item.prompt, return_tensors="pt")
            input_tokens = _input_token_count(inputs)
            inputs = _move_inputs_to_device(inputs, device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=getattr(tokenizer, "eos_token_id", None),
                )
            generated_token_ids = generated_ids[0][input_tokens:]
            generated_text = str(tokenizer.decode(generated_token_ids, skip_special_tokens=True))
            output_tokens = max(0, int(generated_ids.shape[-1]) - input_tokens)
            elapsed_seconds = time.perf_counter() - started
            row.update(
                {
                    "generated_text": generated_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": round(elapsed_seconds * 1000, 6),
                    "end_to_end_latency_ms": round(elapsed_seconds * 1000, 6),
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
        except Exception as exc:  # noqa: BLE001
            elapsed_seconds = time.perf_counter() - started
            row.update(
                {
                    "generated_text": "",
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "latency_ms": round(elapsed_seconds * 1000, 6),
                    "end_to_end_latency_ms": round(elapsed_seconds * 1000, 6),
                    "throughput_tokens_per_second": None,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "final_status": "failed_validation",
                }
            )
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
    return Path(output_path).with_name("phase4_hf_local_smoke_manifest.json")


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
        backend="huggingface_local",
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
    limit: int,
    max_new_tokens: int,
    dry_run: bool = False,
    manifest_path: str | None = None,
    command: str = "run_local_hf_smoke",
) -> tuple[list[dict[str, Any]], Path]:
    """Run or dry-run local HF smoke and write outputs."""

    validate_limit(limit)
    validate_max_new_tokens(max_new_tokens)
    model_id = resolve_model_id(model_alias)
    run_id = "phase4-hf-local-smoke-dry-run" if dry_run else "phase4-hf-local-smoke"
    start_time = utc_now()
    items = load_smoke_items(input_path, limit)
    if dry_run:
        rows = [
            dry_run_result(
                item=item,
                run_id=run_id,
                model_alias=model_alias,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )
            for item in items
        ]
    else:
        rows = run_real_local_hf(
            items=items,
            run_id=run_id,
            model_alias=model_alias,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
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
    """Run the local HF smoke CLI."""

    args = build_parser().parse_args(argv)
    command = " ".join([Path(sys.executable).name, *sys.argv])
    try:
        rows, manifest = run_smoke(
            input_path=args.input_path,
            output_path=args.output_path,
            model_alias=args.model_alias,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens,
            dry_run=args.dry_run,
            manifest_path=args.manifest_path,
            command=command,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Local HF smoke failed: {exc}", file=sys.stderr)
        return 1

    success_count = sum(1 for row in rows if bool(row["success"]))
    print(f"Rows written: {len(rows)}")
    print(f"Successful generations: {success_count}")
    print(f"Output path: {args.output_path}")
    print(f"Run manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
