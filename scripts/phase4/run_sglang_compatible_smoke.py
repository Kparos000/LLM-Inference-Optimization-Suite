"""Run or dry-run a tiny SGLang OpenAI-compatible smoke.

This scaffold delegates request and output handling to the established
OpenAI-compatible Phase 4 smoke path. It does not require SGLang to be
installed for dry-run tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENAI_SMOKE_PATH = REPO_ROOT / "scripts/phase4/run_openai_compatible_smoke.py"
DEFAULT_BASE_URL = "http://localhost:30000/v1"
DEFAULT_API_KEY = "EMPTY"


def _load_openai_smoke_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_phase4_openai_compatible_smoke",
        OPENAI_SMOKE_PATH,
    )
    if spec is None or spec.loader is None:
        msg = f"Unable to load OpenAI-compatible smoke module from {OPENAI_SMOKE_PATH}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    """Build the SGLang-compatible CLI parser."""

    parser = argparse.ArgumentParser(description="Run a tiny SGLang OpenAI-compatible smoke test.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-alias", default="model1_0_5b")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate workload and result plumbing without contacting SGLang.",
    )
    return parser


def validate_sglang_result_row(row: dict[str, Any]) -> None:
    """Validate SGLang-specific identity on the shared smoke result schema."""

    required_fields = {
        "run_id",
        "timestamp_utc",
        "backend",
        "optimization",
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
        msg = f"SGLang smoke result row missing required fields: {missing}"
        raise ValueError(msg)
    if row["backend"] != "sglang_openai_compatible":
        msg = "backend must be sglang_openai_compatible"
        raise ValueError(msg)
    if bool(row["paid_api_call_triggered"]):
        msg = "SGLang local smoke must not trigger paid API calls"
        raise ValueError(msg)


def _rewrite_manifest(path: Path) -> None:
    if not path.is_file():
        return
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return
    loaded["backend"] = "sglang_openai_compatible"
    run_id = str(loaded.get("run_id") or "")
    loaded["run_id"] = run_id.replace("openai-compatible", "sglang-compatible")
    path.write_text(
        json.dumps(loaded, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    dry_run: bool,
    manifest_path: str | None = None,
    command: str = "run_sglang_compatible_smoke",
) -> tuple[list[dict[str, Any]], Path]:
    """Delegate to the shared OpenAI-compatible path and label SGLang rows."""

    module = _load_openai_smoke_module()
    resolved_manifest_path = manifest_path or str(
        Path(output_path).with_name("phase4_sglang_compatible_smoke_manifest.json")
    )
    try:
        rows, written_manifest = module.run_smoke(
            input_path=input_path,
            output_path=output_path,
            model_alias=model_alias,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            limit=limit,
            max_new_tokens=max_new_tokens,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            manifest_path=resolved_manifest_path,
            command=command,
        )
    except RuntimeError as exc:
        if "not reachable" in str(exc):
            msg = (
                f"SGLang OpenAI-compatible server is unavailable at {base_url}. "
                "Start SGLang with its OpenAI-compatible API, then rerun without --dry-run. "
                f"Underlying error: {exc}"
            )
            raise RuntimeError(msg) from exc
        raise

    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        row["backend"] = "sglang_openai_compatible"
        row["optimization"] = "sglang_openai_compatible_smoke"
        row["server_type"] = "sglang"
        row["run_id"] = str(row["run_id"]).replace(
            "openai-compatible",
            "sglang-compatible",
        )
        validate_sglang_result_row(row)
        normalized_rows.append(row)
    module.write_jsonl_rows(normalized_rows, output_path)
    _rewrite_manifest(Path(written_manifest))
    return normalized_rows, Path(written_manifest)


def _sanitized_command(argv: list[str]) -> str:
    sanitized: list[str] = []
    hide_next = False
    for argument in argv:
        if hide_next:
            sanitized.append("***")
            hide_next = False
            continue
        sanitized.append(argument)
        if argument == "--api-key":
            hide_next = True
    return " ".join([Path(sys.executable).name, *sanitized])


def main(argv: list[str] | None = None) -> int:
    """Run the SGLang-compatible smoke CLI."""

    args = build_parser().parse_args(argv)
    command = _sanitized_command(sys.argv)
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
        print(f"SGLang-compatible smoke failed: {exc}", file=sys.stderr)
        return 1

    print(f"Rows written: {len(rows)}")
    print(f"Successful generations: {sum(1 for row in rows if bool(row['success']))}")
    print(f"Output path: {args.output_path}")
    print(f"Run manifest: {manifest}")
    if args.dry_run:
        print("Dry-run mode: SGLang server checks and model calls were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
