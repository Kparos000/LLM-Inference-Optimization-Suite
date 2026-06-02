"""Tiny Hugging Face Inference Provider smoke test.

This script is intentionally hard to run accidentally:
- requires HF_TOKEN,
- requires --allow-paid-api-call,
- caps max_new_tokens at 32,
- refuses to run without a pricing snapshot.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_pricing import (  # noqa: E402
    estimate_api_cost_from_pricing,
    resolve_api_pricing,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.runners.mock_runner import count_whitespace_tokens  # noqa: E402

HF_ROUTER_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


def utc_now_compact() -> str:
    """Return a compact UTC timestamp for output file names."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def request_chat_completion(
    *,
    hf_token: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], float]:
    """Call the Hugging Face router chat completion API once."""

    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    request = Request(
        HF_ROUTER_CHAT_COMPLETIONS_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        msg = f"HTTP {exc.code}: {body_text[:500]}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Network error: {exc.reason}"
        raise RuntimeError(msg) from exc
    latency_ms = (time.perf_counter() - started) * 1000
    if not isinstance(payload, dict):
        msg = "Unexpected non-object response from Hugging Face router"
        raise RuntimeError(msg)
    return payload, latency_ms


def extract_generated_text(response_payload: dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible chat completion response."""

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def usage_tokens(
    response_payload: dict[str, Any], prompt: str, generated_text: str
) -> tuple[int, int]:
    """Return input/output token counts from response usage or local approximation."""

    usage = response_payload.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            return max(0, prompt_tokens), max(0, completion_tokens)
    return count_whitespace_tokens(prompt), count_whitespace_tokens(generated_text)


def write_outputs(output_dir: str | Path, row: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and CSV smoke results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_compact()
    json_path = output_path / f"hf_api_tiny_smoke_{timestamp}.json"
    csv_path = output_path / f"hf_api_tiny_smoke_{timestamp}.csv"
    json_path.write_text(
        json.dumps(row, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return json_path, csv_path


def run_tiny_smoke(
    *,
    model_alias: str,
    prompt: str,
    max_new_tokens: int,
    allow_paid_api_call: bool,
    pricing_config: str | Path,
    output_dir: str | Path,
) -> int:
    """Run one tiny gated API smoke request."""

    if not allow_paid_api_call:
        print("Refusing to run: pass --allow-paid-api-call to permit one paid API request.")
        return 1
    if max_new_tokens <= 0 or max_new_tokens > 32:
        print("Refusing to run: max_new_tokens must be between 1 and 32.")
        return 1
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Refusing to run: HF_TOKEN is required but was not found.")
        return 1

    project_config = load_project_config()
    model_config = project_config.resolve_model_config(model_alias)
    pricing = resolve_api_pricing(model_alias, pricing_config)
    input_token_estimate = count_whitespace_tokens(prompt)
    max_cost = estimate_api_cost_from_pricing(
        input_tokens=input_token_estimate,
        output_tokens=max_new_tokens,
        pricing=pricing,
    )
    print(
        "Estimated maximum API token cost: "
        f"${max_cost['total_api_cost_usd']:.8f} "
        f"for {input_token_estimate} input tokens and <= {max_new_tokens} output tokens."
    )

    row: dict[str, Any] = {
        "model_alias": model_alias,
        "model_id": model_config.model_id,
        "provider": pricing.provider,
        "input_tokens": input_token_estimate,
        "output_tokens": 0,
        "input_cost_usd": max_cost["input_cost_usd"],
        "output_cost_usd": 0.0,
        "total_cost_usd": max_cost["input_cost_usd"],
        "latency_ms": None,
        "success": False,
        "error_type": "",
    }
    try:
        payload, latency_ms = request_chat_completion(
            hf_token=hf_token,
            model_id=model_config.model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        generated_text = extract_generated_text(payload)
        input_tokens, output_tokens = usage_tokens(payload, prompt, generated_text)
        actual_cost = estimate_api_cost_from_pricing(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        )
        row.update(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": actual_cost["input_cost_usd"],
                "output_cost_usd": actual_cost["output_cost_usd"],
                "total_cost_usd": actual_cost["total_api_cost_usd"],
                "latency_ms": round(latency_ms, 6),
                "success": True,
                "error_type": "",
            }
        )
    except Exception as exc:
        row["error_type"] = exc.__class__.__name__

    json_path, csv_path = write_outputs(output_dir, row)
    print(f"Wrote smoke result JSON: {json_path}")
    print(f"Wrote smoke result CSV: {csv_path}")
    return 0 if row["success"] else 1


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run one tiny Hugging Face Inference Provider API smoke test."
    )
    parser.add_argument("--model", default="model5_gated")
    parser.add_argument(
        "--prompt",
        default="Answer in one short sentence: what is API token cost?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--output-dir", default="results/raw/hf_api_tiny_smoke")
    parser.add_argument("--allow-paid-api-call", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run CLI."""

    args = build_parser().parse_args(argv)
    return run_tiny_smoke(
        model_alias=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        allow_paid_api_call=args.allow_paid_api_call,
        pricing_config=args.pricing_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
