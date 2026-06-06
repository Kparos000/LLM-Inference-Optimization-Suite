"""Audit model5 gated access, provider capabilities, and pricing route.

This command uses metadata and access GET requests only. It never sends a model
generation request.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.api_priced_validation import (  # noqa: E402
    check_hf_token,
    check_model_access,
)
from inference_bench.config import load_project_config  # noqa: E402
from inference_bench.env import load_local_env  # noqa: E402
from inference_bench.model5_pricing_routing import (  # noqa: E402
    audit_model5_route,
    write_model5_route_audit,
)

ROUTER_MODEL_URL = "https://router.huggingface.co/v1/models/{model_id}"


def request_router_metadata(model_id: str) -> dict[str, Any]:
    """Fetch public Hugging Face router metadata for one model."""

    url = ROUTER_MODEL_URL.format(model_id=model_id)
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        msg = f"Router metadata request failed with HTTP {exc.code}"
        raise RuntimeError(msg) from exc
    except URLError as exc:
        msg = f"Router metadata request failed: {exc.reason}"
        raise RuntimeError(msg) from exc
    if not isinstance(payload, dict):
        msg = "Router metadata response must be a JSON object"
        raise RuntimeError(msg)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the model5 route audit CLI."""

    parser = argparse.ArgumentParser(
        description=("Audit model5 gated access, provider support, and pricing without generation.")
    )
    parser.add_argument("--model-alias", default="model5_gated")
    parser.add_argument("--pricing-config", default="configs/api_pricing.yaml")
    parser.add_argument("--output-root", default="results/processed")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the model5 pricing route audit."""

    args = build_parser().parse_args(argv)
    load_local_env()
    config = load_project_config()
    model = config.resolve_model_config(args.model_alias)
    report = audit_model5_route(
        model_id=model.model_id,
        pricing_config=args.pricing_config,
        hf_token=os.environ.get("HF_TOKEN", ""),
        model_alias=args.model_alias,
        token_checker=check_hf_token,
        access_checker=check_model_access,
        metadata_fetcher=request_router_metadata,
    )
    report["router_metadata_url"] = ROUTER_MODEL_URL.format(model_id=model.model_id)
    report_path, summary_path = write_model5_route_audit(report, args.output_root)
    print(
        f"{report['model_alias']}: access={report['model_access_granted']} "
        f"provider={report['selected_provider'] or 'none'} "
        f"pricing={report['resolved_pricing_status'] or 'unavailable'} "
        f"costed_smoke_allowed={report['costed_smoke_allowed']}"
    )
    print(f"Route report: {report_path}")
    print(f"Route summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
