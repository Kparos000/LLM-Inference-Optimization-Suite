"""Audit model5 gated access, provider capabilities, and pricing route.

This command uses metadata and access GET requests only. It never sends a model
generation request.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    HF_ROUTER_MODEL_URL,
    audit_model5_route,
    request_router_metadata,
    write_model5_route_audit,
)


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
    report["router_metadata_url"] = HF_ROUTER_MODEL_URL.format(model_id=model.model_id)
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
