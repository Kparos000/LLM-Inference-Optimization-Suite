"""Run Deployability_Repair_Validation_V1 without live inference."""

from __future__ import annotations

import json

from inference_bench.deployability_repair_validation import (
    run_deployability_repair_validation,
)


def main() -> None:
    summary = run_deployability_repair_validation()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
