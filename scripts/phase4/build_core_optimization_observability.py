"""Generate core optimization observability planning artifacts.

This script is CPU-only. It builds registry, readiness, inventory, event-schema,
and static prefix-opportunity artifacts without running inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_bench.core_optimization_observability import (  # noqa: E402
    write_core_optimization_observability_artifacts,
)


def main() -> None:
    paths = write_core_optimization_observability_artifacts()
    print(json.dumps(paths, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
