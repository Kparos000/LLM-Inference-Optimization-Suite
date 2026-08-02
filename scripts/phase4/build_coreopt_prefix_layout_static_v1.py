"""Build static artifacts for coreopt_prefix_layout_static_v1."""

from __future__ import annotations

import json

from inference_bench.coreopt_prefix_layout_static import (
    write_coreopt_prefix_layout_static_artifacts,
)


def main() -> None:
    result = write_coreopt_prefix_layout_static_artifacts(update_registry=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
