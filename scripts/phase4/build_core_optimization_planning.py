"""Build core optimization planning artifacts without running inference."""

from __future__ import annotations

from inference_bench.core_optimization_planning import (
    write_core_optimization_planning_artifacts,
)


def main() -> None:
    outputs = write_core_optimization_planning_artifacts()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
