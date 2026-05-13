from pathlib import Path

import yaml  # type: ignore[import-untyped]


def test_scaled_benchmark_plan_files_exist() -> None:
    assert Path("docs/09_scaled_benchmark_plan.md").exists()
    assert Path("configs/stress_plan.yaml").exists()


def test_stress_plan_contains_expected_values() -> None:
    with Path("configs/stress_plan.yaml").open(encoding="utf-8") as file:
        stress_plan = yaml.safe_load(file)

    assert stress_plan["concurrency_levels"] == [1, 4, 8, 16, 32]
    assert stress_plan["workload_scales"]["small"] == 50
    assert "vllm" in stress_plan["planned_backends"]


def test_scaled_benchmark_plan_contains_key_terms() -> None:
    content = Path("docs/09_scaled_benchmark_plan.md").read_text(encoding="utf-8")

    assert "50 prompts" in content
    assert "concurrency" in content
    assert "vLLM" in content
    assert "p95" in content
