from pathlib import Path


def test_vllm_plan_docs_exist() -> None:
    assert Path("docs/07_vllm_baseline_plan.md").exists()
    assert Path("scripts/vllm_planned_commands.md").exists()
