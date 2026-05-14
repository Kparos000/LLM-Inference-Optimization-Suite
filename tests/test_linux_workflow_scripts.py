from pathlib import Path


def test_linux_vllm_workflow_scripts_exist() -> None:
    assert Path("scripts/run_vllm_smoke_client.sh").exists()
    assert Path("scripts/run_vllm_expanded_baseline_client.sh").exists()
    assert Path("scripts/promote_sample_artifacts.sh").exists()


def test_linux_vllm_workflow_scripts_contain_expected_commands() -> None:
    smoke_script = Path("scripts/run_vllm_smoke_client.sh").read_text(encoding="utf-8")
    expanded_script = Path("scripts/run_vllm_expanded_baseline_client.sh").read_text(
        encoding="utf-8"
    )
    promote_script = Path("scripts/promote_sample_artifacts.sh").read_text(encoding="utf-8")

    assert "openai-compatible-run" in smoke_script
    assert "vllm_workload_comparison.csv" in expanded_script
    assert "vllm_workload_comparison_sample.csv" in promote_script


def test_planned_commands_doc_references_linux_workflow_scripts() -> None:
    planned_commands = Path("scripts/vllm_planned_commands.md").read_text(encoding="utf-8")

    assert "bash scripts/run_vllm_smoke_client.sh" in planned_commands
    assert "bash scripts/run_vllm_expanded_baseline_client.sh" in planned_commands
    assert "bash scripts/promote_sample_artifacts.sh" in planned_commands
