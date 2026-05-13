from pathlib import Path


def test_vllm_client_workflow_files_exist() -> None:
    assert Path("scripts/run_vllm_smoke_client.ps1").exists()
    assert Path("scripts/run_vllm_expanded_baseline_client.ps1").exists()
    assert Path("configs/vllm_baseline_experiments.yaml").exists()


def test_vllm_client_workflow_scripts_reference_expected_commands() -> None:
    smoke_script = Path("scripts/run_vllm_smoke_client.ps1").read_text(encoding="utf-8")
    expanded_script = Path("scripts/run_vllm_expanded_baseline_client.ps1").read_text(
        encoding="utf-8"
    )

    assert "openai-compatible-run" in smoke_script
    assert "vllm_short_chat_results.csv" in expanded_script
    assert "vllm_workload_comparison.csv" in expanded_script
    assert "score-structured-jsonl" in expanded_script


def test_vllm_planned_commands_references_client_workflow_scripts() -> None:
    planned_commands = Path("scripts/vllm_planned_commands.md").read_text(encoding="utf-8")

    assert "scripts/run_vllm_smoke_client.ps1" in planned_commands
    assert "scripts/run_vllm_expanded_baseline_client.ps1" in planned_commands
