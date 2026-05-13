from pathlib import Path


def test_sample_artifact_policy_files_exist() -> None:
    assert Path("results/samples/README.md").exists()
    assert Path("docs/06_result_promotion_policy.md").exists()
    assert Path("scripts/promote_sample_artifacts.ps1").exists()


def test_gitignore_contains_results_samples_exceptions() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "!results/samples/" in gitignore
    assert "!results/samples/.gitkeep" in gitignore
    assert "!results/samples/README.md" in gitignore
    assert "!results/samples/raw/" in gitignore
    assert "!results/samples/raw/.gitkeep" in gitignore
    assert "!results/samples/figures/" in gitignore
    assert "!results/samples/figures/.gitkeep" in gitignore


def test_promotion_script_references_expanded_hf_artifacts() -> None:
    script = Path("scripts/promote_sample_artifacts.ps1").read_text(encoding="utf-8")

    assert "hf_short_chat_results.csv" in script
    assert "hf_code_helpdesk_results.csv" in script
    assert "hf_long_context_results.csv" in script
    assert "hf_shared_prefix_results.csv" in script
    assert "hf_workload_comparison.csv" in script
