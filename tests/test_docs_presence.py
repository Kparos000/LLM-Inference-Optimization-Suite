from pathlib import Path


def test_key_documentation_files_exist() -> None:
    expected_paths = [
        Path("docs/00_project_scope.md"),
        Path("docs/01_reproducibility.md"),
        Path("docs/02_dry_run_plan.md"),
        Path("docs/03_decision_log.md"),
        Path("docs/04_publication_notes.md"),
    ]

    for path in expected_paths:
        assert path.exists(), f"Missing documentation file: {path}"
