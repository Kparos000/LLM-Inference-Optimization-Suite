from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_GITIGNORE_PATTERNS = {
    "pytest_run_tmp_*/",
    "pytest_tmp/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".mypy_cache/",
    ".tmp/",
    ".tmp_pytest/",
    "*.log",
    "*.log.*",
    "logs/",
    ".venv/",
    ".venv*/",
    "venv/",
    "venv*/",
    "env/",
    "env*/",
    "backups/",
    "results/raw/*",
    "results/processed/*",
    "results/raw/**/*api*",
    "results/processed/**/*api*",
}

PROJECT_TEMP_DIRS = (
    "pytest_tmp",
    ".tmp",
    ".tmp_pytest",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines()]


def test_gitignore_covers_local_temp_and_generated_paid_api_artifacts() -> None:
    ignored = set(REPO_ROOT.joinpath(".gitignore").read_text(encoding="utf-8").splitlines())

    missing = sorted(REQUIRED_GITIGNORE_PATTERNS - ignored)
    assert missing == []


def test_local_project_temp_run_folders_are_absent() -> None:
    for dirname in PROJECT_TEMP_DIRS:
        assert not REPO_ROOT.joinpath(dirname).exists()
    assert list(REPO_ROOT.glob("pytest_run_tmp_*")) == []


def test_temp_and_cache_artifacts_are_not_tracked() -> None:
    disallowed_prefixes = (
        "pytest_run_tmp_",
        "pytest_tmp/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".mypy_cache/",
        ".tmp/",
        ".tmp_pytest/",
    )

    tracked = _tracked_files()
    offenders = [
        path
        for path in tracked
        if path.startswith(disallowed_prefixes) or "/pytest_run_tmp_" in path
    ]
    assert offenders == []


def test_large_raw_result_artifacts_are_not_tracked_outside_samples() -> None:
    allowed_raw_or_figure_files = {
        "results/README.md",
        "results/raw/.gitkeep",
        "results/raw/README.md",
        "results/figures/.gitkeep",
        "results/figures/README.md",
    }
    tracked = _tracked_files()
    raw_or_figure_offenders = [
        path
        for path in tracked
        if (
            path.startswith(("results/raw/", "results/figures/"))
            and path not in allowed_raw_or_figure_files
            and not path.startswith("results/samples/")
        )
    ]
    large_processed_offenders = [
        path
        for path in tracked
        if path.startswith("results/processed/")
        and path not in {"results/processed/.gitkeep", "results/processed/README.md"}
        and REPO_ROOT.joinpath(path).stat().st_size > 1_000_000
    ]

    assert raw_or_figure_offenders == []
    assert large_processed_offenders == []
