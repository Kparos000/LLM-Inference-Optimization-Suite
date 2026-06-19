from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

EXPECTED_COMMAND_ORDER = [
    "pytest tests/test_config_validation.py",
    "pytest tests/test_repo_hygiene.py",
    "pytest tests/test_ci_config_audit.py",
    "mypy src tests",
    "pytest",
    "ruff check .",
    "ruff format --check .",
    "python scripts/audit_repo_public_content.py",
    "inference-bench doctor",
    "inference-bench validate-config",
]


def test_ci_runs_local_validation_commands_in_requested_order() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    command_lines = [line.strip() for line in workflow.splitlines()]

    positions = []
    for command in EXPECTED_COMMAND_ORDER:
        matching_positions = [
            index for index, command_line in enumerate(command_lines) if command_line == command
        ]
        assert matching_positions, command
        position = matching_positions[0]
        positions.append(position)
    assert positions == sorted(positions)


def test_ci_uses_cross_platform_repository_relative_commands() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "C:\\" not in workflow
    assert "\\scripts\\" not in workflow
    assert "\\tests\\" not in workflow
    assert 'python-version: "3.10"' in workflow


def test_pyproject_keeps_test_and_type_check_entrypoints_stable() -> None:
    pyproject = REPO_ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8")

    assert 'testpaths = ["tests"]' in pyproject
    assert 'pythonpath = ["src"]' in pyproject
    assert "[tool.setuptools.packages.find]" in pyproject
    assert 'where = ["src"]' in pyproject
