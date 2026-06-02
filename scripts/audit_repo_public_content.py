"""Audit committed text content for public-facing repository concerns."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "raw",
    "processed",
    "figures",
    "models",
    ".cache",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".csv",
    ".toml",
    ".ps1",
    ".sh",
    ".txt",
}

TEXT_FILENAMES = {".gitignore", ".env.example"}


@dataclass(frozen=True)
class AuditFinding:
    """A possible public-content issue found in a committed text file."""

    path: Path
    line_number: int
    concern: str


def _phrase(*parts: str) -> str:
    return "".join(parts)


PROHIBITED_PHRASES = {
    _phrase("explain like I am in 6th", " grade"): "tutorial-style language",
    _phrase("explain to a 6th", " grader"): "tutorial-style language",
    _phrase("beginner", " tutorial"): "tutorial-style language",
    _phrase("learning", " diary"): "informal public-facing wording",
    _phrase("LinkedIn", " draft"): "private draft language",
    _phrase("Twitter", " draft"): "private draft language",
    _phrase("X thread", " draft"): "private draft language",
    _phrase("resume", " bullet"): "private career-note language",
    _phrase("personal learning", " note"): "private note language",
    _phrase("claims", " policy"): "defensive public-facing wording",
    _phrase("not allowed", " claim"): "defensive public-facing wording",
    _phrase("do not over", "claim"): "defensive public-facing wording",
    _phrase("PASTE_", "YOUR_TOKEN"): "unsafe token placeholder",
}

TOKEN_ASSIGNMENT_RE = re.compile(
    r"^\s*(HF_TOKEN|HUGGINGFACE_HUB_TOKEN)\s*[:=]\s*(?P<value>.+\S)\s*$",
)
HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
OPENAI_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
WINDOWS_USER_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\")


def _is_skipped_dir(path: Path) -> bool:
    path_parts = path.parts
    for index, part in enumerate(path_parts[:-1]):
        if part == "results" and path_parts[index + 1] == "samples":
            return False
    if path.name in {"raw", "processed", "figures"} and path.parent.name == "results":
        return True
    return path.name in SKIP_DIRS


def _is_text_like(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES


def _iter_tracked_files(repo_root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=False,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [path.relative_to(repo_root) for path in repo_root.rglob("*") if path.is_file()]

    return [Path(raw_path.decode("utf-8")) for raw_path in result.stdout.split(b"\0") if raw_path]


def _iter_text_files(repo_root: Path) -> list[Path]:
    text_files: list[Path] = []
    for relative_path in _iter_tracked_files(repo_root):
        path = repo_root / relative_path
        if any(_is_skipped_dir(parent) for parent in relative_path.parents):
            continue
        if _is_text_like(path):
            text_files.append(path)
    return sorted(text_files)


def _scan_line(path: Path, line_number: int, line: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    stripped_line = line.strip()
    lower_line = line.lower()

    token_assignment = TOKEN_ASSIGNMENT_RE.match(stripped_line)
    if token_assignment is not None:
        value = token_assignment.group("value").strip()
        if value:
            findings.append(AuditFinding(path, line_number, "non-empty Hugging Face token value"))

    if HF_TOKEN_RE.search(line):
        findings.append(AuditFinding(path, line_number, "Hugging Face token-looking string"))

    if OPENAI_TOKEN_RE.search(line):
        findings.append(AuditFinding(path, line_number, "OpenAI token-looking string"))

    if WINDOWS_USER_PATH_RE.search(line):
        findings.append(AuditFinding(path, line_number, "Windows local absolute path"))

    for phrase, concern in PROHIBITED_PHRASES.items():
        if phrase.lower() in lower_line:
            findings.append(AuditFinding(path, line_number, concern))

    return findings


def audit_repository(repo_root: Path) -> list[AuditFinding]:
    """Return public-content audit findings for text-like committed files."""

    findings: list[AuditFinding] = []
    for path in _iter_text_files(repo_root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        relative_path = path.relative_to(repo_root)
        for line_number, line in enumerate(lines, start=1):
            findings.extend(_scan_line(relative_path, line_number, line))

    return findings


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    findings = audit_repository(repo_root)

    if not findings:
        print("Repository public-content audit passed.")
        return 0

    print("Repository public-content audit found possible issues:")
    for finding in findings:
        print(f"- {finding.path}:{finding.line_number}: {finding.concern}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
