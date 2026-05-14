import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
OPENAI_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
TUTORIAL_PHRASES = (
    "explain to a 6th" + " grader",
    "explain like I am in 6th" + " grade",
)
PRIVATE_DRAFT_PHRASES = (
    "LinkedIn" + " draft",
    "Twitter" + " draft",
    "X thread" + " draft",
    "resume" + " bullet",
    "personal learning" + " note",
)


def _load_audit_module() -> ModuleType:
    module_path = Path("scripts/audit_repo_public_content.py")
    spec = importlib.util.spec_from_file_location("audit_repo_public_content", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo_text_files() -> list[Path]:
    excluded_parts = {
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
    }
    text_suffixes = {
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
    text_names = {".gitignore", ".env.example"}
    files: list[Path] = []
    for path in Path(".").rglob("*"):
        if not path.is_file():
            continue
        if any(part in excluded_parts for part in path.parts):
            if not path.as_posix().startswith("results/samples/"):
                continue
        if path.suffix in text_suffixes or path.name in text_names:
            files.append(path)
    return files


def test_public_content_audit_script_exists() -> None:
    assert Path("scripts/audit_repo_public_content.py").exists()


def test_env_example_contains_blank_placeholders_only() -> None:
    env_example = Path(".env.example")

    assert env_example.exists()
    assert env_example.read_text(encoding="utf-8").splitlines() == [
        "HF_TOKEN=",
        "HUGGINGFACE_HUB_TOKEN=",
    ]


def test_committed_text_files_do_not_contain_obvious_token_values() -> None:
    for path in _repo_text_files():
        content = path.read_text(encoding="utf-8")
        assert TOKEN_RE.search(content) is None
        assert OPENAI_TOKEN_RE.search(content) is None


def test_committed_docs_do_not_contain_tutorial_style_phrases() -> None:
    for path in Path("docs").glob("*.md"):
        content = path.read_text(encoding="utf-8")
        for phrase in TUTORIAL_PHRASES:
            assert phrase not in content


def test_public_content_audit_flags_private_draft_phrases() -> None:
    audit_module = _load_audit_module()

    for phrase in (*TUTORIAL_PHRASES, *PRIVATE_DRAFT_PHRASES):
        findings = audit_module._scan_line(Path("docs/example.md"), 1, phrase)

        assert findings


def test_curated_samples_do_not_contain_token_references_or_values() -> None:
    sample_root = Path("results/samples")
    if not sample_root.exists():
        return

    for path in sample_root.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        assert "HF_TOKEN" not in content
        assert "HUGGINGFACE_HUB_TOKEN" not in content
        assert TOKEN_RE.search(content) is None
        assert OPENAI_TOKEN_RE.search(content) is None
