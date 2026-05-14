from pathlib import Path


def test_scaled_workload_generation_doc_exists_and_contains_key_terms() -> None:
    doc_path = Path("docs/14_scaled_workload_generation.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    assert "1,000 prompts" in content
    assert "5,000 prompts" in content
    assert "10,000 prompts" in content
    assert "seed" in content
    assert "short_chat" in content
    assert "structured_output" in content


def test_readme_references_scaled_workload_generation_doc() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/14_scaled_workload_generation.md" in readme
