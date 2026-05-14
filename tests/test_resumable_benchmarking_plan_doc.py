from pathlib import Path


def test_resumable_benchmarking_plan_doc_exists_and_contains_key_terms() -> None:
    doc_path = Path("docs/15_resumable_benchmarking_plan.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    assert "chunk" in content
    assert "checkpoint" in content
    assert "resume" in content
    assert "RunPod network volume" in content
    assert "100 prompts" in content
    assert "progress" in content


def test_readme_references_resumable_benchmarking_plan_doc() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docs/15_resumable_benchmarking_plan.md" in readme
