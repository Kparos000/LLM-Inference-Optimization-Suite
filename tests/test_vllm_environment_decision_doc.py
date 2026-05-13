from pathlib import Path


def test_vllm_environment_decision_doc_exists_and_contains_key_terms() -> None:
    decision_path = Path("docs/10_vllm_environment_decision.md")

    assert decision_path.exists()

    content = decision_path.read_text(encoding="utf-8")
    assert "Linux cloud GPU" in content
    assert "WSL2" in content
    assert "Windows" in content
    assert "paid GPU" in content
    assert "Qwen/Qwen2.5-7B-Instruct" in content
