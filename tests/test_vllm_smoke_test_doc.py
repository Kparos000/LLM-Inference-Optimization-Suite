from pathlib import Path


def test_vllm_smoke_test_doc_exists_and_contains_key_commands() -> None:
    smoke_doc = Path("docs/11_vllm_smoke_test.md")

    assert smoke_doc.exists()

    content = smoke_doc.read_text(encoding="utf-8")
    assert "vllm serve" in content
    assert "openai-compatible-run" in content
    assert "curl http://localhost:8000/v1/models" in content
    assert "Qwen/Qwen2.5-0.5B-Instruct" in content


def test_vllm_planned_commands_use_vllm_serve() -> None:
    content = Path("scripts/vllm_planned_commands.md").read_text(encoding="utf-8")

    assert "vllm serve" in content
