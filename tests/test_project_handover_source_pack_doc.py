from pathlib import Path


def test_project_handover_source_pack_doc_contains_required_terms() -> None:
    doc_path = Path("docs/24_project_handover_source_pack.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Project Handover Source Pack",
        "Phase 1",
        "Phase 2",
        "vLLM",
        "Hugging Face",
        "TTFT",
        "TPOT",
        "75,000",
        "RAG",
        "groundedness",
        "knowledge base",
        "real-world data",
        "RunPod",
        "learned router",
    ]

    for term in required_terms:
        assert term in content
