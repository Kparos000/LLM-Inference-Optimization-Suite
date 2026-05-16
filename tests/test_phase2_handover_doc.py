from pathlib import Path


def test_phase2_handover_doc_contains_required_terms() -> None:
    doc_path = Path("docs/28_project_handover_phase2.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Project Handover",
        "Phase 1",
        "Phase 2",
        "75,000",
        "SEC",
        "Canada Air",
        "Amazon Reviews",
        "AI Research",
        "Healthcare Administrative",
        "knowledge base",
        "gold",
        "context engineering",
        "RunPod",
    ]

    for term in required_terms:
        assert term in content
