from pathlib import Path


def test_phase2_master_plan_doc_contains_required_terms() -> None:
    doc_path = Path("docs/27_phase2_master_plan.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Phase 2 Master Plan",
        "Phase 2A",
        "Phase 2B",
        "Phase 2C",
        "Finance",
        "Airline",
        "Retail",
        "AI Research",
        "Healthcare Administrative",
        "context engineering",
        "no_context",
        "bm25",
        "dense",
        "hybrid",
        "RunPod",
        "learned router",
    ]

    for term in required_terms:
        assert term in content
