from pathlib import Path


def test_phase2_data_strategy_doc_contains_required_terms() -> None:
    doc_path = Path("docs/29_phase2_data_strategy.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    required_terms = [
        "Phase 2 Data Strategy",
        "SEC",
        "Canada Air",
        "Amazon Reviews",
        "arXiv",
        "Healthcare Administrative",
        "source registry",
        "KB registry",
        "gold",
        "commit policy",
        "Phase 2A Completion Criteria",
    ]

    for term in required_terms:
        assert term in content
