from pathlib import Path


def test_phase1_experiment_inventory_doc_contains_required_terms() -> None:
    doc_path = Path("docs/19_phase1_experiment_inventory.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    assert "Phase 1 Experiment Inventory" in content
    assert "TTFT" in content
    assert "TPOT" in content
    assert "vLLM" in content
    assert "Hugging Face" in content
    assert "concurrency" in content
    assert "checkpoint" in content
    assert "synthetic workload" in content
    assert "correctness evaluation" in content
    assert "real-world data" in content
    assert "5,000-prompt" in content
