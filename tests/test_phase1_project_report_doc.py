from pathlib import Path


def test_phase1_project_report_doc_contains_required_terms() -> None:
    doc_path = Path("docs/20_phase1_project_report.md")

    assert doc_path.exists()

    content = doc_path.read_text(encoding="utf-8")
    assert "Phase 1 Project Report" in content
    assert "TTFT" in content
    assert "TPOT" in content
    assert "vLLM" in content
    assert "Hugging Face" in content
    assert "RunPod" in content
    assert "concurrency" in content
    assert "checkpoint" in content
    assert "5,000-prompt" in content
    assert "correctness" in content
    assert "real-world data" in content
    assert "Phase 2" in content
    assert "bottleneck" in content
