from pathlib import Path


def test_benchmark_methodology_doc_exists_and_contains_key_terms() -> None:
    methodology_path = Path("docs/08_benchmark_methodology.md")

    assert methodology_path.exists()

    content = methodology_path.read_text(encoding="utf-8")
    assert "TTFT" in content
    assert "TPOT" in content
    assert "vLLM" in content
    assert "Calibration baseline" in content
    assert "concurrency" in content
