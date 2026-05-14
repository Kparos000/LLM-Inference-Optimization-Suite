from pathlib import Path


def test_hf_vs_vllm_comparison_doc_exists_and_contains_key_terms() -> None:
    comparison_doc = Path("docs/13_hf_vs_vllm_calibration_comparison.md")

    assert comparison_doc.exists()

    content = comparison_doc.read_text(encoding="utf-8")
    assert "Hugging Face" in content
    assert "vLLM" in content
    assert "calibration" in content
    assert "TPOT" in content
    assert "RunPod L40S" in content
    assert "Limitations" in content
