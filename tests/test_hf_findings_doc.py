from pathlib import Path


def test_hf_baseline_findings_doc_exists_and_contains_key_terms() -> None:
    findings_path = Path("docs/16_hf_baseline_findings.md")

    assert findings_path.exists()

    content = findings_path.read_text(encoding="utf-8")
    assert "long_context" in content
    assert "TTFT" in content
    assert "TPOT" in content
    assert "vLLM" in content
