from pathlib import Path


def test_experiment_log_doc_exists_and_contains_key_terms() -> None:
    experiment_log = Path("docs/12_experiment_log.md")

    assert experiment_log.exists()

    content = experiment_log.read_text(encoding="utf-8")
    assert "vLLM RunPod L40S baseline calibration" in content
    assert "TTFT" in content
    assert "TPOT" in content
    assert "structured_output_smoke" in content
    assert "quality" in content
    assert "concurrency" in content
