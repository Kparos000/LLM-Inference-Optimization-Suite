from pathlib import Path

import pytest

from inference_bench.workloads.loader import load_jsonl_workload


def test_loads_sample_workload() -> None:
    items = load_jsonl_workload(Path("data/prompts/smoke_workload.jsonl"))

    assert len(items) == 3
    assert items[0].prompt_id == "smoke-chat-001"
    assert items[1].metadata["category"] == "code-helpdesk"


def test_loads_structured_output_smoke_workload() -> None:
    items = load_jsonl_workload(Path("data/prompts/structured_output_smoke.jsonl"))

    assert len(items) == 3
    assert items[0].workload_name == "structured_output_smoke"
    assert items[0].metadata["output_format"] == "json"
    assert items[0].metadata["required_fields"] == "category,answer,confidence"


def test_ignores_blank_lines_in_jsonl(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text(
        '\n{"prompt_id":"prompt-1","workload_name":"smoke","prompt":"Hello"}\n\n',
        encoding="utf-8",
    )

    items = load_jsonl_workload(workload_path)

    assert len(items) == 1
    assert items[0].prompt == "Hello"


def test_invalid_json_raises_value_error(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.jsonl"
    workload_path.write_text("{bad-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl_workload(workload_path)
