from pathlib import Path

from inference_bench.workloads.loader import load_jsonl_workload


def test_expanded_workload_files_load_with_expected_counts() -> None:
    expected_workloads = {
        "short_chat": ("data/prompts/short_chat.jsonl", 5),
        "code_helpdesk": ("data/prompts/code_helpdesk.jsonl", 5),
        "long_context": ("data/prompts/long_context.jsonl", 3),
        "shared_prefix": ("data/prompts/shared_prefix.jsonl", 5),
    }

    for workload_name, (path_text, expected_count) in expected_workloads.items():
        path = Path(path_text)

        assert path.exists()
        items = load_jsonl_workload(path)

        assert len(items) == expected_count
        assert {item.workload_name for item in items} == {workload_name}


def test_shared_prefix_workload_records_mark_shared_prefix_metadata() -> None:
    items = load_jsonl_workload(Path("data/prompts/shared_prefix.jsonl"))

    assert all(item.metadata["shared_prefix"] == "true" for item in items)
