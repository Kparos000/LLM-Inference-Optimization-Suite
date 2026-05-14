import json
from pathlib import Path

from inference_bench.workloads.scaled_generator import (
    DEFAULT_SCALED_WORKLOADS,
    SHARED_IT_SUPPORT_PREFIX,
    generate_scaled_workloads,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_scaled_workload_generation_writes_expected_files_and_rows(tmp_path: Path) -> None:
    written_paths = generate_scaled_workloads(output_dir=tmp_path, count=10, seed=123)

    assert len(written_paths) == len(DEFAULT_SCALED_WORKLOADS)

    for workload_name in DEFAULT_SCALED_WORKLOADS:
        output_path = tmp_path / f"{workload_name}_10.jsonl"
        assert output_path.exists()

        rows = _read_jsonl(output_path)
        assert len(rows) == 10

        for row in rows:
            assert row["prompt_id"]
            assert row["workload_name"] == workload_name
            assert row["prompt"]
            assert isinstance(row["metadata"], dict)


def test_scaled_workload_generation_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    generate_scaled_workloads(output_dir=first_dir, count=10, seed=42)
    generate_scaled_workloads(output_dir=second_dir, count=10, seed=42)

    for workload_name in DEFAULT_SCALED_WORKLOADS:
        first_content = (first_dir / f"{workload_name}_10.jsonl").read_text(encoding="utf-8")
        second_content = (second_dir / f"{workload_name}_10.jsonl").read_text(encoding="utf-8")
        assert first_content == second_content


def test_structured_output_prompts_request_valid_json(tmp_path: Path) -> None:
    generate_scaled_workloads(
        output_dir=tmp_path,
        count=10,
        seed=42,
        workloads=["structured_output"],
    )

    rows = _read_jsonl(tmp_path / "structured_output_10.jsonl")

    assert all("valid JSON" in str(row["prompt"]) for row in rows)


def test_shared_prefix_prompts_contain_repeated_it_support_prefix(tmp_path: Path) -> None:
    generate_scaled_workloads(
        output_dir=tmp_path,
        count=10,
        seed=42,
        workloads=["shared_prefix"],
    )

    rows = _read_jsonl(tmp_path / "shared_prefix_10.jsonl")

    assert all(SHARED_IT_SUPPORT_PREFIX in str(row["prompt"]) for row in rows)
