from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from inference_bench.research_ai_capacity_validation import (
    load_research_ai_capacity_replay,
    normalize_research_ai_replay_row,
)


def _replay_row(
    *,
    prompt: str = "USER QUESTION:\nSummarize the supplied evidence.",
) -> dict[str, object]:
    return {
        "prompt_id": "research_ai_scaleup_2000_0001",
        "vertical": "research_ai",
        "b6_evaluation": {"groundedness": False},
        "unexpected_audit_field": {"kept": True},
        "runner_input": {
            "prompt_id": "research_ai_scaleup_2000_0001",
            "workload_name": "smoke_500_mm2_hybrid_top5_b6",
            "prompt": prompt,
            "expected_output": "generation_contract_json",
            "metadata": {
                "vertical": "research_ai",
                "memory_mode": "mm2_hybrid_top5",
                "citation_id_aliases": json.dumps({"E1": ["canonical-gold-id"]}),
                "gold_evidence_ids": json.dumps(["canonical-gold-id"]),
            },
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loader_accepts_extra_audit_fields_and_strips_workload_kwargs(tmp_path: Path) -> None:
    input_path = tmp_path / "replay.jsonl"
    _write_jsonl(input_path, [_replay_row()])

    loaded = load_research_ai_capacity_replay(input_path, limit=1)

    assert len(loaded) == 1
    item = loaded[0]
    workload_item = item.to_workload_item()
    assert workload_item.prompt_id == "research_ai_scaleup_2000_0001"
    assert "unexpected_audit_field" not in workload_item.metadata
    assert item.source_metadata["source_row"]["unexpected_audit_field"] == {"kept": True}


def test_source_metadata_preserves_original_b6_fields() -> None:
    item = normalize_research_ai_replay_row(_replay_row())

    assert item.source_metadata["b6_evaluation"] == {"groundedness": False}
    assert item.source_metadata["runner_input"]["prompt_id"] == item.prompt_id


def test_loader_blocks_canonical_gold_id_leakage_into_prompt() -> None:
    with pytest.raises(ValueError, match="Canonical gold evidence IDs leaked"):
        normalize_research_ai_replay_row(
            _replay_row(prompt="USER QUESTION:\nExplain canonical-gold-id."),
        )


def test_missing_prompt_fields_fail_clearly() -> None:
    row = _replay_row()
    runner_input = row["runner_input"]
    assert isinstance(runner_input, dict)
    runner_input.pop("prompt")

    with pytest.raises(ValueError, match="missing required prompt"):
        normalize_research_ai_replay_row(row)


def test_capacity_loader_does_not_import_or_call_evaluator() -> None:
    import inference_bench.research_ai_capacity_validation as module

    source = inspect.getsource(module)

    assert "evaluate_result_row" not in source
    assert "evaluate_generated_answers" not in source
