from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from inference_bench.research_ai_capacity_validation import (
    build_b6r3_manifest_payload,
    completed_prompt_ids_from_jsonl,
    load_research_ai_capacity_replay,
    pending_replay_items,
    validate_b6r3_cli_limits,
)

SCRIPT_PATH = Path("scripts/phase4/run_b6r3_model6_research_ai_capacity.py")


def _load_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location("b6r3_runner", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _replay_row(prompt_id: str = "research_ai_scaleup_2000_0001") -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "vertical": "research_ai",
        "runner_input": {
            "prompt_id": prompt_id,
            "workload_name": "smoke_500_mm2_hybrid_top5_b6",
            "prompt": (
                "SYSTEM:\nAnswer only from supplied evidence.\n\n"
                "RETRIEVED EVIDENCE:\n[EVIDENCE 1]\nevidence_id: E1\n"
                "title: Paper - Limitations\ntext: A limitation is described.\n\n"
                "USER QUESTION:\nWhat limitation is described?\n\nOUTPUT CONTRACT:\nold"
            ),
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


def test_cli_limits_allow_b6r3_defaults() -> None:
    validate_b6r3_cli_limits(limit=26, max_new_tokens=320)


def test_resume_skips_completed_prompt_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_jsonl(
        input_path,
        [_replay_row("research_ai_1"), _replay_row("research_ai_2")],
    )
    _write_jsonl(output_path, [{"prompt_id": "research_ai_1", "success": True}])

    items = load_research_ai_capacity_replay(input_path)
    pending = pending_replay_items(
        items,
        completed_prompt_ids=completed_prompt_ids_from_jsonl(output_path),
    )

    assert [item.prompt_id for item in pending] == ["research_ai_2"]


def test_dry_run_does_not_touch_paid_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_runner_module()
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    _write_jsonl(input_path, [_replay_row()])

    def fail_if_called(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("paid API credential lookup should not run in dry-run")

    monkeypatch.setattr(module, "api_key_for_route", fail_if_called)

    exit_code = module.main(
        [
            "--dry-run",
            "--input-path",
            str(input_path),
            "--report-path",
            str(report_path),
            "--summary-path",
            str(summary_path),
            "--limit",
            "1",
            "--max-new-tokens",
            "320",
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["paid_api_call_triggered"] is False
    assert report["normalized_replay_row_count"] == 1


def test_missing_paid_flag_blocks_live_execution(tmp_path: Path) -> None:
    module = _load_runner_module()
    input_path = tmp_path / "input.jsonl"
    report_path = tmp_path / "report.json"
    summary_path = tmp_path / "summary.csv"
    _write_jsonl(input_path, [_replay_row()])

    exit_code = module.main(
        [
            "--input-path",
            str(input_path),
            "--report-path",
            str(report_path),
            "--summary-path",
            str(summary_path),
            "--limit",
            "1",
            "--max-new-tokens",
            "320",
        ]
    )

    assert exit_code == 1


def test_completed_manifest_requires_all_expected_rows() -> None:
    with pytest.raises(ValueError, match="Cannot mark B6R3 run completed"):
        build_b6r3_manifest_payload(
            run_id="run",
            model_alias="model6_gated",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            provider="novita",
            backend="hf_inference_provider",
            input_path="input.jsonl",
            output_path="output.jsonl",
            limit=26,
            max_new_tokens=320,
            start_time="2026-06-17T00:00:00Z",
            end_time="2026-06-17T00:01:00Z",
            expected_count=26,
            completed_count=25,
            error_count=0,
            total_cost_usd=0.0,
            status="completed",
            command="python runner.py",
        )
