import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/plan_phase2a_large_scale.py"
DOC_PATH = ROOT / "docs/51_phase2a_large_scale_scaffolding.md"


def _run_large_scale_plan(tmp_path: Path) -> dict[str, Any]:
    report_path = tmp_path / "phase2a_large_scale_plan.json"
    matrix_path = tmp_path / "phase2a_large_scale_matrix.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--write-report",
            "--output-report",
            str(report_path),
            "--output-matrix-csv",
            str(matrix_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert isinstance(summary, dict)
    assert report_path.exists()
    assert matrix_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert isinstance(report, dict)
    return cast(dict[str, Any], report)


def test_large_scale_plan_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_large_scale_plan_cli(tmp_path: Path) -> None:
    report = _run_large_scale_plan(tmp_path)

    assert report["phase"] == "2A-16A"
    assert report["current_checkpoint"] == "checkpoint_2000"
    assert report["total_current_prompts"] == 10000
    assert report["total_current_gold"] == 10000
    assert report["can_plan_4000"] is True
    assert report["can_plan_5000"] is True


def test_large_scale_plan_includes_4000_and_5000(tmp_path: Path) -> None:
    report = _run_large_scale_plan(tmp_path)

    assert report["future_checkpoints"]["checkpoint_4000"]["prompts_per_vertical"] == 4000
    assert report["future_checkpoints"]["checkpoint_4000"]["total_prompts"] == 20000
    assert report["future_checkpoints"]["checkpoint_5000"]["prompts_per_vertical"] == 5000
    assert report["future_checkpoints"]["checkpoint_5000"]["total_prompts"] == 25000
    for vertical, row in report["per_vertical"].items():
        assert row["current_2000_prompt_count"] == 2000, vertical
        assert row["current_2000_gold_count"] == 2000, vertical
        assert row["target_4000_additional_prompts_needed"] == 2000
        assert row["target_5000_additional_prompts_needed"] == 3000
        assert row["generator_implemented_for_4000"] is False
        assert row["generator_implemented_for_5000"] is False


def test_large_scale_plan_does_not_recommend_generation_now(tmp_path: Path) -> None:
    report = _run_large_scale_plan(tmp_path)

    assert report["should_generate_now"] is False
    assert "before generating 4,000/5,000" in report["next_step"]


def test_large_scale_plan_research_ai_mentions_section_pool(tmp_path: Path) -> None:
    report = _run_large_scale_plan(tmp_path)
    research_ai = report["per_vertical"]["research_ai"]
    notes = " ".join(research_ai["notes"]).lower()

    assert "2,590 extracted sections" in " ".join(research_ai["notes"])
    assert "promoted benchmark kb is a selected" in notes
    assert "full retrieval corpus" in notes
    assert report["research_ai_source_pool"]["extracted_section_count_guidance"] == 2590


def test_large_scale_docs_include_command() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-16A" in text
    assert "plan_phase2a_large_scale.py --write-report" in text
    assert "4,000" in text
    assert "5,000" in text
