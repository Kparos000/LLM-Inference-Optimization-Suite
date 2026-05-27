from pathlib import Path

from inference_bench.memory_workloads import write_csv, write_json
from inference_bench.run_safety_audit import build_run_safety_audit


def test_run_safety_audit_report_is_generated(tmp_path: Path) -> None:
    report, rows = build_run_safety_audit()

    write_json(tmp_path / "run_safety_audit_report.json", report)
    write_csv(tmp_path / "run_safety_audit_summary.csv", rows, list(rows[0]))

    assert (tmp_path / "run_safety_audit_report.json").exists()
    assert (tmp_path / "run_safety_audit_summary.csv").exists()
    assert report["no_model_inference_triggered"] is True


def test_run_safety_audit_identifies_checkpoint_resume_support() -> None:
    report, rows = build_run_safety_audit()
    areas = {row["area"]: row for row in rows}

    assert "openai_load_runner_checkpointing" in areas
    assert areas["openai_load_runner_checkpointing"]["reusable"] is True
    assert "resume" in areas["openai_load_runner_checkpointing"]["current_capability"]
    assert "completed_prompt_ids" in areas["openai_load_runner_checkpointing"]["current_capability"]
    assert report["artifacts_inspected"]["openai_load_runner"]["exists"] is True


def test_run_safety_audit_identifies_logging_support_and_gaps() -> None:
    report, rows = build_run_safety_audit()
    areas = {row["area"]: row for row in rows}

    assert "progress_logging" in areas
    assert "memory_mode" in areas["progress_logging"]["phase4_gap"]
    assert any("structured JSONL" in item for item in report["missing_before_main_gpu_experiments"])


def test_run_safety_audit_uses_inference_terms_not_training_terms() -> None:
    report, rows = build_run_safety_audit()
    combined = " ".join(
        [
            report["terminology_note"],
            *[str(row["current_capability"]) for row in rows],
            *[str(row["phase4_gap"]) for row in rows],
        ]
    ).lower()

    assert "training checkpoint" not in combined
    assert "inference checkpointing" in combined


def test_no_gpu_or_api_calls_are_triggered() -> None:
    report, _ = build_run_safety_audit()

    assert report["no_model_inference_triggered"] is True
    assert report["no_gpu_work_triggered"] is True
