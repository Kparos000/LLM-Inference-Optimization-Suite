from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "results" / "processed" / "long_run_recovery_dry_run_report.json"
SUMMARY_PATH = ROOT / "results" / "processed" / "long_run_recovery_dry_run_summary.csv"


def test_long_run_recovery_dry_run_script_completes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/phase4/test_long_run_recovery_dry_run.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "long_run_recovery_dry_run" in result.stdout
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "PASSED"
    assert report["simulated_prompt_count"] == 20
    assert report["initial_written_count"] == 10
    assert report["resume_skipped_count"] == 10
    assert report["resumed_written_count"] == 10
    assert report["final_row_count"] == 20
    assert report["failed_row_count"] == 1
    assert report["duplicate_prompt_ids"] == []
    assert report["backup_verification"]["passed"] is True
    assert report["backup_verification"]["backup_completeness_score"] == 1.0
    assert SUMMARY_PATH.exists()
