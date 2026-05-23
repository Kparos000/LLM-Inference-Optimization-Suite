import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/plan_phase2a_1000_scaleup.py"
DOC_PATH = ROOT / "docs/44_phase2a_1000_scaleup_plan.md"
REPORT_PATH = (
    ROOT / "data/generated/phase2a/scaleup_reports/phase2a_1000_scaleup_readiness_report.json"
)
MATRIX_PATH = ROOT / "data/generated/phase2a/scaleup_reports/phase2a_1000_scaleup_matrix.csv"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_phase2a_1000_scaleup", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_1000_plan_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_1000_plan_requires_promoted_250_manifest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--write-report",
            "--promoted-250-manifest",
            str(tmp_path / "missing_manifest.json"),
            "--output-report",
            str(tmp_path / "report.json"),
            "--output-matrix-csv",
            str(tmp_path / "matrix.csv"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing promoted 250 manifest" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_1000_plan_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--write-report"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-12A"
    assert summary["target_per_vertical"] == 1000
    assert summary["promoted_250_found"] is True
    assert REPORT_PATH.exists()
    assert MATRIX_PATH.exists()


def test_1000_plan_reports_5000_total() -> None:
    module = _load_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")

    report = module.build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
        promoted_manifest_path=Path("data/scaleup/phase2a_250_manifest.json"),
    )

    assert report["target_per_vertical"] == 1000
    assert report["total_target_prompts"] == 5000
    assert report["previous_checkpoint"] == "checkpoint_250"
    assert report["promoted_250_found"] is True


def test_1000_plan_per_vertical_additional_prompts_750() -> None:
    module = _load_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")

    per_vertical = module.build_per_vertical_readiness(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
    )

    assert set(per_vertical) == set(manifest["verticals"])
    for metrics in per_vertical.values():
        assert metrics["current_250_prompts"] == 250
        assert metrics["target_1000_prompts"] == 1000
        assert metrics["additional_prompts_needed"] == 750


def test_1000_plan_docs_include_command() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-12A" in docs
    assert "python scripts/phase2/plan_phase2a_1000_scaleup.py --write-report" in docs
    assert "no RAG" in docs
    assert "promoted 250 dataset" in docs
