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


def test_1000_planner_clears_retail_source_blocker_when_report_ready() -> None:
    module = _load_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")
    retail_report = {
        "phase": "2A-12B",
        "categories": ["All_Beauty", "Home_and_Kitchen", "Electronics"],
        "retail_ready_for_1000_source_expansion": True,
    }

    report = module.build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
        promoted_manifest_path=Path("data/scaleup/phase2a_250_manifest.json"),
        retail_multicategory_report=retail_report,
        retail_multicategory_report_path=Path(
            "data/generated/retail/multicategory/retail_multicategory_source_report.json"
        ),
    )

    retail = report["per_vertical_readiness"]["retail"]
    assert retail["source_expansion_ready"] is True
    assert retail["ready_for_1000_generation"] is True
    assert retail["blockers"] == []
    assert not any("retail:" in blocker for blocker in report["blockers"])


def test_1000_planner_reports_research_ai_missing_requirements() -> None:
    module = _load_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")
    research_ai_report = {
        "phase": "2A-12C",
        "expansion_ready_for_1000": False,
        "missing_requirements": [
            "additional_approved_papers_needed:20",
            "section_coverage_below_target_min:436",
        ],
    }

    report = module.build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
        promoted_manifest_path=Path("data/scaleup/phase2a_250_manifest.json"),
        research_ai_expansion_report=research_ai_report,
        research_ai_expansion_report_path=Path(
            "data/generated/research_ai/research_ai_40_paper_expansion_report.json"
        ),
    )

    research_ai = report["per_vertical_readiness"]["research_ai"]
    assert research_ai["source_expansion_ready"] is False
    assert "source_expansion_required_before_1000_generation" in research_ai["blockers"]
    assert report["research_ai_missing_requirements"] == [
        "additional_approved_papers_needed:20",
        "section_coverage_below_target_min:436",
    ]


def test_1000_planner_clears_research_ai_blocker_when_expansion_ready() -> None:
    module = _load_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")
    retail_report = {"retail_ready_for_1000_source_expansion": True}
    research_ai_report = {
        "phase": "2A-12C",
        "expansion_ready_for_1000": True,
        "missing_requirements": [],
    }

    report = module.build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
        promoted_manifest_path=Path("data/scaleup/phase2a_250_manifest.json"),
        retail_multicategory_report=retail_report,
        research_ai_expansion_report=research_ai_report,
        research_ai_expansion_report_path=Path(
            "data/generated/research_ai/research_ai_40_paper_expansion_report.json"
        ),
    )

    research_ai = report["per_vertical_readiness"]["research_ai"]
    assert research_ai["source_expansion_ready"] is True
    assert research_ai["ready_for_1000_generation"] is True
    assert research_ai["blockers"] == []
    assert not any("research_ai:" in blocker for blocker in report["blockers"])


def test_1000_planner_adds_finance_blocker_when_reuse_risk_is_high() -> None:
    module = _load_module()
    scaleup_plan = module.read_json(ROOT / "data/sources/phase2a_scaleup_plan.json")
    manifest = module.require_promoted_250_manifest(ROOT / "data/scaleup/phase2a_250_manifest.json")
    finance_report = {
        "phase": "2A-12D",
        "evidence_reuse_risk": "high",
        "ready_for_1000_finance_generation": False,
    }

    report = module.build_report(
        scaleup_plan=scaleup_plan,
        promoted_manifest=manifest,
        promoted_manifest_path=Path("data/scaleup/phase2a_250_manifest.json"),
        finance_evidence_reuse_report=finance_report,
        finance_evidence_reuse_report_path=Path(
            "data/generated/phase2a/scaleup_reports/finance_evidence_reuse_audit_report.json"
        ),
    )

    finance = report["per_vertical_readiness"]["finance"]
    assert finance["evidence_reuse_risk"] == "high"
    assert "finance_evidence_reuse_high_risk" in finance["blockers"]
    assert "finance:finance_evidence_reuse_high_risk" in report["blockers"]


def test_1000_plan_docs_include_command() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-12A" in docs
    assert "python scripts/phase2/plan_phase2a_1000_scaleup.py --write-report" in docs
    assert "no RAG" in docs
    assert "promoted 250 dataset" in docs


def test_docs_include_40_paper_expansion_command() -> None:
    plan_docs = DOC_PATH.read_text(encoding="utf-8")
    research_docs = (ROOT / "docs/35_phase2_research_ai_curated_seed.md").read_text(
        encoding="utf-8"
    )

    command = "python scripts/phase2/prepare_research_ai_papers.py --build-40-paper-expansion"
    assert command in plan_docs
    assert command in research_docs
    assert "40 papers" in research_docs
    assert "do not fake PDFs or sections" in research_docs
