import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "data/sources/phase2a_scaleup_plan.json"
SCRIPT_PATH = ROOT / "scripts/phase2/plan_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/39_phase2a_progressive_scaleup_plan.md"
REPORT_PATH = ROOT / "data/generated/phase2a/phase2a_scaleup_plan_report.json"
MATRIX_PATH = ROOT / "data/generated/phase2a/phase2a_scaleup_matrix.csv"


def _load_plan() -> dict[str, Any]:
    parsed = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("plan_phase2a_scaleup", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaleup_plan_json_exists_and_parses() -> None:
    assert PLAN_PATH.exists()
    plan = _load_plan()
    assert plan["phase"] == "2A-8"
    assert plan["approved_total_prompt_cap"] == 25000
    assert plan["approved_max_total_prompts"] == 25000


def test_scaleup_plan_total_cap() -> None:
    plan = _load_plan()
    checkpoints = plan["checkpoints"]

    assert checkpoints["checkpoint_2000"]["total_prompts"] == 10000
    assert checkpoints["checkpoint_5000"]["total_prompts"] == 25000
    for checkpoint in checkpoints.values():
        assert checkpoint["prompts_per_vertical"] != 10000


def test_scaleup_plan_supports_5000_per_vertical() -> None:
    plan = _load_plan()
    checkpoint = plan["checkpoints"]["checkpoint_5000"]

    assert checkpoint["prompts_per_vertical"] == 5000
    assert checkpoint["total_prompts"] == 25000
    assert plan["approved_max_prompts_per_vertical"] == 5000


def test_scaleup_plan_supports_20000_total_stress_tier() -> None:
    plan = _load_plan()
    checkpoint = plan["checkpoints"]["checkpoint_4000"]

    assert checkpoint["prompts_per_vertical"] == 4000
    assert checkpoint["total_prompts"] == 20000
    assert plan["gpu_stress_checkpoint"] == "checkpoint_4000"


def test_near_term_main_checkpoint() -> None:
    plan = _load_plan()

    assert plan["near_term_main_checkpoint"] == "checkpoint_2000"
    assert plan["checkpoints"]["checkpoint_2000"]["total_prompts"] == 10000


def test_max_expanded_checkpoint() -> None:
    plan = _load_plan()

    assert plan["max_expanded_checkpoint"] == "checkpoint_5000"
    assert plan["checkpoints"]["checkpoint_5000"]["total_prompts"] == 25000


def test_scaleup_plan_has_all_five_verticals() -> None:
    plan = _load_plan()

    assert set(plan["vertical_scale_strategy"]) == {
        "finance",
        "airline",
        "healthcare_admin",
        "research_ai",
        "retail",
    }


def test_gold_review_subset_targets() -> None:
    plan = _load_plan()

    assert plan["gold_strategy"]["gold_review_subset"]["target_count"] == 1000
    assert plan["gold_strategy"]["deep_review_subset"]["target_count"] == 300
    assert (
        plan["gold_strategy"]["review_targets_by_checkpoint"]["checkpoint_4000"][
            "gold_review_subset_target"
        ]
        == 1500
    )
    assert (
        plan["gold_strategy"]["review_targets_by_checkpoint"]["checkpoint_5000"][
            "deep_review_subset_target"
        ]
        == 750
    )


def test_status_distribution_targets_exist() -> None:
    plan = _load_plan()
    targets = plan["status_distribution_targets"]

    for vertical in ["finance", "airline", "healthcare_admin", "research_ai", "retail"]:
        assert vertical in targets
    assert "global" in targets


def test_plan_script_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--write-report"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-8"
    assert summary["approved_max_total_prompts"] == 25000
    assert REPORT_PATH.exists()
    assert MATRIX_PATH.exists()


def test_matrix_includes_all_checkpoints() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--write-report"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    matrix = MATRIX_PATH.read_text(encoding="utf-8")
    for checkpoint in [
        "checkpoint_seed",
        "checkpoint_250",
        "checkpoint_1000",
        "checkpoint_2000",
        "checkpoint_4000",
        "checkpoint_5000",
    ]:
        assert checkpoint in matrix


def test_report_recommends_250_when_ready() -> None:
    module = _load_module()
    plan = _load_plan()
    qa_report = {
        "critical_issue_count": 0,
        "scale_up_readiness": {
            vertical: {"ready_for_250_scale": True} for vertical in plan["vertical_scale_strategy"]
        },
    }

    report = module.build_report(plan, qa_report, [], [])

    assert report["recommend_generation"] is True
    assert "Phase 2A-9" in report["next_step"]
    assert report["approved_max_total_prompts"] == 25000
    assert report["max_expanded_checkpoint"] == "checkpoint_5000"


def test_docs_include_progressive_scaleup() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")
    lowered = docs.lower()

    assert "10,000 total prompts" in docs
    assert "25,000 total" in docs
    assert "5,000" in docs
    assert "250" in docs
    assert "1,000" in docs
    assert "2,000" in docs
    assert "gold-review subset" in lowered
    assert "deep-review subset" in lowered
    assert "no rag" in lowered or "does not build rag" in lowered


def test_docs_no_longer_claim_10000_is_final_cap() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")
    lowered = docs.lower()

    assert "25,000 total" in docs
    assert "5,000 prompts per vertical" in docs
    assert "no longer capped at 10,000 total prompts" in lowered
