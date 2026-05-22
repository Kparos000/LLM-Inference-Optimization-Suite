import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/40_phase2a_250_scaleup_generator.md"
GITIGNORE_PATH = ROOT / ".gitignore"
REPORT_DIR = ROOT / "data/generated/phase2a/scaleup_reports"
OUTPUT_DIR = ROOT / "data/generated/phase2a/scaleup"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_phase2a_scaleup", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            rows.append(parsed)
    return rows


def test_scaleup_generator_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_distribution_counts_sum_to_target() -> None:
    module = _load_module()

    for vertical in ["finance", "airline", "healthcare_admin", "research_ai", "retail"]:
        distributions = module.calculate_distribution_counts(vertical, 250)
        assert sum(distributions["expected_status"].values()) == 250
        assert sum(distributions["task_type"].values()) == 250
        assert sum(distributions["expected_output_format"].values()) == 250
        assert sum(distributions["difficulty"].values()) == 250


def test_dry_run_cli() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-9"
    assert summary["mode"] == "dry_run"
    assert set(summary["verticals"]) == {
        "finance",
        "airline",
        "healthcare_admin",
        "research_ai",
        "retail",
    }


def test_generate_plan_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-plan",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["phase"] == "2A-9"
    assert summary["mode"] == "generate_plan"
    assert summary["manifest_count"] == 5
    assert (REPORT_DIR / "airline_scaleup_250_manifest.json").exists()
    assert (REPORT_DIR / "phase2a_scaleup_generation_plan_250.json").exists()


def test_airline_pilot_generation_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--generate-vertical",
            "--vertical",
            "airline",
            "--target-per-vertical",
            "250",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    prompts_path = ROOT / summary["prompts_path"]
    gold_path = ROOT / summary["gold_path"]
    kb_path = ROOT / summary["kb_path"]
    assert prompts_path.exists()
    assert gold_path.exists()
    assert kb_path.exists()

    prompts = _read_jsonl(prompts_path)
    gold = _read_jsonl(gold_path)
    kb = _read_jsonl(kb_path)
    assert len(prompts) == 250
    assert len(gold) == 250
    assert summary["status_counts"] == {"answer": 225, "escalate": 20, "spam_or_fraud": 5}

    module = _load_module()
    assert module.validate_prompt_gold_alignment(prompts, gold) == []
    assert module.validate_evidence_coverage(gold, kb) == []


def test_generated_outputs_are_ignored() -> None:
    gitignore = GITIGNORE_PATH.read_text(encoding="utf-8")

    assert "data/generated/phase2a/scaleup/" in gitignore
    assert "data/generated/phase2a/scaleup_reports/" in gitignore


def test_docs_include_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "python scripts/phase2/generate_phase2a_scaleup.py --dry-run" in docs
    assert "--generate-plan --target-per-vertical 250" in docs
    assert "--generate-vertical --vertical airline --target-per-vertical 250" in docs
    assert "no RAG" in docs
