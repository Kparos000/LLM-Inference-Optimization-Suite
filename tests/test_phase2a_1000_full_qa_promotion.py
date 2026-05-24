import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
QA_SCRIPT_PATH = ROOT / "scripts/phase2/audit_phase2a_scaleup_1000_full.py"
PROMOTE_SCRIPT_PATH = ROOT / "scripts/phase2/promote_phase2a_scaleup_1000_full.py"
DOC_PATH = ROOT / "docs/47_phase2a_1000_full_qa_promotion.md"
VERTICALS = ["airline", "healthcare_admin", "retail", "finance", "research_ai"]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _ensure_research_ai_generated() -> None:
    prompts = ROOT / "data/generated/phase2a/scaleup/research_ai/research_ai_prompts_1000.jsonl"
    gold = ROOT / "data/generated/phase2a/scaleup/research_ai/research_ai_gold_1000.jsonl"
    kb = ROOT / "data/generated/phase2a/scaleup/research_ai/research_ai_kb_1000.jsonl"
    if prompts.exists() and gold.exists() and kb.exists():
        return
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR_PATH),
            "--generate-vertical",
            "--vertical",
            "research_ai",
            "--target-per-vertical",
            "1000",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _run_full_qa() -> dict[str, Any]:
    _ensure_research_ai_generated()
    result = subprocess.run(
        [sys.executable, str(QA_SCRIPT_PATH), "--run-audit"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert isinstance(summary, dict)
    return summary


def _create_clean_qa_report(path: Path) -> None:
    _write_json(
        path,
        {
            "partial_dataset": False,
            "promotion_ready": True,
            "critical_issue_count": 0,
            "warning_count": 0,
            "included_verticals": VERTICALS,
            "total_prompt_count": 5000,
            "total_gold_count": 5000,
        },
    )


def _create_fake_source_trees(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    partial_root = tmp_path / "partial"
    generated_root = tmp_path / "generated"
    promoted_root = tmp_path / "promoted"
    qa_report = tmp_path / "qa_report.json"
    promotion_report = tmp_path / "promotion_report.json"
    _create_clean_qa_report(qa_report)
    for vertical in VERTICALS:
        root = generated_root if vertical == "research_ai" else partial_root
        for kind in ["prompts", "gold", "kb"]:
            _write_jsonl(
                root / vertical / f"{vertical}_{kind}_1000.jsonl",
                [{"record_type": kind, "vertical": vertical}],
            )
    return partial_root, generated_root, promoted_root, qa_report, promotion_report


def test_full_1000_qa_cli() -> None:
    summary = _run_full_qa()

    assert summary["partial_dataset"] is False
    assert summary["included_verticals"] == VERTICALS
    assert summary["total_prompt_count"] == 5000
    assert summary["total_gold_count"] == 5000
    assert summary["critical_issue_count"] == 0
    assert summary["warning_count"] == 0
    assert summary["promotion_ready"] is True
    assert (ROOT / summary["report_path"]).exists()


def test_full_1000_promotion_requires_clean_qa(tmp_path: Path) -> None:
    qa_report = tmp_path / "qa_report.json"
    _write_json(
        qa_report,
        {
            "partial_dataset": False,
            "promotion_ready": False,
            "critical_issue_count": 1,
            "warning_count": 0,
            "included_verticals": VERTICALS,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTE_SCRIPT_PATH),
            "--promote",
            "--qa-report",
            str(qa_report),
            "--partial-root",
            str(tmp_path / "partial"),
            "--generated-root",
            str(tmp_path / "generated"),
            "--promoted-root",
            str(tmp_path / "promoted"),
            "--promotion-report",
            str(tmp_path / "promotion_report.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "promotion_ready" in result.stderr
    assert not (tmp_path / "promoted").exists()


def test_full_1000_promotion_manifest_shape(tmp_path: Path) -> None:
    partial_root, generated_root, promoted_root, qa_report, promotion_report = (
        _create_fake_source_trees(tmp_path)
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTE_SCRIPT_PATH),
            "--promote",
            "--qa-report",
            str(qa_report),
            "--partial-root",
            str(partial_root),
            "--generated-root",
            str(generated_root),
            "--promoted-root",
            str(promoted_root),
            "--promotion-report",
            str(promotion_report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
    assert summary["copied_file_count"] == 15
    assert manifest["phase"] == "2A-13H"
    assert manifest["dataset_name"] == "phase2a_1000_full"
    assert manifest["partial_dataset"] is False
    assert manifest["verticals"] == VERTICALS
    assert "total_prompt_count" in manifest
    assert "total_gold_count" in manifest
    assert "per_vertical" in manifest
    assert manifest["quality_summary"]["promotion_ready"] is True
    assert manifest["next_step"] == "Begin 2,000-per-vertical generator planning."


def test_docs_include_full_1000_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-13G" in docs
    assert "Phase 2A-13H" in docs
    assert "generate_phase2a_scaleup.py --generate-vertical --vertical research_ai" in docs
    assert "audit_phase2a_scaleup_1000_full.py --run-audit" in docs
    assert "promote_phase2a_scaleup_1000_full.py --promote" in docs
    assert "promotion_ready" in docs
    assert "no RAG" in docs
