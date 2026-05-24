import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
QA_SCRIPT_PATH = ROOT / "scripts/phase2/audit_phase2a_scaleup_2000_full.py"
PROMOTE_SCRIPT_PATH = ROOT / "scripts/phase2/promote_phase2a_scaleup_2000_full.py"
GENERATOR_DOC_PATH = ROOT / "docs/49_phase2a_2000_scaleup_generator.md"
QA_DOC_PATH = ROOT / "docs/50_phase2a_2000_full_qa_promotion.md"
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


def _ensure_generated_2000_candidates() -> None:
    for vertical in VERTICALS:
        prompts = ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_prompts_2000.jsonl"
        gold = ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_gold_2000.jsonl"
        kb = ROOT / f"data/generated/phase2a/scaleup/{vertical}/{vertical}_kb_2000.jsonl"
        if prompts.exists() and gold.exists() and kb.exists():
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--generate-vertical",
                "--vertical",
                vertical,
                "--target-per-vertical",
                "2000",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def _run_qa() -> dict[str, Any]:
    _ensure_generated_2000_candidates()
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
            "total_prompt_count": 10000,
            "total_gold_count": 10000,
        },
    )


def _create_fake_generated_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    generated_root = tmp_path / "generated"
    promoted_root = tmp_path / "promoted"
    qa_report = tmp_path / "qa_report.json"
    promotion_report = tmp_path / "promotion_report.json"
    _create_clean_qa_report(qa_report)
    for vertical in VERTICALS:
        for kind in ["prompts", "gold", "kb"]:
            _write_jsonl(
                generated_root / vertical / f"{vertical}_{kind}_2000.jsonl",
                [{"record_type": kind, "vertical": vertical}],
            )
    return generated_root, promoted_root, qa_report, promotion_report


def test_2000_full_qa_cli() -> None:
    summary = _run_qa()

    assert summary["phase"] == "2A-15"
    assert summary["partial_dataset"] is False
    assert summary["included_verticals"] == VERTICALS
    assert summary["critical_issue_count"] == 0
    assert summary["warning_count"] == 0
    assert summary["promotion_ready"] is True
    assert (ROOT / summary["report_path"]).exists()


def test_2000_full_qa_expected_totals() -> None:
    summary = _run_qa()
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["dataset_name"] == "phase2a_2000_full_candidate"
    assert report["total_prompt_count"] == 10000
    assert report["total_gold_count"] == 10000
    assert report["vertical_count"] == 5
    for vertical in VERTICALS:
        assert report["per_vertical"][vertical]["prompt_count"] == 2000
        assert report["per_vertical"][vertical]["gold_count"] == 2000


def test_2000_promotion_requires_clean_qa(tmp_path: Path) -> None:
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


def test_2000_promotion_manifest_shape(tmp_path: Path) -> None:
    generated_root, promoted_root, qa_report, promotion_report = _create_fake_generated_tree(
        tmp_path
    )
    result = subprocess.run(
        [
            sys.executable,
            str(PROMOTE_SCRIPT_PATH),
            "--promote",
            "--qa-report",
            str(qa_report),
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
    assert manifest["phase"] == "2A-15"
    assert manifest["dataset_name"] == "phase2a_2000_full"
    assert manifest["total_prompt_count"] in {5, 10000}
    assert manifest["total_gold_count"] in {5, 10000}
    assert manifest["verticals"] == VERTICALS
    assert "per_vertical" in manifest
    assert manifest["quality_summary"]["promotion_ready"] is True
    assert manifest["next_step"] == (
        "Run comprehensive Phase 2A EDA before Phase 2B context engineering."
    )


def test_docs_include_2000_commands() -> None:
    generator_docs = GENERATOR_DOC_PATH.read_text(encoding="utf-8")
    qa_docs = QA_DOC_PATH.read_text(encoding="utf-8")
    combined = f"{generator_docs}\n{qa_docs}"

    assert "Phase 2A-14" in generator_docs
    assert "Phase 2A-15" in qa_docs
    assert "--generate-vertical --vertical airline --target-per-vertical 2000" in combined
    assert "--generate-vertical --vertical research_ai --target-per-vertical 2000" in combined
    assert "audit_phase2a_scaleup_2000_full.py --run-audit" in combined
    assert "promote_phase2a_scaleup_2000_full.py --promote" in combined
    assert "10,000" in combined
    assert "no RAG" in combined
