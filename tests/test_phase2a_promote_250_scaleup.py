import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/phase2/promote_phase2a_scaleup_250.py"
DOC_PATH = ROOT / "docs/43_phase2a_250_scaleup_promotion.md"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("promote_phase2a_scaleup_250", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _create_fake_generated_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    module = _load_module()
    generated_root = tmp_path / "generated"
    promoted_root = tmp_path / "promoted"
    qa_report = tmp_path / "qa_report.json"
    promotion_report = tmp_path / "promotion_report.json"
    _write_json(
        qa_report,
        {
            "promotion_ready": True,
            "critical_issue_count": 0,
            "warning_count": 0,
            "total_prompt_count": 1250,
            "total_gold_count": 1250,
        },
    )
    for vertical in module.VERTICALS:
        for kind in module.FILE_KINDS:
            _write_jsonl(
                generated_root / vertical / f"{vertical}_{kind}_250.jsonl",
                [{"record_type": kind, "vertical": vertical}],
            )
    return generated_root, promoted_root, qa_report, promotion_report


def _run_temp_promotion(tmp_path: Path) -> dict[str, Any]:
    generated_root, promoted_root, qa_report, promotion_report = _create_fake_generated_tree(
        tmp_path
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
    assert isinstance(summary, dict)
    return summary


def test_promotion_script_exists() -> None:
    assert SCRIPT_PATH.exists()


def test_promote_requires_clean_qa(tmp_path: Path) -> None:
    qa_report = tmp_path / "qa_report.json"
    _write_json(
        qa_report,
        {
            "promotion_ready": False,
            "critical_issue_count": 1,
            "warning_count": 0,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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


def test_promote_copies_expected_files(tmp_path: Path) -> None:
    generated_root, promoted_root, qa_report, promotion_report = _create_fake_generated_tree(
        tmp_path
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
    assert summary["copied_file_count"] == 15
    for vertical in _load_module().VERTICALS:
        assert (promoted_root / vertical / f"{vertical}_prompts_250.jsonl").exists()
        assert (promoted_root / vertical / f"{vertical}_gold_250.jsonl").exists()
        assert (promoted_root / vertical / f"{vertical}_kb_250.jsonl").exists()


def test_manifest_shape(tmp_path: Path) -> None:
    summary = _run_temp_promotion(tmp_path)
    manifest = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["phase"] == "2A-11"
    assert manifest["dataset_name"] == "phase2a_250_scaleup"
    assert "total_prompt_count" in manifest
    assert "total_gold_count" in manifest
    assert set(manifest["per_vertical"]) == set(_load_module().VERTICALS)
    assert manifest["quality_summary"] == {
        "critical_issue_count": 0,
        "warning_count": 0,
        "promotion_ready": True,
    }


def test_data_scaleup_readme_exists_after_promotion(tmp_path: Path) -> None:
    summary = _run_temp_promotion(tmp_path)
    readme = Path(summary["readme_path"])

    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "phase2a_250_scaleup" in text
    assert "not the 1,000, 2,000, 4,000, or 5,000" in text


def test_docs_include_promotion_command() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-11" in docs
    assert "python scripts/phase2/promote_phase2a_scaleup_250.py --promote" in docs
    assert "no RAG" in docs
    assert "1,000-per-vertical" in docs
