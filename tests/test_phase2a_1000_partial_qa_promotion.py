import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QA_SCRIPT_PATH = ROOT / "scripts/phase2/audit_phase2a_scaleup_1000_partial.py"
PROMOTE_SCRIPT_PATH = ROOT / "scripts/phase2/promote_phase2a_scaleup_1000_partial.py"
GENERATOR_PATH = ROOT / "scripts/phase2/generate_phase2a_scaleup.py"
DOC_PATH = ROOT / "docs/46_phase2a_1000_partial_qa_promotion.md"

INCLUDED_VERTICALS = ["airline", "healthcare_admin", "retail", "finance"]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
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


def _candidate_path(vertical: str, kind: str) -> Path:
    return ROOT / "data/generated/phase2a/scaleup" / vertical / f"{vertical}_{kind}_1000.jsonl"


def _ensure_generated_candidates() -> None:
    for vertical in INCLUDED_VERTICALS:
        if all(_candidate_path(vertical, kind).exists() for kind in ["prompts", "gold", "kb"]):
            continue
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--generate-vertical",
                "--vertical",
                vertical,
                "--target-per-vertical",
                "1000",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def _run_partial_qa() -> dict[str, Any]:
    _ensure_generated_candidates()
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
            "partial_dataset": True,
            "promotion_ready": True,
            "critical_issue_count": 0,
            "warning_count": 0,
            "included_verticals": INCLUDED_VERTICALS,
            "excluded_verticals": ["research_ai"],
            "total_prompt_count": 4000,
            "total_gold_count": 4000,
        },
    )


def _create_fake_generated_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    generated_root = tmp_path / "generated"
    promoted_root = tmp_path / "promoted"
    qa_report = tmp_path / "qa_report.json"
    promotion_report = tmp_path / "promotion_report.json"
    _create_clean_qa_report(qa_report)
    for vertical in INCLUDED_VERTICALS:
        for kind in ["prompts", "gold", "kb"]:
            _write_jsonl(
                generated_root / vertical / f"{vertical}_{kind}_1000.jsonl",
                [{"record_type": kind, "vertical": vertical}],
            )
    return generated_root, promoted_root, qa_report, promotion_report


def test_partial_qa_script_exists() -> None:
    assert QA_SCRIPT_PATH.exists()


def test_partial_promotion_script_exists() -> None:
    assert PROMOTE_SCRIPT_PATH.exists()


def test_partial_qa_cli() -> None:
    summary = _run_partial_qa()

    assert summary["partial_dataset"] is True
    assert summary["included_verticals"] == INCLUDED_VERTICALS
    assert summary["excluded_verticals"] == ["research_ai"]
    assert summary["critical_issue_count"] == 0
    assert summary["warning_count"] == 0
    assert summary["promotion_ready"] is True
    assert (ROOT / summary["report_path"]).exists()


def test_partial_qa_expected_counts() -> None:
    summary = _run_partial_qa()
    report = json.loads((ROOT / summary["report_path"]).read_text(encoding="utf-8"))

    assert report["phase"] == "2A-13D"
    assert report["partial_dataset"] is True
    assert report["total_prompt_count"] == 4000
    assert report["total_gold_count"] == 4000
    assert report["vertical_count"] == 4
    assert set(report["per_vertical"]) == set(INCLUDED_VERTICALS)
    for vertical in INCLUDED_VERTICALS:
        assert report["per_vertical"][vertical]["prompt_count"] == 1000
        assert report["per_vertical"][vertical]["gold_count"] == 1000


def test_partial_qa_detects_hygiene_issue(tmp_path: Path) -> None:
    module = _load_module(QA_SCRIPT_PATH, "audit_phase2a_scaleup_1000_partial")
    generated_root = tmp_path / "generated"
    _write_jsonl(
        generated_root / "retail" / "retail_prompts_1000.jsonl",
        [
            {
                "expected_output_format": "text",
                "expected_status": "answer",
                "prompt_id": "retail_scaleup_1000_0001",
                "question": "C:\\Users\\example appears in this bad prompt.",
                "task_type": "answer_grounded",
            }
        ],
    )
    _write_jsonl(
        generated_root / "retail" / "retail_gold_1000.jsonl",
        [
            {
                "expected_status": "answer",
                "must_include": ["retail_doc_1"],
                "must_not_include": [],
                "prompt_id": "retail_scaleup_1000_0001",
                "reference_answer": "Use retail_doc_1.",
                "required_doc_ids": ["retail_doc_1"],
            }
        ],
    )
    _write_jsonl(
        generated_root / "retail" / "retail_kb_1000.jsonl",
        [{"doc_id": "retail_doc_1", "body": "Retail evidence"}],
    )

    report = module.audit_dataset(
        generated_root=generated_root,
        output_report=tmp_path / "report.json",
        output_summary_csv=tmp_path / "summary.csv",
        output_issue_log=tmp_path / "issues.jsonl",
    )
    issues = [
        json.loads(line)
        for line in (tmp_path / "issues.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["critical_issue_count"] > 0
    assert any(issue["check_name"] == "hygiene_scan" for issue in issues)


def test_partial_promotion_requires_clean_qa(tmp_path: Path) -> None:
    qa_report = tmp_path / "qa_report.json"
    _write_json(
        qa_report,
        {
            "partial_dataset": True,
            "promotion_ready": False,
            "critical_issue_count": 1,
            "warning_count": 0,
            "included_verticals": INCLUDED_VERTICALS,
            "excluded_verticals": ["research_ai"],
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


def test_partial_promotion_copies_expected_files(tmp_path: Path) -> None:
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
    assert summary["copied_file_count"] == 12
    for vertical in INCLUDED_VERTICALS:
        assert (promoted_root / vertical / f"{vertical}_prompts_1000.jsonl").exists()
        assert (promoted_root / vertical / f"{vertical}_gold_1000.jsonl").exists()
        assert (promoted_root / vertical / f"{vertical}_kb_1000.jsonl").exists()
    assert not (promoted_root / "research_ai").exists()


def test_partial_manifest_shape(tmp_path: Path) -> None:
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

    assert manifest["phase"] == "2A-13E"
    assert manifest["dataset_name"] == "phase2a_1000_partial"
    assert manifest["partial_dataset"] is True
    assert manifest["included_verticals"] == INCLUDED_VERTICALS
    assert manifest["excluded_verticals"] == ["research_ai"]
    assert manifest["reason_excluded"] == "Research AI 1,000 generator pending"
    assert "total_prompt_count" in manifest
    assert "total_gold_count" in manifest
    assert "per_vertical" in manifest
    assert manifest["quality_summary"]["promotion_ready"] is True


def test_docs_include_partial_commands() -> None:
    docs = DOC_PATH.read_text(encoding="utf-8")

    assert "Phase 2A-13D" in docs
    assert "Phase 2A-13E" in docs
    assert "audit_phase2a_scaleup_1000_partial.py --run-audit" in docs
    assert "promote_phase2a_scaleup_1000_partial.py --promote" in docs
    assert "Research AI" in docs
    assert "promotion_ready" in docs
